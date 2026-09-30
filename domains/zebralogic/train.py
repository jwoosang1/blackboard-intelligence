"""
=============================================================================
Blackboard SFT Trainer
=============================================================================

Headless SFT trains the backbone logits to recover masked answer tokens.

Data:  Puzzle JSONs (puzzle + solution). Noise applied on-the-fly.
  - puzzles_train.json: training puzzles (different masking each epoch)
  - puzzles_eval.json:  val puzzles (fixed noise for reproducible loss)
"""

# --- repo path shim: make repo root importable so `core.* and datagen.*` resolves ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), _os.pardir, _os.pardir)))
# --- end shim ---


import argparse
import collections
import json
import math
import os
import random
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from pathlib import Path
from typing import List, Tuple, Dict, Any
from torch.nn.parallel import DistributedDataParallel as DDP

from omegaconf import OmegaConf

from domains.zebralogic.encode import build_tokenized_sample


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Blackboard SFT Trainer")
    p.add_argument("--config", type=str, default="configs/train.yaml",
                    help="OmegaConf YAML path.")
    p.add_argument(
        "overrides",
        nargs="*",
        help=("OmegaConf dotlist overrides. "
              "Example: lr=5e-5 num_epochs=3 use_wandb=true ddp=true"),
    )
    cli = p.parse_args()

    cfg_path = Path(cli.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    file_cfg = OmegaConf.load(cfg_path)
    if not OmegaConf.is_config(file_cfg):
        raise ValueError(f"Config must be a mapping: {cfg_path}")
    file_cfg_dict = OmegaConf.to_container(file_cfg, resolve=True)
    if not isinstance(file_cfg_dict, dict):
        raise ValueError(f"Config must be a mapping: {cfg_path}")

    cfg = file_cfg

    if cli.overrides:
        override_cfg = OmegaConf.from_dotlist(cli.overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise ValueError("Merged config is not a mapping.")
    cfg_dict["config"] = str(cfg_path)

    return argparse.Namespace(**cfg_dict)


# =============================================================================
# Distributed
# =============================================================================

def setup_distributed(enable_ddp):
    if enable_ddp and "RANK" in os.environ:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP needs CUDA.")
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_distributed(ws):
    if ws > 1 and dist.is_initialized():
        dist.destroy_process_group()


def seed_everything(seed, rank=0):
    s = seed + rank
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# =============================================================================
# Model
# =============================================================================

def build_model(args, rank=0):
    """
    Load LLaDA and attach LoRA adapters with the required initialization order:
      1. Load backbone (LLaDAModel) + tokenizer
      2. Apply LoRA
      3. Wrap in LLaDAModel
    """
    from core.model_utils import (
        _resolve_dtype, _apply_lora,
        _infer_hidden_size, _load_backbone_with_fallback, LLaDAModelBundle,
    )
    from core.model import LLaDAModel
    from transformers import AutoTokenizer

    dtype = _resolve_dtype(args.torch_dtype)
    lora_targets = tuple(
        x.strip() for x in args.lora_target_modules.split(",") if x.strip()
    )

    # ── 1. Load raw backbone + tokenizer ──
    if rank == 0:
        print("[model] Loading backbone...")
    backbone = _load_backbone_with_fallback(
        model_name=args.model_name,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype,
        return_dict=True,
        rank=rank,
    )
    backbone.config.output_hidden_states = True
    backbone.config.return_dict = True

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if rank == 0:
        print(f"[model] LoRA targets: {','.join(lora_targets)}")

    # ── 2. Apply LoRA ──
    backbone = _apply_lora(
        backbone=backbone,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=lora_targets,
    )

    # ── 3. Resolve mask token + wrap in LLaDAModel ──
    mask_token_id = int(args.mask_token_id)
    hidden_size = _infer_hidden_size(backbone)
    model = LLaDAModel(backbone=backbone, hidden_size=hidden_size)

    return LLaDAModelBundle(
        model=model,
        tokenizer=tokenizer,
        hidden_size=hidden_size,
        mask_token_id=int(mask_token_id),
    )


# =============================================================================
# Data
# =============================================================================

class ZebraPuzzleDataset(torch.utils.data.Dataset):
    """
    On-the-fly noise application.

    Stores fixed puzzles (puzzle + solution), applies random masking
    each time __getitem__ is called → different noise every epoch.

    For val: fixed_seed makes noise deterministic across epochs.
    """

    def __init__(
        self,
        puzzles: List[dict],
        tokenizer,
        mask_token_id: int,
        max_length: int = 512,
        t_range: Tuple[float, float] = (0.1, 0.9),
        fixed_seed: int = None,
        masking_only: bool = False,
    ):
        self.puzzles = puzzles
        self.tokenizer = tokenizer
        self.mask_token_id = mask_token_id
        self.max_length = max_length
        self.t_range = t_range
        self.fixed_seed = fixed_seed
        self.masking_only = masking_only

    def __len__(self):
        return len(self.puzzles)

    def __getitem__(self, idx):
        if self.fixed_seed is not None:
            random.seed(self.fixed_seed + idx)

        t = random.uniform(*self.t_range)
        import domains.zebralogic.encode as _bsd
        _bsd._MASKING_ONLY = self.masking_only
        sample = build_tokenized_sample(
            self.puzzles[idx], t, self.tokenizer,
            self.mask_token_id, self.max_length, sample_id=idx,
        )

        # Retry with different t if sample is None (e.g. truncation)
        if sample is None:
            sample = build_tokenized_sample(
                self.puzzles[idx], 0.5, self.tokenizer,
                self.mask_token_id, self.max_length, sample_id=idx,
            )
        if sample is None:
            # Return dummy (will be rare — only if puzzle > max_length)
            L = self.max_length
            return (
                torch.zeros(L, dtype=torch.long),
                torch.zeros(L, dtype=torch.long),
                torch.zeros(L, dtype=torch.bool),
                torch.full((L,), -100, dtype=torch.long),
            )

        return (
            torch.tensor(sample["ids"], dtype=torch.long),
            torch.tensor(sample["corrupted_ids"], dtype=torch.long),
            torch.tensor(sample["attention_mask"], dtype=torch.bool),
            torch.tensor(sample["labels"], dtype=torch.long),
        )


class WeightedDistributedSampler(torch.utils.data.Sampler):
    """
    Distributed weighted sampler with replacement.

    - Draws global samples according to `weights`.
    - Splits sampled indices across ranks so each rank gets `num_samples`.
    - Supports `set_epoch` for deterministic epoch-wise reshuffling.
    """

    def __init__(
        self,
        weights: List[float],
        num_replicas: int,
        rank: int,
        num_samples: int,
        replacement: bool = True,
        seed: int = 0,
    ):
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be > 0, got {num_replicas}")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")
        if num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, got {num_samples}")

        self.weights = torch.as_tensor(weights, dtype=torch.double)
        if self.weights.numel() == 0:
            raise ValueError("weights must be non-empty")
        if not torch.isfinite(self.weights).all():
            raise ValueError("weights must be finite")
        if (self.weights < 0).any():
            raise ValueError("weights must be non-negative")
        if float(self.weights.sum().item()) <= 0.0:
            raise ValueError("sum(weights) must be > 0")

        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.num_samples = int(num_samples)
        self.total_size = int(self.num_samples * self.num_replicas)
        self.replacement = bool(replacement)
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        sampled = torch.multinomial(
            self.weights,
            self.total_size,
            self.replacement,
            generator=g,
        )
        rank_indices = sampled[self.rank:self.total_size:self.num_replicas]
        return iter(rank_indices.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)


def build_dataloaders(args, world_size, rank, tokenizer, mask_token_id):
    # ── Load puzzle JSONs ──
    train_path = Path(args.train_puzzles)
    eval_path = Path(args.eval_puzzles)
    if not train_path.exists():
        raise FileNotFoundError(f"Not found: {train_path}")
    if not eval_path.exists():
        raise FileNotFoundError(f"Not found: {eval_path}")

    with open(train_path, "r") as f:
        train_puzzles = json.load(f)
    with open(eval_path, "r") as f:
        val_puzzles = json.load(f)

    if not train_puzzles:
        raise ValueError("Empty train puzzle set.")

    def _fingerprint(p: Dict[str, Any]) -> str:
        # Stable hash key for overlap checks across train/eval splits.
        return json.dumps(p, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    train_unique = {_fingerprint(p) for p in train_puzzles}
    val_unique = {_fingerprint(p) for p in val_puzzles}
    overlap_unique = len(train_unique & val_unique)

    split_stats = {
        "train_puzzles": len(train_puzzles),
        "val_puzzles": len(val_puzzles),
        "train_unique_puzzles": len(train_unique),
        "val_unique_puzzles": len(val_unique),
        "train_duplicate_puzzles": len(train_puzzles) - len(train_unique),
        "val_duplicate_puzzles": len(val_puzzles) - len(val_unique),
        "train_val_overlap_unique": overlap_unique,
        "train_val_overlap_ratio_val_unique": (
            overlap_unique / max(len(val_unique), 1)
        ),
    }

    if rank == 0:
        print(f"[data] train puzzles: {len(train_puzzles)}, "
              f"val puzzles: {len(val_puzzles)}")
        p0 = train_puzzles[0]
        print(f"  sample puzzle size: {p0.get('size', '?')}, "
              f"clues: {len(p0.get('clue_texts', []))}")
        print(
            f"  unique overlap(train∩val): {overlap_unique} "
            f"({split_stats['train_val_overlap_ratio_val_unique']*100:.2f}% of val unique)"
        )

    # ── Build datasets ──
    _masking_only = bool(getattr(args, 'masking_only', False))
    if rank == 0 and _masking_only:
        print("[data] masking_only=True: error injection disabled")

    train_ds = ZebraPuzzleDataset(
        train_puzzles, tokenizer, mask_token_id,
        max_length=args.max_length,
        t_range=(args.t_min, args.t_max),
        fixed_seed=None,          # random noise each epoch
        masking_only=_masking_only,
    )
    val_ds = ZebraPuzzleDataset(
        val_puzzles, tokenizer, mask_token_id,
        max_length=args.max_length,
        t_range=(args.t_min, args.t_max),
        fixed_seed=9999,          # deterministic for val
        masking_only=_masking_only,
    )

    # ── Difficulty-weighted sampling ──
    # Target effective ratio: easy 10% / med 20% / hard 40% / expert 30%
    DIFF_WEIGHTS = {
        "easy": 0.44,
        "med": 0.42,
        "hard": 1.0,
        "expert": 2.03,
    }

    def _get_diff_key(p):
        d = p.get("difficulty", p.get("size", ""))
        for key in ["expert", "hard", "med", "easy"]:
            if key in d.lower():
                return key
        return "med"  # fallback

    sample_weights = []
    for p in train_puzzles:
        dk = _get_diff_key(p)
        sample_weights.append(DIFF_WEIGHTS.get(dk, 0.5))

    sampler = None
    if world_size > 1:
        sampler = WeightedDistributedSampler(
            weights=sample_weights,
            num_replicas=world_size,
            rank=rank,
            num_samples=math.ceil(len(train_ds) / world_size),
            replacement=True,
            seed=int(getattr(args, "seed", 0)),
        )
    else:
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_ds),
            replacement=True,
        )

    if rank == 0:
        from collections import Counter
        dk_counts = Counter(_get_diff_key(p) for p in train_puzzles)
        eff_counts = {k: int(dk_counts[k] * DIFF_WEIGHTS.get(k, 0.5)) for k in dk_counts}
        print(f"[data] difficulty weights: {DIFF_WEIGHTS}")
        print(f"  raw counts: {dict(dk_counts)}")
        print(f"  effective ~ {eff_counts}")

    train_ld = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size_device,
        sampler=sampler, shuffle=False, drop_last=False,
        num_workers=0,            # tokenizer not fork-safe
    )
    val_ld = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size_device,
        shuffle=False, drop_last=False,
        num_workers=0,
    )

    if rank == 0:
        print(f"[data] train={len(train_ds)} val={len(val_ds)} "
              f"batches/epoch={len(train_ld)}")
    split_stats["train_batches_per_epoch"] = len(train_ld)
    split_stats["val_batches_per_eval"] = len(val_ld)

    return train_ld, val_ld, sampler, split_stats


# =============================================================================
# Loss
# =============================================================================


def compute_loss(model, batch, device, mask_token_id):
    """Cross-entropy on masked positions for the final headless SFT setup."""
    ids, _corrupted_ids, attn, labels = [b.to(device) for b in batch]
    attn = attn.bool()
    B, L = ids.shape
    bix = torch.arange(B, device=device).unsqueeze(1).expand(B, L)

    # ── Unmasking loss ──
    mask_pos = (ids == mask_token_id) & attn
    logits = model(input_ids=ids, attention_mask=attn)["logits"].float()
    n_m = mask_pos.sum(-1, keepdim=True).clamp_min(1.).float()

    if mask_pos.any():
        ce = F.cross_entropy(
            logits[mask_pos], labels[mask_pos], reduction="none"
        )
        u_loss = (ce / n_m.squeeze(-1)[bix[mask_pos]]).sum() / B
    else:
        u_loss = logits.new_zeros(())

    return u_loss


# =============================================================================
# LR Schedule: Linear Warmup + Cosine Decay
# =============================================================================

def get_lr(step, warmup_steps, max_lr, total_steps, min_lr_ratio=0.1):
    """Linear warmup for `warmup_steps`, then cosine decay to `max_lr * min_lr_ratio`."""
    min_lr = max_lr * min_lr_ratio

    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)

    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(progress, 1.0)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# =============================================================================
# Loss Tracker (smoothed logging)
# =============================================================================

class LossTracker:
    """Track smoothed training and validation loss."""

    def __init__(self, window=50):
        self.losses = collections.deque(maxlen=window)
        self.best_val_loss = float("inf")

    def update(self, loss):
        self.losses.append(loss)

    def smoothed(self):
        return sum(self.losses) / max(len(self.losses), 1)


# =============================================================================
# Validation
# =============================================================================

@torch.no_grad()
def run_val(model, val_ld, device, mask_id):
    model.eval()
    losses = [compute_loss(model, batch, device, mask_id).item() for batch in val_ld]
    model.train()
    return sum(losses) / max(len(losses), 1)


# =============================================================================
# Checkpoint: Save / Load / Resume
# =============================================================================

def _unwrap_model(model):
    """DDP -> LLaDAModel"""
    return model.module if isinstance(model, DDP) else model


def save_checkpoint(model, tokenizer, optimizer, out_dir, step, epoch,
                    args, is_main, is_best=False, max_keep=3):
    """
    Save checkpoint with:
      - LoRA adapter (via PEFT save_pretrained) OR full backbone state
      - Optimizer state
      - Tokenizer + training config
    Keeps only the latest `max_keep` checkpoints (best is never deleted).
    """
    if not is_main:
        return

    d = Path(out_dir) / f"step-{step}"
    d.mkdir(parents=True, exist_ok=True)

    m = _unwrap_model(model)

    # ── Save LoRA adapter (PEFT) or full backbone ──
    if args.use_lora and hasattr(m.backbone, "save_pretrained"):
        m.backbone.save_pretrained(d / "lora_adapter")
    else:
        torch.save(m.backbone.state_dict(), d / "backbone.pt")


    # ── Optimizer + training state ──
    torch.save({
        "optimizer": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
    }, d / "training_state.pt")

    # ── Tokenizer ──
    tokenizer.save_pretrained(d / "tokenizer")

    # ── Training config (for reproducibility) ──
    config = {k: str(v) if isinstance(v, Path) else v
              for k, v in vars(args).items()}
    config["saved_step"] = step
    config["saved_epoch"] = epoch
    with open(d / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ── Best model symlink ──
    if is_best:
        best_link = Path(out_dir) / "best"
        if best_link.is_symlink() or best_link.exists():
            if best_link.is_dir() and not best_link.is_symlink():
                shutil.rmtree(best_link)
            else:
                best_link.unlink()
        best_link.symlink_to(d.name)
        print(f"  [ckpt] ★ New best model -> {d.name}")

    print(f"  [ckpt] Saved step-{step} -> {d}")

    # ── Cleanup: keep only latest max_keep checkpoints ──
    if max_keep > 0:
        out_path = Path(out_dir)
        # Find best target (resolve symlink)
        best_link = out_path / "best"
        best_target = best_link.resolve().name if best_link.is_symlink() else None

        # Collect all step-N directories
        ckpt_dirs = sorted(
            [p for p in out_path.iterdir()
             if p.is_dir() and p.name.startswith("step-")],
            key=lambda p: int(p.name.split("-")[1])
        )

        # Never delete the best checkpoint
        deletable = [p for p in ckpt_dirs if p.name != best_target]

        # Keep latest max_keep (including best which is excluded from deletable)
        n_to_keep = max(max_keep - (1 if best_target else 0), 0)
        to_delete = (
            deletable if n_to_keep == 0
            else deletable[:-n_to_keep] if len(deletable) > n_to_keep else []
        )

        for old_dir in to_delete:
            shutil.rmtree(old_dir)
            print(f"  [ckpt] Removed old checkpoint: {old_dir.name}")


def load_checkpoint_for_resume(ckpt_dir, model, optimizer, device, rank=0):
    """
    Resume training from a checkpoint.

    Returns:
        (start_step, start_epoch) or (0, 0) if not found
    """
    d = Path(ckpt_dir)
    if not d.exists():
        if rank == 0:
            print(f"[resume] Checkpoint dir not found: {d}")
        return 0, 0

    m = _unwrap_model(model)

    # ── Load LoRA adapter ──
    lora_dir = d / "lora_adapter"
    backbone_pt = d / "backbone.pt"
    if lora_dir.exists():
        # Find adapter weights (safetensors or bin)
        adapter_state = None
        for fname in ["adapter_model.safetensors", "adapter_model.bin"]:
            fpath = lora_dir / fname
            if fpath.exists():
                if fname.endswith(".safetensors"):
                    from safetensors.torch import load_file
                    adapter_state = load_file(str(fpath))
                else:
                    adapter_state = torch.load(fpath, map_location="cpu")
                break

        if adapter_state is not None:
            # PEFT checkpoint keys omit the adapter name used by the in-memory
            # PeftModel. Its helper restores the saved adapter without silently
            # dropping LoRA tensors.
            from peft import set_peft_model_state_dict
            incompat = set_peft_model_state_dict(m.backbone, adapter_state)
            if rank == 0:
                unexpected = len(getattr(incompat, "unexpected_keys", []) or [])
                print(f"[resume] Loaded LoRA adapter from {lora_dir} "
                      f"({len(adapter_state)} tensors, unexpected={unexpected})")
        elif rank == 0:
            print(f"[resume] LoRA dir exists but no weights found: {lora_dir}")
    elif backbone_pt.exists():
        state = torch.load(backbone_pt, map_location="cpu")
        m.backbone.load_state_dict(state)
        if rank == 0:
            print(f"[resume] Loaded full backbone from {backbone_pt}")


    # ── Optimizer + training state ──
    state_pt = d / "training_state.pt"
    start_step, start_epoch = 0, 0
    if state_pt.exists():
        state = torch.load(state_pt, map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])
        start_step = state.get("step", 0)
        start_epoch = state.get("epoch", 0)
        if rank == 0:
            print(f"[resume] Restored optimizer, step={start_step}, epoch={start_epoch}")

    return start_step, start_epoch


# =============================================================================
# WandB
# =============================================================================

def init_wandb(args, model, rank):
    """Initialize WandB if requested (rank 0 only)."""
    if not args.use_wandb or rank != 0:
        return None

    try:
        import wandb
        project = getattr(args, "wandb_project", None)
        if project is None:
            project = getattr(args, "wandb_project_name", None)
        entity = getattr(args, "wandb_entity", None)

        run = wandb.init(
            project=project,
            entity=entity,
            name=args.wandb_run_name or args.job_name,
            config=vars(args),
            resume="allow",
        )
        m = _unwrap_model(model) if not isinstance(model, DDP) else model
        n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in m.parameters())
        wandb.config.update({
            "trainable_params": n_train,
            "total_params": n_total,
        }, allow_val_change=True)
        return run
    except ImportError:
        print("[warn] wandb not installed. pip install wandb")
        return None


def log_wandb(wandb_run, metrics: dict, step: int):
    if wandb_run is not None:
        import wandb
        wandb.log(metrics, step=step)


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    rank, ws, local_rank = setup_distributed(args.ddp)
    is_main = rank == 0
    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )
    seed_everything(args.seed, rank)

    try:
        import setproctitle
        setproctitle.setproctitle(f"blackboard-{args.job_name} [init]")
    except ImportError:
        setproctitle = None

    out = Path(args.output_dir) / args.job_name
    if is_main:
        out.mkdir(parents=True, exist_ok=True)
        print(f"{'='*60}")
        print(f"Blackboard SFT Trainer")
        print(f"{'='*60}")
        print(f"  output:   {out}")
        print(f"  device:   {device}")
        print(f"  dtype:    {args.torch_dtype}")
        print(f"  LoRA:     {args.use_lora} (r={args.lora_r})")
        print(f"  lr:       {args.lr} (warmup={args.warmup_steps}, cosine)")

    # ── Model ──
    bundle = build_model(args, rank)
    model = bundle.model.to(device)
    tok = bundle.tokenizer
    mask_id = bundle.mask_token_id

    if is_main:
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"  model:    {args.model_name}")
        print(f"  hidden:   {bundle.hidden_size}")
        print(f"  mask_id:  {mask_id}")
        print(f"  params:   {n_train:,} trainable / {n_total:,} total")

    # ── Data ──
    train_ld, val_ld, sampler, split_stats = build_dataloaders(
        args, ws, rank, tok, mask_id
    )

    # ── Sanity check: model vocab vs tokenizer vocab ──
    if is_main:
        # Find output projection (ff_out in LLaDA) by matching vocab size
        import torch.nn as _nn
        _embed = None
        for _name, _mod in model.named_modules():
            if isinstance(_mod, _nn.Embedding) and _mod.weight.shape[0] > 1000:
                _embed = _mod
                break
        lm_head = None
        if _embed is not None:
            for _name, _mod in model.named_modules():
                if isinstance(_mod, _nn.Linear) and _mod.weight.shape[0] == _embed.weight.shape[0]:
                    lm_head = _mod
                    break
        if lm_head is not None:
            model_vocab = lm_head.weight.shape[0]
            tok_vocab = len(tok)
            print(f"  vocab:    model={model_vocab}, tokenizer={tok_vocab}")
            if model_vocab < tok_vocab:
                raise RuntimeError(
                    f"FATAL: model output vocab ({model_vocab}) < tokenizer vocab "
                    f"({tok_vocab}). Labels will be out of range! "
                    f"Check that tokenizer and model backbone are from the same base model."
                )
        else:
            model_vocab = len(tok)
            print(f"  vocab:    lm_head not found, assuming {model_vocab}")
        # Check a sample batch
        sample_batch = next(iter(train_ld))
        # All release datasets return (ids, corrupted_ids, attention_mask, labels).
        labels = sample_batch[3]
        valid_labels = labels[labels != -100]
        if valid_labels.numel() > 0:
            max_label = valid_labels.max().item()
            print(f"  max_label_id: {max_label} (must be < {model_vocab})")
            if max_label >= model_vocab:
                raise RuntimeError(
                    f"FATAL: max label id ({max_label}) >= model vocab ({model_vocab}). "
                    f"Tokenizer/model vocab mismatch."
                )

    # ── DDP ──
    if ws > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # ── Optimizer ──
    trainable_params = [p for p in _unwrap_model(model).parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    if is_main:
        print(f"  optim:    AdamW (lr={args.lr})")

    # ── Total steps for cosine schedule ──
    steps_per_epoch = math.ceil(len(train_ld) / args.grad_accum_steps)
    total_steps = steps_per_epoch * args.num_epochs
    if is_main:
        print(f"  schedule: warmup({args.warmup_steps}) + cosine -> "
              f"{args.lr * args.min_lr_ratio:.1e}")
        print(f"  epochs:   {args.num_epochs}  steps/ep={steps_per_epoch}  "
              f"total={total_steps}")

    # ── Resume ──
    gs = 0
    start_epoch = 0
    if args.resume:
        gs, start_epoch = load_checkpoint_for_resume(
            args.resume, model, opt, device, rank
        )

    # ── WandB ──
    wandb_run = init_wandb(args, model, rank)

    # ── Mixed precision ──
    use_amp = (args.torch_dtype in ("bf16", "fp16") and device.type == "cuda")
    amp_dtype = torch.bfloat16 if args.torch_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
    if is_main:
        print(f"  amp:      {use_amp} ({amp_dtype if use_amp else 'N/A'})"
              f"{' +GradScaler' if scaler.is_enabled() else ''}")
        print(f"{'='*60}\n")

    # ── Training loop ──
    tracker = LossTracker(window=50)
    model.train()
    t0 = time.time()

    for ep in range(start_epoch, args.num_epochs):
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(ep)

        for itr, batch in enumerate(train_ld):
            # ── LR schedule ──
            lr = get_lr(gs, args.warmup_steps, args.lr, total_steps,
                        args.min_lr_ratio)
            opt.param_groups[0]["lr"] = lr

            # ── Forward + Loss ──
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    loss = compute_loss(model, batch, device, mask_id)
                    total = loss
            else:
                loss = compute_loss(model, batch, device, mask_id)
                total = loss

            scaled = total / args.grad_accum_steps
            if scaler.is_enabled():
                scaler.scale(scaled).backward()
            else:
                scaled.backward()

            # ── Skip until accumulation complete ──
            if (((itr + 1) % args.grad_accum_steps != 0)
                    and ((itr + 1) != len(train_ld))):
                continue

            # ── Gradient clip + step ──
            if args.max_grad_norm > 0:
                if scaler.is_enabled():
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm
                )
            if scaler.is_enabled():
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
            gs += 1

            if setproctitle is not None:
                setproctitle.setproctitle(
                    f"blackboard-{args.job_name} [ep{ep+1}/{args.num_epochs} step{gs}/{total_steps}]"
                )

            tracker.update(total.item())

            # ── Logging ──
            if is_main and gs % args.logging_steps == 0:
                st = tracker.smoothed()
                elapsed = time.time() - t0
                samples_sec = (
                    (gs * args.batch_size_device * ws) / max(elapsed, 1)
                )
                print(
                    f"[train] ep={ep+1} step={gs}/{total_steps} lr={lr:.2e} "
                    f"loss={st:.4f} "
                    f"({samples_sec:.1f} samp/s)"
                )
                log_wandb(wandb_run, {
                    "train/loss": st,
                    "train/lr": lr,
                    "train/samples_per_sec": samples_sec,
                }, gs)

            # ── Eval loss + checkpoint ──
            if (args.eval_steps > 0 and gs % args.eval_steps == 0):
                # Val loss
                is_best = False
                if is_main:
                    vt = run_val(model, val_ld, device, mask_id)
                    is_best = vt < tracker.best_val_loss
                    if is_best:
                        tracker.best_val_loss = vt
                    print(
                        f"[val] step={gs} loss={vt:.4f} {'★ BEST' if is_best else ''}"
                    )
                    log_wandb(wandb_run, {
                        "val/loss": vt,
                    }, gs)

                # Save checkpoint
                save_checkpoint(
                    model, tok, opt, out, gs, ep + 1,
                    args, is_main, is_best,
                    max_keep=args.max_keep_ckpts
                )

        # ── End of epoch: eval + save ──
        is_best = False
        if is_main:
            vt = run_val(model, val_ld, device, mask_id)
            is_best = vt < tracker.best_val_loss
            if is_best:
                tracker.best_val_loss = vt
            print(
                f"\n[epoch {ep+1} done] val loss={vt:.4f} {'★ BEST' if is_best else ''}"
            )
            save_checkpoint(
                model, tok, opt, out, gs, ep + 1,
                args, is_main, is_best,
                max_keep=args.max_keep_ckpts
            )

    # ── Final ──
    if is_main:
        elapsed = time.time() - t0
        print(f"\n{'='*60}")
        print(f"Training complete: {gs} steps in {elapsed/60:.1f} min")
        print(f"Best val loss: {tracker.best_val_loss:.4f}")
        print(f"Output: {out}")
        print(f"{'='*60}")

    if wandb_run is not None:
        import wandb
        wandb.finish()

    cleanup_distributed(ws)


if __name__ == "__main__":
    main()
