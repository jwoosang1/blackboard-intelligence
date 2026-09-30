"""
=============================================================================
Blackboard SFT Trainer — Nurse Rostering
=============================================================================

Headless LLaDA SFT:
  - lm_head: predict [MASK] -> correct value

Loss: cross-entropy on masked answer tokens

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

from domains.nurse_rostering.encode import build_tokenized_sample_gc, GcDataset, decode_cell_value, gc_to_solution, DOOM_LABEL, COLORS
from core.table_encoding import encode_cell_value

PALETTE_IDS = None   # set in main(): token ids of the k color names, for doomed uniform-target loss


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

    backbone = _apply_lora(
        backbone=backbone,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=lora_targets,
    )

    # Optional gradient checkpointing (big activation-memory saver for OOM).
    if getattr(args, "gradient_checkpointing", False):
        backbone.config.use_cache = False
        if hasattr(backbone, "gradient_checkpointing_enable"):
            backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(backbone, "enable_input_require_grads"):
            backbone.enable_input_require_grads()
        if rank == 0:
            print("[model] gradient checkpointing: ON (use_reentrant=False)")

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

# TspDataset is imported


class WeightedDistributedSampler(torch.utils.data.Sampler):
    def __init__(self, weights, num_replicas, rank, num_samples,
                 replacement=True, seed=0):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
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
        sampled = torch.multinomial(self.weights, self.total_size,
                                    self.replacement, generator=g)
        rank_indices = sampled[self.rank:self.total_size:self.num_replicas]
        return iter(rank_indices.tolist())

    def __len__(self): return self.num_samples
    def set_epoch(self, epoch): self.epoch = int(epoch)


def _grid_dim(p):
    """Difficulty proxy for NR: number of grid cells (staff x days)."""
    size = p.get("size")
    if isinstance(size, str) and "x" in size:
        a, b = size.split("x")
        return int(a) * int(b)
    return int(size) if isinstance(size, int) else 0


def build_dataloaders(args, world_size, rank, tokenizer, mask_token_id):
    train_path = Path(args.train_puzzles)
    eval_path  = Path(args.eval_puzzles)
    if not train_path.exists(): raise FileNotFoundError(f"Not found: {train_path}")
    if not eval_path.exists():  raise FileNotFoundError(f"Not found: {eval_path}")

    with open(train_path) as f: train_puzzles = json.load(f)
    with open(eval_path)  as f: val_puzzles   = json.load(f)

    if not train_puzzles: raise ValueError("Empty train puzzle set.")

    # Defensive: drop over-length instances explicitly (else the dataset would
    # silently emit an all -100 zero-tensor for them). NR expert prompts (many
    # clues) can exceed max_length; log the count per split so it is auditable.
    def _fits(p):
        return build_tokenized_sample_gc(p, 0.5, tokenizer, mask_token_id,
                                         args.max_length) is not None
    def _filter(puzzles, name):
        kept = [p for p in puzzles if _fits(p)]
        n_drop = len(puzzles) - len(kept)
        if rank == 0 and n_drop:
            from collections import Counter as _C
            by = _C(p.get("difficulty", "?") for p in puzzles if not _fits(p))
            print(f"[data] {name}: dropped {n_drop}/{len(puzzles)} over-length "
                  f"(> {args.max_length} tok) by config: {dict(by)}")
        return kept
    train_puzzles = _filter(train_puzzles, "train")
    val_puzzles   = _filter(val_puzzles, "val")
    if not train_puzzles: raise ValueError("All train puzzles exceed max_length.")

    def _fingerprint(p):
        return json.dumps(p, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    train_unique   = {_fingerprint(p) for p in train_puzzles}
    val_unique     = {_fingerprint(p) for p in val_puzzles}
    overlap_unique = len(train_unique & val_unique)

    split_stats = {
        "train_puzzles": len(train_puzzles),
        "val_puzzles":   len(val_puzzles),
        "train_val_overlap_unique": overlap_unique,
        "train_val_overlap_ratio_val_unique": overlap_unique / max(len(val_unique), 1),
    }

    if rank == 0:
        from collections import Counter as _Counter
        sizes = dict(sorted(_Counter(_grid_dim(p) for p in train_puzzles).items()))
        print(f"[data] domain=nr  train={len(train_puzzles)}  val={len(val_puzzles)}")
        print(f"  sizes: {sizes}")
        print(f"  unique overlap(train∩val): {overlap_unique} "
              f"({split_stats['train_val_overlap_ratio_val_unique']*100:.2f}% of val unique)")

    train_ds = GcDataset(
        train_puzzles, tokenizer, mask_token_id,
        max_length=args.max_length, t_range=(args.t_min, args.t_max), fixed_seed=None,
        augment_colors=bool(getattr(args, "color_permute_augment", False)),
        doom_frac=float(getattr(args, "doom_negative_frac", 0.0)))
    val_ds = GcDataset(
        val_puzzles, tokenizer, mask_token_id,
        max_length=args.max_length, t_range=(args.t_min, args.t_max), fixed_seed=9999)

    # size-weighted sampling: bigger grids are harder
    DIFF_WEIGHTS = {"easy": 0.5, "med": 0.7, "hard": 1.0}

    def _get_diff_key(p):
        cells = _grid_dim(p)
        if cells <= 15:  return "easy"
        if cells <= 28:  return "med"
        return "hard"

    sample_weights = [DIFF_WEIGHTS.get(_get_diff_key(p), 0.5) for p in train_puzzles]

    if world_size > 1:
        sampler = WeightedDistributedSampler(
            weights=sample_weights, num_replicas=world_size, rank=rank,
            num_samples=math.ceil(len(train_ds) / world_size),
            replacement=True, seed=int(getattr(args, "seed", 0)))
    else:
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights, num_samples=len(train_ds), replacement=True)

    if rank == 0:
        from collections import Counter
        dk_counts = Counter(_get_diff_key(p) for p in train_puzzles)
        print(f"[data] size weights: {DIFF_WEIGHTS}")
        print(f"  counts: {dict(dk_counts)}")

    train_ld = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size_device,
        sampler=sampler, shuffle=False, drop_last=False, num_workers=0)
    val_ld = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size_device,
        shuffle=False, drop_last=False, num_workers=0)

    if rank == 0:
        print(f"[data] train={len(train_ds)} val={len(val_ds)} batches/epoch={len(train_ld)}")

    split_stats["train_batches_per_epoch"] = len(train_ld)
    split_stats["val_batches_per_eval"]    = len(val_ld)
    return train_ld, val_ld, sampler, split_stats


# =============================================================================
# Loss
# =============================================================================


def compute_loss(model, batch, device, mask_token_id):
    """Cross-entropy on masked positions for the final headless SFT setup."""
    ids, _corrupted_ids, attn, _rm_idx, labels, _viol_lbl, _timestep = [b.to(device) for b in batch]
    attn = attn.bool()
    B, L   = ids.shape
    bix    = torch.arange(B, device=device).unsqueeze(1).expand(B, L)

    mask_pos = (ids == mask_token_id) & attn
    logits   = model(input_ids=ids, attention_mask=attn)["logits"].float()
    n_m      = mask_pos.sum(-1, keepdim=True).clamp_min(1.).float()

    ce_pos   = mask_pos & (labels >= 0)
    doom_pos = mask_pos & (labels == DOOM_LABEL)
    u_loss   = logits.sum() * 0.0
    if ce_pos.any():
        ce     = F.cross_entropy(logits[ce_pos], labels[ce_pos], reduction="none")
        u_loss = u_loss + (ce / n_m.squeeze(-1)[bix[ce_pos]]).sum() / B
    if doom_pos.any() and PALETTE_IDS is not None:
        # infeasible cells -> push toward UNIFORM over the k palette colors (low max-softmax):
        # CE to uniform-over-palette = -mean_c log p(c)
        logp = F.log_softmax(logits[doom_pos], dim=-1)
        du   = -logp[:, PALETTE_IDS].mean(dim=-1)
        u_loss = u_loss + (du / n_m.squeeze(-1)[bix[doom_pos]]).sum() / B

    return u_loss


# =============================================================================
# LR Schedule
# =============================================================================

def get_lr(step, warmup_steps, max_lr, total_steps, min_lr_ratio=0.1):
    min_lr = max_lr * min_lr_ratio
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    progress = min((step - warmup_steps) / max(total_steps - warmup_steps, 1), 1.0)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# =============================================================================
# Loss Tracker
# =============================================================================

class LossTracker:
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
    """Average the headless SFT loss over the validation loader."""
    model.eval()
    total, n = 0.0, 0
    for batch in val_ld:
        loss = compute_loss(model, batch, device, mask_id)
        total += loss.item()
        n += 1
    model.train()
    return total / max(n, 1)




# =============================================================================
# Checkpoint
# =============================================================================

def _unwrap_model(model):
    """Return the underlying model when distributed training wraps it in DDP."""
    return model.module if isinstance(model, DDP) else model


def save_checkpoint(model, tokenizer, optimizer, out_dir, step, epoch,
                    args, is_main, is_best=False, max_keep=10):
    if not is_main: return
    d = Path(out_dir) / f"step-{step}"
    d.mkdir(parents=True, exist_ok=True)
    m = _unwrap_model(model)
    if args.use_lora and hasattr(m.backbone, "save_pretrained"):
        m.backbone.save_pretrained(d / "lora_adapter")
    else:
        torch.save(m.backbone.state_dict(), d / "backbone.pt")
    torch.save({"optimizer": optimizer.state_dict(), "step": step, "epoch": epoch},
               d / "training_state.pt")
    tokenizer.save_pretrained(d / "tokenizer")
    with open(d / "config.json", "w") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v
                   for k, v in vars(args).items()}, f, indent=2)
    if is_best:
        best = Path(out_dir) / "best"
        if best.is_symlink() or best.exists():
            best.unlink() if not (best.is_dir() and not best.is_symlink()) else shutil.rmtree(best)
        best.symlink_to(d.name)
        print(f"  [ckpt] ★ New best model -> {d.name}")
    print(f"  [ckpt] Saved step-{step} -> {d}")
    if max_keep > 0:
        out_path    = Path(out_dir)
        best_link   = out_path / "best"
        best_target = best_link.resolve().name if best_link.is_symlink() else None
        ckpt_dirs   = sorted(
            [p for p in out_path.iterdir() if p.is_dir() and p.name.startswith("step-")],
            key=lambda p: int(p.name.split("-")[1]))
        deletable   = [p for p in ckpt_dirs if p.name != best_target]
        n_to_keep   = max(max_keep - (1 if best_target else 0), 0)
        to_delete   = (
            deletable if n_to_keep == 0
            else deletable[:-n_to_keep] if len(deletable) > n_to_keep else []
        )
        for old in to_delete:
            shutil.rmtree(old)
            print(f"  [ckpt] Removed old checkpoint: {old.name}")


def load_checkpoint_for_resume(ckpt_dir, model, optimizer, device, rank=0):
    d = Path(ckpt_dir)
    if not d.exists():
        if rank == 0: print(f"[resume] Not found: {d}")
        return 0, 0
    m = _unwrap_model(model)
    lora_dir = d / "lora_adapter"
    if lora_dir.exists():
        for fname in ["adapter_model.safetensors", "adapter_model.bin"]:
            fp = lora_dir / fname
            if fp.exists():
                from safetensors.torch import load_file
                state = load_file(str(fp)) if fname.endswith(".safetensors") else \
                        torch.load(fp, map_location="cpu")
                # NOTE: PEFT's save_pretrained writes keys as "...lora_A.weight",
                # but PeftModel.state_dict() expects "...lora_A.<adapter>.weight".
                # A plain load_state_dict(strict=False) therefore silently drops EVERY
                # LoRA tensor and the run resumes from a FRESH adapter (loss spike).
                # set_peft_model_state_dict does the key remapping correctly.
                from peft import set_peft_model_state_dict
                incompat = set_peft_model_state_dict(m.backbone, state)
                n_missing = len(getattr(incompat, "unexpected_keys", []) or [])
                if rank == 0:
                    print(f"[resume] LoRA restored from {fp.name} "
                          f"({len(state)} tensors, unexpected={n_missing})")
                break
    elif (d / "backbone.pt").exists():
        m.backbone.load_state_dict(torch.load(d / "backbone.pt", map_location="cpu"))
    state_pt = d / "training_state.pt"
    if state_pt.exists():
        state = torch.load(state_pt, map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])
        if rank == 0: print(f"[resume] step={state.get('step',0)} epoch={state.get('epoch',0)}")
        return state.get("step", 0), state.get("epoch", 0)
    return 0, 0


# =============================================================================
# WandB
# =============================================================================

def init_wandb(args, model, rank):
    if not args.use_wandb or rank != 0: return None
    try:
        import wandb
        run = wandb.init(
            project=getattr(args, "wandb_project", "blackboard-intelligence"),
            entity=getattr(args, "wandb_entity", None),
            name=args.wandb_run_name or args.job_name,
            config=vars(args), resume="allow")
        m = _unwrap_model(model)
        wandb.config.update({
            "trainable_params": sum(p.numel() for p in m.parameters() if p.requires_grad),
            "total_params":     sum(p.numel() for p in m.parameters()),
        }, allow_val_change=True)
        return run
    except ImportError:
        print("[warn] wandb not installed"); return None


def log_wandb(run, metrics, step):
    if run is not None:
        import wandb; wandb.log(metrics, step=step)


# =============================================================================
# Main
# =============================================================================

def main():
    args       = parse_args()
    rank, ws, local_rank = setup_distributed(args.ddp)
    is_main    = rank == 0
    device     = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
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
        print(f"Blackboard SFT Trainer — Nurse Rostering")
        print(f"{'='*60}")
        print(f"  output: {out}  device: {device}  dtype: {args.torch_dtype}")
        print(f"  LoRA: r={args.lora_r}  lr={args.lr} ")

    bundle  = build_model(args, rank)
    model   = bundle.model.to(device)
    tok     = bundle.tokenizer
    mask_id = bundle.mask_token_id

    global PALETTE_IDS
    _k = 3
    PALETTE_IDS = torch.tensor([int(encode_cell_value(c, tok)) for c in COLORS[:_k]],
                               dtype=torch.long, device=device)

    if is_main:
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"  model: {args.model_name}")
        print(f"  model: {args.model_name}")
        print(f"  params: {n_train:,} trainable / {n_total:,} total")

    train_ld, val_ld, sampler, split_stats = build_dataloaders(args, ws, rank, tok, mask_id)


    # Vocab sanity check
    if is_main:
        import torch.nn as _nn
        lm_head = None
        for _, mod in model.named_modules():
            if isinstance(mod, _nn.Linear) and mod.weight.shape[0] > 10000:
                lm_head = mod; break
        if lm_head:
            model_vocab = lm_head.weight.shape[0]
            print(f"  vocab: model={model_vocab}, tokenizer={len(tok)}")

    if ws > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    trainable_params = [p for p in _unwrap_model(model).parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    steps_per_epoch = math.ceil(len(train_ld) / args.grad_accum_steps)
    total_steps     = steps_per_epoch * args.num_epochs
    if is_main:
        print(f"  epochs: {args.num_epochs}  steps/ep: {steps_per_epoch}  total: {total_steps}")
        print(f"{'='*60}\n")

    gs, start_epoch = 0, 0
    if args.resume:
        gs, start_epoch = load_checkpoint_for_resume(args.resume, model, opt, device, rank)

    wandb_run = init_wandb(args, model, rank)

    use_amp   = (args.torch_dtype in ("bf16", "fp16") and device.type == "cuda")
    amp_dtype = torch.bfloat16 if args.torch_dtype == "bf16" else torch.float16
    scaler    = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    tracker = LossTracker(window=50)
    model.train()
    t0 = time.time()

    for ep in range(start_epoch, args.num_epochs):
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(ep)

        for itr, batch in enumerate(train_ld):
            lr = get_lr(gs, args.warmup_steps, args.lr, total_steps, args.min_lr_ratio)
            opt.param_groups[0]["lr"] = lr

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    loss = compute_loss(model, batch, device, mask_id)
                    total = loss
            else:
                loss = compute_loss(model, batch, device, mask_id)
                total = loss

            (scaler.scale(total / args.grad_accum_steps) if scaler.is_enabled()
             else (total / args.grad_accum_steps)).backward()

            if (itr + 1) % args.grad_accum_steps != 0 and (itr + 1) != len(train_ld):
                continue

            if args.max_grad_norm > 0:
                if scaler.is_enabled(): scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            if scaler.is_enabled():
                scaler.step(opt); scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
            gs += 1

            if setproctitle:
                setproctitle.setproctitle(
                    f"blackboard-{args.job_name} [ep{ep+1}/{args.num_epochs} step{gs}/{total_steps}]")

            tracker.update(total.item())

            # Logging
            if is_main and gs % args.logging_steps == 0:
                st = tracker.smoothed()
                elapsed = time.time() - t0
                sps = (gs * args.batch_size_device * ws) / max(elapsed, 1)
                print(f"[train] ep={ep+1} step={gs}/{total_steps} lr={lr:.2e} "
                      f"loss={st:.4f} ({sps:.1f} samp/s)")
                log_wandb(wandb_run, {
                    "train/loss": st, "train/lr": lr,
                    "train/samples_per_sec": sps,
                }, gs)

            # Val loss + checkpoint
            if args.eval_steps > 0 and gs % args.eval_steps == 0:
                is_best = False
                if is_main:
                    vt = run_val(model, val_ld, device, mask_id)
                    is_best = vt < (tracker.best_val_loss if tracker.best_val_loss > 0
                                    else float("inf"))
                    print(f"[val] step={gs} loss={vt:.4f} {'★ BEST' if is_best else ''}")
                    log_wandb(wandb_run, {
                        "val/loss": vt
                    }, gs)
                save_checkpoint(model, tok, opt, out, gs, ep+1,
                                args, is_main, is_best, args.max_keep_ckpts)

        # End of epoch
        if is_main:
            vt = run_val(model, val_ld, device, mask_id)
            is_best = vt < (tracker.best_val_loss if tracker.best_val_loss > 0
                            else float("inf"))
            if is_best: tracker.best_val_loss = vt
            print(f"\n[epoch {ep+1} done] val loss={vt:.4f} {'★ BEST' if is_best else ''}")
            save_checkpoint(model, tok, opt, out, gs, ep+1,
                            args, is_main, is_best, args.max_keep_ckpts)

    if is_main:
        elapsed = time.time() - t0
        print(f"\n{'='*60}")
        print(f"Training complete: {gs} steps in {elapsed/60:.1f} min")
        print(f"Output: {out}")

    if wandb_run:
        import wandb; wandb.finish()
    cleanup_distributed(ws)


if __name__ == "__main__":
    main()
