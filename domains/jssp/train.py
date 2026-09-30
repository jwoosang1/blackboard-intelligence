"""JSSP training entry point using the shared table-SFT optimization loop."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from domains.zebralogic import train as _shared_trainer
from domains.jssp.encode import JSSPDataset


def _build_dataloaders(args, world_size, rank, tokenizer, mask_token_id):
    """Build JSSP loaders using the JSSP table encoder.

    The optimizer, loss, checkpoint, and distributed-training code is shared
    with ZebraLogic. Only the table representation is domain-specific.
    """
    train_path = Path(args.train_puzzles)
    eval_path = Path(args.eval_puzzles)
    if not train_path.exists():
        raise FileNotFoundError(f"Not found: {train_path}")
    if not eval_path.exists():
        raise FileNotFoundError(f"Not found: {eval_path}")

    with train_path.open() as f:
        train_puzzles = json.load(f)
    with eval_path.open() as f:
        eval_puzzles = json.load(f)
    if not train_puzzles:
        raise ValueError("Empty train puzzle set.")

    train_ds = JSSPDataset(
        train_puzzles, tokenizer, mask_token_id,
        max_length=args.max_length, t_range=(args.t_min, args.t_max),
    )
    eval_ds = JSSPDataset(
        eval_puzzles, tokenizer, mask_token_id,
        max_length=args.max_length, t_range=(args.t_min, args.t_max),
        fixed_seed=9999,
    )
    if world_size > 1:
        sampler = torch.utils.data.DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True,
            seed=int(args.seed), drop_last=False,
        )
    else:
        sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size_device, sampler=sampler,
        shuffle=sampler is None, drop_last=False, num_workers=0,
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_ds, batch_size=args.batch_size_device, shuffle=False,
        drop_last=False, num_workers=0,
    )
    if rank == 0:
        print(f"[data] domain=jssp train={len(train_ds)} eval={len(eval_ds)} "
              f"batches/epoch={len(train_loader)}")
    return train_loader, eval_loader, sampler, {
        "train_puzzles": len(train_ds),
        "val_puzzles": len(eval_ds),
        "train_batches_per_epoch": len(train_loader),
        "val_batches_per_eval": len(eval_loader),
    }


if __name__ == "__main__":
    if "--config" not in sys.argv:
        sys.argv[1:1] = ["--config", "configs/train_jssp.yaml"]
    _shared_trainer.build_dataloaders = _build_dataloaders
    _shared_trainer.main()
