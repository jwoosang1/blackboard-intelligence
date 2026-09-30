"""
=============================================================================
LLaDA Discrete Diffusion Inference Engine
=============================================================================

Generation goals:
  1) unmask a small number of tokens each step
  2) optionally stop early when generated grid matches ground truth
 when generated grid matches ground-truth grid
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional

import torch
import torch.nn.functional as F


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Gumbel noise for stochastic sampling."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    u = torch.rand_like(logits, dtype=torch.float64)
    g = -torch.log(-torch.log(u + 1e-20) + 1e-20)
    return logits + g * temperature


@dataclasses.dataclass
class GenerationConfig:
    """Configuration for confidence-ranked any-order decoding."""
    num_unmask: int = 1
    temperature: float = 0.0
    max_steps: int = 100
    stop_on_solve: bool = False
    track_trajectory: bool = False
    track_confidence: bool = False
    track_mask_counts: bool = False

@dataclasses.dataclass
class GenerationResult:
    output_ids: torch.LongTensor          # (B, Lp + gen_length)
    grid_ids: torch.LongTensor            # (B, gen_length)
    n_steps_run: int
    success: torch.FloatTensor            # (B,) 1.0 if solved, else 0.0
    success_step: torch.LongTensor        # (B,) first solved step, -1 if never
    trajectory: Optional[List[torch.LongTensor]] = None
    confidence_history: Optional[List[torch.Tensor]] = None
    mask_count_history: Optional[List[torch.LongTensor]] = None


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt_ids: torch.LongTensor,
    gen_length: int,
    mask_token_id: int,
    config: Optional[GenerationConfig] = None,
    pad_token_id: Optional[int] = None,
    target_grid_ids: Optional[torch.LongTensor] = None,
    prefill_ids: Optional[torch.LongTensor] = None,
) -> GenerationResult:
    """
    generation with optional early-stop on exact target match.

    If prefill_ids is provided (B, gen_length), use it to initialize
    the generation region instead of all [MASK]. Structure tokens are
    pre-filled; only [MASK] positions are generated.

    If target_grid_ids is provided and config.stop_on_solve=True:
      - stop when solved (single sample) or when all solved (batch).
      - mark success=1.0 for solved rows.
    """
    if config is None:
        config = GenerationConfig()

    device = prompt_ids.device
    B, Lp = prompt_ids.shape
    L_total = Lp + gen_length
    if target_grid_ids is not None:
        if target_grid_ids.dim() != 2:
            raise ValueError("target_grid_ids must be shape (B, gen_length)")
        if target_grid_ids.size(0) != B or target_grid_ids.size(1) != gen_length:
            raise ValueError(
                "target_grid_ids shape mismatch: "
                f"expected ({B}, {gen_length}), got {tuple(target_grid_ids.shape)}"
            )
        target_grid_ids = target_grid_ids.to(device)

    # prompt + generation region (prefilled or fully masked)
    x = torch.full((B, L_total), mask_token_id, dtype=torch.long, device=device)
    x[:, :Lp] = prompt_ids
    # Structure tokens are prefilled; only masked cells are decoded.
    if prefill_ids is not None:
        # Structure tokens pre-filled, only cell positions remain [MASK]
        x[:, Lp:] = prefill_ids.to(device)
    attn_mask = torch.ones((B, L_total), dtype=torch.bool, device=device)
    if pad_token_id is not None:
        attn_mask[:, :Lp] = (prompt_ids != pad_token_id)

    trajectory = [] if config.track_trajectory else None
    confidence_history = [] if config.track_confidence else None
    mask_count_history = [] if config.track_mask_counts else None

    success = torch.zeros(B, dtype=torch.float32, device=device)
    success_step = torch.full((B,), -1, dtype=torch.long, device=device)

    steps_run = 0
    while steps_run < config.max_steps:
        steps_run += 1

        out = model(input_ids=x, attention_mask=attn_mask)
        gen_logits = out["logits"][:, Lp:, :]
        logits_with_noise = add_gumbel_noise(gen_logits, config.temperature)
        preds = torch.argmax(logits_with_noise, dim=-1)
        masked_index = (x[:, Lp:] == mask_token_id)
        clean_index = ~masked_index

        # ── [2] Unmasking ──
        p = F.softmax(gen_logits, dim=-1)
        unmasking_score = torch.gather(
            p,
            dim=-1,
            index=preds.unsqueeze(-1),
        ).squeeze(-1).float()
        unmasking_score = unmasking_score.clone()
        unmasking_score[clean_index] = -float("inf")

        # unmask up to config.num_unmask per row
        for j in range(B):
            n_masked = int(masked_index[j].sum().item())
            if n_masked <= 0:
                continue
            n_unmask = n_masked if config.num_unmask <= 0 else min(config.num_unmask, n_masked)
            _, top_unmask_idx = torch.topk(unmasking_score[j], k=n_unmask, dim=-1)
            x[j, Lp + top_unmask_idx] = preds[j, top_unmask_idx]

        masked_after_unmask = (x[:, Lp:] == mask_token_id)

        if mask_count_history is not None:
            mask_count_history.append(masked_after_unmask.sum(dim=1).detach().cpu())

        # ── [4] Diagnostics ──
        if trajectory is not None:
            trajectory.append(x[:, Lp:].clone().cpu())
        if confidence_history is not None:
            confidence_history.append(unmasking_score.clone().cpu())

        if target_grid_ids is not None:
            solved_now = (x[:, Lp:] == target_grid_ids).all(dim=1)
            newly_solved = solved_now & (success == 0)
            success[newly_solved] = 1.0
            success_step[newly_solved] = steps_run

            # user-requested behavior:
            # stop early once puzzle is solved in-loop
            if config.stop_on_solve:
                if B == 1 and bool(solved_now[0].item()):
                    break
                if B > 1 and bool(solved_now.all().item()):
                    break

        if not bool(masked_after_unmask.any().item()):
            break

    grid_ids = x[:, Lp:].clone()
    if target_grid_ids is not None:
        solved_final = (grid_ids == target_grid_ids).all(dim=1)
        newly_solved = solved_final & (success == 0)
        success[newly_solved] = 1.0
        success_step[newly_solved] = steps_run

    return GenerationResult(
        output_ids=x,
        grid_ids=grid_ids,
        n_steps_run=steps_run,
        success=success.detach().cpu(),
        success_step=success_step.detach().cpu(),
        trajectory=trajectory,
        confidence_history=confidence_history,
        mask_count_history=mask_count_history,
    )


def _build_prefill(body_toks, cell_map, mask_token_id):
    """Build prefill: body tokens with cell positions replaced by [MASK]."""
    prefill = list(body_toks)
    cell_positions = {cm["token_pos"] for cm in cell_map}
    for pos in cell_positions:
        if pos < len(prefill):
            prefill[pos] = mask_token_id
    return prefill




def generate_batch(
    model: torch.nn.Module,
    prompt_ids_list: List[torch.LongTensor],
    gen_lengths: List[int],
    mask_token_id: int,
    config: Optional[GenerationConfig] = None,
    pad_token_id: Optional[int] = None,
) -> List[GenerationResult]:
    return [
        generate(
            model=model,
            prompt_ids=prompt_ids,
            gen_length=gen_length,
            mask_token_id=mask_token_id,
            config=config,
            pad_token_id=pad_token_id,
        )
        for prompt_ids, gen_length in zip(prompt_ids_list, gen_lengths)
    ]
