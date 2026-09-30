"""
=============================================================================
JSSP Evaluation  (eval_jssp.py)
=============================================================================

JSSP version of run_evaluation.
- Uses encode_prompt_jssp / encode_table_body_jssp
- JSSP-specific metrics: solve_rate, row_accuracy, cell_accuracy, makespan_ratio

"""

# --- repo path shim: make repo root importable so `core.* and datagen.*` resolves ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), _os.pardir, _os.pardir)))
# --- end shim ---


import collections
import re
from typing import Dict, List, Optional

import torch

# ── lazy imports (heavy) ──────────────────────────────────────────────────────
try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None


# =============================================================================
# 1. JSSP Metrics
# =============================================================================

def _parse_grid_from_ids(
    grid_ids:      List[int],
    cell_map:      List[Dict],
    tokenizer,
    mask_token_id: int,
) -> Dict[int, List[Optional[int]]]:
    """
    grid_ids (body only) + cell_map → {machine_idx: [job_id or None, ...]}
    job_id = int('3') = 3, None = masked or unparseable
    """
    grid: Dict[int, List[Optional[int]]] = collections.defaultdict(list)
    for cm in sorted(cell_map, key=lambda x: (x["row"], x["col"])):
        pos = cm["token_pos"]
        tid = grid_ids[pos] if pos < len(grid_ids) else mask_token_id
        if tid == mask_token_id:
            grid[cm["row"]].append(None)
        else:
            raw = tokenizer.convert_ids_to_tokens(tid)
            val = raw.lstrip("\u0120").strip() if raw else None
            try:
                grid[cm["row"]].append(int(val))
            except (ValueError, TypeError):
                grid[cm["row"]].append(None)
    return dict(grid)


def _simulate_makespan(grid: Dict[int, List[Optional[int]]], puzzle: Dict) -> Optional[int]:
    """
    compact grid → actual makespan.
    grid: {machine: [job_id, ...]} (no None allowed)
    Returns None if grid is incomplete or invalid.

    Jacobi iteration (using f_old): at the start of each iteration, preserve the
    previous values and use only the old values to compute job_prev → guarantees convergence.
    (A Gauss-Seidel style that references values updated within the same iteration may diverge.)
    """
    import re as _re

    clue_texts = puzzle.get("clue_texts", [])
    n_jobs     = puzzle["n_jobs"]
    n_machines = puzzle["n_machines"]

    # Parse job routes + durations from the canonical natural-language format.
    job_routes: Dict[int, List] = {}
    for clue in clue_texts:
        # "Job 0 requires: Machine 3 for 9 time units, ..."
        m = _re.match(r"Job\s+(\d+)\s+requires:\s*(.*)", clue)
        if not m:
            continue
        j   = int(m.group(1))
        ops = [(int(mi), int(di))
               for mi, di in _re.findall(r"Machine\s+(\d+)\s+for\s+(\d+)", m.group(2))]
        job_routes[j] = ops

    if len(job_routes) != n_jobs:
        return None

    # op_info[(j, m)] = (op_index, duration)
    op_info: Dict = {}
    for j, ops in job_routes.items():
        for k, (m, d) in enumerate(ops):
            op_info[(j, m)] = (k, d)

    # f[(j, k)] = finish time of job j's k-th operation
    f = {(j, k): 0
         for j in range(n_jobs)
         for k in range(len(job_routes[j]))}

    max_iter = n_jobs * n_machines * 2
    for _ in range(max_iter):
        f_old = dict(f)   # ← key point: preserve previous iteration values

        for m in range(n_machines):
            machine_free = 0
            for j in grid.get(m, []):
                if j is None:
                    return None
                info = op_info.get((j, m))
                if info is None:
                    return None
                k, dur = info
                # job's previous op finish time — use f_old (not values updated this iter)
                job_prev = f_old.get((j, k - 1), 0) if k > 0 else 0
                start      = max(machine_free, job_prev)
                f[(j, k)]  = start + dur
                machine_free = start + dur

        if f == f_old:
            break

    return max(f[(j, len(job_routes[j]) - 1)] for j in range(n_jobs))


def evaluate_single_jssp(
    sample:        Dict,
    grid_ids:      List[int],
    cell_map:      List[Dict],
    tokenizer,
    mask_token_id: int,
) -> Dict:
    """
    JSSP single-puzzle evaluation.

    Returns dict with:
        cell_accuracy, row_accuracy, puzzle_solved,
        makespan_actual, makespan_optimal, makespan_ratio,
        permutation_valid, mask_remaining,
        pred_table, true_table
    """
    solution   = sample.get("solution", {})
    gt_rows    = solution.get("rows", [])      # [['M0','1','3',...], ...]
    n_jobs     = sample["n_jobs"]
    n_machines = sample["n_machines"]
    target_ms  = sample.get("makespan", None)

    # Build GT grid
    gt_grid: Dict[int, List[int]] = {}
    for m, row in enumerate(gt_rows):
        gt_grid[m] = [int(v) for v in row[1:]]

    # Build predicted grid
    pred_grid = _parse_grid_from_ids(grid_ids, cell_map, tokenizer, mask_token_id)

    # ── Cell accuracy ──
    n_total = n_machines * n_jobs
    n_mask  = sum(
        1 for m in range(n_machines)
        for j in pred_grid.get(m, []) if j is None
    )
    n_correct = sum(
        1 for m in range(n_machines)
        for slot, pred_j in enumerate(pred_grid.get(m, []))
        if pred_j is not None and pred_j == gt_grid.get(m, [])[slot]
        if slot < len(gt_grid.get(m, []))
    )
    cell_acc = n_correct / max(n_total, 1)

    # ── Row accuracy (machine row = exact match) ──
    n_row_correct = sum(
        1 for m in range(n_machines)
        if pred_grid.get(m) == gt_grid.get(m)
    )
    row_acc = n_row_correct / max(n_machines, 1)

    # ── Permutation validity (each row is a valid permutation, even if wrong order) ──
    expected_set = set(range(n_jobs))
    n_perm_valid = sum(
        1 for m in range(n_machines)
        if set(pred_grid.get(m, [])) == expected_set
        and None not in pred_grid.get(m, [])
    )
    perm_valid = n_perm_valid == n_machines

    # ── Makespan ──
    makespan_actual = None
    makespan_ratio  = None
    if perm_valid and target_ms is not None:
        makespan_actual = _simulate_makespan(pred_grid, sample)
        if makespan_actual is not None:
            makespan_ratio = makespan_actual / max(target_ms, 1)

    # ── Two solved criteria ──
    # solved_gt      : exact match with GT (existing criterion)
    # puzzle_solved  : makespan optimal (= true solved criterion)
    # → there can be multiple optimal schedules, so solved_gt ⊆ puzzle_solved
    solved_gt     = (n_row_correct == n_machines)
    puzzle_solved = (makespan_actual is not None and makespan_actual == target_ms)

    # ── Tables for wandb logging ──
    pred_table = [
        [f"M{m}"] + [str(j) if j is not None else "[M]"
                     for j in pred_grid.get(m, [])]
        for m in range(n_machines)
    ]
    true_table = list(gt_rows)

    return {
        "cell_accuracy":    cell_acc,
        "row_accuracy":     row_acc,
        "puzzle_solved":    puzzle_solved,   # makespan optimal criterion
        "solved_gt":        solved_gt,       # GT exact match criterion
        "permutation_valid": perm_valid,
        "makespan_actual":  makespan_actual,
        "makespan_optimal": target_ms,
        "makespan_ratio":   makespan_ratio if makespan_ratio is not None else -1.0,
        "mask_remaining":   n_mask / max(n_total, 1),
        "pred_table":       pred_table,
        "true_table":       true_table,
        "difficulty":       sample.get("difficulty", sample.get("size", "?")),
        "n_row_correct":    n_row_correct,
    }


# =============================================================================
