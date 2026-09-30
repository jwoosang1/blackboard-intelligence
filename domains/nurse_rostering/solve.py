"""
Nurse Rostering head-free inference primitives for

Nurse Rostering has:
  - NO givens (every constraint lives in the prompt as a natural-language clue),
  - a RECTANGULAR grid (rows = staff, cols = days), not a square,
  - "solved" = exact match to the unique canonical solution (uniqueness is
    guaranteed by the generator, so proper == ground-truth).

Exposes the inference primitives used by
the Blackboard evaluator below.
"""

# --- repo path shim: make repo root importable so `core.* and datagen.*` resolves ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), _os.pardir, _os.pardir)))
# --- end shim ---


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from core.model import LLaDAModel
from domains.nurse_rostering.encode import (
    nr_to_solution, encode_table_body, setup_tokenizer, SHIFTS,
)
from core.table_encoding import encode_cell_value, make_prompt
from core.engine import _build_prefill
from core.trigger import blackboard_infer

MASK_ID = 126336


# -- model loading (head OFF; merge LoRA) ---------------------------------------

def load_model(checkpoint, model_name, device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    setup_tokenizer(tokenizer)
    backbone = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)
    backbone = PeftModel.from_pretrained(backbone, f'{checkpoint}/lora_adapter')
    backbone = backbone.merge_and_unload()
    model = LLaDAModel(backbone, backbone.config.hidden_size).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"[nr] loaded {model_name} + LoRA({checkpoint}), head OFF", flush=True)
    return model, tokenizer


def shift_tok2idx(tokenizer):
    """token-id -> shift index 0..len(SHIFTS)-1 (day/late/night/off, single-token)."""
    t2i = {}
    for i, w in enumerate(SHIFTS):
        try:
            t2i[int(encode_cell_value(w, tokenizer))] = i
        except Exception:
            pass
    return t2i


def make_nr_prompt(inst):
    """Prompt exactly as used at training time (nurse_rostering_encode)."""
    return make_prompt(inst["puzzle"], nr_to_solution(inst))


# -- puzzle preparation (all grid cells masked; no givens) ----------------------

def prepare_puzzle(inst, tokenizer, device):
    sol = nr_to_solution(inst)
    rows = sol['rows']
    n_rows = len(rows)
    n_cols = len(sol['header']) - 1
    prompt_ids = tokenizer.encode(make_nr_prompt(inst), add_special_tokens=False)
    body_toks, cell_map = encode_table_body(sol, tokenizer)
    Lp = len(prompt_ids); gen_length = len(body_toks)
    prefill = _build_prefill(body_toks, cell_map, MASK_ID)
    cell_lookup = {}; cell_positions = []
    for cm in cell_map:
        cell_lookup[cm['token_pos']] = {
            'row': cm['row'], 'col': cm['col'],
            'true_id': cm['token_id'],
            'true_val': rows[cm['row']][cm['col'] + 1],
        }
        cell_positions.append(cm['token_pos'])
    return {
        'Lp': Lp, 'gen_length': gen_length,
        'cell_lookup': cell_lookup, 'cell_positions': cell_positions,
        'n_cells': len(cell_positions), 'n_rows': n_rows, 'n_cols': n_cols,
        'solution': inst['solution'],
        'attn': torch.ones((1, Lp + gen_length), dtype=torch.bool, device=device),
        'prompt_t': torch.tensor(prompt_ids, dtype=torch.long, device=device),
        'prefill_t': torch.tensor(prefill, dtype=torch.long, device=device),
    }


def make_fresh_x(prep, device):
    x = torch.full((1, prep['Lp'] + prep['gen_length']), MASK_ID,
                   dtype=torch.long, device=device)
    x[0, :prep['Lp']] = prep['prompt_t']
    x[0, prep['Lp']:] = prep['prefill_t']
    return x


# =============================================================================
# Blackboard inference + evaluation
# =============================================================================

"""Nurse Rostering greedy and confidence-guided Blackboard inference."""

# --- repo path shim: make repo root importable so `core.* and datagen.*` resolves ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), _os.pardir, _os.pardir)))
# --- end shim ---


import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F



def nr_solved(x, Lp, cell_lookup):
    """NR unique -> solved iff every committed cell equals its canonical token."""
    return all(x[0, Lp + pos].item() == info["true_id"]
               for pos, info in cell_lookup.items())


def read_state(model, x, prep, cell_positions, Lp):
    out = model.backbone(input_ids=x, attention_mask=prep['attn'],
                         use_cache=False, return_dict=True)
    logits = out.logits[:, Lp:, :]
    probs = F.softmax(logits, dim=-1)
    confs = {}
    for pos in cell_positions:
        tid = x[0, Lp + pos].item()
        confs[pos] = (probs[0, pos].max().item() if tid == MASK_ID
                      else probs[0, pos, tid].item())
    mc_vals = [c for pos, c in confs.items() if x[0, Lp + pos].item() == MASK_ID]
    mean_conf = float(np.mean(mc_vals)) if mc_vals else float(np.mean(list(confs.values())))
    return logits, probs, confs, mean_conf


@torch.no_grad()
def solve_cascade_lookahead(model, prep, device, theta=0.90, search_k=5,
                            n_lookahead=3, max_iter=300):
    Lp = prep['Lp']; cell_positions = prep['cell_positions']
    x = make_fresh_x(prep, device)
    fp = 0

    def _read(x_in):
        nonlocal fp
        r = read_state(model, x_in, prep, cell_positions, Lp); fp += 1
        return r

    def _mean(x_in):
        return _read(x_in)[3]

    def _greedy_steps(x_in, n_steps):
        nonlocal fp
        x_cur = x_in.clone()
        for _ in range(n_steps):
            remaining = [p for p in cell_positions if x_cur[0, Lp + p].item() == MASK_ID]
            if not remaining:
                break
            _, pr, _, _ = read_state(model, x_cur, prep, cell_positions, Lp); fp += 1
            best = max(remaining, key=lambda p: pr[0, p].max().item())
            x_cur[0, Lp + best] = pr[0, best].argmax().item()
        return x_cur

    phase1_next = False
    for it in range(max_iter):
        if sum(1 for p in cell_positions if x[0, Lp + p].item() == MASK_ID) == 0:
            break
        gl, probs, confs, mean_conf = _read(x)
        use_phase1 = (mean_conf < theta) or phase1_next
        if use_phase1:
            # ZL-EXACT anchor: search ALL masked cells x top-k vals, commit the
            # (pos,val) path with the best mean_conf delta.
            phase1_next = False
            masked = [p for p in cell_positions if x[0, Lp + p].item() == MASK_ID]
            best_delta = 0.0; best_x = None
            for pos in masked:
                for val in probs[0, pos].topk(search_k).indices.tolist():
                    x_try = x.clone(); x_try[0, Lp + pos] = val
                    if n_lookahead > 1:
                        x_try = _greedy_steps(x_try, n_lookahead - 1)
                    delta = _mean(x_try) - mean_conf
                    if delta > best_delta:
                        best_delta = delta; best_x = x_try
            if best_x is not None:
                x = best_x
            else:
                pos = masked[0]; x[0, Lp + pos] = probs[0, pos].argmax().item()
        else:
            masked = [(p, confs[p]) for p in cell_positions if x[0, Lp + p].item() == MASK_ID]
            masked.sort(key=lambda t: -t[1])
            filled = False
            for pos, conf in masked:
                x_try = x.clone(); x_try[0, Lp + pos] = probs[0, pos].argmax().item()
                if _mean(x_try) >= mean_conf:
                    x = x_try; filled = True; break
            if not filled:
                phase1_next = True

    return bool(nr_solved(x, Lp, prep['cell_lookup'])), fp


@torch.no_grad()
def solve_greedy(model, prep, device):
    """Greedy decode plus the masked-position mean-confidence trace."""
    Lp, cells = prep["Lp"], prep["cell_positions"]
    x, fp, trace = make_fresh_x(prep, device), 0, []
    for _ in range(prep["n_cells"] * 2):
        masked = [p for p in cells if x[0, Lp + p].item() == MASK_ID]
        if not masked:
            break
        _, probs, _, mean_conf = read_state(model, x, prep, cells, Lp)
        fp += 1
        trace.append(mean_conf)
        pos = max(masked, key=lambda p: probs[0, p].max().item())
        x[0, Lp + pos] = probs[0, pos].argmax().item()
    return bool(nr_solved(x, Lp, prep["cell_lookup"])), fp, trace


@torch.no_grad()
def solve_blackboard(model, prep, device, rho=0.90, tau=0.95,
                     anchor=0.90, depth=3, width=5):
    """Paper feasibility policy: trigger greedy, then restart with cascade."""
    greedy_fp = 0
    def greedy():
        nonlocal greedy_fp
        solved, fp, trace = solve_greedy(model, prep, device)
        greedy_fp = fp
        return (solved, fp), trace

    def correct():
        return solve_cascade_lookahead(
            model, prep, device, theta=anchor, search_k=width, n_lookahead=depth)

    result, fired = blackboard_infer(greedy, correct, rho=rho, tau=tau, stat="mean")
    return result[0], result[1] + (greedy_fp if fired else 0), fired


def main():
    ap = argparse.ArgumentParser(description="Nurse Rostering Blackboard evaluation")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model_name", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--eval_path", required=True)
    ap.add_argument("--out", "--output", dest="out",
                    default="results/nurse_rostering/blackboard.json")
    ap.add_argument("--methods", nargs="+", default=["greedy", "blackboard"],
                    choices=["greedy", "blackboard_always_on", "blackboard"])
    ap.add_argument("--anchor", type=float, default=0.90)
    ap.add_argument("--width", type=int, default=5)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model, tokenizer = load_model(args.checkpoint, args.model_name, args.device)
    puzzles = json.load(open(args.eval_path))
    if args.max_samples > 0:
        puzzles = puzzles[:args.max_samples]
    rows = {method: [] for method in args.methods}
    for i, puzzle in enumerate(puzzles):
        prep = prepare_puzzle(puzzle, tokenizer, args.device)
        for method in args.methods:
            if method == "greedy":
                solved, fp, _ = solve_greedy(model, prep, args.device)
                fired = False
            elif method == "blackboard_always_on":
                solved, fp = solve_cascade_lookahead(
                    model, prep, args.device, theta=args.anchor,
                    search_k=args.width, n_lookahead=args.depth)
                fired = True
            else:
                solved, fp, fired = solve_blackboard(
                    model, prep, args.device, rho=0.90, tau=0.95,
                    anchor=args.anchor, depth=args.depth, width=args.width)
            rows[method].append({"puzzle_id": puzzle.get("puzzle_id", i),
                                 "ss_category": puzzle.get("ss_category", "?"),
                                 "solved": bool(solved), "forward_passes": fp,
                                 "triggered": fired})
        if (i + 1) % 10 == 0:
            print(f"[{i + 1}/{len(puzzles)}]", flush=True)

    summary = {}
    for method, method_rows in rows.items():
        n = len(method_rows)
        summary[method] = {"solved": sum(r["solved"] for r in method_rows),
                           "total": n,
                           "rate": sum(r["solved"] for r in method_rows) / max(n, 1),
                           "avg_forward_passes": sum(r["forward_passes"] for r in method_rows) / max(n, 1),
                           "trigger_rate": sum(r["triggered"] for r in method_rows) / max(n, 1)}
        print(f"{method}: {summary[method]['solved']}/{n} "
              f"({100 * summary[method]['rate']:.1f}%), "
              f"avg_fp={summary[method]['avg_forward_passes']:.1f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "summary": summary, "per_method": rows}, f, indent=2)


if __name__ == "__main__":
    main()
