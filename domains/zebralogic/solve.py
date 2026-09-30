"""ZebraLogic evaluation harness for greedy and triggered Blackboard inference."""

# --- repo path shim: make repo root importable so `core.* and datagen.*` resolves ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), _os.pardir, _os.pardir)))
# --- end shim ---


import sys, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from core.model import LLaDAModel
from domains.zebralogic.encode import (
    encode_table_body, make_prompt, setup_tokenizer,
)
from core.engine import _build_prefill
from core.trigger import blackboard_infer

MASK_ID = 126336


# ============================================================================
# Model loading
# ============================================================================

def load_model(checkpoint, model_name, device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    setup_tokenizer(tokenizer)
    backbone = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)
    backbone = PeftModel.from_pretrained(backbone, f'{checkpoint}/lora_adapter')
    backbone = backbone.merge_and_unload()
    model = LLaDAModel(backbone, backbone.config.hidden_size)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    print("  Loaded merged LoRA adapter")
    return model, tokenizer


# ============================================================================
# Puzzle preparation
# ============================================================================

def prepare_puzzle(puzzle, tokenizer, device):
    sol = puzzle['solution']
    categories = sol['header'][1:]
    rows = sol['rows']
    prompt_str = make_prompt(puzzle['puzzle'], sol)
    prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
    body_toks, cell_map = encode_table_body(sol, tokenizer)
    gen_length = len(body_toks)
    Lp = len(prompt_ids)
    prefill = _build_prefill(body_toks, cell_map, MASK_ID)
    cell_lookup = {}
    cell_positions = []
    for cm in cell_map:
        cell_lookup[cm['token_pos']] = {
            'row': cm['row'], 'col': cm['col'],
            'true_val': rows[cm['row']][cm['col'] + 1],
            'true_id': cm['token_id'],
            'cat': categories[cm['col']],
        }
        cell_positions.append(cm['token_pos'])
    return {
        'Lp': Lp, 'gen_length': gen_length,
        'cell_lookup': cell_lookup, 'cell_positions': cell_positions,
        'n_cells': len(cell_positions),
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


def grid_stats(x, Lp, cell_lookup):
    c = i = m = 0
    for pos, info in cell_lookup.items():
        tid = x[0, Lp + pos].item()
        if tid == MASK_ID:   m += 1
        elif tid == info['true_id']: c += 1
        else: i += 1
    return c, i, m


# ============================================================================
# DLM signals
# ============================================================================

@torch.no_grad()
def get_cell_confs(model, x, attn, Lp, cell_positions):
    out = model.backbone(input_ids=x, attention_mask=attn,
                         use_cache=False, return_dict=True)
    gl = out.logits[:, Lp:, :]
    probs = F.softmax(gl, dim=-1)
    confs = {}
    for pos in cell_positions:
        tid = x[0, Lp + pos].item()
        confs[pos] = (probs[0, pos].max().item() if tid == MASK_ID
                      else probs[0, pos, tid].item())
    return confs



# ============================================================================
# Core: greedy decoding with confidence tracking
# ============================================================================

@torch.no_grad()
def sequential_fill(model, prep, device):
    """
    Greedy any-order decoding with a confidence trace for trigger evaluation.
    """
    Lp = prep['Lp']
    nc = prep['n_cells']
    cell_positions = prep['cell_positions']
    x = make_fresh_x(prep, device)
    prev_confs = None
    last_fp = None
    fill_history = []

    for step in range(nc * 2):
        n_m = sum(1 for pos in cell_positions
                  if x[0, Lp + pos].item() == MASK_ID)
        if n_m == 0:
            break

        curr_confs = get_cell_confs(model, x, prep['attn'], Lp, cell_positions)
        if prev_confs is not None and last_fp is not None:
            n_drops = sum(1 for pos in cell_positions
                          if pos != last_fp and
                          curr_confs[pos] - prev_confs[pos] < -0.1)
            if fill_history:
                fill_history[-1]['drops'] = n_drops

        out = model.backbone(input_ids=x, attention_mask=prep['attn'],
                             use_cache=False, return_dict=True)
        gl = out.logits[:, Lp:, :]
        preds = torch.argmax(gl, dim=-1)
        probs = F.softmax(gl, dim=-1)
        score = torch.gather(probs, -1, preds.unsqueeze(-1)).squeeze(-1).float()
        masked = (x[0, Lp:] == MASK_ID)
        score[0, ~masked] = -float('inf')
        if score[0].max().item() == -float('inf'):
            break

        fp = torch.argmax(score[0]).item()
        fv = preds[0, fp].item()
        fi = prep['cell_lookup'][fp]
        x[0, Lp + fp] = fv
        fill_history.append({
            'pos': fp, 'val': fv, 'drops': 0,
            'correct': (fv == fi['true_id']),
            'mean_conf': (float(np.mean([curr_confs[pos] for pos in cell_positions if x[0, Lp + pos].item() == MASK_ID])) if any(x[0, Lp + pos].item() == MASK_ID for pos in cell_positions) else 1.0),
        })
        prev_confs = curr_confs
        last_fp = fp

    if fill_history and last_fp is not None and prev_confs is not None:
        curr_confs = get_cell_confs(model, x, prep['attn'], Lp, cell_positions)
        n_drops = sum(1 for pos in cell_positions
                      if pos != last_fp and
                      curr_confs[pos] - prev_confs[pos] < -0.1)
        fill_history[-1]['drops'] = n_drops

    return x, fill_history


# =============================================================================
# Shared helpers
# =============================================================================

def read_state(model, x, prep, cell_positions, Lp):
    out = model.backbone(input_ids=x, attention_mask=prep['attn'],
                         use_cache=False, return_dict=True)
    logits = out.logits[:, Lp:, :]
    probs = F.softmax(logits, dim=-1)
    confs = {}
    for pos in cell_positions:
        tid = x[0, Lp+pos].item()
        confs[pos] = (probs[0,pos].max().item() if tid == MASK_ID
                      else probs[0,pos,tid].item())
    # mask-only mean_conf: measure only positions the MDM was trained on
    mc_vals   = [c for pos, c in confs.items()
                 if x[0, Lp+pos].item() == MASK_ID]
    mean_conf = float(np.mean(mc_vals)) if mc_vals else float(np.mean(list(confs.values())))
    return logits, probs, confs, mean_conf


def compute_mean_conf(model, x, prep, cell_positions, Lp):
    _, _, _, mc = read_state(model, x, prep, cell_positions, Lp)
    return mc


# =============================================================================
# Method 1: greedy (baseline)
# =============================================================================

@torch.no_grad()
def solve_greedy(model, prep, device):
    x, fh = sequential_fill(model, prep, device)
    c, w, m = grid_stats(x, prep['Lp'], prep['cell_lookup'])
    return w == 0 and m == 0, len(fh) + len(fh)  # ~2 fp per step




# =============================================================================
# Method 3/4/5: cascade (Phase 1 + Phase 2)
# =============================================================================


# =============================================================================
# Method: cascade lookahead (N-step trajectory delta)
# =============================================================================

@torch.no_grad()
def solve_cascade_lookahead(model, prep, device, theta=0.90, search_k=5,
                            n_lookahead=2, max_iter=300):
    """
    Cascade inference with N-step lookahead anchor search.

    Two-mode structure (TRIGGER -> SEARCH + BACKTRACK):
      Anchor mode  (mean_conf < theta):
        For each candidate (pos, val), simulate n_lookahead greedy steps
        and measure trajectory delta = mean_conf(t+N) - mean_conf(t).
        Commit the full N-step path of the best candidate.
      Flow mode (mean_conf >= theta):
        Greedy fill with confidence gate.
        Gate failure re-triggers anchor mode.

    n_lookahead=1 is a single-step delta; the paper uses n_lookahead=3 (depth d=3).
    n_lookahead=2,3 look further ahead before committing.
    """
    Lp             = prep['Lp']
    cell_positions = prep['cell_positions']
    cell_lookup    = prep['cell_lookup']
    x              = make_fresh_x(prep, device)
    fp             = 0

    def _read(x_in):
        nonlocal fp
        gl, pr, co, mc = read_state(model, x_in, prep, cell_positions, Lp)
        fp += 1
        return gl, pr, co, mc

    def _mean(x_in):
        _, _, _, mc = _read(x_in)
        return mc

    def _greedy_steps(x_in, n_steps):
        """Advance n_steps via argmax greedy. fp count included."""
        x_cur = x_in.clone()
        for _ in range(n_steps):
            remaining = [p for p in cell_positions
                         if x_cur[0, Lp+p].item() == MASK_ID]
            if not remaining:
                break
            gl, pr, co, mc = read_state(model, x_cur, prep, cell_positions, Lp)
            nonlocal fp
            fp += 1
            best = max(remaining, key=lambda p: pr[0, p].max().item())
            x_cur[0, Lp + best] = pr[0, best].argmax().item()
        return x_cur

    phase1_next = False

    for it in range(max_iter):
        n_masked = sum(1 for p in cell_positions if x[0, Lp+p].item() == MASK_ID)
        if n_masked == 0:
            break

        gl, probs, confs, mean_conf = _read(x)
        use_phase1 = (mean_conf < theta) or phase1_next

        if use_phase1:
            phase1_next = False
            masked = [p for p in cell_positions if x[0, Lp+p].item() == MASK_ID]

            best_delta = 0.0
            best_x     = None

            for pos in masked:
                for val in probs[0, pos].topk(search_k).indices.tolist():
                    x_try = x.clone()
                    x_try[0, Lp + pos] = val
                    if n_lookahead > 1:
                        x_try = _greedy_steps(x_try, n_lookahead - 1)
                    final_mc = _mean(x_try)
                    delta    = final_mc - mean_conf
                    if delta > best_delta:
                        best_delta = delta
                        best_x     = x_try

            if best_x is not None:
                x = best_x
            else:
                pos = masked[0]
                x[0, Lp + pos] = probs[0, pos].argmax().item()
        else:
            masked = [(p, confs[p]) for p in cell_positions
                      if x[0, Lp+p].item() == MASK_ID]
            masked.sort(key=lambda t: -t[1])

            filled = False
            for pos, conf in masked:
                val   = probs[0, pos].argmax().item()
                x_try = x.clone()
                x_try[0, Lp + pos] = val
                if _mean(x_try) >= mean_conf:
                    x      = x_try
                    filled = True
                    break

            if not filled:
                phase1_next = True

    c, w, m = grid_stats(x, Lp, cell_lookup)
    return w == 0 and m == 0, fp

@torch.no_grad()
def solve_blackboard(model, prep, device, rho=0.80, tau=1.0,
                     anchor=0.90, depth=3, width=5):
    """Paper Algorithm 1: trigger greedy, then restart with cascade correction."""
    greedy_fp = 0
    def greedy():
        nonlocal greedy_fp
        x, history = sequential_fill(model, prep, device)
        _, wrong, masked = grid_stats(x, prep["Lp"], prep["cell_lookup"])
        trace = [step["mean_conf"] for step in history]
        greedy_fp = 2 * len(history)
        return (wrong == 0 and masked == 0, greedy_fp), trace

    def correct():
        return solve_cascade_lookahead(
            model, prep, device, theta=anchor, search_k=width, n_lookahead=depth)

    result, fired = blackboard_infer(greedy, correct, rho=rho, tau=tau, stat="min")
    return result[0], result[1] + (greedy_fp if fired else 0)


def make_methods(depth=3, width=5, anchor=0.90):
    """Method registry: blackboard_always_on is the always-on ablation; blackboard is
    the paper policy (greedy trigger followed by confidence-guided correction)."""
    return {
        "greedy": lambda model, prep, device: solve_greedy(model, prep, device),
        "blackboard_always_on": lambda model, prep, device: solve_cascade_lookahead(
            model, prep, device, theta=anchor, search_k=width, n_lookahead=depth),
        "blackboard": lambda model, prep, device: solve_blackboard(
            model, prep, device, rho=0.80, tau=1.0, anchor=anchor,
            depth=depth, width=width),
    }


def cascade_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/zebralogic')
    parser.add_argument('--model_name', default='GSAI-ML/LLaDA-8B-Instruct')
    parser.add_argument('--eval_path', default='data/zebralogic/zebralogic_hard_eval.json')
    parser.add_argument('--results_path', default='results/zebralogic/zl_results.json')
    parser.add_argument('--methods', nargs='+', default=['greedy', 'blackboard'])
    parser.add_argument('--depth', type=int, default=3, help='SEARCH lookahead depth d (paper 3)')
    parser.add_argument('--width', type=int, default=5, help='SEARCH branch width k (paper 5)')
    parser.add_argument('--anchor', type=float, default=0.90, help='anchor threshold alpha (paper 0.90)')
    parser.add_argument('--puzzles', nargs='+', type=int, default=None,
                        help='Specific puzzle indices. Default: all failed.')
    parser.add_argument('--all', action='store_true', default=True,
                        help='Run on all puzzles (the release default).')
    parser.add_argument('--max_samples', type=int, default=0,
                        help='Optional cap after selecting evaluation indices.')
    parser.add_argument('--output', default='results/zebralogic/zl_blackboard.json')
    args = parser.parse_args()
    methods = make_methods(args.depth, args.width, args.anchor)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    print('Loading model...')
    model, tokenizer = load_model(args.checkpoint, args.model_name, device)

    with open(args.eval_path) as f:
        puzzles = json.load(f)
    print(f'Loaded {len(puzzles)} puzzles')

    # Get indices
    if args.puzzles:
        indices = args.puzzles
    elif args.all:
        indices = list(range(len(puzzles)))
    else:
        with open(args.results_path) as f:
            data = json.load(f)
        pp = data['per_puzzle']
        first_method = list(pp.keys())[0]
        indices = [r['idx'] for r in pp[first_method] if not r['solved']]
    if args.max_samples > 0:
        indices = indices[:args.max_samples]
    print(f'Testing on {len(indices)} puzzles\n')

    # Validate methods
    unknown = [m for m in args.methods if m not in methods]
    if unknown:
        print(f"Unknown methods: {unknown}")
        print(f"Available: {sorted(methods.keys())}")
        sys.exit(1)

    # Run all methods
    results = {m: {'solved': 0, 'total_fp': 0, 'per_puzzle': []} for m in args.methods}

    # Header
    header = f"{'idx':>4} {'cat':<8}"
    for m in args.methods:
        header += f" {m[:14]:>14}"
    print(header)
    print('-' * len(header))

    for i, idx in enumerate(indices):
        puzzle = puzzles[idx]
        cat = puzzle.get('ss_category', '?')
        row = f"{idx:>4} {cat:<8}"

        for method_name in args.methods:
            prep = prepare_puzzle(puzzle, tokenizer, device)
            t0 = time.time()
            solved, fp = methods[method_name](model, prep, device)
            elapsed = time.time() - t0

            results[method_name]['per_puzzle'].append({
                'idx': idx, 'solved': solved, 'fp': fp, 'time': elapsed
            })
            if solved:
                results[method_name]['solved'] += 1
            results[method_name]['total_fp'] += fp

            mark = '✅' if solved else '❌'
            row += f" {mark}({fp:>5}fp)"

        print(row)

        if (i+1) % 10 == 0:
            counts = " | ".join(
                f"{m[:8]}={results[m]['solved']}"
                for m in args.methods
            )
            print(f"  [{i+1}/{len(indices)}] {counts}")

    # Summary
    n = len(indices)
    label = "all puzzles" if args.all else f"{n} failed puzzles"
    print(f"\n{'='*70}")
    print(f"SUMMARY — {label}")
    print(f"{'='*70}")
    print(f"{'method':<22} {'solved':>8} {'rate':>8} {'avg_fp':>8}")
    print('-' * 60)
    for m in args.methods:
        s = results[m]['solved']
        avg_fp = results[m]['total_fp'] // n if n > 0 else 0
        print(f"{m:<22} {s:>5}/{n} {100*s/n:>7.1f}% {avg_fp:>8}")
    print(f"{'='*70}")

    # Save
    save_data = {
        'args': vars(args),
        'n_puzzles': n,
        'indices': indices,
        'summary': {
            m: {
                'solved': results[m]['solved'],
                'rate': round(100 * results[m]['solved'] / n, 1) if n > 0 else 0,
                'avg_fp': results[m]['total_fp'] // n if n > 0 else 0,
            }
            for m in args.methods
        },
        'per_puzzle': {m: results[m]['per_puzzle'] for m in args.methods},
    }
    _os.makedirs(_os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f'\nSaved to {args.output}')


if __name__ == "__main__":
    cascade_cli()
