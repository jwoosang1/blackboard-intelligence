"""
JSSP inference-time evaluation harness (greedy + Blackboard).

Methods (--methods):
  greedy      greedy any-order decoding (baseline)
  blackboard  triggered Best-of-N (lowest-makespan selection); N is the --n_bon flag (paper N=10)

The puzzle-level trigger (rho, tau) lives in core/trigger.py.
Core metric: makespan_ratio (1.0 = optimal).

Usage (from the repository root):
  python domains/jssp/solve.py \\
      --checkpoint checkpoints/jssp \\
      --eval_path data/jssp/jssp_eval.json \\
      --methods greedy blackboard --n_bon 10
"""

# --- repo path shim: make repo root importable so `core.* and datagen.*` resolves ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), _os.pardir, _os.pardir)))
# --- end shim ---


import sys, json, time, argparse, pickle, math
import numpy as np
from collections import defaultdict
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, '.')

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from core.model import LLaDAModel
from core.trigger import blackboard_infer
from domains.zebralogic.encode import setup_tokenizer
from domains.jssp.encode import (
    encode_prompt_jssp, encode_table_body_jssp,
)
from core.engine import _build_prefill
from metrics import evaluate_single_jssp

MASK_ID = 126336
USE_MASK_ONLY = True


# =============================================================================
# Model loading
# =============================================================================

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
    return model, tokenizer


def prepare_puzzle(puzzle, tokenizer, device):
    solution    = puzzle['solution']
    puzzle_text = puzzle['puzzle']
    prompt_ids          = encode_prompt_jssp(puzzle_text, solution, tokenizer)
    body_toks, cell_map = encode_table_body_jssp(solution, tokenizer)
    Lp         = len(prompt_ids)
    gen_length = len(body_toks)
    prefill    = _build_prefill(body_toks, cell_map, MASK_ID)
    cell_positions = [cm['token_pos'] for cm in cell_map]
    return {
        'Lp':             Lp,
        'gen_length':     gen_length,
        'cell_map':       cell_map,
        'cell_positions': cell_positions,
        'n_cells':        len(cell_positions),
        'attn':      torch.ones((1, Lp + gen_length), dtype=torch.bool, device=device),
        'prompt_t':  torch.tensor(prompt_ids, dtype=torch.long, device=device),
        'prefill_t': torch.tensor(prefill,    dtype=torch.long, device=device),
        'puzzle':    puzzle,
    }


def make_fresh_x(prep, device):
    x = torch.full((1, prep['Lp'] + prep['gen_length']), MASK_ID,
                   dtype=torch.long, device=device)
    x[0, :prep['Lp']]  = prep['prompt_t']
    x[0, prep['Lp']:]  = prep['prefill_t']
    return x


@torch.no_grad()
def read_state(model, x, prep, cell_positions, Lp):
    out    = model.backbone(input_ids=x, attention_mask=prep['attn'],
                            use_cache=False, return_dict=True)
    logits = out.logits[:, Lp:, :]
    probs  = F.softmax(logits, dim=-1)
    confs  = {p: float(probs[0, p].max().item()) for p in cell_positions}

    if USE_MASK_ONLY:
        mc_vals = [c for p, c in confs.items() if x[0, Lp+p].item() == MASK_ID]
        mean_conf = float(np.mean(mc_vals)) if mc_vals else float(np.mean(list(confs.values())))
    else:
        mean_conf = float(np.mean(list(confs.values())))
    return logits, probs, confs, mean_conf


@torch.no_grad()
def get_mean_conf(model, x, prep, cell_positions, Lp):
    _, _, _, mc = read_state(model, x, prep, cell_positions, Lp)
    return mc


def evaluate(x, prep, tokenizer):
    Lp         = prep['Lp']
    cell_map   = prep['cell_map']
    grid_ids   = x[0, Lp:].tolist()
    result     = evaluate_single_jssp(prep['puzzle'], grid_ids, cell_map,
                                      tokenizer, MASK_ID)
    return result.get('puzzle_solved', False), result.get('makespan_ratio', -1.0)


# =============================================================================
# Method 1: Greedy
# =============================================================================

@torch.no_grad()
def solve_greedy(model, prep, device, tokenizer, return_trace=False):
    Lp             = prep['Lp']
    cell_positions = prep['cell_positions']
    x              = make_fresh_x(prep, device)
    fp             = 0
    conf_trace     = []
    for _ in range(prep['n_cells'] * 2):
        masked = [p for p in cell_positions if x[0, Lp+p].item() == MASK_ID]
        if not masked:
            break
        out    = model.backbone(input_ids=x, attention_mask=prep['attn'],
                                use_cache=False, return_dict=True)
        probs  = F.softmax(out.logits[:, Lp:, :], dim=-1)
        fp    += 1
        scores = {p: float(probs[0, p].max().item()) for p in masked}
        conf_trace.append(float(np.mean(list(scores.values()))))
        best_pos = max(scores, key=scores.get)
        best_val = int(probs[0, best_pos].argmax().item())
        x[0, Lp + best_pos] = best_val
    solved, ms_ratio = evaluate(x, prep, tokenizer)
    result = (solved, ms_ratio, fp, x)
    return (*result, conf_trace) if return_trace else result


# =============================================================================
# Method 2: Greedy + gate
# =============================================================================


# =============================================================================
# Method 3: Cascade
# =============================================================================


# =============================================================================
# Method 4: Best-of-N (mean_conf, random, or makespan selection)
# =============================================================================

@torch.no_grad()
def solve_bon(model, prep, device, tokenizer, n_samples=4, noise_temp=0.5,
              selection='mean_conf'):
    Lp             = prep['Lp']
    cell_positions = prep['cell_positions']
    total_fp       = 0
    candidates     = []

    for _ in range(n_samples):
        x          = make_fresh_x(prep, device)
        fp         = 0
        mc_history = []
        for _ in range(prep['n_cells'] * 2):
            masked = [p for p in cell_positions if x[0, Lp+p].item() == MASK_ID]
            if not masked:
                break
            out   = model.backbone(input_ids=x, attention_mask=prep['attn'],
                                   use_cache=False, return_dict=True)
            probs = F.softmax(out.logits[:, Lp:, :], dim=-1)
            fp   += 1
            mc_vals = [float(probs[0, p].max().item()) for p in masked]
            mc_history.append(float(np.mean(mc_vals)))
            logits  = out.logits[0, :, :].float()
            u       = torch.rand_like(logits)
            noisy   = logits + (-torch.log(-torch.log(u + 1e-20) + 1e-20)) * noise_temp
            scores  = torch.tensor([noisy[p].max().item() for p in masked])
            chosen_pos = masked[scores.argmax().item()]
            chosen_val = int(probs[0, chosen_pos].argmax().item())
            x[0, Lp + chosen_pos] = chosen_val

        total_fp += fp
        solved, ms_ratio = evaluate(x, prep, tokenizer)

        if selection == 'mean_conf':
            score = float(np.mean(mc_history)) if mc_history else 0.0
        elif selection == 'random':
            score = float(torch.rand(1).item())
        elif selection == 'makespan':
            score = -float(ms_ratio) if ms_ratio > 0 else -1e9
        else:
            score = 0.0

        candidates.append((score, solved, ms_ratio, x.clone()))

    candidates.sort(key=lambda t: -t[0])
    _, best_solved, best_ratio, best_x = candidates[0]
    return best_solved, best_ratio, total_fp, best_x


# =============================================================================
# Method 5: Beam search with c̄ pruning
# =============================================================================

class _Beam:
    """Lightweight beam state."""
    __slots__ = ('x', 'cum_score')

    def __init__(self, x, cum_score):
        self.x = x
        self.cum_score = cum_score


# =============================================================================
# Method 6: Greedy-first wrapper
# =============================================================================


# =============================================================================
# Method registry
# =============================================================================

@torch.no_grad()
def solve_blackboard(model, prep, device, tokenizer, n_bon=10, rho=0.50, tau=0.70):
    """Paper JSSP policy: trigger greedy, then objective-selected BoN-10."""
    greedy_fp = 0
    def greedy():
        nonlocal greedy_fp
        solved, ratio, fp, x, trace = solve_greedy(
            model, prep, device, tokenizer, return_trace=True)
        greedy_fp = fp
        return (solved, ratio, fp, x), trace

    def correct():
        return solve_bon(model, prep, device, tokenizer, n_samples=n_bon, selection="makespan")

    result, fired = blackboard_infer(greedy, correct, rho=rho, tau=tau, stat="min")
    if fired:
        return result[0], result[1], result[2] + greedy_fp, result[3]
    return result


def make_methods(n_bon=10, tokenizer=None):
    """Build the method table. `blackboard` = triggered Best-of-N with `n_bon` samples,
    selecting the valid schedule with the lowest makespan (paper: N=10)."""
    return {
        # greedy any-order decoding (baseline)
        'greedy':     lambda m,p,d,t: solve_greedy(m, p, d, t),
        # Blackboard (paper main result)
        'blackboard': lambda m,p,d,t: solve_blackboard(m, p, d, t, n_bon=n_bon),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',
                        default='checkpoints/jssp')
    parser.add_argument('--model_name', default='GSAI-ML/LLaDA-8B-Instruct')
    parser.add_argument('--eval_path',
                        default='data/jssp/jssp_eval.json')
    parser.add_argument('--greedy_traces',
                        default='results/jssp_greedy_traces_eval.pkl')
    parser.add_argument('--failed_only', action='store_true')
    parser.add_argument('--methods', nargs='+',
                        default=['greedy', 'blackboard'])
    parser.add_argument('--n_puzzles', type=int, default=None)
    parser.add_argument('--n_bon', type=int, default=10, help='Best-of-N samples N (paper 10)')
    parser.add_argument('--no_mask_only', action='store_true')
    parser.add_argument('--output', default='results/jssp_blackboard.json')
    args = parser.parse_args()

    global USE_MASK_ONLY
    USE_MASK_ONLY = not args.no_mask_only
    mc_mode = 'mask-only' if USE_MASK_ONLY else 'all'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}  |  mc: {mc_mode}  |  N_bon: {args.n_bon}')
    print('Loading model...')
    model, tokenizer = load_model(args.checkpoint, args.model_name, device)

    with open(args.eval_path) as f:
        all_puzzles = json.load(f)

    if args.failed_only and args.greedy_traces:
        with open(args.greedy_traces, 'rb') as f:
            traces = pickle.load(f)
        indices = [t['idx'] for t in traces if not t['solved']]
        print(f'Failed-only mode: {len(indices)} instances')
    elif args.n_puzzles:
        indices = list(range(min(args.n_puzzles, len(all_puzzles))))
    else:
        indices = list(range(len(all_puzzles)))

    methods = make_methods(args.n_bon, tokenizer)
    selected_methods = {k: methods[k] for k in args.methods if k in methods}
    missing = set(args.methods) - set(selected_methods.keys())
    if missing:
        print(f'WARNING: unknown methods skipped: {missing}')
    print(f'Puzzles: {len(indices)}  Methods: {list(selected_methods.keys())}')
    print()

    results = {name: [] for name in selected_methods}

    for idx in tqdm(indices):
        puzzle = all_puzzles[idx]
        prep   = prepare_puzzle(puzzle, tokenizer, device)
        size   = puzzle.get('size', puzzle.get('difficulty', '?'))

        for name, fn in selected_methods.items():
            t0 = time.time()
            solved, ms_ratio, fp, x = fn(model, prep, device, tokenizer)
            elapsed = time.time() - t0
            results[name].append({
                'idx':      idx,
                'size':     size,
                'solved':   bool(solved),
                'ms_ratio': float(ms_ratio) if ms_ratio > 0 else -1.0,
                'fp':       fp,
                'elapsed':  round(elapsed, 2),
            })

    # Summary
    print('\n' + '='*70)
    print(f'{"Method":<22} {"Solved":>8} {"Rate":>7} {"ms_ratio":>10} {"avg_fp":>8}')
    print('-'*70)

    summary = {}
    for name, rlist in results.items():
        n        = len(rlist)
        n_solved = sum(1 for r in rlist if r['solved'])
        ratios   = [r['ms_ratio'] for r in rlist if r['ms_ratio'] > 0]
        avg_ms   = np.mean(ratios) if ratios else float('nan')
        avg_fp   = np.mean([r['fp'] for r in rlist])
        print(f'{name:<22} {n_solved:>4}/{n:<4} {100*n_solved/max(n,1):>6.1f}%'
              f' {avg_ms:>10.4f} {avg_fp:>8.1f}')
        summary[name] = {
            'n': n, 'n_solved': n_solved,
            'solve_rate':   round(n_solved/max(n,1), 4),
            'avg_ms_ratio': round(avg_ms, 4),
            'avg_fp':       round(avg_fp, 1),
        }

    # Size-stratified
    print()
    print('Size-stratified summary:')
    by_size_method = {name: defaultdict(list) for name in selected_methods}
    for name, rlist in results.items():
        for r in rlist:
            by_size_method[name][r['size']].append(r)
    sizes = sorted({s for d in by_size_method.values() for s in d}, key=str)
    print(f'  {"size":<8}', end='')
    for name in selected_methods:
        print(f' {name[:14]:>16}', end='')
    print()
    for sz in sizes:
        print(f'  {str(sz):<8}', end='')
        for name in selected_methods:
            rs = by_size_method[name].get(sz, [])
            if rs:
                sr = sum(r['solved'] for r in rs) / len(rs) * 100
                ratios = [r['ms_ratio'] for r in rs if r['ms_ratio'] > 0]
                ms = np.mean(ratios) if ratios else float('nan')
                print(f' {sr:>4.0f}% {ms:>6.3f}   ', end='')
            else:
                print(f' {"--":>16}', end='')
        print()

    out = {
        'args': vars(args),
        'indices': indices,
        'summary': summary,
        'per_puzzle': results,
    }
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved → {args.output}')


if __name__ == '__main__':
    main()