"""Parallel Nurse-Rostering (NR) train/val generator.

Produces NORMALIZED problem dicts (problem["puzzle"], problem["solution"]
{header, rows}) ready for nurse_rostering_encode.build_tokenized_sample_nr.

Uniqueness is guaranteed by the generator (clues are added until propagation
solves the grid, then minimized while count_solutions(limit=2)==1), so Z3 is
NOT run here -- keep it for the held-out stratified benchmark
(the held-out stratified benchmark helper). Difficulty is spread over the module CURRICULUM.

Usage (from repo root):
  python domains/nurse_rostering/generate.py --per_cfg_train 400 \
      --per_cfg_eval 40 --out_dir data/nurse_rostering --workers 64
"""

# --- repo path shim: make repo root importable so `core.* and datagen.*` resolves ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), _os.pardir, _os.pardir)))
# --- end shim ---

import argparse, json, random
import sys
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), _os.pardir))
sys.path.insert(0, ROOT)
from multiprocessing import Pool
from pathlib import Path

from domains.nurse_rostering._problem import (
    CURRICULUM, generate_roster_problem)
from domains.nurse_rostering._normalize import normalize_generated_problem


def _cfg_key(problem):
    """Dedup key: solution grid + sorted clue texts."""
    return (json.dumps(problem["solution"], sort_keys=True) + "|" +
            json.dumps(sorted(problem["clue_texts"])))


def _chunk(task):
    cfg_idx, seed, count, minimize_budget = task
    cfg = CURRICULUM[cfg_idx]
    random.seed(seed)
    seen, out = set(), []
    attempts = 0
    while len(out) < count:
        attempts += 1
        if attempts > count * 40:
            break                       # give up on this shard (avoid hang)
        try:
            raw = generate_roster_problem(
                n_staff=cfg.n_staff, n_days=cfg.n_days, n_shifts=cfg.n_shifts,
                level=cfg.level, minimal_conditions=True,
                max_seconds_for_minimizing=minimize_budget)
        except Exception:
            continue
        prob = normalize_generated_problem(raw)
        prob["difficulty"] = cfg.name
        k = _cfg_key(prob)
        if k in seen:
            continue
        seen.add(k)
        out.append(prob)
    return cfg_idx, out


def _train_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_cfg_train", type=int, default=400)
    ap.add_argument("--per_cfg_eval", type=int, default=40)
    ap.add_argument("--out_dir", default="data/nurse_rostering")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--minimize_budget", type=float, default=6.0)
    args = ap.parse_args()

    n_cfg = len(CURRICULUM)
    per_cfg = args.per_cfg_train + args.per_cfg_eval
    shards = max(1, args.workers // n_cfg)
    per_shard = per_cfg // shards + 1
    tasks = [(ci, args.seed + 1000 * ci + s, per_shard, args.minimize_budget)
             for ci in range(n_cfg) for s in range(shards)]
    print(f"[nr] {n_cfg} configs x per_cfg={per_cfg}; "
          f"{len(tasks)} tasks, workers={args.workers}", flush=True)

    with Pool(args.workers) as pool:
        results = pool.map(_chunk, tasks)

    by_cfg = {ci: [] for ci in range(n_cfg)}
    for ci, out in results:
        by_cfg[ci].extend(out)

    rng = random.Random(args.seed)
    train, ev, pid = [], [], 0
    for ci in range(n_cfg):
        cfg = CURRICULUM[ci]
        seen, uniq = set(), []
        for prob in by_cfg[ci]:
            k = _cfg_key(prob)
            if k in seen:
                continue
            seen.add(k)
            uniq.append(prob)
        rng.shuffle(uniq)
        for prob in uniq:
            prob["puzzle_id"] = f"nr_{cfg.name}_{pid}"
            pid += 1
        ev.extend(uniq[:args.per_cfg_eval])
        train.extend(uniq[args.per_cfg_eval:args.per_cfg_eval + args.per_cfg_train])
        nclues = [p["n_clues"] if "n_clues" in p else len(p["clue_texts"]) for p in uniq] or [0]
        import statistics as st
        print(f"  {cfg.name:<16}: have {len(uniq):4d}  clues_med={st.median(nclues):.0f}", flush=True)

    rng.shuffle(train); rng.shuffle(ev)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    json.dump(train, open(f"{args.out_dir}/nurse_rostering_train.json", "w"))
    json.dump(ev, open(f"{args.out_dir}/nurse_rostering_eval.json", "w"))
    print(f"[nr] train={len(train)} eval={len(ev)} -> {args.out_dir}", flush=True)


# =============================================================================
# eval split (held-out, difficulty-banded)
# =============================================================================

"""Generate held-out nurse-roster evaluation instances in Z3-conflict bands."""
import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path
import importlib.util

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from domains.nurse_rostering._problem import generate_roster_problem
_spec = importlib.util.spec_from_file_location("release_nr_normalize", HERE / "_normalize.py")
genr = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(genr)

# (name, lo, hi) Z3-conflict bands, with equal target count per band.
BANDS = [("d0", 0, 1), ("d1", 2, 4), ("d2", 5, 9), ("d3", 10, 10**9)]

# grid pool -> the grids likely to hit each band (efficiency hint).
# low bands: small grids; high bands: 4x7/4x6 which carry the conflict tail.
GRID_HINT = {
    "d0": [(3, 5, 4), (4, 4, 4), (4, 5, 4)],
    "d1": [(4, 5, 4), (4, 6, 4), (3, 7, 4)],
    "d2": [(4, 6, 4), (3, 7, 4), (4, 7, 4)],
    "d3": [(4, 7, 4), (4, 6, 4)],
}
LEVELS = [5, 6, 7]  # full vocab (relational A + succession B + cardinality C)


def _worker(task):
    import random
    band, lo, hi, seed, need = task
    rng = random.Random(seed)
    grids = GRID_HINT[band]
    out = []; guard = 0; t0 = time.time()
    while len(out) < need and guard < need * 300 + 500 and time.time() - t0 < 900:
        guard += 1
        s, d, sh = rng.choice(grids)
        L = rng.choice(LEVELS)
        try:
            raw = generate_roster_problem(
                n_staff=s, n_days=d, n_shifts=sh, level=L,
                minimal_conditions=True, max_seconds_for_minimizing=10.0)
        except Exception:
            continue
        uniq, match, conf, _ = genr.z3_verify_and_measure(raw, 1)
        if not (uniq and match):
            continue
        if not (lo <= conf <= hi):
            continue
        prob = genr.normalize_generated_problem(raw)
        prob["z3_conflicts"] = conf
        prob["ss_category"] = band
        prob["difficulty"] = f"{band}-{s}x{d}-L{L}-c{int(conf)}"
        out.append(prob)
    return out


def _eval_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_band", type=int, default=125)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--out", default="data/nurse_rostering/nurse_rostering_eval.json")
    args = ap.parse_args()

    shards = args.workers
    per = args.per_band // (shards // len(BANDS) or 1) + 1
    tasks = [(b, lo, hi, 1000 * bi + s, per)
             for bi, (b, lo, hi) in enumerate(BANDS)
             for s in range(shards // len(BANDS) or 1)]
    print(f"[gen-nr] bands={[b[0] for b in BANDS]} per_band={args.per_band} "
          f"tasks={len(tasks)}", flush=True)

    with Pool(args.workers) as pool:
        chunks = pool.map(_worker, tasks)

    from collections import defaultdict
    byb = defaultdict(list)
    for ch in chunks:
        for p in ch:
            byb[p["ss_category"]].append(p)

    allp = []; pid = 0
    for b, _, _ in BANDS:
        items = byb[b][:args.per_band]
        for p in items:
            p["puzzle_id"] = pid; pid += 1
        allp += items
        confs = [p["z3_conflicts"] for p in items]
        mc = sum(confs) / len(confs) if confs else 0
        print(f"  {b:8}: {len(items):4d}  (mean conf {mc:.1f})", flush=True)

    import random as _r
    _r.Random(0).shuffle(allp)
    for i, p in enumerate(allp):
        p["puzzle_idx"] = i
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(allp, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"[gen-nr] total={len(allp)} -> {args.out}", flush=True)


# =============================================================================
# entry point
# =============================================================================

def main():
    import sys
    split = 'train'
    if '--split' in sys.argv:
        i = sys.argv.index('--split'); split = sys.argv[i + 1]; del sys.argv[i:i + 2]
    (_train_main if split == 'train' else _eval_main)()


if __name__ == '__main__':
    main()
