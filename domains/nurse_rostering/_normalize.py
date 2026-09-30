"""
Internal Nurse-Rostering normalization and stratified evaluation-set helper.

Stratified eval-set generation for the Nurse-Rostering (NR) task.
Uses the same normalized puzzle schema as ZebraLogic.

  search_space = n_shifts ^ (n_staff * n_days)   (free cell assignment)

NR-calibrated log10 bins (free assignment blows past ZL's permutation
bins, so boundaries are re-drawn; Z3 conflict count remains the primary
difficulty metric, per the ZebraLogic-Hard recipe):
  Small:   log_ss <  7
  Medium:  7  <= log_ss < 13
  Large:   13 <= log_ss < 19
  X-Large: log_ss >= 19

Every emitted instance is cross-verified with Z3 (sat + uniqueness via a
blocking clause); Z3 conflict counts are recorded as the difficulty metric.

"""

import sys, json, math, random, argparse, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from collections import Counter

sys.path.insert(0, '.')

try:
    from .nurse_rostering_problem import (
        generate_roster_problem, DifficultyConfig,
        ForbiddenSucc, RequiredSucc, RowCount, ColCount,
        NeverSamePair, AlwaysSamePair, MaxConsecWork,
        FixedAssign, NotAssign, SameCell, DiffCell, serialize_clue,
    )
except ImportError:
    from domains.nurse_rostering._problem import (
        generate_roster_problem, DifficultyConfig,
        ForbiddenSucc, RequiredSucc, RowCount, ColCount,
        NeverSamePair, AlwaysSamePair, MaxConsecWork,
        FixedAssign, NotAssign, SameCell, DiffCell, serialize_clue,
    )

try:
    from z3 import Int, Solver, And, Or, Not, Implies, If, Sum, sat, unsat
    _HAS_Z3 = True
except ImportError:
    _HAS_Z3 = False


# -- normalize: raw problem -> ZL-shaped problem dict --------------
def normalize_generated_problem(problem):
    grid     = problem["grid"]
    shifts   = problem["shifts"]
    n_staff  = problem["n_staff"]
    n_days   = problem["n_days"]
    clue_texts = list(problem["clue_texts"])

    header = ["Staff"] + [f"D{d+1}" for d in range(n_days)]
    rows   = [[f"S{s+1}"] + [shifts[grid[s][d]] for d in range(n_days)]
              for s in range(n_staff)]

    lines = [
        f"There are {n_staff} staff members (S1 to S{n_staff}) and "
        f"{n_days} days (D1 to D{n_days}).",
        f"Each day, each staff member is assigned exactly one shift.",
        f"Shifts: {', '.join(shifts)}",
        "",
        "Rules:",
    ]
    for idx, clue in enumerate(clue_texts, 1):
        lines.append(f"{idx}. {clue}")

    return {
        "puzzle":     "\n".join(lines),
        "solution":   {"header": header, "rows": rows},
        "size":       f"{n_staff}x{n_days}",
        "clue_texts": clue_texts,
        # machine-readable clue metadata (kind/family/cells) for H1a lag-by-family
        "clues_meta": [serialize_clue(c) for c in problem["clues"]],
        "level":      problem["level"],
        "n_shifts":   problem["n_shifts"],
    }


# -- search space & bins -------------------------------------------
def get_log_search_space(n_staff, n_days, n_shifts):
    return n_staff * n_days * math.log10(n_shifts)


def ss_label(log_ss):
    if log_ss < 7:    return "Small"
    elif log_ss < 13: return "Medium"
    elif log_ss < 19: return "Large"
    else:             return "X-Large"


# -- stratified curriculum (verify() checks bin consistency) -------
STRATIFIED_CURRICULUM = {
    "Small": [
        DifficultyConfig("small-3x4-s3-L2", 3, 4, 3, 2, weight=1.0),
        DifficultyConfig("small-3x4-s3-L3", 3, 4, 3, 3, weight=1.5),
        DifficultyConfig("small-4x3-s3-L3", 4, 3, 3, 3, weight=1.0),
    ],
    "Medium": [
        DifficultyConfig("med-3x5-s4-L4",  3, 5, 4, 4, weight=1.5),
        DifficultyConfig("med-4x5-s4-L5",  4, 5, 4, 5, weight=2.0),
        DifficultyConfig("med-3x7-s4-L5",  3, 7, 4, 5, weight=1.5),
        DifficultyConfig("med-4x5-s4-L7",  4, 5, 4, 7, weight=1.5),
    ],
    "Large": [
        DifficultyConfig("large-4x6-s4-L7", 4, 6, 4, 7, weight=2.0),
        DifficultyConfig("large-5x6-s4-L7", 5, 6, 4, 7, weight=2.0),
        DifficultyConfig("large-4x7-s4-L8", 4, 7, 4, 8, weight=1.5),
        DifficultyConfig("large-5x6-s4-L8", 5, 6, 4, 8, weight=1.5),
    ],
    "X-Large": [
        DifficultyConfig("xl-5x7-s4-L7",  5, 7, 4, 7, weight=1.5),
        DifficultyConfig("xl-5x7-s4-L8",  5, 7, 4, 8, weight=1.5),
        DifficultyConfig("xl-6x7-s4-L8",  6, 7, 4, 8, weight=1.5),
        DifficultyConfig("xl-6x7-s4-L9",  6, 7, 4, 9, weight=1.0),
    ],
}


def verify():
    ok = True
    for cat, cfgs in STRATIFIED_CURRICULUM.items():
        for d in list(cfgs):
            log_ss = get_log_search_space(d.n_staff, d.n_days, d.n_shifts)
            actual = ss_label(log_ss)
            if actual != cat:
                print(f"  MISMATCH: {cat} {d.name} -> actual={actual} "
                      f"log_ss={log_ss:.2f}  (dropping config)")
                cfgs.remove(d)
                ok = False
    print("Config verification: " + ("OK" if ok else "UPDATED"))
    return ok


# -- Z3 cross-verification + conflict measurement ------------------
def build_z3_model(problem):
    """Encode the NR clue set in Z3. Cell var c[s][d] in [0, n_shifts)."""
    n_staff, n_days = problem["n_staff"], problem["n_days"]
    n_shifts = problem["n_shifts"]
    off = n_shifts - 1
    c = [[Int(f"c_{s}_{d}") for d in range(n_days)] for s in range(n_staff)]
    solver = Solver()
    for s in range(n_staff):
        for d in range(n_days):
            solver.add(c[s][d] >= 0, c[s][d] < n_shifts)

    for cl in problem["clues"]:
        k = cl.kind
        if k == "fixed_assign":
            solver.add(c[cl.s][cl.d] == cl.v)
        elif k == "not_assign":
            solver.add(c[cl.s][cl.d] != cl.v)
        elif k == "same_cell":
            solver.add(c[cl.s1][cl.d1] == c[cl.s2][cl.d2])
        elif k == "diff_cell":
            solver.add(c[cl.s1][cl.d1] != c[cl.s2][cl.d2])
        elif k == "forbidden_succ":
            for d in range(n_days - 1):
                solver.add(Not(And(c[cl.s][d] == cl.x, c[cl.s][d + 1] == cl.y)))
        elif k == "required_succ":
            for d in range(n_days - 1):
                solver.add(Implies(c[cl.s][d] == cl.x, c[cl.s][d + 1] == cl.y))
        elif k in ("row_count", "col_count"):
            solver.add(Sum([If(c[s][d] == cl.x, 1, 0)
                            for (s, d) in cl.cell_list]) == cl.k)
        elif k == "never_same_pair":
            for d in range(n_days):
                solver.add(c[cl.s1][d] != c[cl.s2][d])
        elif k == "always_same_pair":
            for d in range(n_days):
                solver.add(c[cl.s1][d] == c[cl.s2][d])
        elif k == "max_consec_work":
            for start in range(n_days - cl.k):
                solver.add(Or([c[cl.s][d] == off
                               for d in range(start, start + cl.k + 1)]))
        else:
            raise ValueError(f"unknown clue kind: {k}")
    return solver, c


def z3_verify_and_measure(problem, n_runs=1):
    """Returns (unique: bool, matches_truth: bool, mean_conflicts, solve_s).
    Uniqueness: sat -> add blocking clause -> unsat."""
    grid = problem["grid"]
    n_staff, n_days = problem["n_staff"], problem["n_days"]

    conflicts, t_total = [], 0.0
    unique = matches = True
    for run in range(n_runs):
        solver, c = build_z3_model(problem)
        solver.set("random_seed", run)
        t0 = time.monotonic()
        r = solver.check()
        t_total += time.monotonic() - t0
        if r != sat:
            return False, False, -1.0, t_total
        st = solver.statistics()
        conf = 0
        for i in range(len(st)):
            if st[i][0] == "conflicts":
                conf = st[i][1]
        conflicts.append(conf)
        if run == 0:
            m = solver.model()
            sol = [[m.evaluate(c[s][d]).as_long() for d in range(n_days)]
                   for s in range(n_staff)]
            matches = (sol == grid)
            solver.add(Or([c[s][d] != grid[s][d]
                           for s in range(n_staff) for d in range(n_days)]))
            unique = (solver.check() == unsat)
    return unique, matches, sum(conflicts) / len(conflicts), t_total


# -- generation loop (mirrors generate_stratified_eval) ------------
def generate_stratified_eval(per_category, seed, output, categories,
                             z3_check=True, z3_runs=1, minimize_budget=15.0):
    random.seed(seed)
    verify()
    print()
    if z3_check and not _HAS_Z3:
        print("WARNING: z3-solver not installed; skipping Z3 verification.\n")
        z3_check = False

    all_puzzles, puzzle_id = [], 0
    for cat in categories:
        curriculum = STRATIFIED_CURRICULUM[cat]
        weights = [d.weight for d in curriculum]
        print(f"[{cat}] target={per_category}")
        count = attempts = 0
        while count < per_category:
            attempts += 1
            if attempts > per_category * 30:
                print(f"  WARNING: stopping at {count} (attempts={attempts})")
                break
            try:
                diff = random.choices(curriculum, weights=weights, k=1)[0]
                raw = generate_roster_problem(
                    n_staff=diff.n_staff, n_days=diff.n_days,
                    n_shifts=diff.n_shifts, level=diff.level,
                    minimal_conditions=True,
                    max_seconds_for_minimizing=minimize_budget,
                )
            except Exception:
                continue

            if z3_check:
                unique, matches, conf, _ = z3_verify_and_measure(raw, z3_runs)
                if not (unique and matches):
                    continue                      # reject non-unique instances
            else:
                conf = -1.0

            problem = normalize_generated_problem(raw)
            log_ss = get_log_search_space(diff.n_staff, diff.n_days,
                                          diff.n_shifts)
            problem.update({
                "puzzle_id":        puzzle_id,
                "eval_id":          f"nr_{cat.lower()}_{puzzle_id:04d}",
                "difficulty":       diff.name,
                "ss_category":      cat,
                "log_search_space": log_ss,
                "z3_conflicts":     conf,
                "n_clues":          len(problem["clue_texts"]),
            })
            all_puzzles.append(problem)
            puzzle_id += 1
            count += 1
            if count % 10 == 0:
                print(f"  {count}/{per_category}  (attempts={attempts})")
        print(f"  done: {count}")

    random.shuffle(all_puzzles)
    for i, p in enumerate(all_puzzles):
        p["puzzle_idx"] = i

    with open(output, "w", encoding="utf-8") as f:
        json.dump(all_puzzles, f, ensure_ascii=False, indent=2)

    print(f'\n{"="*55}')
    print(f"Total: {len(all_puzzles)} puzzles -> {output}")
    cat_cnt = Counter(p["ss_category"] for p in all_puzzles)
    for cat in ["Small", "Medium", "Large", "X-Large"]:
        print(f"  {cat:<10}: {cat_cnt.get(cat, 0)}")
    if all_puzzles and all_puzzles[0]["z3_conflicts"] >= 0:
        for cat in ["Small", "Medium", "Large", "X-Large"]:
            confs = [p["z3_conflicts"] for p in all_puzzles
                     if p["ss_category"] == cat]
            if confs:
                print(f"  Z3 conflicts {cat:<10}: "
                      f"mean={sum(confs)/len(confs):.1f}")
    size_cnt = Counter(p["size"] for p in all_puzzles)
    print("\nSize distribution:")
    for sz, n in sorted(size_cnt.items()):
        print(f"  {sz}: {n}")
    print("=" * 55)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_category", type=int, default=100)
    parser.add_argument("--out", type=str,
                        default="data/nurse_rostering/nurse_rostering_eval.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--categories", nargs="+",
                        default=["Small", "Medium", "Large", "X-Large"])
    parser.add_argument("--no_z3", action="store_true")
    parser.add_argument("--z3_runs", type=int, default=1,
                        help="Z3 seeds per instance for conflict averaging "
                             "(use the desired sample count)")
    parser.add_argument("--minimize_budget", type=float, default=15.0)
    args = parser.parse_args()

    print(f"Target: {args.per_category} x {len(args.categories)} = "
          f"{args.per_category * len(args.categories)} puzzles")
    print(f"Output: {args.out}\n")
    generate_stratified_eval(args.per_category, args.seed, args.out,
                             args.categories, z3_check=not args.no_z3,
                             z3_runs=args.z3_runs,
                             minimize_budget=args.minimize_budget)


if __name__ == "__main__":
    main()
