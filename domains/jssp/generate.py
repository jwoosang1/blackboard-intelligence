"""
JSSP Compact Grid Generator.

Two modes:
  convert:  existing time-unit data → compact grid
  generate: from scratch with OR-Tools / dispatch solver

Compact grid (Machine × Slot → JobID):
  | Machine | Slot_1 | Slot_2 | Slot_3 |
  | M0      | J2     | J0     | J1     |
  | M1      | J0     | J1     | J2     |
  = n_machines × n_jobs cells (same scale as ZebraLogic!)

Usage:
    python gen_jssp_compact.py convert \
        --input data/jssp/jssp_train.json \
        --output data/jssp/jssp_train.json

    python gen_jssp_compact.py generate \
        --output_dir data/jssp --n_train 30000 --n_eval 200
"""

import argparse, json, os, random
from collections import defaultdict


# ============================================================
# Solvers
# ============================================================

def solve_dispatch(jobs, n_machines):
    """Priority dispatching: earliest start, shortest duration tiebreak."""
    n_jobs = len(jobs)
    next_op = [0] * n_jobs
    machine_free = [0] * n_machines
    job_free = [0] * n_jobs
    schedule = {}

    for _ in range(n_jobs * n_machines):
        cands = []
        for j in range(n_jobs):
            if next_op[j] >= len(jobs[j]):
                continue
            o = next_op[j]
            m, dur = jobs[j][o]
            start = max(machine_free[m], job_free[j])
            cands.append((start, dur, j, o, m))
        if not cands:
            break
        cands.sort()
        start, dur, j, o, m = cands[0]
        end = start + dur
        schedule[(j, o)] = (m, start, end, dur)
        machine_free[m] = end
        job_free[j] = end
        next_op[j] += 1

    return schedule, max(machine_free) if machine_free else 0


def solve_ortools(jobs, n_machines, timeout=30):
    """OR-Tools CP-SAT optimal solver."""
    try:
        from ortools.sat.python import cp_model
    except ImportError:
        return None, None

    n_jobs = len(jobs)
    horizon = sum(d for job in jobs for _, d in job)
    mdl = cp_model.CpModel()
    starts, ends, ivs = {}, {}, {}
    mach_ivs = defaultdict(list)

    for j, job in enumerate(jobs):
        for o, (m, dur) in enumerate(job):
            s = mdl.NewIntVar(0, horizon, f's{j}_{o}')
            e = mdl.NewIntVar(0, horizon, f'e{j}_{o}')
            iv = mdl.NewIntervalVar(s, dur, e, f'i{j}_{o}')
            starts[(j,o)], ends[(j,o)], ivs[(j,o)] = s, e, iv
            mach_ivs[m].append(iv)

    for j, job in enumerate(jobs):
        for o in range(len(job) - 1):
            mdl.Add(starts[(j, o+1)] >= ends[(j, o)])
    for m in range(n_machines):
        if mach_ivs[m]:
            mdl.AddNoOverlap(mach_ivs[m])

    ms = mdl.NewIntVar(0, horizon, 'ms')
    for j, job in enumerate(jobs):
        mdl.Add(ms >= ends[(j, len(job)-1)])
    mdl.Minimize(ms)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout
    st = solver.Solve(mdl)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        sched = {}
        for j, job in enumerate(jobs):
            for o, (m, dur) in enumerate(job):
                sched[(j,o)] = (m, solver.Value(starts[(j,o)]),
                                solver.Value(ends[(j,o)]), dur)
        return sched, solver.Value(ms)
    return None, None


def solve_ortools_multi(jobs, n_machines, max_solutions=8, timeout=60):
    """
    OR-Tools CP-SAT: find the optimal makespan, then collect up to
    max_solutions alternative optimal solutions with the same makespan.

    Multiple optimal augmentation:
      - use N different optimal schedules from the same puzzle as GT
      - the MDM learns the "solution space" → fewer false attractors
    """
    try:
        from ortools.sat.python import cp_model
    except ImportError:
        return None, None, []

    n_jobs  = len(jobs)
    horizon = sum(d for job in jobs for _, d in job)
    mdl     = cp_model.CpModel()
    starts, ends, ivs = {}, {}, {}
    mach_ivs = defaultdict(list)

    for j, job in enumerate(jobs):
        for o, (m, dur) in enumerate(job):
            s  = mdl.NewIntVar(0, horizon, f's{j}_{o}')
            e  = mdl.NewIntVar(0, horizon, f'e{j}_{o}')
            iv = mdl.NewIntervalVar(s, dur, e, f'i{j}_{o}')
            starts[(j,o)], ends[(j,o)], ivs[(j,o)] = s, e, iv
            mach_ivs[m].append(iv)

    for j, job in enumerate(jobs):
        for o in range(len(job) - 1):
            mdl.Add(starts[(j, o+1)] >= ends[(j, o)])
    for m in range(n_machines):
        if mach_ivs[m]:
            mdl.AddNoOverlap(mach_ivs[m])

    ms_var = mdl.NewIntVar(0, horizon, 'ms')
    for j, job in enumerate(jobs):
        mdl.Add(ms_var >= ends[(j, len(job)-1)])
    mdl.Minimize(ms_var)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout // 2
    st = solver.Solve(mdl)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, None, []

    optimal_ms = solver.Value(ms_var)

    # first solution
    def extract_schedule(slv):
        sched = {}
        for j, job in enumerate(jobs):
            for o, (m, dur) in enumerate(job):
                sched[(j,o)] = (m, slv.Value(starts[(j,o)]),
                                slv.Value(ends[(j,o)]), dur)
        return sched

    all_schedules = [extract_schedule(solver)]

    # fix the optimal_ms constraint, then search for additional solutions
    mdl.Add(ms_var == optimal_ms)
    mdl.ClearObjective()

    class SolutionCollector(cp_model.CpSolverSolutionCallback):
        def __init__(self, limit):
            super().__init__()
            self._limit   = limit
            self.solutions = []
        def on_solution_callback(self):
            if len(self.solutions) >= self._limit:
                self.StopSearch()
                return
            self.solutions.append(extract_schedule(self))

    collector = SolutionCollector(max_solutions - 1)
    solver2   = cp_model.CpSolver()
    solver2.parameters.max_time_in_seconds     = timeout // 2
    solver2.parameters.enumerate_all_solutions = True
    solver2.Solve(mdl, collector)
    all_schedules.extend(collector.solutions)

    return all_schedules[0], optimal_ms, all_schedules



    sched, ms = solve_ortools(jobs, n_machines)
    if sched is None:
        sched, ms = solve_dispatch(jobs, n_machines)
    return sched, ms


# ============================================================
# Grid conversion
# ============================================================

def schedule_to_compact(schedule, n_jobs, n_machines):
    """Schedule → compact grid (sort ops per machine by start time)."""
    mops = defaultdict(list)
    for (j, o), (m, start, end, dur) in schedule.items():
        mops[m].append((start, j))
    for m in range(n_machines):
        mops[m].sort()

    header = ['Machine'] + [f'Slot_{s+1}' for s in range(n_jobs)]
    rows = []
    for m in range(n_machines):
        row = [f'M{m}'] + [str(j) for _, j in mops[m]]
        while len(row) < len(header):
            row.append('_')
        rows.append(row)
    return {'header': header, 'rows': rows}


def timeunit_to_schedule(rows, jobs, n_machines):
    """Parse time-unit grid → schedule dict."""
    schedule = {}
    for m in range(n_machines):
        row = rows[m]
        cur, start = None, None
        for t in range(1, len(row)):
            cell = row[t]
            if cell != cur:
                if cur and cur != '_':
                    j = int(cur[1:])
                    for o, (om, od) in enumerate(jobs[j]):
                        if om == m and (j, o) not in schedule:
                            schedule[(j, o)] = (m, start-1, t-1, od)
                            break
                cur, start = cell, t
        if cur and cur != '_':
            j = int(cur[1:])
            for o, (om, od) in enumerate(jobs[j]):
                if om == m and (j, o) not in schedule:
                    schedule[(j, o)] = (m, start-1, len(row)-1, od)
                    break
    return schedule


# ============================================================
# Text generation
# ============================================================

def make_puzzle_text(jobs, n_jobs, n_machines):
    """
    Natural language JSSP description — following Starjob (Abgaryan et al., 2025).
    Unlike Starjob (AR + explicit start/end times), output is a compact grid
    compatible with MDM's masked diffusion objective.
    """
    machine_names = ', '.join(f'Machine {m}' for m in range(n_machines))
    lines = [
        f'You have {n_jobs} jobs to schedule on {n_machines} machines '
        f'({machine_names}).',
        'Each job must visit every machine exactly once, in a fixed order.',
        'No two jobs can use the same machine at the same time.',
        'Goal: minimize the total completion time (makespan).',
        '',
    ]
    for j, job in enumerate(jobs):
        ops = ', '.join(
            f'Machine {m} for {d} time unit{"s" if d > 1 else ""}'
            for m, d in job
        )
        lines.append(f'Job {j} requires: {ops}.')
    lines += [
        '',
        'Fill in the processing order for each machine.',
        'Each row lists the jobs processed on that machine, in order.',
    ]
    return '\n'.join(lines)


def make_clues(jobs):
    """Natural language clue per job — matches make_puzzle_text format."""
    return [
        'Job {j} requires: {ops}.'.format(
            j=j,
            ops=', '.join(
                f'Machine {m} for {d} time unit{"s" if d > 1 else ""}'
                for m, d in job
            )
        )
        for j, job in enumerate(jobs)
    ]


def make_entry(jobs, n_jobs, n_machines, schedule, makespan, difficulty, size):
    return {
        'puzzle': make_puzzle_text(jobs, n_jobs, n_machines),
        'solution': schedule_to_compact(schedule, n_jobs, n_machines),
        'jobs': jobs,
        'n_jobs': n_jobs,
        'n_machines': n_machines,
        'makespan': makespan,
        'clue_texts': make_clues(jobs),
        'difficulty': difficulty,
        'size': size,
    }


# ============================================================
# Mode 1: Convert time-unit → compact
# ============================================================

def cmd_convert(args):
    print(f'Loading {args.input}...')
    with open(args.input) as f:
        data = json.load(f)
    print(f'  {len(data)} instances')

    out = []
    for i, inst in enumerate(data):
        jobs = inst['jobs']
        nj, nm = inst['n_jobs'], inst['n_machines']
        sched = timeunit_to_schedule(inst['solution']['rows'], jobs, nm)
        out.append(make_entry(jobs, nj, nm, sched,
                              inst['makespan'],
                              inst.get('difficulty','?'),
                              inst.get('size', f'{nj}x{nm}')))
        if (i+1) % 5000 == 0:
            print(f'  {i+1}/{len(data)}')

    with open(args.output, 'w') as f:
        json.dump(out, f)
    print(f'Saved {len(out)} → {args.output}')
    show_sample(out[0])


# ============================================================
# Mode 2: Generate from scratch
# ============================================================

CONFIGS = {
    'easy-2x2': (2,2,1,5), 'easy-3x2': (3,2,1,5), 'easy-3x3': (3,3,1,5),
    'med-4x3': (4,3,1,8), 'med-3x4': (3,4,1,8), 'med-4x4': (4,4,1,8),
    'hard-5x3': (5,3,2,10), 'hard-5x4': (5,4,2,10),
    'hard-5x5': (5,5,2,10), 'hard-6x4': (6,4,2,10),
    'expert-6x6': (6,6,2,12), 'expert-8x8': (8,8,3,15),
}

TRAIN_W = {
    'easy-2x2':.5, 'easy-3x2':1, 'easy-3x3':1.5,
    'med-4x3':2, 'med-3x4':1.5, 'med-4x4':2.5,
    'hard-5x3':2, 'hard-5x4':2.5, 'hard-5x5':2, 'hard-6x4':1.5,
    'expert-6x6':1, 'expert-8x8':.3,
}

EVAL_W = {
    'easy-3x3':1, 'med-4x3':1.5, 'med-4x4':2,
    'hard-5x4':2.5, 'hard-5x5':2, 'expert-6x6':1.5, 'expert-8x8':.5,
}


def pick(weights, rng):
    names, ws = list(weights.keys()), list(weights.values())
    r, c = rng.random() * sum(ws), 0
    for n, w in zip(names, ws):
        c += w
        if r <= c:
            return n
    return names[-1]


def solve_best(jobs, n_machines):
    sched, ms = solve_ortools(jobs, n_machines)
    if sched is None:
        sched, ms = solve_dispatch(jobs, n_machines)
    return sched, ms


def _gen_one(args_tuple):
    """Single-puzzle generation worker (for multiprocessing)."""
    i, name, seed_i, multi_opt = args_tuple
    rng = random.Random(seed_i)
    nj, nm, lo, hi = CONFIGS[name]
    jobs = []
    for j in range(nj):
        ms = list(range(nm))
        rng.shuffle(ms)
        jobs.append([(m, rng.randint(lo, hi)) for m in ms])

    sched, makespan = solve_best(jobs, nm)
    if sched is None:
        return None

    all_scheds = []
    if multi_opt:
        multi_timeout = 3 if nj * nm <= 20 else (5 if nj * nm <= 35 else 8)
        _, _, all_scheds = solve_ortools_multi(
            jobs, nm, max_solutions=6, timeout=multi_timeout)
    alt_solutions = [
        schedule_to_compact(s, nj, nm)
        for s in all_scheds[1:]
    ] if len(all_scheds) > 1 else []

    entry = make_entry(jobs, nj, nm, sched, makespan, name, f'{nj}x{nm}')
    entry['alt_solutions'] = alt_solutions
    return entry


def gen_dataset(n, weights, seed=42, multi_opt=False, n_workers=8):
    # The unified CLI generates one split at a time and explicitly requests
    # zero examples for the other split. Treat that as an empty split rather
    # than falling through to the summary statistics below.
    if n <= 0:
        return []

    # first, generate the full task list with the RNG (deterministic)
    rng = random.Random(seed)
    tasks = []
    for i in range(n):
        name = pick(weights, rng)
        # independent seed per puzzle (deterministic via seed + i)
        tasks.append((i, name, seed * 31337 + i, multi_opt))

    # parallel generation with multiprocessing
    import multiprocessing as mp
    n_cpu = min(n_workers, mp.cpu_count())
    if multi_opt and n_cpu > 1:
        print(f'  Using {n_cpu} workers (multi_opt=True)')
        with mp.Pool(n_cpu) as pool:
            results = []
            for j, entry in enumerate(pool.imap_unordered(_gen_one, tasks, chunksize=20)):
                results.append(entry)
                if (j+1) % 5000 == 0:
                    done = sum(1 for r in results if r is not None)
                    print(f'  {j+1}/{n} (valid: {done})', flush=True)
        data = [r for r in results if r is not None]
    else:
        # single thread (multi_opt=False is fast enough)
        data = []
        for j, task in enumerate(tasks):
            entry = _gen_one(task)
            if entry is not None:
                data.append(entry)
            if (j+1) % 5000 == 0:
                print(f'  {j+1}/{n}', flush=True)

    counts = defaultdict(int)
    for d in data:
        counts[d['difficulty']] += 1

    print(f'\n  Total: {len(data)}')
    for k in sorted(counts):
        print(f'    {k:15s}: {counts[k]}')
    cells = [d['n_jobs']*d['n_machines'] for d in data]
    print(f'  Grid cells: {min(cells)}~{max(cells)} '
          f'(mean {sum(cells)/len(cells):.0f})')
    return data



def cmd_generate(args):
    os.makedirs(args.output_dir, exist_ok=True)

    multi_opt  = getattr(args, 'multi_opt',  False)
    n_workers  = getattr(args, 'n_workers',  8)

    print('=== Train ===')
    train = gen_dataset(args.n_train, TRAIN_W, args.seed,
                        multi_opt=multi_opt, n_workers=n_workers)
    if args.n_train > 0:
        p = os.path.join(args.output_dir, 'jssp_train.json')
        with open(p, 'w') as f:
            json.dump(train, f)
        print(f'  → {p}')
        show_sample(train[0])

    print('\n=== Eval ===')
    ev = gen_dataset(args.n_eval, EVAL_W, args.seed + 100000,
                     multi_opt=False, n_workers=1)   # eval uses a single GT
    if args.n_eval > 0:
        p = os.path.join(args.output_dir, 'jssp_eval.json')
        with open(p, 'w') as f:
            json.dump(ev, f)
        print(f'  → {p}')
        show_sample(ev[0])


# ============================================================
# Util
# ============================================================

def show_sample(s):
    print(f'\n  Sample ({s["size"]}, makespan={s["makespan"]}):')
    print(f'    Grid ({s["n_machines"]}×{s["n_jobs"]} '
          f'= {s["n_machines"]*s["n_jobs"]} cells):')
    print(f'      {s["solution"]["header"]}')
    for row in s['solution']['rows']:
        print(f'      {row}')
    print(f'    Clues ({len(s["clue_texts"])}): {s["clue_texts"][:3]}')


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode')

    p1 = sub.add_parser('convert')
    p1.add_argument('--input', required=True)
    p1.add_argument('--output', required=True)

    p2 = sub.add_parser('generate')
    p2.add_argument('--output_dir', default='data/jssp')
    p2.add_argument('--n_train',   type=int, default=30000)
    p2.add_argument('--n_eval',    type=int, default=200)
    p2.add_argument('--seed',      type=int, default=42)
    p2.add_argument('--n_workers', type=int, default=8,
                    help='Number of parallel workers (only used when multi_opt=True)')
    p2.add_argument('--multi_opt', action='store_true', default=False,
                    help='Multiple optimal solution augmentation (slow, requires OR-Tools)')

    args = parser.parse_args()
    if args.mode == 'convert':
        cmd_convert(args)
    elif args.mode == 'generate':
        cmd_generate(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
