"""Confidence trigger for Blackboard inference (shared across all domains).

Blackboard runs greedy any-order decoding first, then a puzzle-level TRIGGER decides
whether the greedy output is already coherent (keep it) or the corrective search /
Best-of-N should fire:

    keep greedy  iff   late_phase_stat({C_theta(x_ti)}, rho)  >=  tau
    fire         otherwise

Two hyperparameters:
  * rho in [0, 1)  -- the *late-phase fraction*. The trigger reads confidence only over
                      commit steps i >= rho * N. Early steps are measured over a mostly
                      empty canvas and are not yet discriminative, so a late window is
                      used. Each domain supplies its selected rho.
  * tau            -- the confidence threshold.

Both rho and tau are selected jointly on held-out TRAINING traces (max F-score for the
"greedy-failed" class) and frozen before test; see `select_rho_tau`.

The late-phase statistic is task-dependent:
  * 'min'  -- ZebraLogic / JSSP use the minimum late-phase confidence (paper Algorithm 1).
  * 'mean' -- Nurse rostering uses the mean late-phase confidence.

`VALIDATION_SELECTED` records the (rho, tau, stat) chosen by joint validation for each
domain. These are the final paper's
reported trigger settings.
"""
from typing import Dict, List, Sequence, Tuple

DEFAULT_RHO = 2.0 / 3.0  # fallback only; use a domain policy for paper reproduction.

# domain -> (rho*, tau*, statistic) selected jointly on held-out training traces.
VALIDATION_SELECTED: Dict[str, Tuple[float, float, str]] = {
    "zebralogic":       (0.80, 1.00, "min"),
    "nurse_rostering":  (0.90, 0.95, "mean"),
    "jssp":             (0.50, 0.70, "min"),
}


def late_phase_stat(conf_trace: Sequence[float], rho: float = DEFAULT_RHO,
                    stat: str = "min") -> float:
    """Aggregate confidence over the late phase i >= rho*N.

    conf_trace : per-commit-step mean confidence C_theta(x_ti), in decoding order.
    rho        : late-phase fraction in [0, 1). rho = DEFAULT_RHO is only a fallback; paper callers pass their domain policy.
    stat       : 'min' or 'mean'.

    Returns 1.0 for an empty trace / empty late window (i.e. "fully confident").
    """
    n = len(conf_trace)
    if n == 0:
        return 1.0
    lo = max(1, int(rho * n))
    tail = [c for c in conf_trace[lo:] if c is not None]
    if not tail:
        return 1.0
    return min(tail) if stat == "min" else sum(tail) / len(tail)


def fires(conf_trace: Sequence[float], rho: float = DEFAULT_RHO, tau: float = 1.0,
          stat: str = "min") -> bool:
    """True if the trigger fires (greedy looks unreliable -> run correction)."""
    return late_phase_stat(conf_trace, rho, stat) < tau


def blackboard_infer(greedy, correct, rho: float = DEFAULT_RHO, tau: float = 1.0,
                     stat: str = "min"):
    """Online Blackboard inference — paper Algorithm 1, as a single call.

    Run greedy once; if the late-phase confidence trigger fires, spend the correction,
    otherwise keep the greedy output. This is the inference-time path (as opposed to
    `apply_trigger`, which re-assembles the same decision offline from collected traces).

        greedy()  -> (result, conf_trace)   # conf_trace = per-commit-step mean confidence
        correct() -> result                 # SEARCH+BACKTRACK (feasibility) or Best-of-N (optimization)

    Returns (result, fired). The correction closure is only evaluated when the trigger
    fires, so greedy-only puzzles cost nothing extra.
    """
    result, conf_trace = greedy()
    if fires(conf_trace, rho, tau, stat):
        return correct(), True
    return result, False


def select_rho_tau(train_traces: List[dict], stat: str = "min",
                   beta: float = 1.0,
                   rho_grid: Sequence[float] = tuple(i / 20 for i in range(0, 20)),
                   ) -> Tuple[float, float, float]:
    """Jointly select (rho, tau) on held-out training traces.

    Positive class = greedy FAILED. We grid over rho and, for each, over the observed
    late-statistic values as tau candidates, and keep the pair maximizing F_beta of the
    "predict-fail-if stat < tau" rule. beta < 1 favors precision (avoid firing on solved
    puzzles); beta = 1 is plain F1 (used in the paper's joint selection).

    Each trace dict must provide `conf` (the per-step confidence list) and `solved` (bool).
    Returns (rho*, tau*, best_fscore).
    """
    beta2 = beta * beta
    n_fail = sum(1 for t in train_traces if not t["solved"])
    best = (-1.0, DEFAULT_RHO, 1.0)  # (fscore, rho, tau)
    for rho in rho_grid:
        stats = [(late_phase_stat(t["conf"], rho, stat), t["solved"]) for t in train_traces]
        for tau in sorted({round(s, 4) for s, _ in stats}):
            tp = sum(1 for s, solved in stats if (not solved) and s < tau)
            fp = sum(1 for s, solved in stats if solved and s < tau)
            if tp == 0:
                continue
            prec = tp / (tp + fp)
            rec = tp / max(n_fail, 1)
            denom = beta2 * prec + rec
            fb = (1 + beta2) * prec * rec / denom if denom > 0 else 0.0
            if fb > best[0]:
                best = (fb, rho, tau)
    return best[1], best[2], best[0]


def apply_trigger(per_puzzle: List[dict], rho: float, tau: float,
                  stat: str = "min") -> dict:
    """Assemble the triggered result: keep greedy where the trigger stays silent, use
    the corrected output where it fires.

    per_puzzle : list of dicts with keys
        `conf`            -- greedy per-step confidence trace,
        `greedy_solved`   -- bool (or a greedy metric),
        `corrected_solved`-- bool (or a corrected metric: cascade / BoN output).
    Returns aggregate counts + the fire rate, and the per-puzzle final outcome.
    """
    final, n_fire = [], 0
    for p in per_puzzle:
        fire = fires(p["conf"], rho, tau, stat)
        n_fire += int(fire)
        final.append(p["corrected_solved"] if fire else p["greedy_solved"])
    n = len(per_puzzle)
    return {
        "rho": rho, "tau": tau, "stat": stat,
        "n": n, "fire_rate": n_fire / n if n else 0.0,
        "solve_rate": sum(bool(x) for x in final) / n if n else 0.0,
        "final": final,
    }
