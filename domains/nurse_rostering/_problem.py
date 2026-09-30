"""
Nurse-Rostering (NR) puzzle generator.

Structural counterpart to the ZebraLogic generator (same generation recipe:
cell-domain propagation -> add ground-truth-consistent clues at ambiguous
cells until the grid is uniquely determined -> minimize clues under a time
budget so that difficulty rises while uniqueness is preserved), but with a
constraint vocabulary disjoint from ZebraLogic's positional predicates:

  Family A (relational, ZL-analogue):
      FixedAssign, NotAssign, SameCell, DiffCell,
      NeverSamePair, AlwaysSamePair
  Family B (temporal / succession, directional -- absent in ZL):
      ForbiddenSucc  ("night is never immediately followed by day"),
      RequiredSucc   ("night is always immediately followed by off")
  Family C (cardinality / coverage, non-binary -- absent in ZL):
      RowCount   ("S2 works night on exactly 2 days"),
      ColCount   ("exactly 1 staff member works night on D3"),
      MaxConsecWork ("S1 never works more than 3 consecutive days")

Grid semantics differ from ZL: rows = staff, columns = days, cell = shift.
Cells are FREE assignments (a shift may repeat within a row and within a
column), so ZL's permutation-based propagation does not apply; each clue
class carries its own arc/cardinality/window propagator.

Uniqueness guarantee:
  The generation loop adds clues until pure propagation solves the grid,
  which implies a unique solution. Minimization then removes clues while
  re-verifying uniqueness with branch-and-propagate solution counting
  (capped at 2). The minimized instance is typically no longer solvable by
  propagation alone -- this is the knob that raises solver conflict counts
  at fixed search-space size, mirroring the ZebraLogic-Hard recipe.

Interface parity with the ZL module:
  DifficultyConfig, CURRICULUM, sample_difficulty, generate_roster_problem
"""

import random
import time
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

# -----------------------------------------------------------------
# Shift vocabularies.
# All values are common lowercase English words expected to encode as a
# single token with a leading space (G-form) in both the LLaDA and LLaMA
# tokenizers.  Run nurse_rostering_encode.verify_shift_tokens(tokenizer) once per
# tokenizer before training (same convention as ZL's single-token check).
# "off" MUST be last (index n_shifts-1); it is the designated rest shift
# used by MaxConsecWork.
# -----------------------------------------------------------------
SHIFTS_3 = ["day", "night", "off"]
SHIFTS_4 = ["day", "late", "night", "off"]

CONTRADICTION = "CONTRADICTION"

# Clue family taxonomy used by the generator.
#   A = relational (ZL analog), B = temporal succession, C = cardinality/coverage
KIND_FAMILY = {
    "fixed_assign": "A", "not_assign": "A", "same_cell": "A", "diff_cell": "A",
    "never_same_pair": "A", "always_same_pair": "A",
    "forbidden_succ": "B", "required_succ": "B",
    "row_count": "C", "col_count": "C", "max_consec_work": "C",
}


def serialize_clue(clue):
    """JSON-safe clue metadata for trace/actionability analysis (H1a family
    attribution): kind, family (A/B/C), and the grid cells it touches."""
    return {
        "kind": clue.kind,
        "family": KIND_FAMILY.get(clue.kind, "?"),
        "cells": sorted([list(c) for c in clue.cells()]),
        "text": clue.text(),
    }


def _staff(s):
    return f"S{s + 1}"


def _day(d):
    return f"D{d + 1}"


def _a(word):
    return f"an {word}" if word[0] in "aeiou" else f"a {word}"


# -----------------------------------------------------------------
# Clue classes.
# Each clue implements:
#   holds(grid)      -> bool     : true of the ground-truth grid?
#   propagate(dom)   -> bool|CONTRADICTION : one round of domain pruning;
#                                  returns True if any domain changed
#   text()           -> str      : natural-language clue
#   cells()          -> set[(s,d)]: grid cells the clue touches
#                                  (constraint-graph analyses, cf. Fig. 6)
#   kind             -> str
# `dom` is a list-of-list of sets of shift indices, dom[s][d].
# -----------------------------------------------------------------

class Clue:
    kind = "base"

    def holds(self, grid):
        raise NotImplementedError

    def propagate(self, dom):
        raise NotImplementedError

    def text(self):
        raise NotImplementedError

    def cells(self):
        raise NotImplementedError

    def _assign(self, dom, s, d, allowed: Set[int]):
        """Intersect dom[s][d] with `allowed`. Returns changed | CONTRADICTION."""
        cur = dom[s][d]
        new = cur & allowed
        if not new:
            return CONTRADICTION
        if new != cur:
            dom[s][d] = new
            return True
        return False


class FixedAssign(Clue):
    kind = "fixed_assign"

    def __init__(self, s, d, v, shifts):
        self.s, self.d, self.v, self.shifts = s, d, v, shifts

    def holds(self, grid):
        return grid[self.s][self.d] == self.v

    def propagate(self, dom):
        return self._assign(dom, self.s, self.d, {self.v})

    def text(self):
        return f"{_staff(self.s)} works {self.shifts[self.v]} on {_day(self.d)}."

    def cells(self):
        return {(self.s, self.d)}


class NotAssign(Clue):
    kind = "not_assign"

    def __init__(self, s, d, v, shifts):
        self.s, self.d, self.v, self.shifts = s, d, v, shifts

    def holds(self, grid):
        return grid[self.s][self.d] != self.v

    def propagate(self, dom):
        cur = dom[self.s][self.d]
        if self.v in cur:
            if len(cur) == 1:
                return CONTRADICTION
            dom[self.s][self.d] = cur - {self.v}
            return True
        return False

    def text(self):
        return f"{_staff(self.s)} does not work {self.shifts[self.v]} on {_day(self.d)}."

    def cells(self):
        return {(self.s, self.d)}


class SameCell(Clue):
    kind = "same_cell"

    def __init__(self, s1, d1, s2, d2, shifts):
        self.s1, self.d1, self.s2, self.d2, self.shifts = s1, d1, s2, d2, shifts

    def holds(self, grid):
        return grid[self.s1][self.d1] == grid[self.s2][self.d2]

    def propagate(self, dom):
        inter = dom[self.s1][self.d1] & dom[self.s2][self.d2]
        if not inter:
            return CONTRADICTION
        changed = False
        for (s, d) in [(self.s1, self.d1), (self.s2, self.d2)]:
            r = self._assign(dom, s, d, inter)
            if r == CONTRADICTION:
                return CONTRADICTION
            changed |= r
        return changed

    def text(self):
        if self.d1 == self.d2:
            return (f"{_staff(self.s1)} and {_staff(self.s2)} work the same "
                    f"shift on {_day(self.d1)}.")
        if self.s1 == self.s2:
            return (f"{_staff(self.s1)} works the same shift on {_day(self.d1)} "
                    f"and {_day(self.d2)}.")
        return (f"{_staff(self.s1)}'s shift on {_day(self.d1)} is the same as "
                f"{_staff(self.s2)}'s shift on {_day(self.d2)}.")

    def cells(self):
        return {(self.s1, self.d1), (self.s2, self.d2)}


class DiffCell(Clue):
    kind = "diff_cell"

    def __init__(self, s1, d1, s2, d2, shifts):
        self.s1, self.d1, self.s2, self.d2, self.shifts = s1, d1, s2, d2, shifts

    def holds(self, grid):
        return grid[self.s1][self.d1] != grid[self.s2][self.d2]

    def propagate(self, dom):
        changed = False
        a, b = dom[self.s1][self.d1], dom[self.s2][self.d2]
        if len(a) == 1 and len(b) == 1 and a == b:
            return CONTRADICTION
        if len(a) == 1:
            v = next(iter(a))
            if v in b:
                if len(b) == 1:
                    return CONTRADICTION
                dom[self.s2][self.d2] = b - {v}
                changed = True
        b = dom[self.s2][self.d2]
        if len(b) == 1:
            v = next(iter(b))
            a = dom[self.s1][self.d1]
            if v in a:
                if len(a) == 1:
                    return CONTRADICTION
                dom[self.s1][self.d1] = a - {v}
                changed = True
        return changed

    def text(self):
        if self.d1 == self.d2:
            return (f"{_staff(self.s1)} and {_staff(self.s2)} work different "
                    f"shifts on {_day(self.d1)}.")
        if self.s1 == self.s2:
            return (f"{_staff(self.s1)} works different shifts on "
                    f"{_day(self.d1)} and {_day(self.d2)}.")
        return (f"{_staff(self.s1)}'s shift on {_day(self.d1)} is different "
                f"from {_staff(self.s2)}'s shift on {_day(self.d2)}.")

    def cells(self):
        return {(self.s1, self.d1), (self.s2, self.d2)}


class ForbiddenSucc(Clue):
    """For staff s, shift y never immediately follows shift x (directional)."""
    kind = "forbidden_succ"

    def __init__(self, s, x, y, n_days, shifts):
        self.s, self.x, self.y, self.n_days, self.shifts = s, x, y, n_days, shifts

    def holds(self, grid):
        row = grid[self.s]
        return not any(row[d] == self.x and row[d + 1] == self.y
                       for d in range(self.n_days - 1))

    def propagate(self, dom):
        changed = False
        for d in range(self.n_days - 1):
            a, b = dom[self.s][d], dom[self.s][d + 1]
            if a == {self.x} and self.y in b:
                if len(b) == 1:
                    return CONTRADICTION
                dom[self.s][d + 1] = b - {self.y}
                changed = True
            b = dom[self.s][d + 1]
            if b == {self.y} and self.x in dom[self.s][d]:
                a = dom[self.s][d]
                if len(a) == 1:
                    return CONTRADICTION
                dom[self.s][d] = a - {self.x}
                changed = True
        return changed

    def text(self):
        return (f"For {_staff(self.s)}, {_a(self.shifts[self.x])} shift is never "
                f"immediately followed by {_a(self.shifts[self.y])} shift.")

    def cells(self):
        return {(self.s, d) for d in range(self.n_days)}


class RequiredSucc(Clue):
    """For staff s, whenever x is worked (before the last day), the next day is y."""
    kind = "required_succ"

    def __init__(self, s, x, y, n_days, shifts):
        self.s, self.x, self.y, self.n_days, self.shifts = s, x, y, n_days, shifts

    def holds(self, grid):
        row = grid[self.s]
        return all(row[d] != self.x or row[d + 1] == self.y
                   for d in range(self.n_days - 1))

    def propagate(self, dom):
        changed = False
        for d in range(self.n_days - 1):
            if dom[self.s][d] == {self.x}:
                r = self._assign(dom, self.s, d + 1, {self.y})
                if r == CONTRADICTION:
                    return CONTRADICTION
                changed |= r
            if self.y not in dom[self.s][d + 1]:
                a = dom[self.s][d]
                if self.x in a:
                    if len(a) == 1:
                        return CONTRADICTION
                    dom[self.s][d] = a - {self.x}
                    changed = True
        return changed

    def text(self):
        return (f"For {_staff(self.s)}, {_a(self.shifts[self.x])} shift is always "
                f"immediately followed by {_a(self.shifts[self.y])} shift.")

    def cells(self):
        return {(self.s, d) for d in range(self.n_days)}


class _CountBase(Clue):
    """Exactly-k cardinality over a fixed cell set for a fixed shift value."""

    def __init__(self, cell_list, x, k, shifts):
        self.cell_list, self.x, self.k, self.shifts = cell_list, x, k, shifts

    def holds(self, grid):
        return sum(grid[s][d] == self.x for (s, d) in self.cell_list) == self.k

    def propagate(self, dom):
        fixed, open_cells = 0, []
        for (s, d) in self.cell_list:
            cur = dom[s][d]
            if cur == {self.x}:
                fixed += 1
            elif self.x in cur:
                open_cells.append((s, d))
        possible = fixed + len(open_cells)
        if fixed > self.k or possible < self.k:
            return CONTRADICTION
        changed = False
        if fixed == self.k:
            for (s, d) in open_cells:
                dom[s][d] = dom[s][d] - {self.x}
                if not dom[s][d]:
                    return CONTRADICTION
                changed = True
        elif possible == self.k:
            for (s, d) in open_cells:
                dom[s][d] = {self.x}
                changed = True
        return changed

    def cells(self):
        return set(self.cell_list)


class RowCount(_CountBase):
    kind = "row_count"

    def __init__(self, s, x, k, n_days, shifts):
        super().__init__([(s, d) for d in range(n_days)], x, k, shifts)
        self.s = s

    def text(self):
        name = self.shifts[self.x]
        if self.k == 0:
            return f"{_staff(self.s)} never works {_a(name)} shift."
        unit = "day" if self.k == 1 else "days"
        return f"{_staff(self.s)} works {name} on exactly {self.k} {unit}."


class ColCount(_CountBase):
    kind = "col_count"

    def __init__(self, d, x, c, n_staff, shifts):
        super().__init__([(s, d) for s in range(n_staff)], x, c, shifts)
        self.d = d

    def text(self):
        name = self.shifts[self.x]
        if self.k == 0:
            return f"No staff member works {name} on {_day(self.d)}."
        unit = "staff member" if self.k == 1 else "staff members"
        return f"Exactly {self.k} {unit} work {name} on {_day(self.d)}."


class NeverSamePair(Clue):
    kind = "never_same_pair"

    def __init__(self, s1, s2, n_days, shifts):
        self.s1, self.s2, self.n_days, self.shifts = s1, s2, n_days, shifts
        self._sub = [DiffCell(s1, d, s2, d, shifts) for d in range(n_days)]

    def holds(self, grid):
        return all(grid[self.s1][d] != grid[self.s2][d] for d in range(self.n_days))

    def propagate(self, dom):
        changed = False
        for c in self._sub:
            r = c.propagate(dom)
            if r == CONTRADICTION:
                return CONTRADICTION
            changed |= r
        return changed

    def text(self):
        return (f"{_staff(self.s1)} and {_staff(self.s2)} never work the same "
                f"shift on the same day.")

    def cells(self):
        return {(s, d) for s in (self.s1, self.s2) for d in range(self.n_days)}


class AlwaysSamePair(Clue):
    kind = "always_same_pair"

    def __init__(self, s1, s2, n_days, shifts):
        self.s1, self.s2, self.n_days, self.shifts = s1, s2, n_days, shifts
        self._sub = [SameCell(s1, d, s2, d, shifts) for d in range(n_days)]

    def holds(self, grid):
        return all(grid[self.s1][d] == grid[self.s2][d] for d in range(self.n_days))

    def propagate(self, dom):
        changed = False
        for c in self._sub:
            r = c.propagate(dom)
            if r == CONTRADICTION:
                return CONTRADICTION
            changed |= r
        return changed

    def text(self):
        return (f"{_staff(self.s1)} and {_staff(self.s2)} always work the same "
                f"shift as each other.")

    def cells(self):
        return {(s, d) for s in (self.s1, self.s2) for d in range(self.n_days)}


class MaxConsecWork(Clue):
    """Staff s never works more than k consecutive days (every window of
    k+1 consecutive days contains at least one 'off')."""
    kind = "max_consec_work"

    def __init__(self, s, k, n_days, shifts, off_idx):
        self.s, self.k, self.n_days, self.shifts = s, k, n_days, shifts
        self.off = off_idx

    def holds(self, grid):
        run = 0
        for d in range(self.n_days):
            run = run + 1 if grid[self.s][d] != self.off else 0
            if run > self.k:
                return False
        return True

    def propagate(self, dom):
        changed = False
        for start in range(self.n_days - self.k):
            window = range(start, start + self.k + 1)
            can_off = [d for d in window if self.off in dom[self.s][d]]
            if not can_off:
                return CONTRADICTION
            if len(can_off) == 1:
                d = can_off[0]
                if dom[self.s][d] != {self.off}:
                    dom[self.s][d] = {self.off}
                    changed = True
        return changed

    def text(self):
        unit = "day" if self.k == 1 else "days"
        return (f"{_staff(self.s)} never works more than {self.k} "
                f"consecutive {unit}.")

    def cells(self):
        return {(self.s, d) for d in range(self.n_days)}


# -----------------------------------------------------------------
# Propagation / solving
# -----------------------------------------------------------------

def full_domains(n_staff, n_days, n_shifts):
    return [[set(range(n_shifts)) for _ in range(n_days)] for _ in range(n_staff)]


def copy_domains(dom):
    return [[set(c) for c in row] for row in dom]


def propagate_all(clues, dom):
    """Run all propagators to fixpoint. Returns True | CONTRADICTION."""
    changed = True
    while changed:
        changed = False
        for c in clues:
            r = c.propagate(dom)
            if r == CONTRADICTION:
                return CONTRADICTION
            changed |= r
    return True


def domain_status(dom):
    """Returns ('solved'|'open'|'dead', list of ambiguous (s,d))."""
    ambiguous = []
    for s, row in enumerate(dom):
        for d, cell in enumerate(row):
            if len(cell) == 0:
                return "dead", []
            if len(cell) > 1:
                ambiguous.append((s, d))
    return ("solved" if not ambiguous else "open"), ambiguous


def count_solutions(clues, n_staff, n_days, n_shifts, limit=2):
    """Branch-and-propagate solution counting, capped at `limit`.
    Used for the uniqueness check during minimization (ZL recipe)."""
    dom0 = full_domains(n_staff, n_days, n_shifts)
    if propagate_all(clues, dom0) == CONTRADICTION:
        return 0
    count = 0
    stack = [dom0]
    while stack and count < limit:
        dom = stack.pop()
        status, ambiguous = domain_status(dom)
        if status == "dead":
            continue
        if status == "solved":
            count += 1
            continue
        # branch on the smallest-domain ambiguous cell
        s, d = min(ambiguous, key=lambda sd: len(dom[sd[0]][sd[1]]))
        for v in sorted(dom[s][d]):
            child = copy_domains(dom)
            child[s][d] = {v}
            if propagate_all(clues, child) != CONTRADICTION:
                stack.append(child)
    return count


# -----------------------------------------------------------------
# Candidate clue enumeration (level-gated, mirroring ZL's rule table)
#
#   L1: FixedAssign, SameCell/DiffCell (same-day partners)
#   L2: + NotAssign, SameCell/DiffCell (arbitrary partners)
#   L3: + ForbiddenSucc, RequiredSucc          (Family B)
#   L4: + RowCount                              (Family C, row)
#   L5: + ColCount                              (Family C, column)
#   L6: + NeverSamePair / AlwaysSamePair
#   L7: + MaxConsecWork
#   L8: drop FixedAssign   (forces harder clue mixes, cf. ZL pop())
#   L9: drop same-day SameCell/DiffCell
# -----------------------------------------------------------------

def _partner_cells(s, d, n_staff, n_days, same_day_only):
    out = [(s2, d) for s2 in range(n_staff) if s2 != s]
    if not same_day_only:
        out += [(s, d2) for d2 in range(n_days) if d2 != d]
        others = [(s2, d2) for s2 in range(n_staff) for d2 in range(n_days)
                  if s2 != s and d2 != d]
        out += random.sample(others, min(4, len(others)))
    return out


def _true_succ_pairs(row, n_shifts, required):
    """(x, y) pairs making ForbiddenSucc / RequiredSucc true and non-vacuous."""
    n_days = len(row)
    succ = set()
    xs_present = set()
    for d in range(n_days - 1):
        succ.add((row[d], row[d + 1]))
        xs_present.add(row[d])
    pairs = []
    for x in xs_present:                      # non-vacuous: x occurs before last day
        followers = {y for (a, y) in succ if a == x}
        for y in range(n_shifts):
            if required:
                if followers == {y}:
                    pairs.append((x, y))
            else:
                if y not in followers:
                    pairs.append((x, y))
    return pairs


def _max_run(row, off_idx):
    run = best = 0
    for v in row:
        run = run + 1 if v != off_idx else 0
        best = max(best, run)
    return best


def candidate_clues(grid, s, d, level, n_staff, n_days, n_shifts, shifts):
    """Enumerate clues that are true of `grid`, touch ambiguous cell (s, d),
    and are admitted by `level`."""
    v = grid[s][d]
    off = n_shifts - 1
    cands = []

    if level < 8:
        cands.append(FixedAssign(s, d, v, shifts))

    if level >= 2:
        for u in range(n_shifts):
            if u != v:
                cands.append(NotAssign(s, d, u, shifts))

    same_day_only = level < 2
    if level < 9 or not same_day_only:
        for (s2, d2) in _partner_cells(s, d, n_staff, n_days,
                                       same_day_only=same_day_only):
            if level >= 9 and d2 == d and s2 != s:
                continue                      # same-day relational dropped at L9
            if grid[s2][d2] == v:
                cands.append(SameCell(s, d, s2, d2, shifts))
            else:
                cands.append(DiffCell(s, d, s2, d2, shifts))

    if level >= 3:
        row = grid[s]
        for (x, y) in _true_succ_pairs(row, n_shifts, required=False):
            cands.append(ForbiddenSucc(s, x, y, n_days, shifts))
        for (x, y) in _true_succ_pairs(row, n_shifts, required=True):
            cands.append(RequiredSucc(s, x, y, n_days, shifts))

    if level >= 4:
        k = sum(1 for u in grid[s] if u == v)
        cands.append(RowCount(s, v, k, n_days, shifts))
        u = random.randrange(n_shifts)
        cands.append(RowCount(s, u, sum(1 for w in grid[s] if w == u),
                              n_days, shifts))

    if level >= 5:
        c = sum(1 for s2 in range(n_staff) if grid[s2][d] == v)
        cands.append(ColCount(d, v, c, n_staff, shifts))
        u = random.randrange(n_shifts)
        cands.append(ColCount(d, u,
                              sum(1 for s2 in range(n_staff) if grid[s2][d] == u),
                              n_staff, shifts))

    if level >= 6:
        for s2 in range(n_staff):
            if s2 == s:
                continue
            if all(grid[s][dd] != grid[s2][dd] for dd in range(n_days)):
                cands.append(NeverSamePair(min(s, s2), max(s, s2), n_days, shifts))
            if all(grid[s][dd] == grid[s2][dd] for dd in range(n_days)):
                cands.append(AlwaysSamePair(min(s, s2), max(s, s2), n_days, shifts))

    if level >= 7:
        k = _max_run(grid[s], off)
        if 0 < k < n_days:                    # non-trivial only
            cands.append(MaxConsecWork(s, k, n_days, shifts, off))

    return cands


# -----------------------------------------------------------------
# Generation (mirrors generate_puzzle in the ZL module)
# -----------------------------------------------------------------

def _dedup_key(clue):
    return (clue.kind, clue.text())


def generate_roster(n_staff, n_days, n_shifts, level=3, *,
                    minimal_conditions=True, max_seconds_for_minimizing=15.0,
                    max_clue_attempts=400):
    """
    Returns (grid, clues) where `grid` is the unique solution of `clues`.

    Recipe (== ZL): sample ground truth; add true clues at ambiguous cells
    until propagation solves the grid; then greedily remove clues while
    uniqueness (checked by branch-and-propagate, cap 2) is preserved.
    """
    if n_shifts == 3:
        shifts = SHIFTS_3
    elif n_shifts == 4:
        shifts = SHIFTS_4
    else:
        raise ValueError("n_shifts must be 3 or 4")

    grid = [[random.randrange(n_shifts) for _ in range(n_days)]
            for _ in range(n_staff)]

    dom = full_domains(n_staff, n_days, n_shifts)
    clues: List[Clue] = []
    seen = set()

    attempts = 0
    while True:
        attempts += 1
        if attempts > max_clue_attempts:
            raise RuntimeError("clue insertion did not converge")
        propagate_all(clues, dom)             # true clues cannot kill the truth
        status, ambiguous = domain_status(dom)
        assert status != "dead", "true clues eliminated the ground truth"
        if status == "solved":
            break
        s, d = random.choice(ambiguous)
        cands = [c for c in candidate_clues(grid, s, d, level, n_staff,
                                            n_days, n_shifts, shifts)
                 if _dedup_key(c) not in seen]
        if not cands:
            continue
        clue = random.choice(cands)
        assert clue.holds(grid), f"generated clue false of truth: {clue.text()}"
        clues.append(clue)
        seen.add(_dedup_key(clue))

    if minimal_conditions:
        clues = _minimize(clues, grid, n_staff, n_days, n_shifts,
                          max_seconds_for_minimizing)

    return grid, clues, shifts


def _minimize(clues, grid, n_staff, n_days, n_shifts, budget_s):
    """Randomized greedy clue removal under a time budget; uniqueness is
    re-verified by count_solutions(limit=2) after each tentative removal."""
    start = time.monotonic()
    clues = list(clues)
    improved = True
    while improved and time.monotonic() - start < budget_s:
        improved = False
        order = list(range(len(clues)))
        random.shuffle(order)
        for i in order:
            if time.monotonic() - start >= budget_s:
                break
            trial = clues[:i] + clues[i + 1:]
            if count_solutions(trial, n_staff, n_days, n_shifts, limit=2) == 1:
                clues = trial
                improved = True
                break
    return clues


def generate_roster_problem(n_staff=4, n_days=5, n_shifts=4, level=3,
                            minimal_conditions=True,
                            max_seconds_for_minimizing=15.0):
    """ZL-parity entry point. Returns a raw problem dict."""
    grid, clues, shifts = generate_roster(
        n_staff, n_days, n_shifts, level,
        minimal_conditions=minimal_conditions,
        max_seconds_for_minimizing=max_seconds_for_minimizing,
    )
    clue_texts = [c.text() for c in clues]
    random.shuffle(clue_texts)
    return {
        "grid": grid,                          # [[shift_idx]*n_days]*n_staff
        "clues": clues,                        # Clue objects (Z3 / graph use)
        "clue_texts": clue_texts,              # shuffled NL clues
        "shifts": shifts,
        "n_staff": n_staff,
        "n_days": n_days,
        "n_shifts": n_shifts,
        "level": level,
    }


# -----------------------------------------------------------------
# Difficulty curriculum (ZL parity)
# -----------------------------------------------------------------

@dataclass
class DifficultyConfig:
    name: str
    n_staff: int
    n_days: int
    n_shifts: int
    level: int
    weight: float = 1.0


# The held-out evaluator groups instances by constraint difficulty.
CURRICULUM = [
    # reachable, relational+succession core
    DifficultyConfig("3x5-s4-L4", 3, 5, 4, 4, weight=1.5),
    DifficultyConfig("4x4-s4-L5", 4, 4, 4, 5, weight=1.5),
    DifficultyConfig("4x5-s4-L5", 4, 5, 4, 5, weight=2.0),
    # full vocab, harder clue mixes at the SAME reachable grids
    DifficultyConfig("4x5-s4-L7", 4, 5, 4, 7, weight=2.0),
    DifficultyConfig("3x7-s4-L6", 3, 7, 4, 6, weight=1.5),   # a week, 3 staff
    DifficultyConfig("4x6-s4-L7", 4, 6, 4, 7, weight=2.0),
    # a week, 4 staff -- the high-conflict tail (search-hard, still 28 cells)
    DifficultyConfig("4x7-s4-L7", 4, 7, 4, 7, weight=2.0),
    DifficultyConfig("4x7-s4-L9", 4, 7, 4, 9, weight=1.5),
]



def sample_difficulty(curriculum: Optional[List[DifficultyConfig]] = None):
    if curriculum is None:
        curriculum = CURRICULUM
    weights = [d.weight for d in curriculum]
    return random.choices(curriculum, weights=weights, k=1)[0]


__all__ = [
    "SHIFTS_3", "SHIFTS_4",
    "DifficultyConfig", "CURRICULUM", "sample_difficulty",
    "generate_roster_problem", "generate_roster",
    "count_solutions", "full_domains", "propagate_all", "domain_status",
    "KIND_FAMILY", "serialize_clue",
]
