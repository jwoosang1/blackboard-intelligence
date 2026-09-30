import os
"""
=============================================================================
Blackboard SFT data builder — JSSP
=============================================================================

Cell value: J0→'a', J1→'b', ..., J9→'j'  (LLaDA single token ✅)
Clue text is kept as fixed natural-language context (not masked).

Grid format:
  | Machine | Slot_1 | Slot_2 | ... |
  | M0      |   b    |   d    | ... |
  | M1      |   a    |   c    | ... |

Output: same dict structure as ZebraLogic's build_tokenized_sample.
"""

import random
from typing import Dict, List, Optional, Tuple

import torch



def jssp_encode_cell(val: str, tokenizer) -> int:
    """
    JSSP cell value ('0'-'9') → token id.
    Encoded without a space prefix (differs from ZebraLogic's encode_cell_value).
    '0'→15, '1'→16, ... (based on the LLaDA tokenizer)
    """
    ids = tokenizer.encode(val, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(
            f"JSSP cell '{val}' encodes to {len(ids)} tokens {ids}, expected 1"
        )
    return ids[0]


def jssp_decode_cell(token_id: int, tokenizer) -> Optional[str]:
    """token id → cell value string ('0'-'9'). None if not valid."""""
    raw = tokenizer.convert_ids_to_tokens(token_id)
    if raw is None:
        return None
    val = raw.lstrip('\u0120').strip()
    return val if val.isdigit() else None


# =============================================================================
# 1. Text + Token Encoding
# =============================================================================

def make_prompt_jssp(puzzle_text: str, solution: dict) -> str:
    header = solution["header"]
    header_row    = " | ".join(header)
    separator_row = " | ".join(["---"] * len(header))
    return (
        f"### SYSTEM:\n"
        f"You are a precision scheduling solver engine.\n"
        f"Output ONLY a valid Markdown table as the solution.\n"
        f"\n"
        f"### PUZZLE CONTEXT:\n"
        f"{puzzle_text}\n"
        f"\n"
        f"### FINAL SOLUTION:\n"
        f"| {header_row} |\n"
        f"| {separator_row} |\n"
    )


def encode_prompt_jssp(puzzle_text: str, solution: dict, tokenizer) -> List[int]:
    return tokenizer.encode(make_prompt_jssp(puzzle_text, solution),
                            add_special_tokens=False)


def encode_table_body_jssp(solution: dict, tokenizer) -> Tuple[List[int], List[Dict]]:
    """
    solution['rows'] = [['M0', 'b', 'd', 'a', ...], ...]
    cell value = single alphabet char = 1 token.
    Returns: token_ids, cell_map
    """
    rows    = solution["rows"]
    n_slots = len(rows[0]) - 1  # skip machine label

    tokens   = []
    cell_map = []

    for machine_idx, row in enumerate(rows):
        tokens.extend(tokenizer.encode(f"| {row[0]} |", add_special_tokens=False))

        for slot_idx in range(n_slots):
            val    = row[slot_idx + 1]
            val_id = jssp_encode_cell(val, tokenizer)

            cell_map.append({
                "token_pos": len(tokens),
                "row":       machine_idx,
                "col":       slot_idx,
                "value":     val,
                "token_id":  val_id,
            })
            tokens.append(val_id)

            sep = " |" if slot_idx < n_slots - 1 else " |\n"
            tokens.extend(tokenizer.encode(sep, add_special_tokens=False))

    return tokens, cell_map


# =============================================================================
# 2. Error Injection
# =============================================================================

def _inject_errors_jssp(
    corrupted_ids: List[int],
    cell_map:      List[Dict],
    mask_cells:    set,
    n_jobs:        int,
    n_error:       int,
    tokenizer,
) -> Tuple[set, List[Dict]]:
    """Row-wise swap/cycle/single error injection."""
    pos_lookup = {(cm["row"], cm["col"]): cm for cm in cell_map}
    domain     = [str(j) for j in range(n_jobs)]

    error_cells         = set()
    error_details       = []
    corrupted_positions = set(mask_cells)
    cells_used          = 0

    clean_by_machine: Dict[int, List] = {}
    for cm in cell_map:
        key = (cm["row"], cm["col"])
        if key not in mask_cells:
            clean_by_machine.setdefault(cm["row"], []).append(key)

    eligible = [m for m, cells in clean_by_machine.items() if len(cells) >= 2]
    random.shuffle(eligible)
    n_corrupt = min(len(eligible), random.choice([1, 1, 2]))

    def _true_val(m, s):
        return pos_lookup[(m, s)]["value"]

    def _swap(m, sa, sb):
        nonlocal cells_used
        va, vb = _true_val(m, sa), _true_val(m, sb)
        corrupted_ids[pos_lookup[(m, sa)]["token_pos"]] = jssp_encode_cell(vb, tokenizer)
        corrupted_ids[pos_lookup[(m, sb)]["token_pos"]] = jssp_encode_cell(va, tokenizer)
        for s, inj in [(sa, vb), (sb, va)]:
            corrupted_positions.add((m, s)); error_cells.add((m, s))
            error_details.append({"cell": [m, s], "true": _true_val(m, s),
                                   "injected": inj, "corruption_type": "swap"})
        cells_used += 2

    def _cycle(m, slots):
        nonlocal cells_used
        vals = [_true_val(m, s) for s in slots]
        rot  = [vals[-1]] + vals[:-1]
        for i, s in enumerate(slots):
            corrupted_ids[pos_lookup[(m, s)]["token_pos"]] = jssp_encode_cell(rot[i], tokenizer)
            corrupted_positions.add((m, s)); error_cells.add((m, s))
            error_details.append({"cell": [m, s], "true": vals[i],
                                   "injected": rot[i], "corruption_type": "cycle"})
        cells_used += len(slots)

    def _single(m, s):
        nonlocal cells_used
        tv = _true_val(m, s)
        cands = [v for v in domain
                 if v != tv and (m, domain.index(v)) not in corrupted_positions]
        if not cands:
            cands = [v for v in domain if v != tv]
        if not cands:
            return
        wrong = random.choice(cands)
        corrupted_ids[pos_lookup[(m, s)]["token_pos"]] = jssp_encode_cell(wrong, tokenizer)
        corrupted_positions.add((m, s)); error_cells.add((m, s))
        error_details.append({"cell": [m, s], "true": tv,
                               "injected": wrong, "corruption_type": "single"})
        cells_used += 1

    for m in eligible[:n_corrupt]:
        if cells_used >= n_error:
            break
        group = [(mi, si) for (mi, si) in clean_by_machine[m]
                 if (mi, si) not in corrupted_positions]
        if not group:
            continue
        random.shuffle(group)
        rem = n_error - cells_used

        if len(group) >= 3 and rem >= 3:
            ch = random.choice(["swap", "cycle", "single"])
            if ch == "cycle":
                n_c = min(len(group), rem, random.choice([3, 4]))
                _cycle(m, [s for _, s in group[:n_c]])
            elif ch == "swap":
                _swap(m, group[0][1], group[1][1])
            else:
                _single(m, group[0][1])
        elif len(group) >= 2 and rem >= 2:
            _swap(m, group[0][1], group[1][1]) if random.random() < 0.6 \
                else _single(m, group[0][1])
        elif rem >= 1:
            _single(m, group[0][1])

    return error_cells, error_details


# =============================================================================
# 3. Main
# =============================================================================

def build_tokenized_sample_jssp(
    problem:       dict,
    t:             float,
    tokenizer,
    mask_token_id: int,
    max_length:    int = 512,
    sample_id:     int = 0,
) -> Optional[Dict]:
    puzzle_text = problem["puzzle"]
    solution    = problem["solution"]
    n_jobs      = int(problem["n_jobs"])
    n_machines  = int(problem["n_machines"])

    # Step 1: encode
    prompt_tokens         = encode_prompt_jssp(puzzle_text, solution, tokenizer)
    body_tokens, cell_map = encode_table_body_jssp(solution, tokenizer)
    prompt_len = len(prompt_tokens)
    for cm in cell_map:
        cm["token_pos"] += prompt_len

    clean_ids = prompt_tokens + body_tokens
    if len(clean_ids) > max_length:
        clean_ids = clean_ids[:max_length]
        cell_map  = [cm for cm in cell_map if cm["token_pos"] < max_length]
    if not cell_map:
        return None

    pad_len        = max_length - len(clean_ids)
    clean_ids      = clean_ids + [tokenizer.pad_token_id or 0] * pad_len
    attention_mask = [1] * (max_length - pad_len) + [0] * pad_len

    # Step 2: mask budget
    data_cells = [(cm["row"], cm["col"]) for cm in cell_map]
    random.shuffle(data_cells)
    total   = len(data_cells)
    n_mask  = max(1, int(total * t)) if t > 0.05 else 0
    n_clean = total - n_mask
    # JSSP_USE_ERROR=0 → no injection (wo_head), 1 → original w_head
    if os.environ.get("JSSP_USE_ERROR", "0") == "1":
        n_error = min(max(2, int(n_clean * random.uniform(0.15, 0.40))), n_clean)
    else:
        n_error = 0
    mask_cells = set(data_cells[:n_mask])

    # Step 3: ids (mask)
    pos_lookup = {(cm["row"], cm["col"]): cm for cm in cell_map}
    ids = list(clean_ids)
    for (m, s) in mask_cells:
        ids[pos_lookup[(m, s)]["token_pos"]] = mask_token_id

    # Step 4: corrupted_ids (errors)
    corrupted_ids = list(ids)
    error_cells, error_details = _inject_errors_jssp(
        corrupted_ids, cell_map, mask_cells, n_jobs, n_error, tokenizer)

    # Labels are defined only at answer-cell positions.
    labels = [-100] * max_length
    for cm in cell_map:
        labels[cm["token_pos"]] = cm["token_id"]

    n_mask_tokens = sum(1 for x in ids if x == mask_token_id)
    if len(mask_cells) > 0 and n_mask_tokens == 0:
        return None

    return {
        "sample_id":        sample_id,
        "ids":              ids,
        "corrupted_ids":    corrupted_ids,
        "attention_mask":   attention_mask,
        "labels":           labels,
        "timestep":         t,
        "_debug": {
            "timestep":              round(t, 3),
            "n_mask_cells":          len(mask_cells),
            "n_error_cells":         len(error_cells),
            "n_mask_tokens":         n_mask_tokens,
            "n_label_tokens":        sum(1 for x in labels if x != -100),
            "n_grid_cells":          len(cell_map),
            "seq_length_before_pad": max_length - pad_len,
            "error_details":         error_details,
            "size":                  problem.get("size", f"{n_machines}x{n_jobs}"),
        },
    }


# =============================================================================
# 4. Dataset
# =============================================================================

class JSSPDataset(torch.utils.data.Dataset):
    def __init__(self, puzzles, tokenizer, mask_token_id,
                 max_length=512, t_range=(0.1, 0.9), fixed_seed=None):
        self.puzzles       = puzzles
        self.tokenizer     = tokenizer
        self.mask_token_id = mask_token_id
        self.max_length    = max_length
        self.t_range       = t_range
        self.fixed_seed    = fixed_seed

    def __len__(self):
        return len(self.puzzles)

    def __getitem__(self, idx):
        if self.fixed_seed is not None:
            random.seed(self.fixed_seed + idx)
        t = random.uniform(*self.t_range)

        puzzle = self.puzzles[idx]

        # Multiple optimal augmentation:
        # if alt_solutions exist, use an alternative optimal as GT with 50% probability
        # → the model learns diverse optimal contexts → fewer false attractors
        alt = puzzle.get('alt_solutions', [])
        if alt and random.random() < 0.5 and self.fixed_seed is None:
            puzzle = dict(puzzle)
            puzzle['solution'] = random.choice(alt)

        sample = build_tokenized_sample_jssp(
            puzzle, t, self.tokenizer,
            self.mask_token_id, self.max_length, sample_id=idx)
        if sample is None:
            sample = build_tokenized_sample_jssp(
                puzzle, 0.5, self.tokenizer,
                self.mask_token_id, self.max_length, sample_id=idx)
        if sample is None:
            L = self.max_length
            return (
                torch.zeros(L, dtype=torch.long),
                torch.zeros(L, dtype=torch.long),
                torch.zeros(L, dtype=torch.bool),
                torch.full((L,), -100, dtype=torch.long),
            )
        return (
            torch.tensor(sample["ids"], dtype=torch.long),
            torch.tensor(sample["corrupted_ids"], dtype=torch.long),
            torch.tensor(sample["attention_mask"], dtype=torch.bool),
            torch.tensor(sample["labels"], dtype=torch.long),
        )
