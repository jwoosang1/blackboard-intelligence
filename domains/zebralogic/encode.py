"""
=============================================================================
Blackboard SFT data builder
=============================================================================

ZebraLogic-compatible format. Every grid value is a SINGLE token from
the base LLaDA vocabulary (no special tokens added).

All entity values in CATEGORY_DOMAINS are chosen so that they encode to
exactly 1 token with a space prefix (Ġ-form in BPE).

Prompt format (ZebraLogic style):
  ### SYSTEM:
  You are a precision logic solver engine.
  ...
  ### PUZZLE CONTEXT:
  There are 3 houses...
  1. Beverage: coffee == Food: pizza
  ...
  ### FINAL SOLUTION:
  | House | Beverage | Food | Pet |
  | --- | --- | --- | --- |

Table body (generation target, position-major):
  | 1 | [cola]    | [pizza]   | [cat]   |
  | 2 | [water]   | [burger]  | [dog]   |
  | 3 | [juice]   | [pasta]   | [fish]  |

Token-level surgery (cell values are 1 base-vocab token each):
  clean:     ...| 1 | Ġcoffee   | Ġpizza    | Ġcat    |...
  ids:       ...| 1 | [MASK]    | Ġpizza    | Ġcat    |...
  corrupted: ...| 1 | [MASK]    | Ġsteak    | Ġcat    |...
  labels:    ...-100   Ġcoffee    Ġpizza     Ġcat     ...

The public training pipeline constructs samples on the fly; evaluation data uses puzzle-and-solution JSON.
"""

import random
import copy

# masking_only flag — set by trainer to disable error injection
# Usage: import domains.zebralogic.encode as zebra_encode; zebra_encode._MASKING_ONLY = True
_MASKING_ONLY = False
from typing import List, Optional, Tuple, Dict, Any, Set

try:
    from ._problem import CATEGORY_DOMAINS
except ImportError:
    from domains.zebralogic._problem import CATEGORY_DOMAINS


# =============================================================================
# PART 1: Single-Token Cell Value Encoding/Decoding
# =============================================================================
# All CATEGORY_DOMAINS values are chosen so that they encode to exactly
# 1 token in the base LLaDA vocabulary (with Ġ prefix, i.e. space-prefixed).
# No special tokens are added — the base tokenizer is used as-is.


def encode_cell_value(val: str, tokenizer) -> int:
    """
    Encode a cell value string to a single base-vocab token ID.

    In BPE tokenizers, a word preceded by whitespace gets a Ġ prefix
    and maps to a single token (e.g., " french" → Ġfrench → ID 50190).
    We prepend a space to ensure this encoding.
    """
    ids = tokenizer.encode(" " + val, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(
            f"Cell value '{val}' encodes to {len(ids)} tokens {ids}, "
            f"expected exactly 1.  Is it in CATEGORY_DOMAINS?"
        )
    return ids[0]


def decode_cell_value(token_id: int, tokenizer) -> str:
    """
    Decode a token ID back to a clean cell value string.

    Strips the Ġ (space) prefix that BPE tokenizers prepend.
    """
    raw = tokenizer.convert_ids_to_tokens(token_id)
    if raw is None:
        return f"<unk:{token_id}>"
    return raw.lstrip("\u0120").strip()  # \u0120 = Ġ


# =============================================================================
# PART 2: Tokenizer & Embedding Setup
# =============================================================================

def setup_tokenizer(tokenizer):
    """
    Validate tokenizer for single-token cell values (no special tokens added).

    With the single-token entity strategy, all CATEGORY_DOMAINS values
    already exist in the base vocabulary. No tokens are added, no
    embedding resize is needed.

    Returns:
        (vocab_size, bpe_map)  — bpe_map is always empty.
    """
    print(f"[tokenizer] Using base vocab only (size: {len(tokenizer)}). "
          f"No special tokens added.")
    return len(tokenizer), {}



# =============================================================================
# PART 3: Text Formatting (human-readable)
# =============================================================================

def make_prompt(puzzle: str, solution: dict) -> str:
    """Build a ZebraLogic prompt from puzzle text and a solution header."""
    header = solution["header"]
    header_row = " | ".join(header)
    separator_row = " | ".join(["---"] * len(header))
    return (
        "### SYSTEM:\n"
        "You are a precision logic solver engine.\n"
        "Output ONLY a valid Markdown table as the solution.\n"
        "\n"
        "### PUZZLE CONTEXT:\n"
        f"{puzzle}\n"
        "\n"
        "### FINAL SOLUTION:\n"
        f"| {header_row} |\n"
        f"| {separator_row} |\n"
    )

# =============================================================================
# PART 4: Token-Level Data Construction
# =============================================================================

def encode_prompt(puzzle: str, solution: dict, tokenizer) -> List[int]:
    """
    Encode the ZebraLogic prompt (system + puzzle + header + separator).
    This is the FIXED conditioning — never masked.
    """
    text = make_prompt(puzzle, solution)
    return tokenizer.encode(text, add_special_tokens=False)


def encode_table_body(solution: dict, tokenizer) -> Tuple[List[int], List[Dict]]:
    """
    Encode ZebraLogic table body as tokens.

    Each cell value = exactly 1 token (base vocab, Ġ-prefixed form).
    Position numbers and markdown structure (|, spaces, newlines) are fixed.

    Args:
        solution: {"header": ["House", "Bev", ...], "rows": [["1", "cola", ...], ...]}

    Returns:
        token_ids: flat list of token ids for the table body
        cell_map: [{token_pos, row, col, value, token_id, category}, ...]
                  for data cells only (not position numbers or structure)
    """
    header = solution["header"]
    categories = header[1:]        # skip "House"
    rows = solution["rows"]
    n_cats = len(categories)

    tokens = []
    cell_map = []

    for pos, row in enumerate(rows):
        # Structural prefix: "| 1 | " (position number)
        row_prefix = f"| {row[0]} |"
        tokens.extend(tokenizer.encode(row_prefix, add_special_tokens=False))

        for cat_idx in range(n_cats):
            val = row[cat_idx + 1]           # +1: skip position number
            val_id = encode_cell_value(val, tokenizer)

            cell_map.append({
                "token_pos": len(tokens),    # position in token_ids
                "row": pos,                  # house/position index (0-based)
                "col": cat_idx,              # category index
                "category": categories[cat_idx],
                "value": val,
                "token_id": val_id,
            })

            tokens.append(val_id)

            # Structural suffix: " | " or " |\n"
            if cat_idx < n_cats - 1:
                sep = " |"
            else:
                sep = " |\n"
            tokens.extend(tokenizer.encode(sep, add_special_tokens=False))

    return tokens, cell_map


def build_tokenized_sample(
    problem: dict,
    t: float,
    tokenizer: Any,
    mask_token_id: int,
    max_length: int = 512,
    sample_id: int = 0,
) -> Optional[Dict]:
    """
    Build a single tokenized training sample.

    Args:
        problem: {"puzzle": str, "solution": {"header": [...], "rows": [...]}, ...}

    Layout:
      [prompt tokens] [table body tokens]
       ^ fixed          ^ contains masking targets (cell values)
    """
    puzzle = problem["puzzle"]
    solution = problem["solution"]
    header = solution["header"]
    rows = solution["rows"]
    categories = header[1:]          # skip "House"
    n_pos = len(rows)
    n_cats = len(categories)

    # --- Step 1: Encode prompt + table body ---
    prompt_tokens = encode_prompt(puzzle, solution, tokenizer)
    body_tokens, cell_map = encode_table_body(solution, tokenizer)

    # Offset cell_map positions by prompt length
    prompt_len = len(prompt_tokens)
    for cm in cell_map:
        cm["token_pos"] += prompt_len

    clean_ids = prompt_tokens + body_tokens

    # Truncate if needed
    if len(clean_ids) > max_length:
        clean_ids = clean_ids[:max_length]
        cell_map = [cm for cm in cell_map if cm["token_pos"] < max_length]

    if not cell_map:
        return None

    # Pad
    pad_len = max_length - len(clean_ids)
    pad_token_id = tokenizer.pad_token_id or 0
    clean_ids = clean_ids + [pad_token_id] * pad_len
    attention_mask = [1] * (max_length - pad_len) + [0] * pad_len

    # --- Step 2: Cell-level noise ---
    # Build domain lookup per category (column values in solution)
    cat_domain = {}    # cat_name -> [val, val, ...]
    val_to_pos = {}    # (val, cat_name) -> position index
    for cat_idx, cat in enumerate(categories):
        vals = [rows[pos][cat_idx + 1] for pos in range(n_pos)]
        cat_domain[cat] = vals
        for pos in range(n_pos):
            val_to_pos[(vals[pos], cat)] = pos

    data_cells = [(cm["row"], cm["col"]) for cm in cell_map]
    random.shuffle(data_cells)
    total = len(data_cells)

    # Mask budget: follow original LLaDA diffusion ratio
    n_mask = max(1, int(total * t)) if t > 0.05 else 0

    # Optional corruption budget for training examples
    # Pattern: concentrated in 1-2 categories (swap/cycle), realistic amount
    n_clean = total - n_mask
    import domains.zebralogic.encode as _bsd_self
    if getattr(_bsd_self, '_MASKING_ONLY', False):
        # masking_only mode: no error injection
        # backbone learns from masking only — no violation labels needed
        n_error = 0
    else:
        error_frac = random.uniform(0.15, 0.4)  # 15-40% of clean cells
        n_error = max(2, int(n_clean * error_frac))
        n_error = min(n_error, n_clean)

    mask_cells = set(data_cells[:n_mask])

    # Build error candidates: pick 1-2 categories, concentrate errors there
    # (mirrors real failures where swap/cycle happen within a category)
    clean_cells_by_cat = {}  # cat_idx -> [(pos_idx, cat_idx), ...]
    for (pos_idx, cat_idx) in data_cells:
        if (pos_idx, cat_idx) in mask_cells:
            continue
        if cat_idx not in clean_cells_by_cat:
            clean_cells_by_cat[cat_idx] = []
        clean_cells_by_cat[cat_idx].append((pos_idx, cat_idx))

    # Pick 1-2 categories that have enough clean cells for swap/cycle
    eligible_cats = [ci for ci, cells in clean_cells_by_cat.items() if len(cells) >= 2]
    random.shuffle(eligible_cats)
    n_cats_to_corrupt = min(len(eligible_cats), random.choice([1, 1, 2]))  # bias toward 1

    error_candidates = []
    for ci in eligible_cats[:n_cats_to_corrupt]:
        cells = clean_cells_by_cat[ci]
        random.shuffle(cells)
        # Take as many as budget allows from this category
        take = min(len(cells), n_error - len(error_candidates))
        error_candidates.extend(cells[:take])
        if len(error_candidates) >= n_error:
            break

    pos_lookup = {(cm["row"], cm["col"]): cm for cm in cell_map}

    # --- Step 3: Build ids (mask injection) ---
    ids = list(clean_ids)
    for (pos_idx, cat_idx) in mask_cells:
        cm = pos_lookup[(pos_idx, cat_idx)]
        ids[cm["token_pos"]] = mask_token_id

    # --- Step 4: Build corrupted_ids (masks + errors) ---
    # Realistic corruption: simulate actual inference error patterns.
    # In real failures, errors are always correlated within a category:
    #   - swap:  two cells exchange values (A↔B)
    #   - cycle: 3+ cells rotate values (A→B→C→A)
    # Single-cell independent errors rarely occur in practice.

    corrupted_ids = list(ids)
    error_cells = set()
    error_details = []
    corrupted_positions = set(mask_cells)

    # Group error_candidates by category
    cat_candidates = {}  # cat_idx -> [(pos_idx, cat_idx), ...]

    for (pos_idx, cat_idx) in error_candidates:
        if (pos_idx, cat_idx) in corrupted_positions:
            continue
        if cat_idx not in cat_candidates:
            cat_candidates[cat_idx] = []
        cat_candidates[cat_idx].append((pos_idx, cat_idx))

    cells_used = 0
    max_errors = n_error

    def _apply_swap(pos_a, pos_b, cat_idx_):
        nonlocal cells_used
        cat_ = categories[cat_idx_]
        val_a = rows[pos_a][cat_idx_ + 1]
        val_b = rows[pos_b][cat_idx_ + 1]
        cm_a = pos_lookup[(pos_a, cat_idx_)]
        cm_b = pos_lookup[(pos_b, cat_idx_)]
        corrupted_ids[cm_a["token_pos"]] = encode_cell_value(val_b, tokenizer)
        corrupted_ids[cm_b["token_pos"]] = encode_cell_value(val_a, tokenizer)
        for pos_, injected_ in [(pos_a, val_b), (pos_b, val_a)]:
            corrupted_positions.add((pos_, cat_idx_))
            error_cells.add((pos_, cat_idx_))
            error_details.append({
                "cell": [pos_, cat_idx_],
                "category": cat_,
                "true": rows[pos_][cat_idx_ + 1],
                "injected": injected_,
                "corruption_type": "swap",
            })
        cells_used += 2

    def _apply_cycle(positions_, cat_idx_):
        nonlocal cells_used
        cat_ = categories[cat_idx_]
        vals_ = [rows[p_][cat_idx_ + 1] for p_ in positions_]
        rotated_ = [vals_[-1]] + vals_[:-1]
        for i_, pos_ in enumerate(positions_):
            cm_ = pos_lookup[(pos_, cat_idx_)]
            corrupted_ids[cm_["token_pos"]] = encode_cell_value(rotated_[i_], tokenizer)
            corrupted_positions.add((pos_, cat_idx_))
            error_cells.add((pos_, cat_idx_))
            error_details.append({
                "cell": [pos_, cat_idx_],
                "category": cat_,
                "true": rows[pos_][cat_idx_ + 1],
                "injected": rotated_[i_],
                "corruption_type": "cycle",
            })
        cells_used += len(positions_)

    def _apply_single_cell(pos_idx_, cat_idx_):
        nonlocal cells_used
        cat_ = categories[cat_idx_]
        true_val_ = rows[pos_idx_][cat_idx_ + 1]
        cm_ = pos_lookup[(pos_idx_, cat_idx_)]
        cands_ = [v for v in cat_domain[cat_]
                  if v != true_val_ and (val_to_pos[(v, cat_)], cat_idx_) not in corrupted_positions]
        if not cands_:
            return
        wrong_val_ = random.choice(cands_)
        corrupted_ids[cm_["token_pos"]] = encode_cell_value(wrong_val_, tokenizer)
        corrupted_positions.add((pos_idx_, cat_idx_))
        error_cells.add((pos_idx_, cat_idx_))
        error_details.append({
            "cell": [pos_idx_, cat_idx_],
            "category": cat_,
            "true": true_val_,
            "injected": wrong_val_,
            "corruption_type": "single",
        })
        cells_used += 1

    cat_indices_shuffled = list(cat_candidates.keys())
    random.shuffle(cat_indices_shuffled)

    for ci_ in cat_indices_shuffled:
        if cells_used >= max_errors:
            break
        group_ = [(p, c) for (p, c) in cat_candidates[ci_]
                   if (p, c) not in corrupted_positions]
        if not group_:
            continue
        remaining_ = max_errors - cells_used
        random.shuffle(group_)

        if len(group_) >= 3 and remaining_ >= 3:
            choice_ = random.choice(["swap", "cycle", "single"])
            if choice_ == "cycle":
                n_cyc = min(len(group_), remaining_, random.choice([3, 4]))
                _apply_cycle([p for (p, c) in group_[:n_cyc]], ci_)
            elif choice_ == "swap":
                _apply_swap(group_[0][0], group_[1][0], ci_)
            else:
                _apply_single_cell(group_[0][0], ci_)
        elif len(group_) >= 2 and remaining_ >= 2:
            if random.random() < 0.5:
                _apply_swap(group_[0][0], group_[1][0], ci_)
            else:
                # single: corrupt only one cell
                _apply_single_cell(group_[0][0], ci_)
        elif len(group_) >= 1 and remaining_ >= 1:
            _apply_single_cell(group_[0][0], ci_)
        else:
            pass

    # Labels are defined only at answer-cell positions.
    labels = [-100] * max_length
    for cm in cell_map:
        labels[cm["token_pos"]] = cm["token_id"]

    # --- Validation ---
    n_mask_tokens = sum(1 for x in ids if x == mask_token_id)
    if len(mask_cells) > 0 and n_mask_tokens == 0:
        return None

    return {
        "sample_id": sample_id,
        "ids": ids,
        "corrupted_ids": corrupted_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "timestep": t,
        "_debug": {
            "timestep": round(t, 3),
            "n_mask_cells": len(mask_cells),
            "n_error_cells": len(error_cells),
            "n_mask_tokens": n_mask_tokens,
            "n_label_tokens": sum(1 for x in labels if x != -100),
            "n_grid_cells": len(cell_map),
            "seq_length_before_pad": max_length - pad_len,
            "error_details": error_details,
            "size": problem.get("size", f"{n_pos}x{n_cats}"),
        },
    }


def apply_cell_noise_to_solution(solution: dict, t: float):
    """
    Apply cell-level noise to a ZebraLogic solution for text-only inspection.

    Args:
        solution: {"header": [...], "rows": [["1", val, ...], ...]}

    Returns: (masked_rows, corrupted_rows, clean_rows, mask_cells, error_cells, error_map)
    """
    rows = solution["rows"]
    header = solution["header"]
    categories = header[1:]
    n_pos = len(rows)
    n_cats = len(categories)

    clean_rows = copy.deepcopy(rows)
    masked_rows = copy.deepcopy(rows)
    corrupted_rows = copy.deepcopy(rows)

    # Domain per category (column values)
    cat_domain = {}
    val_to_pos = {}
    for cat_idx, cat in enumerate(categories):
        vals = [rows[pos][cat_idx + 1] for pos in range(n_pos)]
        cat_domain[cat] = vals
        for pos in range(n_pos):
            val_to_pos[(vals[pos], cat)] = pos

    # All data cell coordinates: (pos_idx, cat_idx)
    data_coords = [(pos, cat_idx) for pos in range(n_pos) for cat_idx in range(n_cats)]
    random.shuffle(data_coords)
    total = len(data_coords)

    n_corrupt = max(0, int(total * t))
    n_mask = max(1, int(n_corrupt * 0.5)) if n_corrupt > 0 else 0
    n_error = max(0, n_corrupt - n_mask)

    if t > 0.05 and n_mask == 0:
        n_mask = 1
    if t > 0.15 and n_error == 0 and total > 2:
        n_error = 1
    if n_mask + n_error > total:
        n_error = max(0, total - n_mask)

    mask_cells = data_coords[:n_mask]
    error_candidates = data_coords[n_mask:n_mask + n_error]

    for pos, cat_idx in mask_cells:
        masked_rows[pos][cat_idx + 1] = "[MASK]"
        corrupted_rows[pos][cat_idx + 1] = "[MASK]"

    corrupted_positions = set(mask_cells)
    error_cells = []
    error_map = {}

    for pos, cat_idx in error_candidates:
        cat = categories[cat_idx]
        true_val = clean_rows[pos][cat_idx + 1]
        candidates = []
        for x in cat_domain[cat]:
            if x == true_val:
                continue
            ox_pos = val_to_pos[(x, cat)]
            if (ox_pos, cat_idx) not in corrupted_positions:
                candidates.append(x)
        if not candidates:
            continue
        wrong_val = random.choice(candidates)
        corrupted_rows[pos][cat_idx + 1] = wrong_val
        corrupted_positions.add((pos, cat_idx))
        error_cells.append((pos, cat_idx))
        error_map[(pos, cat_idx)] = {"true": true_val, "injected": wrong_val}

    return masked_rows, corrupted_rows, clean_rows, mask_cells, error_cells, error_map


def _rows_to_body(rows: List[List[str]]) -> str:
    """Format solution rows as table body text."""
    return "\n".join(f"| {' | '.join(row)} |" for row in rows)
