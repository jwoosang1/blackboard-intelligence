"""
SFT data builder for the Nurse-Rostering (NR) task.

Mirrors the rotating-roster builder: 1 row per staff, columns = days,
cell = shift word (single token). Consumes normalized problem dicts from
nurse_rostering_normalize.normalize_generated_problem (ZL-shaped:
problem["puzzle"], problem["solution"]{header, rows}). Reuses the ZL
prompt/table encoders so tt/gc trainers plug in unchanged.

Label convention matches zebralogic_encode.build_tokenized_sample:
labels are set at ALL grid cells; the trainer restricts the unmask loss to
masked positions (input_ids == mask_token_id).

IMPORTANT: run verify_shift_tokens(tokenizer) once for BOTH the LLaDA and
LLaMA tokenizers before training (ZL single-token convention, paper C.1.1).
"""

import random

from core.table_encoding import (
    encode_cell_value, decode_cell_value, setup_tokenizer,
    encode_table_body, make_prompt,
)
from domains.nurse_rostering._problem import SHIFTS_3, SHIFTS_4

SHIFTS = list(SHIFTS_4)          # superset vocabulary: day, late, night, off


def verify_shift_tokens(tokenizer):
    """Assert every shift word encodes to exactly 1 token in G-form.
    Run once per tokenizer (LLaDA AND LLaMA) before any training."""
    bad = []
    for w in SHIFTS:
        ids = tokenizer.encode(" " + w, add_special_tokens=False)
        if len(ids) != 1:
            bad.append((w, ids))
    if bad:
        raise ValueError(f"multi-token shift words for this tokenizer: {bad}")
    return True


def nr_to_solution(problem):
    """Normalized NR problem -> ZL-style solution dict (identity passthrough;
    kept for interface parity with rr_to_solution / gc_to_solution)."""
    return problem["solution"]


def build_tokenized_sample_nr(problem, t, tokenizer, mask_token_id,
                              max_length=1024, augment_colors=False):
    sol = nr_to_solution(problem)
    prompt_text = make_prompt(problem["puzzle"], sol)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    body_ids, cell_map = encode_table_body(sol, tokenizer)
    Lp = len(prompt_ids)
    for cm in cell_map:
        cm["token_pos"] += Lp
    clean_ids = prompt_ids + body_ids
    if len(clean_ids) > max_length or not cell_map:
        return None
    pad_id = tokenizer.pad_token_id or 0
    pad_len = max_length - len(clean_ids)
    clean_ids = clean_ids + [pad_id] * pad_len
    attention_mask = [1] * (max_length - pad_len) + [0] * pad_len

    cells = list(range(len(cell_map)))
    random.shuffle(cells)
    n_mask = max(1, int(len(cells) * t)) if t > 0.05 else 0
    mask_set = set(cells[:n_mask])

    input_ids = list(clean_ids)
    labels = [-100] * max_length
    for i, cm in enumerate(cell_map):
        pos = cm["token_pos"]
        if pos >= max_length:
            continue
        labels[pos] = cm["token_id"]         # all cells (ZL convention)
        if i in mask_set:
            input_ids[pos] = mask_token_id

    n_rows = len(sol["rows"])
    n_cols = len(sol["header"]) - 1
    return {"input_ids": input_ids, "attention_mask": attention_mask,
            "labels": labels, "cell_map": cell_map,
            "n_rows": n_rows, "n_cols": n_cols,
            "prompt_len": Lp, "task_id": problem.get("puzzle_id")}


# gc-trainer aliases (tt_train / tt_eval / tt_pipeline reuse)
build_tokenized_sample_gc = build_tokenized_sample_nr
gc_to_solution = nr_to_solution
DOOM_LABEL = -200
COLORS = list(SHIFTS)


try:
    import torch
    _TorchDataset = torch.utils.data.Dataset
except ImportError:
    torch = None
    _TorchDataset = object


class NrDataset(_TorchDataset):
    def __init__(self, puzzles, tokenizer, mask_token_id,
                 max_length=1024, t_range=(0.1, 0.9), fixed_seed=None,
                 augment_colors=False, doom_frac=0.0):
        self.puzzles = puzzles; self.tokenizer = tokenizer
        self.mask_token_id = mask_token_id; self.max_length = max_length
        self.t_range = t_range
        # per-instance RNG avoids global-seed collisions with num_workers > 0
        self._rng = random.Random(fixed_seed)
        self.fixed_seed = fixed_seed

    def __len__(self):
        return len(self.puzzles)

    def __getitem__(self, idx):
        rng = (random.Random(self.fixed_seed + idx)
               if self.fixed_seed is not None else self._rng)
        t = rng.uniform(*self.t_range)
        s = build_tokenized_sample_nr(self.puzzles[idx], t, self.tokenizer,
                                      self.mask_token_id, self.max_length)
        if s is None:
            s = build_tokenized_sample_nr(self.puzzles[idx], 0.5,
                                          self.tokenizer, self.mask_token_id,
                                          self.max_length)
        L = self.max_length
        if s is None:
            return (torch.zeros(L, dtype=torch.long),
                    torch.zeros(L, dtype=torch.long),
                    torch.zeros(L, dtype=torch.bool),
                    torch.zeros(L, dtype=torch.bool),
                    torch.full((L,), -100, dtype=torch.long),
                    torch.zeros(L, dtype=torch.float32),
                    torch.tensor(0.5, dtype=torch.float32))
        ids = torch.tensor(s["input_ids"], dtype=torch.long)
        attn = torch.tensor(s["attention_mask"], dtype=torch.bool)
        lbl = torch.tensor(s["labels"], dtype=torch.long)
        return (ids, ids.clone(), attn, torch.zeros(L, dtype=torch.bool),
                lbl, torch.zeros(L, dtype=torch.float32),
                torch.tensor(t, dtype=torch.float32))


GcDataset = NrDataset
RrDataset = NrDataset
