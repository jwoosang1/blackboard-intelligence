# Domain implementations

Each domain follows the same public workflow.

| File | Purpose |
| --- | --- |
| `generate.py` | Generate puzzle instances. |
| `encode.py` | Convert tables into masked-token SFT examples. |
| `train.py` | Train and save a LoRA adapter. |
| `solve.py` | Run greedy and triggered Blackboard inference. |

ZebraLogic and Nurse Rostering use confidence-guided cascade search and
backtracking after the trigger fires. JSSP uses Best-of-N and selects the
feasible candidate with the best makespan.

Files prefixed with `_` are internal generator helpers. `jssp/metrics.py`
contains the feasibility and makespan checks used by the JSSP solver.

Shared model loading, any-order decoding, trigger logic, and table tokenization
live in `../core/`.

## Using a trained adapter

Training writes checkpoints below `checkpoints/<domain>/step-<N>/` and updates
`checkpoints/<domain>/best` to the best validation checkpoint. To evaluate a
locally trained adapter, pass that directory to the domain solver, for example:

```bash
python domains/nurse_rostering/solve.py \
  --checkpoint checkpoints/nurse_rostering/best \
  --eval_path data/nurse_rostering/nurse_rostering_eval.json
```

The released inference adapters instead live directly under
`checkpoints/<domain>/lora_adapter/`; `scripts/evaluate.sh` handles their
download and uses that layout automatically.
