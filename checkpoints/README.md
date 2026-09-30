# Checkpoints

Model weights are distributed separately from the Git repository. The solver
loads `GSAI-ML/LLaDA-8B-Instruct` and applies the released LoRA adapter.

## Hugging Face adapters

The released adapters are hosted at [6uvsoomJ/blackboard-intelligence](https://huggingface.co/6uvsoomJ/blackboard-intelligence). Download these three directories into `checkpoints/`:

- `zebralogic/`
- `nurse_rostering/`
- `jssp/`

Each directory must preserve this structure:

```text
checkpoints/zebralogic/lora_adapter/adapter_model.safetensors
checkpoints/zebralogic/lora_adapter/adapter_config.json
checkpoints/zebralogic/training_config.json
```

The `nurse_rostering/` and `jssp/` directories use the same layout. No auxiliary head,
optimizer state, or tokenizer copy is needed for the released inference code.

With the Hugging Face CLI:

```bash
hf download 6uvsoomJ/blackboard-intelligence --include "zebralogic/*" --local-dir checkpoints
hf download 6uvsoomJ/blackboard-intelligence --include "nurse_rostering/*" --local-dir checkpoints
hf download 6uvsoomJ/blackboard-intelligence --include "jssp/*" --local-dir checkpoints
```

## Integrity

| Domain | Adapter SHA-256 |
| --- | --- |
| ZebraLogic-Hard | `9d1115a9e2198eb227579b4ea7c07fb1f490cade3319c7cb743d15082207dc40` |
| Nurse Rostering | `a6e218433f4a2f8b89bf0203832c90fc7dc140ca4c3d69995dfc480e6c3e2fd1` |
| JSSP | `367e4715fe4655320116996da53fda35ce73f7ef919385faf176ab8b59df8f1a` |

Each adapter is about 805 MB and is hosted in the Hugging Face model repository.
