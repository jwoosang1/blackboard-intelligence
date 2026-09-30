<h1 align="center">Blackboard Intelligence Can Surpass Autoregressive<br>on Globally Constrained Problems</h1>

<p align="center">
  <b>Woosang Jeon<sup>1,*</sup> · Jaeyeon Kim<sup>2,*</sup> · Sham Kakade<sup>2</sup> · Yilun Du<sup>2</sup> · Amrit Singh Bedi<sup>3</sup><br>
  Arun Kumar Chithanar<sup>4</sup> · Chul Lee<sup>4</sup> · Taehyeong Kim<sup>1,†</sup> · Sitan Chen<sup>2,†</sup>
  <br><br>
  <sup>1</sup>Seoul National University &nbsp; <sup>2</sup>Harvard University &nbsp; <sup>3</sup>University of Central Florida &nbsp; <sup>4</sup>Independent Researcher
  <br>
  <sup>*</sup>Equal contribution; lead junior authors &nbsp; <sup>†</sup>Lead senior authors
</p>

<p align="center">
  <a href="https://huggingface.co/6uvsoomJ/blackboard-intelligence"><img src="https://img.shields.io/badge/Checkpoints-Hugging%20Face-f9a03c?logo=huggingface" alt="Checkpoints"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-2ea44f.svg" alt="MIT License"></a>
  <img src="https://img.shields.io/badge/arXiv-coming%20soon-B31B1B?logo=arxiv" alt="arXiv coming soon">
</p>

> Reference implementation of Blackboard inference for ZebraLogic-Hard, Nurse Rostering, and Job-Shop Scheduling.

## Quick start

Install the dependencies in a CUDA-enabled PyTorch environment:

```bash
pip install -r requirements.txt
```

Run one domain from the repository root. The launcher downloads its released
adapter from [Hugging Face](https://huggingface.co/6uvsoomJ/blackboard-intelligence)
on first use and writes results under `results/`:

```bash
bash scripts/evaluate.sh zebralogic --max-samples 5
```

Use `nurse_rostering`, `jssp`, or `all` in place of `zebralogic`. Omit
`--max-samples` to evaluate the full committed set.

## Contents

| Path | Description |
| --- | --- |
| [`core/`](core/) | Shared any-order decoding, trigger, model, and table utilities. |
| [`domains/`](domains/README.md) | Domain-specific generation, training, and inference. |
| [`data/`](data/) | Evaluation sets. |
| [`checkpoints/`](checkpoints/README.md) | Adapter layout, download instructions, and checksums. |

## License

This repository is released under the [MIT License](LICENSE). Use of the base model is subject to its own license and terms.

## Citation

The arXiv link and BibTeX entry will be added upon publication.
