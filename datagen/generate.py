"""Unified data generation entry point.

    python datagen/generate.py --domain <d> --split {train,eval} --num_samples N \
        [--seed S] [--out PATH]

Dispatches to the per-domain generators in this folder. `--num_samples` is the target
total number of instances (mapped to each generator's native per-bucket knob); `--split`
selects the training set vs. the held-out evaluation set. For finer control (extra knobs,
difficulty bands, workers), call the underlying `generate_*` / `gen_*` scripts directly.

Examples:
    python datagen/generate.py --domain zebralogic --split train --num_samples 30000
    python datagen/generate.py --domain jssp       --split eval  --num_samples 400
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# domain -> short file prefix used for the default output name (matches configs/*.yaml)
PREFIX = {
    "zebralogic": "zebralogic_hard", "jssp": "jssp",
    "nurse_rostering": "nurse_rostering",
}
DOMAINS = list(PREFIX)


def _run(script, *args):
    subprocess.run([PY, os.path.join(HERE, script)] + [str(a) for a in args], check=True)


def _place_generated(out_dir, generated_name, out):
    """Honor the unified CLI's explicit ``--out`` path.

    Domain generators use their config-compatible default filenames internally.
    When a caller supplies another filename, move that single requested split
    into the requested location after generation.
    """
    generated = os.path.join(out_dir, generated_name)
    if os.path.abspath(generated) != os.path.abspath(out):
        os.replace(generated, out)


def generate(domain, split, n, seed, out):
    """Route to the right per-domain generator. `n` is the target total instance count."""
    out_dir = os.path.dirname(out) or "."
    if domain == "zebralogic":
        # 4 difficulty categories -> per_category = n / 4
        _run("../domains/zebralogic/generate.py", "--per_category", max(1, n // 4), "--seed", seed, "--out", out)
    elif domain == "jssp":
        # one generator emits both splits; ask for n of the requested split, 0 of the other
        _run("../domains/jssp/generate.py", "generate", "--output_dir", out_dir, "--seed", seed,
             "--n_train", n if split == "train" else 0,
             "--n_eval",  n if split == "eval" else 0)
        _place_generated(out_dir, f"jssp_{split}.json", out)
    elif domain == "nurse_rostering":
        if split == "train":
            _run("../domains/nurse_rostering/generate.py", "--split", "train", "--per_cfg_train", max(1, n // 8),
                 "--per_cfg_eval", 0, "--workers", min(8, max(1, n)),
                 "--out_dir", out_dir, "--seed", seed)
            _place_generated(out_dir, "nurse_rostering_train.json", out)
        else:  # 4 Z3-conflict bands -> per_band = n / 4
            _run("../domains/nurse_rostering/generate.py", "--split", "eval", "--per_band", max(1, n // 4),
                 "--workers", min(4, max(1, n)), "--out", out)
    else:
        raise SystemExit(f"unknown domain: {domain}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", required=True, choices=DOMAINS)
    ap.add_argument("--split", default="train", choices=["train", "eval"])
    ap.add_argument("--num_samples", type=int, default=30000,
                    help="target total number of instances (default 30000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="output path (default: data/<domain>/<prefix>_<split>.json)")
    a = ap.parse_args()
    out = a.out or f"data/{a.domain}/{PREFIX[a.domain]}_{a.split}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    generate(a.domain, a.split, a.num_samples, a.seed, out)


if __name__ == "__main__":
    main()
