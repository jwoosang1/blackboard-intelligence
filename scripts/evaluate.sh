#!/usr/bin/env bash
# Download a released adapter when needed and evaluate one Blackboard domain.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 {zebralogic|nurse_rostering|jssp|all} [--max-samples N]"
  exit 2
fi

DOMAIN="$1"
shift
MAX_SAMPLES=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-samples)
      MAX_SAMPLES="${2:?--max-samples requires an integer}"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
HF_REPO="6uvsoomJ/blackboard-intelligence"

download_adapter() {
  local domain="$1"
  local adapter="checkpoints/${domain}/lora_adapter/adapter_model.safetensors"
  if [[ -f "$adapter" ]]; then
    return
  fi
  command -v hf >/dev/null 2>&1 || {
    echo "The Hugging Face CLI ('hf') is required. Install requirements.txt first." >&2
    exit 1
  }
  echo "Downloading ${domain} adapter from ${HF_REPO}..."
  hf download "$HF_REPO" --repo-type model --include "${domain}/*" --local-dir checkpoints
}

evaluate_domain() {
  local domain="$1"
  download_adapter "$domain"
  mkdir -p "results/${domain}"

  case "$domain" in
    zebralogic)
      python domains/zebralogic/solve.py \
        --checkpoint checkpoints/zebralogic \
        --eval_path data/zebralogic/zebralogic_hard_eval.json \
        --methods greedy blackboard \
        --max_samples "$MAX_SAMPLES" \
        --output results/zebralogic/eval.json
      ;;
    nurse_rostering)
      python domains/nurse_rostering/solve.py \
        --checkpoint checkpoints/nurse_rostering \
        --eval_path data/nurse_rostering/nurse_rostering_eval.json \
        --methods greedy blackboard \
        --max_samples "$MAX_SAMPLES" \
        --out results/nurse_rostering/eval.json
      ;;
    jssp)
      python domains/jssp/solve.py \
        --checkpoint checkpoints/jssp \
        --eval_path data/jssp/jssp_eval.json \
        --methods greedy blackboard \
        --n_puzzles "$MAX_SAMPLES" \
        --n_bon 10 \
        --output results/jssp/eval.json
      ;;
    *)
      echo "Unknown domain: ${domain}" >&2
      exit 2
      ;;
  esac
}

if [[ "$DOMAIN" == "all" ]]; then
  for domain in zebralogic nurse_rostering jssp; do
    evaluate_domain "$domain"
  done
else
  evaluate_domain "$DOMAIN"
fi
