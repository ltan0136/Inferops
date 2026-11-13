#!/usr/bin/env bash
# Cross-check `inferops bench` with GuideLLM (https://github.com/vllm-project/guidellm).
# Two independent tools measuring the same server should agree on the shape of the curve.
#
#   pip install "guidellm==0.8.1"
#   TARGET=http://localhost:8000 ./scripts/guidellm_sweep.sh
#
# GuideLLM sizes its synthetic prompts with the served model's tokenizer, which it fetches from
# Hugging Face. Set TOKENIZER to a local tokenizer directory to stay offline.
set -euo pipefail

TARGET="${TARGET:-http://localhost:8000}"
STREAMS="${STREAMS:-1,2,4,8,16,32,64}"
SECONDS_PER_LEVEL="${SECONDS_PER_LEVEL:-60}"
PROMPT_TOKENS="${PROMPT_TOKENS:-512}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-128}"
TOKENIZER="${TOKENIZER:-}"
OUT_DIR="${OUT_DIR:-results/guidellm-$(date -u +%Y%m%dT%H%M%SZ)}"

mkdir -p "${OUT_DIR}"

tokenizer_args=()
if [[ -n "${TOKENIZER}" ]]; then
  tokenizer_args=(--tokenizer "kind=huggingface_auto,model=${TOKENIZER}")
fi

guidellm run \
  --backend "kind=openai_http,target=${TARGET}" \
  --profile kind=concurrent \
  --override profile.streams "${STREAMS}" \
  --constraint "kind=max_duration,seconds=${SECONDS_PER_LEVEL}" \
  --data "kind=synthetic_text,prompt_tokens=${PROMPT_TOKENS},output_tokens=${OUTPUT_TOKENS}" \
  "${tokenizer_args[@]}" \
  --output "kind=json,path=${OUT_DIR}/benchmark.json"

echo "wrote ${OUT_DIR}/benchmark.json"
