#!/usr/bin/env bash
# Serve the released phyjudge judge via vLLM's OpenAI-compatible API.
#
# vLLM cannot serve this adapter as a LoRA (it fails on Qwen3.5's packed
# linear-attention projections and ignores the vision-merger LoRA), so on
# first run the LoRA is merged into the base checkpoint with
# scripts/merge_judge.py and the merged model is served. This is the same
# setup that produced the paper's numbers.
#
# Defaults match the model card at
#   https://huggingface.co/NU-World-Model-Embodied-AI/phyjudge-9B
#
# Optional env vars:
#   PHYJUDGE_BASE   HF id (or local path) of the base model the LoRA targets.
#                   Default: Qwen/Qwen3.5-9B (per adapter_config.json on the
#                   model card).
#   PHYJUDGE_LORA   HF id (or local path) of the LoRA adapter.
#                   Default: NU-World-Model-Embodied-AI/phyjudge-9B.
#   PHYJUDGE_MERGED Where the merged checkpoint is cached (~18 GB). Delete it
#                   to force a re-merge after changing the base or adapter.
#                   Default: ~/.cache/phyjudge/<adapter name>-merged.
#   PORT            (default 29673)
#   GPU             CUDA_VISIBLE_DEVICES value (default 0)
#   TP              tensor-parallel size (default 1)
#   GPU_UTIL        gpu-memory-utilization (default 0.9)
#   MAX_LEN         max-model-len (default 32768)
#
# The merged model is served under the name "phyjudge"; scripts/
# score_videos.sh passes --model phyjudge to evals.vlm_eval.
#
# Usage (foreground):  bash scripts/serve_judge.sh
# Usage (background):  bash scripts/serve_judge.sh &
set -euo pipefail

PHYJUDGE_BASE="${PHYJUDGE_BASE:-Qwen/Qwen3.5-9B}"
PHYJUDGE_LORA="${PHYJUDGE_LORA:-NU-World-Model-Embodied-AI/phyjudge-9B}"
PHYJUDGE_MERGED="${PHYJUDGE_MERGED:-${XDG_CACHE_HOME:-$HOME/.cache}/phyjudge/$(basename "${PHYJUDGE_LORA%/}")-merged}"
PORT="${PORT:-29673}"
GPU="${GPU:-0}"
TP="${TP:-1}"
GPU_UTIL="${GPU_UTIL:-0.9}"
MAX_LEN="${MAX_LEN:-32768}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "${PHYJUDGE_MERGED}/config.json" ]]; then
    echo "Merging ${PHYJUDGE_LORA} into ${PHYJUDGE_BASE} -> ${PHYJUDGE_MERGED} (one-time)..."
    python3 "${SCRIPT_DIR}/merge_judge.py" \
        --base "$PHYJUDGE_BASE" --lora "$PHYJUDGE_LORA" --out "$PHYJUDGE_MERGED"
fi

# `exec` so the script's PID becomes the server's PID — caller's
# `kill $SERVER_PID` will reach the actual process.
CUDA_VISIBLE_DEVICES="$GPU" exec python3 -m vllm.entrypoints.openai.api_server \
    --model "$PHYJUDGE_MERGED" \
    --served-model-name phyjudge \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --enforce-eager \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len "$MAX_LEN" \
    --limit-mm-per-prompt '{"video": 1}' \
    --reasoning-parser deepseek_r1
