#!/usr/bin/env bash
# Serve the released phyjudge LoRA on top of its base Qwen-VL checkpoint via
# vLLM's OpenAI-compatible API.
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
#   PORT            (default 29673)
#   GPU             CUDA_VISIBLE_DEVICES value (default 0)
#   TP              tensor-parallel size (default 1)
#   GPU_UTIL        gpu-memory-utilization (default 0.9)
#   MAX_LEN         max-model-len (default 32768)
#
# The LoRA is registered under the served-model name "phyjudge"; scripts/
# score_videos.sh passes --model phyjudge to evals.vlm_eval to route requests
# to the adapter (vs. the bare base model).
#
# Usage (foreground):  bash scripts/serve_judge.sh
# Usage (background):  bash scripts/serve_judge.sh &
set -euo pipefail

PHYJUDGE_BASE="${PHYJUDGE_BASE:-Qwen/Qwen3.5-9B}"
PHYJUDGE_LORA="${PHYJUDGE_LORA:-NU-World-Model-Embodied-AI/phyjudge-9B}"
PORT="${PORT:-29673}"
GPU="${GPU:-0}"
TP="${TP:-1}"
GPU_UTIL="${GPU_UTIL:-0.9}"
MAX_LEN="${MAX_LEN:-32768}"

# `exec` so the script's PID becomes the server's PID — caller's
# `kill $SERVER_PID` will reach the actual process.
CUDA_VISIBLE_DEVICES="$GPU" exec python3 -m vllm.entrypoints.openai.api_server \
    --model "$PHYJUDGE_BASE" \
    --enable-lora \
    --lora-modules "phyjudge=${PHYJUDGE_LORA}" \
    --max-lora-rank 32 \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --enforce-eager \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len "$MAX_LEN" \
    --limit-mm-per-prompt video=1 \
    --reasoning-parser deepseek_r1
