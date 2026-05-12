#!/usr/bin/env bash
# One-click runner: start the phyjudge vLLM server in the background, score
# every .mp4 under --video_dir against the Phyground prompts JSON, then tear
# the server down on exit.
#
# Optional env vars (defaults match the phyjudge-9B model card):
#   PHYJUDGE_BASE  base model HF id or local path (default Qwen/Qwen3.5-9B)
#   PHYJUDGE_LORA  LoRA adapter HF id or local path
#                   (default NU-World-Model-Embodied-AI/phyjudge-9B)
#   PORT           vLLM port (default 29673)
#   PROMPT_CONFIG  YAML template under evals/prompts/ (default default.yaml)
#   PROMPTS_JSON   path to phyground.json (default data/prompts/phyground.json)
#
# CLI args (passed to evals.vlm_eval; --video_dir is the only required one):
#   --video_dir DIR        directory of .mp4 files to score
#   --save_path PATH       where to write the final scores JSON
#   --limit N              smoke-test mode (score first N videos)
#   ... any other evals.vlm_eval arg
#
# Example:
#   bash scripts/score_videos.sh \
#       --video_dir ./videos \
#       --save_path ./scores.json
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PORT="${PORT:-29673}"
PROMPT_CONFIG="${PROMPT_CONFIG:-default.yaml}"
PROMPTS_JSON="${PROMPTS_JSON:-data/prompts/phyground.json}"

if [[ ! -f "${PROMPTS_JSON}" ]]; then
    cat >&2 <<EOF
Benchmark prompts not found at ${PROMPTS_JSON}.

Pull them from the Hugging Face dataset first:

  huggingface-cli download --repo-type dataset \\
      NU-World-Model-Embodied-AI/phyground \\
      --include "prompts/phyground.json" "first_images/*" \\
      --local-dir ./data

(The first_images/ tree holds the first-frame images you feed to your ti2v
model alongside each text prompt — see README §3 for the generation flow.)
EOF
    exit 1
fi

# Start vLLM in the background. Propagates PHYJUDGE_BASE / PHYJUDGE_LORA via env.
echo "Starting phyjudge vLLM server on port ${PORT}..."
PORT="${PORT}" bash "${SCRIPT_DIR}/serve_judge.sh" &
SERVER_PID=$!
trap "echo 'Stopping vLLM server...'; kill ${SERVER_PID} 2>/dev/null || true; wait ${SERVER_PID} 2>/dev/null || true" EXIT

echo "Waiting for vLLM /health on port ${PORT}..."
until curl -fs "http://localhost:${PORT}/health" > /dev/null 2>&1; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "vLLM server exited before becoming ready." >&2
        exit 1
    fi
    sleep 5
done
echo "vLLM server ready."

# --use_training_prompts is REQUIRED for the released LoRA: the adapter was
# fine-tuned against training_prompts, and inference under eval_prompts will
# produce miscalibrated scores. Override at your own risk.
cd "${REPO_ROOT}"
python3 -m evals.vlm_eval \
    --backend qwen9b \
    --prompt_config "${PROMPT_CONFIG}" \
    --use_training_prompts \
    --model phyjudge \
    --prompts_json "${PROMPTS_JSON}" \
    --api_base "http://localhost:${PORT}/v1" \
    "$@"
