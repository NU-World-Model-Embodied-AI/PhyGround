#!/usr/bin/env bash
# Score videos using a cloud API judge instead of the local phyjudge LoRA.
# No vLLM is started; this just runs evals.vlm_eval against the chosen
# closed-source model.
#
# Required env vars (depending on backend):
#   JUDGE_BACKEND   one of: gemini | gpt | claude
#                   (default: gemini)
#   API_KEY_FILE    path to a file containing the key. Alternatively, set
#                   GEMINI_API_KEY / OPENAI_API_KEY.
#   GCP_PROJECT     required for backend=claude (Vertex AI) and for
#                   backend=gemini when --vertexai is on.
#   GCP_LOCATION    default "global"
#
# Optional env vars:
#   PROMPT_CONFIG   YAML template under evals/prompts/ (default default.yaml)
#   PROMPTS_JSON    path to phyground.json (default data/prompts/phyground.json)
#   JUDGE_MODEL     override model name (e.g. gemini-3.1-pro-preview,
#                   gpt-5.4, claude-opus-4-7)
#   USE_VERTEX      "1" to add --vertexai for the gemini backend
#
# CLI args (passed through to evals.vlm_eval):
#   --video_dir DIR        directory of .mp4 files to score
#   --save_path PATH       where to write the final scores JSON
#   --limit N              smoke-test mode
#
# Note: closed-source judges DO NOT need --use_training_prompts (that flag is
# specific to the released LoRA). Eval-time prompts are used by default.
#
# Examples:
#   # Gemini via AI Studio
#   GEMINI_API_KEY=... bash scripts/score_videos_api.sh \\
#       --video_dir ./videos --save_path ./scores_gemini.json
#
#   # GPT
#   JUDGE_BACKEND=gpt OPENAI_API_KEY=... bash scripts/score_videos_api.sh \\
#       --video_dir ./videos --save_path ./scores_gpt.json
#
#   # Claude on Vertex AI
#   JUDGE_BACKEND=claude GCP_PROJECT=my-gcp-project \\
#       bash scripts/score_videos_api.sh --video_dir ./videos --save_path ./scores_claude.json
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

JUDGE_BACKEND="${JUDGE_BACKEND:-gemini}"
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
EOF
    exit 1
fi

extra_args=()
case "${JUDGE_BACKEND}" in
    gemini)
        if [[ "${USE_VERTEX:-0}" == "1" ]]; then
            : "${GCP_PROJECT:?Set GCP_PROJECT when USE_VERTEX=1.}"
            extra_args+=(--vertexai --project "${GCP_PROJECT}" --location "${GCP_LOCATION:-global}")
        fi
        ;;
    claude)
        : "${GCP_PROJECT:?Set GCP_PROJECT for the claude (Vertex AI) backend.}"
        extra_args+=(--project "${GCP_PROJECT}" --location "${GCP_LOCATION:-global}")
        ;;
    gpt)
        : # No extra routing needed; credentials come from env or --api_key_file.
        ;;
    *)
        echo "Unknown JUDGE_BACKEND: ${JUDGE_BACKEND}" >&2
        echo "Expected one of: gemini | gpt | claude" >&2
        exit 1
        ;;
esac

if [[ -n "${API_KEY_FILE:-}" ]]; then
    extra_args+=(--api_key_file "${API_KEY_FILE}")
fi
if [[ -n "${JUDGE_MODEL:-}" ]]; then
    extra_args+=(--model "${JUDGE_MODEL}")
fi

cd "${REPO_ROOT}"
python3 -m evals.vlm_eval \
    --backend "${JUDGE_BACKEND}" \
    --prompt_config "${PROMPT_CONFIG}" \
    --prompts_json "${PROMPTS_JSON}" \
    "${extra_args[@]}" \
    "$@"
