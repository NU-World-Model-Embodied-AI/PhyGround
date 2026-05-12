# End-to-end pipeline

The benchmark is built in five stages. Each stage corresponds to one or two
modules in this repo; downstream stages consume artifacts that live in the
companion dataset. Point `$DATASET_DIR` at wherever you downloaded the
Hugging Face dataset before running the snippets below, e.g.:

```bash
export DATASET_DIR=/path/to/phyground-dataset
```

## 1. Prompt curation

```bash
# Enhance prompts to make the expected physical phenomenon explicit.
# Reads $DATASET_DIR/prompts/*.csv, calls Gemini, writes back enhanced prompts.
python -m dataprocessing.refine.enhance_prompts_physics --dry_run
python -m dataprocessing.refine.enhance_prompts_physics --dataset wmb
```

Requires Gemini API access (Vertex AI or AI Studio) — see
`dataprocessing/common/gemini.py` for client config and CLI flags.

## 2. Video generation

Out of scope for this repo. Use any ti2v model on the curated prompt set;
the dataset card lists the eight models we ran (`wan2.2-ti2v-5b`,
`ltx-2-19b-dev`, `cosmos-predict2.5-{2b,14b}`, `veo-3.1`,
`wan2.2-i2v-a14b`, `omniweaving`, `ltx-2.3-22b-dev`).

Each prompt in `prompts/phyground.json` ships with a corresponding first-
frame conditioning image under
[`first_images/`](https://huggingface.co/datasets/NU-World-Model-Embodied-AI/phyground/tree/main/first_images)
on the HF dataset — feed `(text_prompt, first_image)` to your ti2v model
and save the result as `videos/<video_id>.mp4`, where `<video_id>` matches
the `video` field of the prompts JSON entry. The scorer pairs videos to
prompts by that filename stem.

## 3. VLM-as-judge evaluation

**Three commands to a scored JSON.**

```bash
# 3a. Install the eval runner (one-stop extra: all backends + HF CLI).
pip install -e ".[eval]"
# Plus a system-level ffmpeg if you'll use the local vLLM judge:
#   apt-get install ffmpeg   /   brew install ffmpeg

# 3b. Pull the benchmark prompts (250 entries) and first-frame images.
huggingface-cli download --repo-type dataset \
    NU-World-Model-Embodied-AI/phyground \
    --include "prompts/phyground.json" "first_images/*" \
    --local-dir ./data
#   → data/prompts/phyground.json
#   → data/first_images/*.png

# 3c. Score every videos/*.mp4 with the released phyjudge LoRA via vLLM.
pip install "vllm>=0.6"
bash scripts/score_videos.sh \
    --video_dir ./videos \
    --save_path ./scores.json
```

The wrapper starts vLLM in the background (base `Qwen/Qwen3.5-9B` + LoRA
adapter `NU-World-Model-Embodied-AI/phyjudge-9B`, as recorded in the
[model card](https://huggingface.co/NU-World-Model-Embodied-AI/phyjudge-9B)),
waits for `/health`, runs `python -m evals.vlm_eval` against every
`*.mp4` under `--video_dir`, and tears the server down on exit. Override
the base or adapter with `PHYJUDGE_BASE=…` / `PHYJUDGE_LORA=…` if you've
mirrored them locally.

`scores.json` schema:

```jsonc
{
  "meta": { "evaluator": "qwen9b", "video_model": "videos", ... },
  "num_videos": 250,
  "general_dimensions": ["SA", "PTV", "persistence"],
  "results": [
    {
      "video": "ball_fall_0001",
      "SA": 4, "PTV": 5, "persistence": 5, "general_avg": 4.67,
      "physical": {
        "laws": { "gravity": { "score": 4, "status": "scored" } },
        "avg": 4.0, "coverage": 1.0
      },
      "prompt": "...", "physical_laws": ["gravity", "collision"]
    }
  ]
}
```

### Cloud-API judges (Gemini / OpenAI / Claude)

For reproducing the closed-source baselines you don't need a GPU — use
`scripts/score_videos_api.sh`, which talks to a cloud model directly:

```bash
# Gemini (AI Studio key — fastest path)
GEMINI_API_KEY=… bash scripts/score_videos_api.sh \
    --video_dir ./videos --save_path ./scores_gemini.json

# OpenAI GPT
JUDGE_BACKEND=gpt OPENAI_API_KEY=… bash scripts/score_videos_api.sh \
    --video_dir ./videos --save_path ./scores_gpt.json

# Claude on Vertex AI (uses gcloud default credentials)
JUDGE_BACKEND=claude GCP_PROJECT=my-gcp-project \
    bash scripts/score_videos_api.sh \
    --video_dir ./videos --save_path ./scores_claude.json
```

The closed-source backends sample N frames per video (default 32) and
send them as images. Override the model name (`JUDGE_MODEL=gpt-5.4`,
`JUDGE_MODEL=gemini-3.1-pro-preview`, …), prompt template
(`PROMPT_CONFIG=cotnosubq.yaml`), or any other `evals.vlm_eval` flag by
passing it after the script name — see `python -m evals.vlm_eval --help`.

The five prompt templates under `evals/prompts/` are A/B-comparable: they
share the same scoring keys but differ in whether they elicit
chain-of-thought reasoning and/or intermediate yes/no answers to per-law
sub-questions. The released phyjudge LoRA was fine-tuned against
`default.yaml`'s `training_prompts`, which is why `scripts/score_videos.sh`
passes `--use_training_prompts` by default.

## 4. Human annotation

```bash
pip install flask selenium
python -m evals.human_eval.app          # serves on :5000
```

The Flask app implements the comparison-mode UI: each annotator rates a
random group of 3 models side-by-side along the General dims and the
prompt-specific physical laws. Assignments, sessions, and ratings live in
`human_eval.db` (SQLite). Tests under `evals/human_eval/tests/` cover
assignment, DB, import, and route logic and run with `pytest`.

## 5. Judge LoRA training

```bash
# Build ms-swift SFT JSONL from raw judgement logs.
python -m judge_training.data.build_swift_data convert \
    --base_dir . --val-output train_val.jsonl --val-ratio 0.1
python -m judge_training.data.build_swift_data validate train.jsonl

# Or from Claude CoT eval JSONs:
python -m judge_training.data.build_from_claude_cot convert \
    --prompt-config cotnosubq.yaml \
    --eval-dir data/scores/claude \
    --pattern 'eval_claude_cot_*.json'
```

Train with [`ms-swift`](https://github.com/modelscope/ms-swift) +
DeepSpeed ZeRO-2 against the produced JSONL — see the model card for the
exact swift CLI invocation, hyperparameters, and base checkpoint
(`Qwen/Qwen3.5-9B`).
