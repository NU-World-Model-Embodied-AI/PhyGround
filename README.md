# Phyground — Code

<div align="center">

[![Slide](https://img.shields.io/badge/Slides-07C160?style=for-the-badge&logo=slides&logoColor=white)](assets/phyground-slides.pdf)
[![Paper](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.10806)
[![Hugging Face](https://img.shields.io/badge/dataset-fcd022?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/NU-World-Model-Embodied-AI/phyground)

</div>

Source code for **Phyground**, a benchmark for the *physical plausibility* of
text+image-to-video (ti2v) generations. This repo contains everything needed
to reproduce the benchmark end-to-end: prompt curation, VLM-as-judge
evaluation, LoRA judge training, and the Flask web app used to collect human
ratings.

Companion artifacts:

| Artifact | Where |
| --- | --- |
| Dataset | [🤗 NU-World-Model-Embodied-AI/phyground](https://huggingface.co/datasets/NU-World-Model-Embodied-AI/phyground) |
| judge model | [🤗 Phyjudge-9B](https://huggingface.co/NU-World-Model-Embodied-AI/phyjudge-9B) |
| Paper (rubric, methodology, results) | [PhyGround](https://arxiv.org/abs/2605.10806) |

---

## What this benchmark measures

Each video is rated on a **1–5 ordinal scale** along three families of
dimensions:

**General (always rated, 3 dims)**

| Code | Key | Question |
| --- | --- | --- |
| G1 | `persistence` | Do objects keep consistent appearance, shape, and existence? |
| G2 | `PTV`         | Is the temporal order of physical events plausible? |
| G3 | `SA`          | Does the video align with the text prompt? |

**Physical-law sub-rubric (13 laws, only the laws that apply to the prompt are rated)**

| Domain | Laws |
| --- | --- |
| A. Solid-Body Mechanics | `gravity`, `inertia`, `momentum`, `impenetrability`, `collision`, `material` |
| B. Fluid Dynamics       | `buoyancy`, `displacement`, `flow_dynamics`, `boundary_interaction`, `fluid_continuity` |
| C. Optics               | `reflection`, `shadow` |

The single source of truth for these definitions (English + Chinese, plus
sub-question decompositions used for chain-of-thought judging) is
[`evals/physics_criteria.py`](evals/physics_criteria.py) and
[`evals/sub_questions.py`](evals/sub_questions.py).

---

## What's in this drop

Python source for the eval runner + 5 prompt-template YAMLs + the
HTML/CSS/JS assets for the human-annotation app + two thin shell wrappers
(`scripts/serve_judge.sh` and `scripts/score_videos*.sh`) that turn the
runner into a one-click flow. Secrets, generated dashboards, SQLite
databases, and binary outputs are intentionally excluded — the dataset
and model cards point to the artifacts those scripts consume or produce.

---

## Repository layout

```
dataprocessing/
  common/        # Vertex AI / OpenAI client helpers, video-id utilities,
                 # batched-pipeline runner with quality checks
  refine/        # Prompt-set construction:
                 #   enhance_prompts_physics.py — Gemini-aided physics-aware rewriting
                 #   gen_humaneval_set.py       — sampler for the human-eval subset
                 #                                 (disabled stub in this release)

evals/
  vlm_eval.py          # CLI entry point — scores one --video_dir of *.mp4
  vlm_common.py        # Backends: OpenAI-compatible vLLM, Gemini, GPT,
                       # Claude (Vertex AI), plus response parsing +
                       # summary printing
  eval_types.py        # Typed result containers for VLM-as-judge runs
  physics_criteria.py  # 13 physical laws (EN + ZH) + human-eval rubric definitions
  sub_questions.py     # Per-law observational sub-questions for CoT / SubQ prompts
  prompts/             # 5 judge prompt templates + PromptConfig loader:
                       #   default.yaml      — direct 1-5 score, JSON-only output
                       #   cotnosubq.yaml    — chain-of-thought, no sub-questions
                       #   cot-subq.yaml     — CoT + observational sub-questions
                       #   subq+answer.yaml  — sub-questions answered yes/no/uncertain
                       #   subq+human.yaml   — human-style sub-questions
  human_eval/          # Flask app: assignment, rating UI, coverage reports,
                       # alignment checks, tests, templates, static assets

scripts/
  serve_judge.sh       # Launch phyjudge LoRA on a vLLM OpenAI-compatible server
  score_videos.sh      # One-click: serve_judge.sh + evals.vlm_eval (local LoRA path)
  score_videos_api.sh  # Same, but routes to Gemini / GPT / Claude cloud APIs

judge_training/
  data/          # Build ms-swift SFT data from raw judgement logs:
                 #   schema.py, sample.py, naming.py, prompt_config.py
                 #   build_records_from_db.py     — aggregate human ratings
                 #   build_from_claude_cot.py     — convert Claude CoT logs
                 #   build_swift_data.py          — write/split/validate JSONL
```

---

## End-to-end pipeline

The benchmark is built in five stages. Each stage corresponds to one or two
modules in this repo; downstream stages consume artifacts that live in the
companion dataset. Point `$DATASET_DIR` at wherever you downloaded the
Hugging Face dataset before running the snippets below, e.g.:

```bash
export DATASET_DIR=/path/to/phyground-dataset
```

### 1. Prompt curation

```bash
# Enhance prompts to make the expected physical phenomenon explicit.
# Reads $DATASET_DIR/prompts/*.csv, calls Gemini, writes back enhanced prompts.
python -m dataprocessing.refine.enhance_prompts_physics --dry_run
python -m dataprocessing.refine.enhance_prompts_physics --dataset wmb
```

Requires Gemini API access (Vertex AI or AI Studio) — see
`dataprocessing/common/gemini.py` for client config and CLI flags.

### 2. Video generation

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

### 3. VLM-as-judge evaluation

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

#### Cloud-API judges (Gemini / OpenAI / Claude)

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

### 4. Human annotation

```bash
pip install flask selenium
python -m evals.human_eval.app          # serves on :5000
```

The Flask app implements the comparison-mode UI: each annotator rates a
random group of 3 models side-by-side along the General dims and the
prompt-specific physical laws. Assignments, sessions, and ratings live in
`human_eval.db` (SQLite). Tests under `evals/human_eval/tests/` cover
assignment, DB, import, and route logic and run with `pytest`.

### 5. Judge LoRA training

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

---

## Quick start (judge inference only)

If you just want to score videos with the released judge, jump to §3 of
the pipeline above — `scripts/score_videos.sh` is the one-click runner.
The transformers/peft path documented on the
[model card](https://huggingface.co/NU-World-Model-Embodied-AI/phyjudge-9B)
remains supported for users who'd rather load the LoRA in-process instead
of going through vLLM.

---

## Installation

The repo ships a [`pyproject.toml`](pyproject.toml) with extras grouped by
use case, so you only install what you need.

```bash
# Minimum: just the prompt-template loader and project source
pip install -e .

# Pick any subset of the extras:
pip install -e ".[eval]"                # One-stop for scripts/score_videos*.sh
                                        # (all backend clients + decord + HF CLI)
pip install -e ".[web]"                 # Flask annotation app
pip install -e ".[inference]"           # In-process judge inference (transformers, peft)
pip install -e ".[training]"            # ms-swift + deepspeed for judge fine-tuning
pip install -e ".[gemini,claude,openai]" # Per-API subsets, if you don't want [eval]
pip install -e ".[test]"                # pytest for the human-eval tests

# Or grab everything in one go
pip install -e ".[all]"
```

---

## Reproducing benchmark numbers

Reproducing the table from the paper requires the dataset (videos +
human ratings) and one of the released judges. The steps are:

1. Pull the dataset from Hugging Face (link above) so that
   `data/{prompts/phyground.json,first_images/,annotations/}` exist
   locally (see §3b).
2. Run the judge of your choice on every (video, prompt-template) pair:
   `scripts/score_videos.sh` for the released LoRA, or
   `scripts/score_videos_api.sh` for a closed-source baseline. Vary
   `PROMPT_CONFIG=…` across `default.yaml`, `cotnosubq.yaml`,
   `cot-subq.yaml`, `subq+answer.yaml`, `subq+human.yaml` to reproduce
   the prompt-template ablations.
3. Aggregate per-model means and compute agreement against the human
   ratings in `data/annotations/annotator_*.json`.

The released LoRA adapter targets the `default.yaml` template (and that
template's `training_prompts` block — which is why
`scripts/score_videos.sh` always passes `--use_training_prompts`).
Chain-of-thought and sub-question variants exist for ablations and
require re-running the relevant tables under a different
`PROMPT_CONFIG`.

---

## Citation

Paper: [arXiv:2605.10806](https://arxiv.org/abs/2605.10806)

```bibtex
@misc{lin2026phygroundbenchmarkingphysicalreasoning,
      title={PhyGround: Benchmarking Physical Reasoning in Generative World Models},
      author={Juyi Lin and Arash Akbari and Yumei He and Lin Zhao and Haichao Zhang and Arman Akbari and Xingchen Xu and Zoe Y. Lu and Enfu Nan and Hokin Deng and Edmund Yeh and Sarah Ostadabbas and Yun Fu and Jennifer Dy and Pu Zhao and Yanzhi Wang},
      year={2026},
      eprint={2605.10806},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.10806},
}
```
