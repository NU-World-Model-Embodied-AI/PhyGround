# Phyground — Code

Source code for **Phyground**, a benchmark for the *physical plausibility* of
text+image-to-video (ti2v) generations. This repo contains everything needed
to reproduce the benchmark end-to-end: prompt curation, VLM-as-judge
evaluation, LoRA judge training, and the Flask web app used to collect human
ratings.

Companion artifacts:

| Artifact | Where |
| --- | --- |
| 250 prompts × 8 ti2v models = 2,000 videos + per-rubric human ratings | [🤗 NU-World-Model-Embodied-AI/phyground](https://huggingface.co/datasets/NU-World-Model-Embodied-AI/phyground) |
| LoRA judge adapter (Qwen2.5-VL based) + `infer.py` | [🤗 model card](https://huggingface.co/NU-World-Model-Embodied-AI/phyground) (linked from the dataset card) |
| Paper (rubric, methodology, results) | See dataset card for citation |

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

42 Python source files, 5 prompt-template YAMLs, and the HTML/CSS/JS assets
for the human-annotation app. Shell launchers, secrets/configs, generated
dashboards, SQLite databases, and binary outputs are intentionally excluded —
the dataset and model cards point to the artifacts those scripts consume or
produce.

---

## Repository layout

```
export_human_annotations.py   # Dump SQLite annotations → annotator_*.json (dataset format)

dataprocessing/
  common/        # Vertex AI / OpenAI client helpers, video-id utilities,
                 # batched-pipeline runner with quality checks
  refine/        # Prompt-set construction:
                 #   enhance_prompts_physics.py — Gemini-aided physics-aware rewriting
                 #   gen_humaneval_set.py       — sampler for the human-eval subset
                 #                                 (disabled stub in this release)
  analysis/      # Ablation: effect of prompt enhancement on judge agreement

evals/
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
companion dataset (paths shown below assume the dataset is checked out at
`../datasets/`).

### 1. Prompt curation

```bash
# Enhance prompts to make the expected physical phenomenon explicit.
# Reads ../datasets/prompts/*.csv, calls Gemini, writes back enhanced prompts.
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

### 3. VLM-as-judge evaluation

Pick a prompt template under `evals/prompts/`, then run the judge against
each video. The runner is wired through `dataprocessing/common/pipeline.py`
plus the appropriate API client (`gemini.py`, or your own vLLM /
Anthropic / OpenAI wrapper). Outputs are per-model JSON files of the form
`{"SA": 4, "PTV": 5, "gravity": 3, ...}` — see `evals/eval_types.py` for
the typed schema.

The five prompt templates are A/B-comparable: they share the same scoring
keys but differ in whether they elicit chain-of-thought reasoning and/or
intermediate yes/no answers to per-law sub-questions.

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

After collection, filter and export:

```bash
# Assumes evals/human_eval/human_eval_filtered.db has been produced
# from the raw collection DB by the workflow described in the dataset card.
python export_human_annotations.py \
    --db evals/human_eval/human_eval_filtered.db \
    --out_dir ../datasets/annotations
```

The exporter strips all PII (names, demographics, timestamps); only a
`annotator_NNN` serial id is kept.

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
(Qwen2.5-VL).

For inference at the end, use `infer.py` in the model release.

---

## Quick start (judge inference only)

If you just want to score a video with the released judge:

```bash
pip install transformers peft "qwen-vl-utils[decord]"
# Then follow ../model/infer.py — it loads the LoRA adapter, formats a
# prompt with PromptConfig.load("default.yaml"), and returns a JSON dict
# of per-dimension scores.
```

---

## Dependencies

Versions match those reported in the paper.

| Component | Used for |
| --- | --- |
| `transformers`, `peft`, `qwen-vl-utils[decord]` | Judge inference |
| `ms-swift`, `deepspeed` (ZeRO-2)                | Judge LoRA training |
| `vllm` (OpenAI-compatible server)               | Hosting the base VLM during evaluation |
| `google-genai` / Vertex AI                      | Gemini family runs and prompt enhancement |
| `anthropic` / Vertex AI                         | Claude family runs |
| `openai` Python SDK                             | OpenAI / GPT family runs |
| `flask`, `sqlite3`, `selenium`, `pytest`        | Human-annotation web app & tests |
| `pyyaml`                                        | Loading prompt templates |

Python ≥ 3.10 (uses PEP-604 union types throughout).

---

## Reproducing benchmark numbers

Reproducing the table from the paper requires the dataset (videos +
human ratings) and one of the released judges. The steps are:

1. Pull the dataset from Hugging Face (link above) so that
   `../datasets/{prompts,videos,annotations}/` exist.
2. Run the judge of your choice (released LoRA or any closed-source VLM)
   on every (video, prompt-template) pair using the runners in
   `dataprocessing/common/`.
3. Aggregate per-model means and compute agreement against
   `../datasets/annotations/annotator_*.json` using
   `dataprocessing/analysis/ablation_prompt_enhancement.py` as a worked
   example.

The released LoRA adapter targets the `default.yaml` template;
chain-of-thought and sub-question variants exist for ablations and
require re-running the relevant tables.

---

## Citation

If you use this code or the accompanying dataset/model, please cite the
Phyground paper. The BibTeX entry lives on the
[dataset card](https://huggingface.co/datasets/NU-World-Model-Embodied-AI/phyground)
and is reproduced in the model card. Please also credit the upstream
projects in the dependency table above (Qwen2.5-VL, ms-swift, vLLM, etc.).

---

## License

Released under the **Creative Commons Attribution 4.0 International License
(CC BY 4.0)** — the same terms as the rest of the Phyground release.

You are free to share and adapt the material for any purpose, including
commercially, provided you give appropriate credit, link to the license,
and indicate if changes were made. No additional restrictions may be
applied. The material is provided as-is, without warranties of any kind.

Full legal text: <https://creativecommons.org/licenses/by/4.0/legalcode>
