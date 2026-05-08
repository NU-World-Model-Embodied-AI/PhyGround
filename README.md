# Phyground — Code

Source code for the **Phyground** benchmark: prompt curation, video-model
evaluation with VLM-as-judge, LoRA judge training, and the Flask web app used
to collect human ratings. Companion to:

- **Dataset** — 250 prompts × 8 video models = 2,000 videos plus per-rubric
  human ratings:
  [🤗 NU-World-Model-Embodied-AI/phyground](https://huggingface.co/datasets/NU-World-Model-Embodied-AI/phyground)
  (mirrored locally at [`../datasets/`](../datasets/))
- **Model** — LoRA judge adapter (Qwen2.5-VL based), prompt template, and
  `infer.py`: [`../model/`](../model/)

## What's in this drop

42 Python source files, 5 prompt-template YAMLs, and the HTML/CSS/JS assets
for the human-annotation app. Shell scripts, secrets/configs, generated
dashboards, SQLite databases, and binary outputs are intentionally
excluded — the dataset and model cards point to the artifacts those scripts
consume.

## Layout

```
export_human_annotations.py   # Dump SQLite annotations → annotator_*.json (dataset format)

dataprocessing/
  common/        # Vertex AI / OpenAI client helpers, video-id utilities,
                 # batched-pipeline runner
  refine/        # Prompt-set construction:
                 #   enhance_prompts_physics.py — physics-aware rewriting
                 #   gen_humaneval_set.py       — sample the human-eval subset
  analysis/      # Ablation: effect of prompt enhancement on judge agreement

evals/
  eval_types.py        # Typed result containers for VLM-as-judge runs
  physics_criteria.py  # Physical-law sub-rubric definitions
  sub_questions.py     # Sub-question rendering for CoT / SubQ prompts
  prompts/             # Judge prompt templates (default, CoT, SubQ variants)
  human_eval/          # Flask app: assignment, rating UI, coverage reports,
                       # alignment checks, tests, templates, static assets

judge_training/
  data/          # Build ms-swift SFT data from raw judgement logs:
                 #   schema, sampling, naming, Claude-CoT and DB builders
```

## Quick start

```bash
# Judge inference (see ../model/infer.py for the full path)
pip install transformers peft "qwen-vl-utils[decord]"

# Run the human-annotation app locally
pip install flask selenium
python -m evals.human_eval.app          # serves on :5000

# Build SFT data for ms-swift
python -m judge_training.data.build_swift_data --help
```

The judge prompt templates under `evals/prompts/` are loaded by the runners
described in the paper. The actual VLM-API runners (vLLM / Vertex / OpenAI
clients) are wired through `dataprocessing/common/`.

## Dependencies

Versions match those reported in the paper.

| Component | Used for |
| --- | --- |
| `transformers`, `peft`, `qwen-vl-utils[decord]` | Judge inference |
| `ms-swift`, `deepspeed` (ZeRO-2) | Judge LoRA training |
| `vllm` (OpenAI-compatible server) | Hosting the base VLM during evaluation |
| `google-genai` / Vertex AI | Gemini family runs |
| `anthropic` / Vertex AI | Claude family runs |
| `openai` Python SDK | OpenAI / GPT family runs |
| `flask`, `sqlite3`, `selenium` | Human-annotation web app & tests |

## Citation

If you use this code, please cite the Phyground paper (see the dataset card
for the BibTeX entry) and the upstream VLM / training-stack projects listed
above.

## License

Released under the same terms as the rest of the Phyground release; see
`../datasets/LICENSE`.
