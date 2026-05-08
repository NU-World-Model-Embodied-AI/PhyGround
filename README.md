---
language:
  - en
tags:
  - code
  - video-evaluation
  - benchmark
  - judge
---

# Phyground — Code

Source code for the benchmark, evaluation, and judge-training pipeline that
accompanies the companion data release
[`phyground`](../datasets/) and the LoRA judge adapter under
[`../model/`](../model/).

This drop contains 43 Python source files plus the HTML/CSS/JS assets needed
by the human-annotation app. Shell scripts, configuration YAMLs,
prompts/answers JSONs, generated dashboards, databases, and binary assets are
intentionally excluded — see the dataset and model cards for the artifacts and
prompts those scripts consume.

## Layout

```
dataprocessing/
  common/              # Vertex AI / OpenAI client helpers, video-id utilities
  refine/              # Prompt-set construction: enhance, dedup, hard-subset,
                       # humaneval-set assembly, removal sync
  analysis/            # Ablation: prompt-enhancement effect on judges

evals/
  eval_types.py        # Typed result containers for VLM-as-judge runs
  physics_criteria.py  # Physical-law sub-rubric definitions
  sub_questions.py     # Sub-question rendering for CoT/SubQ prompts
  prompts/             # Prompt-template loaders (YAMLs withheld; see model card)
  human_eval/          # Flask-based human-annotation app + tests + templates

judge_training/
  data/                # Build SFT data for ms-swift from raw judgement logs
                       # (schema, sampling, naming, Claude-CoT/DB builders)
```

## Companion artifacts

- **Dataset** (250 prompts × 8 video models = 2 000 videos, sub-rubric
  ground truth): `../datasets/` — see its `README.md`.
- **Model** (LoRA judge adapter, prompt template, inference script):
  `../model/` — see its `README.md` and `infer.py`.

## Dependencies (top-level)

The pipeline relies on the following open-source components. Versions match
those reported in the paper.

| Component | Used for |
| --- | --- |
| `transformers`, `peft`, `qwen-vl-utils[decord]` | Judge inference |
| `ms-swift`, `deepspeed` | Judge LoRA training (ZeRO-2) |
| `vllm` (OpenAI-compatible server) | Hosting the base VLM for evaluation |
| `google-genai` / Vertex AI | Gemini family runs |
| `anthropic` / Vertex AI | Claude family runs |
| `openai` Python SDK | OpenAI / GPT family runs |
| `flask`, `sqlite3`, `selenium` | Human-annotation web app |

## License

Code is released under the same terms as the rest of this release.
