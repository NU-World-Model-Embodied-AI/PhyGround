#!/usr/bin/env python3
"""Phyground VLM-as-judge evaluation — YAML-config-aligned scoring.

Uses evals/prompts/*.yaml for prompt templates so eval-time prompts match
training-time prompts for the released phyjudge LoRA.

Supported backends:
  qwen9b / qwen27b   local vLLM serving the released LoRA (default)
  gemini             Gemini via AI Studio or Vertex AI
  gpt                OpenAI direct API (frames as images)
  claude             Claude on Vertex AI

Usage (local LoRA):
  python -m evals.vlm_eval \\
      --backend qwen9b --prompt_config default.yaml --use_training_prompts \\
      --video_dir ./videos \\
      --prompts_json data/prompts/phyground.json \\
      --api_base http://localhost:29673/v1 \\
      --save_path ./scores.json

`--use_training_prompts` is REQUIRED when scoring with the released LoRA —
the adapter was fine-tuned against `training_prompts` and will produce
miscalibrated scores under `eval_prompts`. Closed-source backends should
omit it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from evals.eval_types import PromptEntry, parse_prompt_entry
from evals.physics_criteria import CRITERIA
from evals.prompts import PromptConfig
from evals.vlm_common import (
    _BACKEND_CHOICES,
    create_backend,
    parse_single_general_score,
    print_summary,
    save_eval_json,
    VLMBackend,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Video-prompt loading
# ──────────────────────────────────────────────────────────────────────────────


def _prompt_to_filename_stem(prompt: str) -> str:
    safe = prompt[:80].replace(" ", "_").replace(",", "").replace(".", "")
    safe = safe.replace('"', "").replace("'", "")
    return safe


def load_video_prompt_mapping(prompts_json: str) -> dict[str, PromptEntry]:
    with open(prompts_json) as f:
        data = json.load(f)

    mapping: dict[str, PromptEntry] = {}

    if "prompts" not in data:
        return mapping

    raw_prompts = data["prompts"]
    items = raw_prompts.items() if isinstance(raw_prompts, dict) else enumerate(raw_prompts)

    for key, raw in items:
        key = str(key)
        entry = parse_prompt_entry(raw, key=key)
        if entry is None:
            continue
        if entry.first_frame_image:
            stem = Path(entry.first_frame_image).stem
            mapping[stem] = entry
        if entry.domain:
            mapping[f"{entry.domain}_{key}"] = entry
        if entry.video:
            mapping[entry.video] = entry
        mapping[key] = entry
        prompt_stem = _prompt_to_filename_stem(entry.prompt)
        if prompt_stem:
            mapping[prompt_stem] = entry

    return mapping


def resolve_video_prompt(stem: str, mapping: dict[str, PromptEntry]) -> PromptEntry | None:
    entry = mapping.get(stem)
    if entry is None and "_trimmed-" in stem:
        scenario = stem.split("_trimmed-", 1)[1]
        entry = mapping.get(scenario)
    if entry is None and "_" in stem:
        numeric_id = stem.rsplit("_", 1)[1]
        entry = mapping.get(f"__id_{numeric_id}")
    return entry


def _count_expected_results(video_items: list[tuple[Path, PromptEntry]]) -> int:
    return sum(1 for _, entry in video_items if entry.has_physical_laws)


def _count_nonempty_jsonl_lines(path: Path) -> int:
    with path.open() as f:
        return sum(1 for line in f if line.strip())


def _cleanup_incremental_jsonl(
    *,
    jsonl_path: Path,
    results: list[dict],
    expected_results: int,
    save_path: str,
) -> None:
    if not jsonl_path.exists():
        return
    if len(results) != expected_results:
        logger.info(
            "Keeping incremental JSONL %s: results=%d, expected=%d",
            jsonl_path, len(results), expected_results,
        )
        return

    line_count = _count_nonempty_jsonl_lines(jsonl_path)
    if line_count != len(results):
        logger.warning(
            "Keeping incremental JSONL %s: line_count=%d, results=%d",
            jsonl_path, line_count, len(results),
        )
        return

    if not Path(save_path).exists():
        logger.warning("Keeping incremental JSONL %s: final JSON missing at %s", jsonl_path, save_path)
        return

    jsonl_path.unlink()
    logger.info("Removed incremental JSONL after full completion: %s", jsonl_path)


# ──────────────────────────────────────────────────────────────────────────────
# Eval loop with configurable prompts
# ──────────────────────────────────────────────────────────────────────────────


def _score_general_per_metric(
    backend: VLMBackend,
    cfg: PromptConfig,
    video_path: str,
    prompt_text: str,
    *,
    use_training_prompts: bool = False,
    max_retries: int = 3,
) -> tuple[dict[str, int | None], dict[str, str], dict[str, str], dict[str, dict]]:

    def _eval_one(metric: str) -> tuple[str, int | None, str, str, dict | None]:
        if use_training_prompts:
            metric_prompt = cfg.build_training_prompt(prompt_text, metric)
        else:
            metric_prompt = cfg.build_eval_prompt(prompt_text, metric)

        parsed = None
        raw = ""
        rationale = ""
        usage = None
        for attempt in range(max_retries):
            try:
                response = backend.call(video_path, metric_prompt, system_prompt=cfg.system_prompt)
                usage = getattr(response, "usage", None)
                raw = str(response)
                score_val, rationale = parse_single_general_score(raw, metric)
                if score_val is not None:
                    parsed = score_val
                    break
                logger.warning("Unparseable %s response (attempt %d): %s", metric, attempt + 1, raw[:200])
            except Exception as e:
                logger.warning("API error %s (attempt %d/%d): %s", metric, attempt + 1, max_retries, e)

        if parsed is None:
            logger.warning("Failed to score general metric '%s'", metric)
        return metric, parsed, raw, rationale, usage

    scores: dict[str, int | None] = {}
    raws: dict[str, str] = {}
    rationales: dict[str, str] = {}
    usages: dict[str, dict] = {}

    with ThreadPoolExecutor(max_workers=len(cfg.general_keys)) as pool:
        for metric, parsed, raw, rationale, usage in pool.map(_eval_one, cfg.general_keys):
            scores[metric] = parsed
            raws[metric] = raw
            rationales[metric] = rationale
            if usage is not None:
                usages[metric] = usage

    return scores, raws, rationales, usages


def _score_physical_laws(
    backend: VLMBackend,
    cfg: PromptConfig,
    video_path: str,
    prompt_text: str,
    physical_laws: list[str],
    *,
    max_retries: int = 3,
) -> tuple[dict[str, dict | None], dict[str, str], dict[str, str], dict[str, dict]]:
    if not physical_laws:
        return {}, {}, {}, {}

    def _eval_one(law: str) -> tuple[str, dict | None, str, str, dict | None]:
        criteria = CRITERIA.get(law)
        if not criteria:
            logger.warning("Unknown law '%s', skipping", law)
            return law, None, "", "", None

        law_prompt = cfg.build_physical_prompt(prompt_text, law, criteria)

        parsed = None
        raw = ""
        rationale = ""
        usage = None
        for attempt in range(max_retries):
            try:
                response = backend.call(video_path, law_prompt, system_prompt=cfg.system_prompt)
                usage = getattr(response, "usage", None)
                raw = str(response)
                score_val, rationale = parse_single_general_score(raw, law)
                if score_val is not None:
                    parsed = {"score": int(score_val)}
                    break
                logger.warning("Unparseable %s response (attempt %d): %s", law, attempt + 1, raw[:200])
            except Exception as e:
                logger.warning("API error %s (attempt %d/%d): %s", law, attempt + 1, max_retries, e)

        return law, parsed, raw, rationale, usage

    law_results: dict[str, dict | None] = {}
    raws: dict[str, str] = {}
    rationales: dict[str, str] = {}
    usages: dict[str, dict] = {}

    with ThreadPoolExecutor(max_workers=min(len(physical_laws), 6)) as pool:
        for law, parsed, raw, rationale, usage in pool.map(_eval_one, physical_laws):
            law_results[law] = parsed
            raws[law] = raw
            rationales[law] = rationale
            if usage is not None:
                usages[law] = usage

    return law_results, raws, rationales, usages


def _sum_usage(
    general_usage: dict[str, dict],
    physical_usage: dict[str, dict],
) -> dict[str, int]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    seen = False
    for usage_by_name in (general_usage, physical_usage):
        for usage in usage_by_name.values():
            for key in totals:
                value = usage.get(key)
                if isinstance(value, int):
                    totals[key] += value
                    seen = True
    return totals if seen else {}


def score_video(
    backend: VLMBackend,
    cfg: PromptConfig,
    video_path: str,
    prompt_text: str,
    physical_laws: list[str],
    *,
    use_training_prompts: bool = False,
    max_retries: int = 3,
) -> dict | None:
    general_scores, raw_generals, rationale_generals, usage_generals = _score_general_per_metric(
        backend, cfg, video_path, prompt_text,
        use_training_prompts=use_training_prompts,
        max_retries=max_retries,
    )

    if all(v is None for v in general_scores.values()):
        return None

    law_results, raw_physicals, rationale_physicals, usage_physicals = _score_physical_laws(
        backend, cfg, video_path, prompt_text, physical_laws,
        max_retries=max_retries,
    )

    from evals.eval_types import LawScore, PhysicalSummary
    law_scores: dict[str, LawScore] = {}
    for law in physical_laws:
        parsed = law_results.get(law)
        if parsed is None:
            law_scores[law] = LawScore.failed(law)
        elif parsed.get("score") == "na":
            law_scores[law] = LawScore.not_observed(law)
        else:
            score = parsed["score"]
            sub_answers = parsed.get("sub_answers")
            law_scores[law] = LawScore.scored(law, score, sub_answers=sub_answers)

    physical = PhysicalSummary.from_law_scores(law_scores, len(physical_laws))
    gen_vals = [v for v in general_scores.values() if isinstance(v, (int, float))]
    gen_avg = sum(gen_vals) / len(gen_vals) if gen_vals else None

    return {
        **general_scores,
        "general_avg": gen_avg,
        "physical": physical.to_dict(),
        "raw_response_general": json.dumps(raw_generals),
        "raw_response_physical": json.dumps(raw_physicals),
        "rationale_general": json.dumps(rationale_generals),
        "rationale_physical": json.dumps(rationale_physicals),
        "usage": {
            "general": usage_generals,
            "physical": usage_physicals,
            "total": _sum_usage(usage_generals, usage_physicals),
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Phyground VLM-as-judge evaluation against a vLLM-served judge."
    )
    parser.add_argument(
        "--backend", choices=_BACKEND_CHOICES, default="qwen9b",
        help="qwen9b/qwen27b = local vLLM; gemini/gpt/claude = cloud APIs.",
    )
    parser.add_argument(
        "--prompt_config", type=str, default="default.yaml",
        help="YAML prompt config under evals/prompts/ (default: default.yaml)",
    )
    parser.add_argument(
        "--use_training_prompts", action="store_true",
        help="REQUIRED for the released phyjudge LoRA — it was fine-tuned with training_prompts.",
    )
    parser.add_argument("--video_dir", type=str, required=True, help="Directory of .mp4 files to score.")
    parser.add_argument(
        "--prompts_json", type=str, default="data/prompts/phyground.json",
        help="Path to the Phyground prompts JSON. Default expects the HF dataset pulled into ./data/.",
    )
    parser.add_argument("--model", type=str, default=None, help="Override model name sent to vLLM (default: 'phyjudge', matching scripts/serve_judge.sh --lora-modules).")
    parser.add_argument("--save_path", type=str, default=None, help="Output JSON path. Defaults to <video_dir>/eval_<ts>.json.")
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N videos (smoke test).")
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--shard_total", type=int, default=1)
    parser.add_argument(
        "--skip_existing_jsonl", type=str, default=None,
        help="Path to a jsonl whose 'video' stems should be skipped before sharding.",
    )

    parser.add_argument("--api_base", type=str, default="http://localhost:29673/v1", help="Used only by qwen9b/qwen27b backends.")
    parser.add_argument("--fps", type=int, default=None, help="Resample video frames at this FPS before scoring.")
    parser.add_argument("--num_frames", type=int, default=0, help="Frame-based backends: sample exactly this many frames (overrides --fps).")
    parser.add_argument("--max_images", type=int, default=32, help="Max sampled image frames for frame-based backends.")
    parser.add_argument(
        "--max_tokens", type=int, default=None,
        help="Max output tokens (default 1024 vLLM / 4096 Claude).",
    )

    # Cloud-API credentials and routing (consumed by gemini/gpt/claude backends only).
    parser.add_argument("--api_key_file", type=str, default=None, help="Path to a file containing a Gemini or OpenAI API key. Falls back to GEMINI_API_KEY / OPENAI_API_KEY env vars. (claude on Vertex AI uses gcloud default credentials.)")
    parser.add_argument("--vertexai", action="store_true", help="Gemini backend: use Vertex AI instead of AI Studio (requires --project).")
    parser.add_argument("--project", type=str, default="", help="GCP project id for --vertexai or the claude (Vertex) backend.")
    parser.add_argument("--location", type=str, default="global", help="GCP location for --vertexai or claude (Vertex) backend.")
    parser.add_argument("--reasoning_effort", type=str, default=None, help="GPT backend: reasoning effort hint.")
    parser.add_argument("--request_timeout", type=int, default=120, help="HTTP request timeout in seconds (Gemini and Claude backends).")

    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--max_retries", type=int, default=3)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if not Path(args.prompts_json).exists():
        parser.error(
            f"prompts JSON not found at {args.prompts_json}.\n"
            "Pull it from the HF dataset first:\n"
            "  huggingface-cli download --repo-type dataset "
            "NU-World-Model-Embodied-AI/phyground prompts/phyground.json --local-dir ./data"
        )

    cfg = PromptConfig.load(args.prompt_config)
    logger.info(
        "Prompt config: %s (scheme=%s, use_training_prompts=%s)",
        args.prompt_config, cfg.scheme, args.use_training_prompts,
    )

    backend = create_backend(args)
    judge_name = f"{args.backend}:{args.model or backend.model}"
    logger.info("Backend: %s", judge_name)

    mapping = load_video_prompt_mapping(args.prompts_json)
    logger.info("Loaded %d prompt mappings", len(mapping))
    video_dir = Path(args.video_dir)
    video_files = sorted(video_dir.glob("*.mp4"))
    if args.limit:
        video_files = video_files[:args.limit]
    video_items: list[tuple[Path, PromptEntry]] = []
    for vpath in video_files:
        entry = resolve_video_prompt(vpath.stem, mapping)
        if entry is None:
            logger.warning("No prompt found for %s, skipping", vpath.stem)
            continue
        video_items.append((vpath, entry))

    if args.skip_existing_jsonl:
        skip_path = Path(args.skip_existing_jsonl)
        skip_stems: set[str] = set()
        if skip_path.exists():
            with skip_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        skip_stems.add(json.loads(line)["video"])
                    except (json.JSONDecodeError, KeyError):
                        continue
        before = len(video_items)
        video_items = [vi for vi in video_items if vi[0].stem not in skip_stems]
        logger.info(
            "Skipped %d already-done stems from %s; %d/%d remaining",
            before - len(video_items), skip_path, len(video_items), before,
        )

    if args.shard_total > 1:
        if not (0 <= args.shard_idx < args.shard_total):
            parser.error(f"--shard_idx must be in [0, {args.shard_total}), got {args.shard_idx}")
        before = len(video_items)
        video_items = video_items[args.shard_idx::args.shard_total]
        logger.info(
            "Shard %d/%d: %d/%d videos (strided)",
            args.shard_idx, args.shard_total, len(video_items), before,
        )

    if args.save_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_tag = "_train" if args.use_training_prompts else ""
        args.save_path = str(Path(args.video_dir) / f"eval_{args.backend}{mode_tag}_{ts}.json")

    jsonl_path = args.save_path + "l"
    Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    expected_results = _count_expected_results(video_items)
    actual_fps = getattr(args, "fps", None) or getattr(backend, "fps", None)

    with open(jsonl_path, "a") as jsonl_f:
        for i, (vpath, entry) in enumerate(video_items, 1):
            stem = vpath.stem
            effective_laws = list(entry.physical_laws or [])

            if not entry.has_physical_laws:
                logger.warning("[%d/%d] %s | no physical_laws, skipping", i, len(video_items), stem)
                continue

            logger.info(
                "[%d/%d] %s | laws=%s | prompt=%s",
                i, len(video_items), stem, effective_laws, entry.prompt[:60],
            )

            try:
                result = score_video(
                    backend, cfg, str(vpath), entry.prompt, effective_laws,
                    use_training_prompts=args.use_training_prompts,
                    max_retries=args.max_retries,
                )
            finally:
                backend.release_video(str(vpath))

            if result is None:
                logger.warning("  Failed to score, skipping")
                continue

            result["video"] = stem
            result["prompt"] = entry.prompt
            result["physical_laws"] = effective_laws
            results.append(result)

            jsonl_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            jsonl_f.flush()

            phys = result.get("physical", {})
            phys_avg = phys.get("avg", "N/A") if isinstance(phys, dict) else "N/A"
            logger.info(
                "  -> %s general_avg=%s physical=%s",
                " ".join(f"{k}={result.get(k)}" for k in cfg.general_keys),
                result.get("general_avg"), phys_avg,
            )

    logger.info("Incremental results saved to %s", jsonl_path)

    prompt_mode = "training_prompts" if args.use_training_prompts else "eval_prompts"
    visual_config = {
        "max_height": getattr(backend, "max_height", 360),
        "fps": actual_fps,
    }
    if hasattr(backend, "max_images"):
        visual_config["max_images"] = getattr(backend, "max_images")

    inference_config = {
        "max_tokens": getattr(backend, "max_tokens", getattr(args, "max_tokens", None)),
    }

    prompts_stem = Path(args.prompts_json).stem
    eval_meta = {
        "type": "eval_result",
        "prompt_config": args.prompt_config,
        "prompt_mode": prompt_mode,
        "scheme": cfg.scheme,
        "evaluator": args.backend,
        "subset": prompts_stem,
        "video_model": Path(args.video_dir).name,
        "eval_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "source_filename": Path(args.save_path).name,
    }
    save_eval_json(
        args.save_path, results,
        meta=eval_meta,
        judge=judge_name,
        visual_config=visual_config,
        inference_config=inference_config,
        prompt_config=args.prompt_config,
        prompt_mode=prompt_mode,
    )

    print_summary(results, judge_name)

    _cleanup_incremental_jsonl(
        jsonl_path=Path(jsonl_path),
        results=results,
        expected_results=expected_results,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
