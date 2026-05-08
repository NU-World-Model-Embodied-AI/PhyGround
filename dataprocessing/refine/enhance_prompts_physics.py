"""Enhance prompts by adding expected physical phenomena using Gemini.

Many prompts only describe the initial scene setup and action, but don't describe
the expected physical outcome (e.g., liquid overflowing, dominoes falling in
sequence, ball bouncing). This script uses Gemini to append a concise description
of the expected physical phenomenon to each prompt.

Usage:
    # Dry run (preview changes without writing)
    python -m dataprocessing.refine.enhance_prompts_physics --dry_run

    # Run on all datasets
    python -m dataprocessing.refine.enhance_prompts_physics

    # Run on specific dataset
    python -m dataprocessing.refine.enhance_prompts_physics --dataset physics_iq

    # Vertex AI mode
    python -m dataprocessing.refine.enhance_prompts_physics --vertexai
"""

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from dataprocessing.common.gemini import add_gemini_args, make_client

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = """\
You are a physics expert helping improve text prompts for a video generation benchmark.

Your task: given a prompt that describes a physical scene, check whether it explicitly
describes the **expected physical outcome/phenomenon**. If the prompt only describes
the setup and initial action but NOT the resulting physics, add a concise sentence
describing the expected physical phenomenon.

Rules:
1. If the prompt ALREADY describes the physical outcome adequately, return it unchanged.
2. If the prompt is MISSING the physical outcome, insert ONE concise sentence describing
   what physically happens as a result of the described action.
3. Keep the addition natural and concise (1 sentence, ~10-20 words).
4. Do NOT change the existing text — only append/insert the new physics description.
5. Keep any "Static shot with no camera movement." or similar camera notes at the END.
6. Return ONLY the enhanced prompt text, no explanation, no quotes.

Examples:

Input: "A bright red liquid being poured from a dispenser into a glass which is placed on a dark baking tray on a wooden table. Static shot with no camera movement."
Output: "A bright red liquid being poured from a dispenser into a glass which is placed on a dark baking tray on a wooden table. The liquid overflows from the glass and spills onto the baking tray. Static shot with no camera movement."

Input: "A row of colorful wooden blocks lined up on a wooden table with a wooden stick attached to a black rotating platform. The platform rotates clockwise and the wooden stick hits the first block as it rotates. Static shot with no camera movement."
Output: "A row of colorful wooden blocks lined up on a wooden table with a wooden stick attached to a black rotating platform. The platform rotates clockwise and the wooden stick hits the first block as it rotates. The blocks topple over one by one in a domino chain reaction. Static shot with no camera movement."

Input: "A basketball is dropped from a rooftop and bounces on the pavement, each bounce lower than the last."
Output: "A basketball is dropped from a rooftop and bounces on the pavement, each bounce lower than the last."
(Already describes the physics — returned unchanged.)
"""

USER_TEMPLATE = """\
Category: {category}
Prompt: {prompt}"""


def enhance_prompt(client, prompt: str, category: str, max_retries: int = 3) -> str | None:
    """Call Gemini to enhance one prompt with physical phenomenon description."""
    user_msg = USER_TEMPLATE.format(prompt=prompt, category=category)

    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[user_msg],
                config={"system_instruction": SYSTEM_PROMPT},
            )
            text = resp.text.strip().strip('"').strip("'")
            if text:
                return text
            logger.warning("Empty response (attempt %d)", attempt + 1)
        except Exception as e:
            logger.warning("API error (attempt %d/%d): %s", attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None


def process_physics_iq(client, dry_run: bool = False):
    """Enhance Physics-IQ descriptions.csv prompts."""
    csv_path = Path("data/prompts/physics_iq/descriptions.csv")
    logger.info("=== Physics-IQ: %s ===", csv_path)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if "take-2" not in r.get("scenario", "")]

    unique_descs: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        desc = row["description"]
        unique_descs.setdefault(desc, []).append(i)

    logger.info("Total rows: %d, unique descriptions: %d", len(rows), len(unique_descs))

    enhanced_map: dict[str, str] = {}
    changed = 0
    for j, (desc, indices) in enumerate(unique_descs.items()):
        category = rows[indices[0]]["category"]
        logger.info("[%d/%d] %s — %s", j + 1, len(unique_descs), category, desc[:80])

        if dry_run:
            continue

        result = enhance_prompt(client, desc, category)
        if result and result != desc:
            enhanced_map[desc] = result
            changed += 1
            logger.info("  CHANGED → %s", result[:100])
        else:
            logger.info("  unchanged")
        time.sleep(0.3)

    if dry_run:
        logger.info("DRY RUN — no changes written")
        return

    if changed > 0:
        backup = csv_path.with_suffix(f".csv.bak_{datetime.now():%Y%m%d_%H%M%S}")
        backup.write_text(csv_path.read_text())
        logger.info("Backup → %s", backup)

        for row in rows:
            if row["description"] in enhanced_map:
                row["description"] = enhanced_map[row["description"]]

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["scenario", "description", "category", "generated_video_name"])
            writer.writeheader()
            writer.writerows(rows)

        logger.info("Physics-IQ: %d/%d unique descriptions enhanced", changed, len(unique_descs))
    else:
        logger.info("Physics-IQ: no changes needed")


_JSON_DATASETS = {
    "video_phy_2": (Path("data/prompts/video_phy_2/video_phy_2.json"), "VideoPhy-2", ("_domain",)),
    "wmb": (Path("data/prompts/wmb/wmb.json"), "WorldModelBench", ("_domain", "wmb_domain")),
    "openvid": (Path("data/prompts/openvid/openvid.json"), "OpenVid", ("domain",)),
}


def _process_json_dataset(client, json_path: Path, name: str,
                          domain_keys: tuple[str, ...],
                          dry_run: bool = False):
    """Enhance prompts in a JSON dataset file."""
    logger.info("=== %s: %s ===", name, json_path)

    with open(json_path) as f:
        data = json.load(f)

    changed = 0
    total = 0
    for pid, p in data["prompts"].items():
        if p.get("status") != "kept":
            continue
        total += 1
        prompt = p["prompt"]
        domain = next((p[k] for k in domain_keys if p.get(k)), "")
        logger.info("[%s %s] %s", domain, pid, prompt[:80])

        if dry_run:
            continue

        result = enhance_prompt(client, prompt, domain)
        if result and result != prompt:
            p["prompt"] = result
            changed += 1
            logger.info("  CHANGED → %s", result[:100])
        else:
            logger.info("  unchanged")
        time.sleep(0.3)

    if dry_run:
        logger.info("DRY RUN — no changes written")
        return

    if changed > 0:
        backup = json_path.with_suffix(f".json.bak_{datetime.now():%Y%m%d_%H%M%S}")
        backup.write_text(json_path.read_text())
        logger.info("Backup → %s", backup)

        with open(json_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info("%s: %d/%d prompts enhanced", name, changed, total)
    else:
        logger.info("%s: no changes needed", name)


def main():
    parser = argparse.ArgumentParser(description="Enhance prompts with physical phenomena via Gemini")
    parser.add_argument("--dataset", choices=["physics_iq", "video_phy_2", "wmb", "openvid", "all"], default="all")
    add_gemini_args(parser)
    parser.add_argument("--dry_run", action="store_true", help="Preview without making changes")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    client = make_client(args) if not args.dry_run else None

    if args.dataset in ("all", "physics_iq"):
        process_physics_iq(client, dry_run=args.dry_run)
        logger.info("")

    datasets = list(_JSON_DATASETS) if args.dataset == "all" else (
        [args.dataset] if args.dataset in _JSON_DATASETS else []
    )
    for ds in datasets:
        path, name, dk = _JSON_DATASETS[ds]
        _process_json_dataset(client, path, name, dk, dry_run=args.dry_run)
        logger.info("")

    logger.info("Done.")


if __name__ == "__main__":
    main()
