#!/usr/bin/env python3
"""Merge the phyjudge LoRA into its base checkpoint for vLLM serving.

vLLM cannot serve this adapter as a LoRA: it fails to load the LoRA on
Qwen3.5's packed linear-attention projections and ignores the vision-merger
LoRA. The paper's numbers come from serving a merged checkpoint, so
scripts/serve_judge.sh does the same.

The merge works directly on the safetensors shards (W += alpha / r * B @ A)
and keeps the base checkpoint's tensor names, so no model class is loaded and
the result loads in vLLM exactly like the base model.

Usage:
  python scripts/merge_judge.py \\
      --base Qwen/Qwen3.5-9B \\
      --lora NU-World-Model-Embodied-AI/phyjudge-9B \\
      --out ./phyjudge-9B-merged
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ADAPTER_FILES = ["adapter_config.json", "adapter_model.safetensors"]


def resolve_dir(source: str, allow_patterns: list[str] | None = None) -> Path:
    """Return a local directory for a local path or a HF Hub repo id."""
    path = Path(source)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=source, allow_patterns=allow_patterns))


def load_lora_deltas(adapter_dir: Path) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], float]:
    """Map base tensor name -> (lora_A, lora_B), plus the LoRA scale."""
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    if cfg.get("use_dora"):
        raise ValueError("DoRA adapters are not supported by this merge script")
    r, alpha = cfg["r"], cfg["lora_alpha"]
    scale = alpha / math.sqrt(r) if cfg.get("use_rslora") else alpha / r

    weights = load_file(str(adapter_dir / "adapter_model.safetensors"))
    deltas: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for key, lora_a in weights.items():
        if not key.endswith(".lora_A.weight"):
            continue
        prefix = key[: -len(".lora_A.weight")]
        lora_b = weights[f"{prefix}.lora_B.weight"]
        target = prefix.removeprefix("base_model.model.") + ".weight"
        deltas[target] = (lora_a, lora_b)
    return deltas, scale


def merge(base_dir: Path, adapter_dir: Path, out_dir: Path) -> None:
    deltas, scale = load_lora_deltas(adapter_dir)

    index_path = base_dir / "model.safetensors.index.json"
    if index_path.exists():
        shards = sorted(set(json.loads(index_path.read_text())["weight_map"].values()))
    else:
        shards = ["model.safetensors"]

    # Write into a temp dir and rename at the end, so an interrupted merge is
    # never mistaken for a finished one.
    tmp_dir = out_dir.with_name(out_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    merged: set[str] = set()
    for shard in shards:
        tensors = load_file(str(base_dir / shard))
        for name, weight in tensors.items():
            if name not in deltas:
                continue
            lora_a, lora_b = deltas[name]
            delta = scale * (lora_b.float() @ lora_a.float())
            tensors[name] = (weight.float() + delta).to(weight.dtype)
            merged.add(name)
        save_file(tensors, str(tmp_dir / shard), metadata={"format": "pt"})
        print(f"  {shard}: done")

    missing = sorted(set(deltas) - merged)
    if missing:
        shutil.rmtree(tmp_dir)
        raise RuntimeError(
            f"{len(missing)} LoRA targets not found in the base checkpoint, "
            f"e.g. {missing[:3]}"
        )

    for item in base_dir.iterdir():
        if item.suffix == ".safetensors" or item.name.startswith("."):
            continue
        if item.is_file():
            shutil.copy(item, tmp_dir / item.name)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp_dir.rename(out_dir)
    print(f"Merged {len(merged)} LoRA modules (scale={scale}) into {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", default="Qwen/Qwen3.5-9B", help="Base model HF id or local dir.")
    parser.add_argument(
        "--lora", default="NU-World-Model-Embodied-AI/phyjudge-9B",
        help="LoRA adapter HF id or local dir.",
    )
    parser.add_argument("--out", required=True, help="Output dir for the merged checkpoint.")
    args = parser.parse_args()

    base_dir = resolve_dir(args.base)
    adapter_dir = resolve_dir(args.lora, allow_patterns=ADAPTER_FILES)
    print(f"Merging {adapter_dir} into {base_dir}")
    merge(base_dir, adapter_dir, Path(args.out))


if __name__ == "__main__":
    main()
