#!/usr/bin/env python3
"""Validate a local Qwen3-Omni AutoRound model directory before serving."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REQUIRED = {
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "quantization_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--sha256-manifest", type=Path)
    args = parser.parse_args()

    missing = sorted(name for name in REQUIRED if not (args.model_dir / name).is_file())
    if missing:
        raise SystemExit(f"missing required files: {', '.join(missing)}")

    config = json.loads((args.model_dir / "config.json").read_text())
    quant = config.get("quantization_config", {})
    if quant.get("quant_method") != "auto-round":
        raise SystemExit(f"unexpected quant_method: {quant.get('quant_method')!r}")
    if config.get("architectures") != ["Qwen3OmniMoeForConditionalGeneration"]:
        raise SystemExit(f"unexpected architectures: {config.get('architectures')!r}")

    index = json.loads((args.model_dir / "model.safetensors.index.json").read_text())
    weight_files = sorted(set(index["weight_map"].values()))
    absent = [name for name in weight_files if not (args.model_dir / name).is_file()]
    if absent:
        raise SystemExit(f"missing weight shards: {', '.join(absent)}")

    if args.sha256_manifest:
        with args.sha256_manifest.open("w") as out:
            for name in sorted(REQUIRED | set(weight_files)):
                path = args.model_dir / name
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                out.write(f"{digest}  {name}\n")

    total = sum((args.model_dir / name).stat().st_size for name in weight_files)
    print(f"OK: {len(weight_files)} shards, {total / 1024**3:.2f} GiB")
    print(f"quant_method={quant.get('quant_method')} bits={quant.get('bits')} group_size={quant.get('group_size')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
