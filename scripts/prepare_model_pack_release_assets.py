#!/usr/bin/env python3
"""Collect every published NextE model asset into one immutable release directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_runtime_release_assets import fetch_asset


def model_source(artifact: dict) -> dict:
    urls = artifact.get("sourceUrls", [])
    if not urls:
        raise RuntimeError(f"{artifact['fileName']}: published model has no pack source URL")
    return {
        "fileName": artifact["fileName"],
        "url": urls[0],
        "bytes": artifact["bytes"],
        "sha256": artifact["sha256"],
    }


def prepare_pack(models_manifest: dict, runtime_manifest: dict, output_dir: Path) -> list[Path]:
    release_tag = models_manifest["releaseTag"]
    if runtime_manifest["releaseTag"] != release_tag:
        raise RuntimeError("model and runtime manifests use different release tags")
    names: set[str] = set()
    outputs: list[Path] = []

    for entry in models_manifest["models"]:
        if entry["status"] != "published":
            continue
        artifact = entry["artifact"]
        name = artifact["fileName"]
        if name in names:
            raise RuntimeError(f"duplicate release asset file name: {name}")
        names.add(name)
        outputs.append(fetch_asset(model_source(artifact), artifact, output_dir))

    for entry in runtime_manifest["assets"]:
        artifact = entry["artifact"]
        name = artifact["fileName"]
        if name in names:
            raise RuntimeError(f"duplicate release asset file name: {name}")
        names.add(name)
        outputs.append(fetch_asset(entry["source"], artifact, output_dir))

    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-manifest", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    models_manifest = json.loads(args.models_manifest.read_text(encoding="utf-8"))
    runtime_manifest = json.loads(args.runtime_manifest.read_text(encoding="utf-8"))
    for output in prepare_pack(models_manifest, runtime_manifest, args.output):
        print(f"prepared {output.name} bytes={output.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
