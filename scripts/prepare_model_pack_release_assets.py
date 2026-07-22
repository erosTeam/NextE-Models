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


def prepare_pack(
    models_manifest: dict,
    runtime_manifest: dict,
    output_dir: Path,
    comic_manifest: dict | None = None,
    generated_comic_dir: Path | None = None,
) -> list[Path]:
    release_tag = models_manifest["releaseTag"]
    if runtime_manifest["releaseTag"] != release_tag:
        raise RuntimeError("model and runtime manifests use different release tags")
    if comic_manifest is not None and comic_manifest["releaseTag"] != release_tag:
        raise RuntimeError("model and comic manifests use different release tags")
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

    if comic_manifest is not None:
        if generated_comic_dir is None:
            raise RuntimeError("generated comic model directory is required")
        for entry in comic_manifest["models"]:
            if entry["status"] != "published":
                continue
            for artifact in entry["artifacts"]:
                name = artifact["fileName"]
                if name in names:
                    raise RuntimeError(f"duplicate release asset file name: {name}")
                names.add(name)
                source_path = (generated_comic_dir / name).resolve()
                source = {
                    "fileName": name,
                    "url": source_path.as_uri(),
                    "bytes": artifact["bytes"],
                    "sha256": artifact["sha256"],
                }
                outputs.append(fetch_asset(source, artifact, output_dir))
            for corresponding_source in entry.get("correspondingSource", []):
                artifact = corresponding_source["artifact"]
                name = artifact["fileName"]
                if name in names:
                    raise RuntimeError(f"duplicate release asset file name: {name}")
                names.add(name)
                outputs.append(fetch_asset(
                    corresponding_source["source"],
                    artifact,
                    output_dir,
                ))

    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-manifest", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--comic-manifest", type=Path)
    parser.add_argument("--generated-comic-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    models_manifest = json.loads(args.models_manifest.read_text(encoding="utf-8"))
    runtime_manifest = json.loads(args.runtime_manifest.read_text(encoding="utf-8"))
    comic_manifest = None
    if args.comic_manifest is not None:
        comic_manifest = json.loads(args.comic_manifest.read_text(encoding="utf-8"))
    for output in prepare_pack(
        models_manifest,
        runtime_manifest,
        args.output,
        comic_manifest,
        args.generated_comic_dir,
    ):
        print(f"prepared {output.name} bytes={output.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
