#!/usr/bin/env python3
"""Verify a rebuilt artifact against the public manifest and release tag."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--tag")
    args = parser.parse_args()

    manifest = json.loads((root / "manifests/models-v1.json").read_text(encoding="utf-8"))
    matches = [entry for entry in manifest["models"] if entry["id"] == args.model]
    if len(matches) != 1:
        raise RuntimeError(f"manifest model count is {len(matches)} for {args.model}")
    entry = matches[0]
    expected = entry["artifact"]
    artifact = args.artifact.resolve()
    if artifact.name != expected["fileName"]:
        raise RuntimeError("artifact file name does not match the manifest")
    if artifact.stat().st_size != int(expected["bytes"]):
        raise RuntimeError("artifact byte count does not match the manifest")
    if sha256(artifact) != expected["sha256"]:
        raise RuntimeError("artifact SHA-256 does not match the manifest")

    if args.tag is not None:
        if entry["status"] != "published":
            raise RuntimeError("release tags require a published manifest entry")
        expected_url = (
            f"https://github.com/erosTeam/NextE-Models/releases/download/"
            f"{args.tag}/{artifact.name}"
        )
        if expected["urls"] != [expected_url]:
            raise RuntimeError("published URL does not match the immutable release tag")
    print(
        f"verified model={args.model} artifact={artifact.name} "
        f"bytes={artifact.stat().st_size} sha256={expected['sha256']}"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyError, RuntimeError, ValueError) as error:
        print(f"publish verification failed: {error}", file=sys.stderr)
        sys.exit(1)
