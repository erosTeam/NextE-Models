#!/usr/bin/env python3
"""Fetch, rename, and verify the immutable ncnn runtime release assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.request


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_asset(source: dict, artifact: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / artifact["fileName"]
    part = output.with_suffix(output.suffix + ".part")
    if part.exists():
        part.unlink()
    request = urllib.request.Request(source["url"], headers={
        "User-Agent": "NextE-Models-release-builder/1",
    })
    try:
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=120) as response, part.open("wb") as stream:
                    shutil.copyfileobj(response, stream)
                break
            except (TimeoutError, ConnectionError, urllib.error.URLError):
                part.unlink(missing_ok=True)
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        if part.stat().st_size != int(source["bytes"]):
            raise RuntimeError(f"{source['fileName']}: source size mismatch")
        if sha256(part) != source["sha256"]:
            raise RuntimeError(f"{source['fileName']}: source SHA-256 mismatch")
        if int(artifact["bytes"]) != int(source["bytes"]):
            raise RuntimeError(f"{artifact['fileName']}: artifact size does not match source")
        if artifact["sha256"] != source["sha256"]:
            raise RuntimeError(f"{artifact['fileName']}: artifact SHA-256 does not match source")
        part.replace(output)
        return output
    except Exception:
        part.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    names: set[str] = set()
    for entry in manifest["assets"]:
        artifact = entry["artifact"]
        name = artifact["fileName"]
        if name in names:
            raise RuntimeError(f"duplicate artifact file name: {name}")
        names.add(name)
        output = fetch_asset(entry["source"], artifact, args.output)
        print(f"prepared {output.name} bytes={output.stat().st_size} sha256={sha256(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
