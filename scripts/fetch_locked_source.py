#!/usr/bin/env python3
"""Download one immutable checkpoint described by a source lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(path: Path, expected_bytes: int, expected_sha256: str) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == expected_bytes
        and file_sha256(path) == expected_sha256
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("lock", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--entry",
        choices=[
            "checkpoint",
            "converter",
            "inferenceJson",
            "inferenceParams",
            "inferenceYml",
            "parameter",
            "source",
            "sourceOnnx",
            "weights",
        ],
        default="checkpoint",
    )
    args = parser.parse_args()

    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    locked_asset = lock[args.entry]
    expected_bytes = int(locked_asset["bytes"])
    expected_sha256 = str(locked_asset["sha256"])
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if validate(output, expected_bytes, expected_sha256):
        print(f"already verified: {output}")
        return 0

    part = output.with_name(output.name + ".part")
    part.unlink(missing_ok=True)
    request = urllib.request.Request(
        str(locked_asset["url"]),
        headers={"User-Agent": "NextE-Models/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response, part.open("wb") as target:
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
        if not validate(part, expected_bytes, expected_sha256):
            raise RuntimeError("downloaded checkpoint does not match the source lock")
        os.replace(part, output)
    finally:
        part.unlink(missing_ok=True)

    print(
        f"verified: {output} bytes={expected_bytes} sha256={expected_sha256}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
