#!/usr/bin/env python3
"""Create private Float32 NCHW calibration tiles matching NextE preprocessing."""

from __future__ import annotations

import argparse
from array import array
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys

from PIL import Image, ImageOps


TILE_SIZE = 160
PREPADDING = 10
INPUT_SIZE = TILE_SIZE + PREPADDING * 2
EXPECTED_BYTES = 3 * INPUT_SIZE * INPUT_SIZE * 4
# NextE stores verified reader bytes with an opaque .img cache suffix; Pillow still detects the
# encoded format from the file header.
IMAGE_EXTENSIONS = {".avif", ".bmp", ".img", ".jpeg", ".jpg", ".png", ".webp"}
Image.MAX_IMAGE_PIXELS = None


@dataclass(frozen=True)
class TileCandidate:
    path: Path
    width: int
    height: int
    x: int
    y: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def discover_candidates(input_dir: Path) -> list[TileCandidate]:
    candidates: list[TileCandidate] = []
    paths = sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    for path in paths:
        image = open_rgb(path)
        width, height = image.size
        image.close()
        for y in range(0, height, TILE_SIZE):
            for x in range(0, width, TILE_SIZE):
                candidates.append(TileCandidate(path, width, height, x, y))
    return candidates


def edge_padded_page(image: Image.Image) -> Image.Image:
    width, height = image.size
    covered_width = math.ceil(width / TILE_SIZE) * TILE_SIZE
    covered_height = math.ceil(height / TILE_SIZE) * TILE_SIZE
    canvas = Image.new("RGB", (covered_width + PREPADDING * 2, covered_height + PREPADDING * 2))
    canvas.paste(image, (PREPADDING, PREPADDING))

    right = covered_width + PREPADDING - width
    bottom = covered_height + PREPADDING - height
    nearest = Image.Resampling.NEAREST
    canvas.paste(
        image.crop((0, 0, width, 1)).resize((width, PREPADDING), nearest),
        (PREPADDING, 0),
    )
    canvas.paste(
        image.crop((0, height - 1, width, height)).resize((width, bottom), nearest),
        (PREPADDING, PREPADDING + height),
    )
    canvas.paste(
        image.crop((0, 0, 1, height)).resize((PREPADDING, height), nearest),
        (0, PREPADDING),
    )
    canvas.paste(
        image.crop((width - 1, 0, width, height)).resize((right, height), nearest),
        (PREPADDING + width, PREPADDING),
    )
    canvas.paste(image.getpixel((0, 0)), (0, 0, PREPADDING, PREPADDING))
    canvas.paste(
        image.getpixel((width - 1, 0)),
        (PREPADDING + width, 0, canvas.width, PREPADDING),
    )
    canvas.paste(
        image.getpixel((0, height - 1)),
        (0, PREPADDING + height, PREPADDING, canvas.height),
    )
    canvas.paste(
        image.getpixel((width - 1, height - 1)),
        (PREPADDING + width, PREPADDING + height, canvas.width, canvas.height),
    )
    return canvas


def build_runtime_tile(image: Image.Image, input_x: int, input_y: int) -> Image.Image:
    padded = edge_padded_page(image)
    tile = padded.crop((input_x, input_y, input_x + INPUT_SIZE, input_y + INPUT_SIZE))
    padded.close()
    if tile.size != (INPUT_SIZE, INPUT_SIZE):
        raise RuntimeError(f"unexpected tile size: {tile.size}")
    return tile


def write_nchw_float32(tile: Image.Image, output: Path) -> None:
    values = array("f")
    scale = 1.0 / 255.0
    for band in tile.split():
        values.extend(pixel * scale for pixel in band.tobytes())
    if sys.byteorder != "little":
        values.byteswap()
    with output.open("wb") as target:
        values.tofile(target)
    if output.stat().st_size != EXPECTED_BYTES:
        raise RuntimeError(f"unexpected calibration buffer size: {output.stat().st_size}")


def prepare_output(output: Path, force: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    existing = list(output.iterdir())
    if existing and not force:
        raise RuntimeError(f"output directory is not empty: {output}; pass --force to replace it")
    if force:
        for path in existing:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.samples <= 500:
        raise RuntimeError("--samples must be between 1 and 500")
    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    if not input_dir.is_dir():
        raise RuntimeError(f"input directory does not exist: {input_dir}")

    candidates = discover_candidates(input_dir)
    if not candidates:
        raise RuntimeError("no supported images were found")
    count = min(args.samples, len(candidates))
    selected = random.Random(args.seed).sample(candidates, count)
    prepare_output(output_dir, args.force)
    buffers_dir = output_dir / "input"
    buffers_dir.mkdir()

    source_hashes: dict[Path, str] = {}
    records: list[dict] = []
    cached_path: Path | None = None
    cached_image: Image.Image | None = None
    try:
        for index, candidate in enumerate(selected):
            if cached_path != candidate.path:
                if cached_image is not None:
                    cached_image.close()
                cached_path = candidate.path
                cached_image = open_rgb(candidate.path)
            assert cached_image is not None
            source_digest = source_hashes.setdefault(candidate.path, sha256(candidate.path))
            name = f"input-{index:04d}.bin"
            path = buffers_dir / name
            tile = build_runtime_tile(cached_image, candidate.x, candidate.y)
            write_nchw_float32(tile, path)
            tile.close()
            records.append(
                {
                    "file": f"input/{name}",
                    "sha256": sha256(path),
                    "sourceId": source_digest[:16],
                    "sourceWidth": candidate.width,
                    "sourceHeight": candidate.height,
                    "inputX": candidate.x,
                    "inputY": candidate.y,
                }
            )
    finally:
        if cached_image is not None:
            cached_image.close()

    manifest = {
        "schemaVersion": 1,
        "privacy": "local-only; do not commit or upload",
        "seed": args.seed,
        "candidateTileCount": len(candidates),
        "sampleCount": len(records),
        "contract": {
            "dataType": "FLOAT32",
            "byteOrder": "little",
            "format": "NCHW",
            "shape": [1, 3, INPUT_SIZE, INPUT_SIZE],
            "tileSize": TILE_SIZE,
            "prepadding": PREPADDING,
            "pixelRange": [0.0, 1.0],
        },
        "samples": records,
    }
    manifest_path = output_dir / "calibration-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"prepared={output_dir} samples={len(records)} candidates={len(candidates)} "
        f"manifest={manifest_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
