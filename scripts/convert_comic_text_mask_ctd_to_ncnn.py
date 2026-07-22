#!/usr/bin/env python3
"""Extract CTD's segmentation head and rebuild the locked ncnn FP16 text masker."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess


SOURCE_BYTES = 94_669_756
SOURCE_SHA256 = "1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f"
EXTRACTED_BYTES = 65_568_468
EXTRACTED_SHA256 = "0063cc01e9d41844cb3920076a5290040e4dd0f86547dbf37cd1634f9255ed04"
PARAM_BYTES = 15_552
PARAM_SHA256 = "1f7e14f5c327e0341e775bf23fd61f4ec34a4d3661bf13ff06ea32e3c5c8d072"
MODEL_BYTES = 32_790_420
MODEL_SHA256 = "c0bdafdc17e194fa28496532abd860aeb1ee08f60f08b919215abd45191f4deb"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, expected_bytes: int, expected_sha256: str, label: str) -> None:
    actual_bytes = path.stat().st_size
    actual_sha256 = sha256(path)
    if actual_bytes != expected_bytes or actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"{label} does not match the locked artifact: "
            f"bytes={actual_bytes} sha256={actual_sha256}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pnnx", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    verify(args.onnx, SOURCE_BYTES, SOURCE_SHA256, "source ONNX")

    import onnx
    from onnx.utils import extract_model

    if onnx.__version__ != "1.18.0":
        raise RuntimeError(f"onnx 1.18.0 is required, got {onnx.__version__}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extracted = args.output_dir / "comic-text-mask-ctd-1024.extracted.onnx"
    extract_model(
        str(args.onnx),
        str(extracted),
        ["images"],
        ["seg"],
        check_model=True,
    )
    verify(extracted, EXTRACTED_BYTES, EXTRACTED_SHA256, "extracted segmentation ONNX")

    subprocess.run(
        [
            str(args.pnnx),
            str(extracted),
            "inputshape=[1,3,1024,1024]",
            "fp16=1",
            f"pnnxparam={args.output_dir / 'comic-text-mask-ctd-1024.pnnx.param'}",
            f"pnnxbin={args.output_dir / 'comic-text-mask-ctd-1024.pnnx.bin'}",
            f"pnnxpy={args.output_dir / 'comic_text_mask_ctd_1024_pnnx.py'}",
            f"pnnxonnx={args.output_dir / 'comic-text-mask-ctd-1024.pnnx.onnx'}",
            f"ncnnparam={args.output_dir / 'comic-text-mask-ctd-1024.ncnn.param'}",
            f"ncnnbin={args.output_dir / 'comic-text-mask-ctd-1024.ncnn.bin'}",
            f"ncnnpy={args.output_dir / 'comic_text_mask_ctd_1024_ncnn.py'}",
        ],
        check=True,
    )
    param_path = args.output_dir / "comic-text-mask-ctd-1024.ncnn.param"
    model_path = args.output_dir / "comic-text-mask-ctd-1024.ncnn.bin"
    verify(param_path, PARAM_BYTES, PARAM_SHA256, "ncnn param")
    verify(model_path, MODEL_BYTES, MODEL_SHA256, "ncnn model")
    print(f"verified {param_path.name} and {model_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
