#!/usr/bin/env python3
"""Convert the SHA-locked YSGYolo ONNX graph to a fixed-shape ncnn pair."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess


SOURCE_BYTES = 10_838_944
SOURCE_SHA256 = "6f3202925f01fdf045f8c31a3bf62e6c44944f56ce09107eb436bc5a5b185ebe"
PARAM_BYTES = 27_031
PARAM_SHA256 = "f3617c7834bf3f7ae67521db908a53709140aeb1c11a02f8c64b7c091b569987"
MODEL_BYTES = 10_720_760
MODEL_SHA256 = "7658e654db1a2e8a77c387607def85d5d297b26f110cc2b334dd52ae17a4fe00"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, expected_bytes: int, expected_sha256: str, label: str) -> None:
    if path.stat().st_size != expected_bytes or sha256(path) != expected_sha256:
        raise RuntimeError(f"{label} does not match the locked artifact")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pnnx", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    verify(args.onnx, SOURCE_BYTES, SOURCE_SHA256, "source ONNX")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    working_onnx = args.output_dir / "ysgyolo_1.2_OS1.0.onnx"
    if args.onnx.resolve() != working_onnx.resolve():
        shutil.copyfile(args.onnx, working_onnx)
    subprocess.run(
        [str(args.pnnx), working_onnx.name, "inputshape=[1,3,640,640]", "fp16=0"],
        cwd=args.output_dir,
        check=True,
    )
    stem = working_onnx.stem
    generated_param = args.output_dir / f"{stem}.ncnn.param"
    generated_model = args.output_dir / f"{stem}.ncnn.bin"
    target_param = args.output_dir / "ysgyolo_1.2_OS1.0.ncnn.param"
    target_model = args.output_dir / "ysgyolo_1.2_OS1.0.ncnn.bin"
    if generated_param != target_param:
        shutil.replace(generated_param, target_param)
    if generated_model != target_model:
        shutil.replace(generated_model, target_model)
    verify(target_param, PARAM_BYTES, PARAM_SHA256, "ncnn param")
    verify(target_model, MODEL_BYTES, MODEL_SHA256, "ncnn model")
    print(f"verified {target_param.name} and {target_model.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
