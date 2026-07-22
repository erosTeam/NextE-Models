#!/usr/bin/env python3
"""Rebuild the SHA-locked AOT manga inpainting model as fixed-shape ncnn FP32."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess


CHECKPOINT_BYTES = 22_785_303
CHECKPOINT_SHA256 = "878d541c68648969bc1b042a6e997f3a58e49b6c07c5636ad55130736977149f"
SOURCE_BYTES = 11_022
SOURCE_SHA256 = "b3e409540e1f2b61f1a5fee2650ff2b900880ad0841ac17be7724d6c544b99b5"
ONNX_BYTES = 23_061_643
ONNX_SHA256 = "4563b234bd26dd757080b4adcf665103748e9fe1f5990f18d16683fc833dc4fe"
PARAM_BYTES = 35_150
PARAM_SHA256 = "fcaf0b35e6bf19d94f3fe42c98e484bb6e754ede71ffaaa2a48cc9ad5d737b66"
MODEL_BYTES = 22_711_880
MODEL_SHA256 = "356f816992eeb90d50480e96322d6a06dbb8254524180cb951e15975fb60cd7d"


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


def load_generator(source: Path):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from typing import List, Optional

    text = source.read_text(encoding="utf-8")
    marker = "def relu_nf(x):"
    start = text.find(marker)
    if start < 0:
        raise RuntimeError("AOT architecture marker is missing")
    namespace = {
        "__name__": "nexte_aot_manga_architecture",
        "List": List,
        "Optional": Optional,
        "np": np,
        "torch": torch,
        "nn": nn,
        "F": F,
    }
    exec(compile(text[start:], str(source), "exec"), namespace)
    return namespace["AOTGenerator"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pnnx", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    verify(args.checkpoint, CHECKPOINT_BYTES, CHECKPOINT_SHA256, "checkpoint")
    verify(args.source, SOURCE_BYTES, SOURCE_SHA256, "architecture source")

    import torch

    if torch.__version__.split("+")[0] != "2.8.0":
        raise RuntimeError(f"torch 2.8.0 is required, got {torch.__version__}")
    generator = load_generator(args.source)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = generator()
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.output_dir / "aot-manga-inpainting-256.onnx"
    image = torch.zeros((1, 3, 256, 256), dtype=torch.float32)
    mask = torch.zeros((1, 1, 256, 256), dtype=torch.float32)
    torch.onnx.export(
        model,
        (image, mask),
        onnx_path,
        input_names=["image", "mask"],
        output_names=["inpainted"],
        dynamic_axes=None,
        opset_version=17,
        do_constant_folding=True,
        export_params=True,
    )
    verify(onnx_path, ONNX_BYTES, ONNX_SHA256, "exported ONNX")

    subprocess.run(
        [
            str(args.pnnx),
            str(onnx_path),
            "inputshape=[1,3,256,256],[1,1,256,256]",
            "fp16=0",
            f"pnnxparam={args.output_dir / 'aot.pnnx.param'}",
            f"pnnxbin={args.output_dir / 'aot.pnnx.bin'}",
            f"pnnxpy={args.output_dir / 'aot_pnnx.py'}",
            f"pnnxonnx={args.output_dir / 'aot.pnnx.onnx'}",
            f"ncnnparam={args.output_dir / 'aot-manga-inpainting-256.ncnn.param'}",
            f"ncnnbin={args.output_dir / 'aot-manga-inpainting-256.ncnn.bin'}",
            f"ncnnpy={args.output_dir / 'aot_manga_inpainting_256_ncnn.py'}",
        ],
        check=True,
    )
    param_path = args.output_dir / "aot-manga-inpainting-256.ncnn.param"
    model_path = args.output_dir / "aot-manga-inpainting-256.ncnn.bin"
    verify(param_path, PARAM_BYTES, PARAM_SHA256, "ncnn param")
    verify(model_path, MODEL_BYTES, MODEL_SHA256, "ncnn model")
    print(f"verified {param_path.name} and {model_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
