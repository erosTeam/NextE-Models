#!/usr/bin/env python3
"""Reconstruct the pinned waifu2x upconv7 ncnn graph and export fixed-shape ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import onnx
import torch
from torch import nn


FP16_TAG = 0x01306B47
CONVOLUTION_SPECS = (
    (3, 16, 3),
    (16, 32, 3),
    (32, 64, 3),
    (64, 128, 3),
    (128, 128, 3),
    (128, 256, 3),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_locked_file(path: Path, entry: dict, label: str) -> None:
    if path.stat().st_size != int(entry["bytes"]) or sha256(path) != entry["sha256"]:
        raise RuntimeError(f"{label} does not match source.lock.json")


class WeightReader:
    def __init__(self, path: Path) -> None:
        self.data = memoryview(path.read_bytes())
        self.offset = 0

    def fp16(self, count: int) -> torch.Tensor:
        tag = int.from_bytes(self.data[self.offset : self.offset + 4], "little")
        self.offset += 4
        if tag != FP16_TAG:
            raise RuntimeError(f"unexpected ncnn weight tag 0x{tag:08x}")
        byte_count = count * 2
        values = np.frombuffer(
            self.data[self.offset : self.offset + byte_count], dtype="<f2"
        ).astype(np.float32)
        self.offset += byte_count
        return torch.from_numpy(values.copy())

    def fp32(self, count: int) -> torch.Tensor:
        byte_count = count * 4
        values = np.frombuffer(
            self.data[self.offset : self.offset + byte_count], dtype="<f4"
        )
        self.offset += byte_count
        return torch.from_numpy(values.copy())

    def require_eof(self) -> None:
        if self.offset != len(self.data):
            raise RuntimeError(
                f"unconsumed ncnn model bytes: {len(self.data) - self.offset}"
            )


class Waifu2xPhotoNoise0X2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.convolutions = nn.ModuleList(
            nn.Conv2d(input_channels, output_channels, kernel_size)
            for input_channels, output_channels, kernel_size in CONVOLUTION_SPECS
        )
        self.deconvolution = nn.ConvTranspose2d(256, 3, 4, stride=2, padding=3)
        self.activation = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for convolution in self.convolutions:
            value = self.activation(convolution(value))
        return self.deconvolution(value)

    def load_ncnn_weights(self, path: Path) -> None:
        reader = WeightReader(path)
        with torch.no_grad():
            for convolution, (input_channels, output_channels, kernel_size) in zip(
                self.convolutions, CONVOLUTION_SPECS
            ):
                convolution.weight.copy_(
                    reader.fp16(
                        input_channels * output_channels * kernel_size * kernel_size
                    ).reshape(output_channels, input_channels, kernel_size, kernel_size)
                )
                convolution.bias.copy_(reader.fp32(output_channels))
            self.deconvolution.weight.copy_(
                reader.fp16(256 * 3 * 4 * 4).reshape(256, 3, 4, 4)
            )
            self.deconvolution.bias.copy_(reader.fp32(3))
        reader.require_eof()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameter", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--lock",
        type=Path,
        default=root / "models/waifu2x-photo-noise0-x2/source.lock.json",
    )
    args = parser.parse_args()

    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    verify_locked_file(args.parameter, lock["parameter"], "ncnn parameter")
    verify_locked_file(args.weights, lock["weights"], "ncnn weights")
    parameter_lines = args.parameter.read_text(encoding="utf-8").splitlines()
    if parameter_lines[:2] != ["7767517", "8 8"]:
        raise RuntimeError("unexpected ncnn graph header")
    layer_types = [line.split()[0] for line in parameter_lines[2:]]
    if layer_types != ["Input", *("Convolution" for _ in range(6)), "Deconvolution"]:
        raise RuntimeError(f"unexpected ncnn graph layers: {layer_types}")

    model = Waifu2xPhotoNoise0X2().eval()
    model.load_ncnn_weights(args.weights)
    contract = lock["runtimeContract"]
    sample = torch.linspace(
        0.0,
        1.0,
        steps=int(np.prod(contract["inputShape"])),
        dtype=torch.float32,
    ).reshape(contract["inputShape"])
    with torch.no_grad():
        output = model(sample)
    expected_output = tuple(int(value) for value in contract["outputShape"])
    if tuple(output.shape) != expected_output:
        raise RuntimeError(
            f"unexpected output shape {tuple(output.shape)}, expected {expected_output}"
        )
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("reconstructed model produced non-finite output")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        sample,
        args.output,
        export_params=True,
        opset_version=int(lock["export"]["opset"]),
        do_constant_folding=True,
        input_names=[contract["inputName"]],
        output_names=[contract["outputName"]],
    )
    exported = onnx.load(args.output)
    onnx.checker.check_model(exported)
    print(
        f"exported={args.output.resolve()} bytes={args.output.stat().st_size} "
        f"sha256={sha256(args.output)} nodes={len(exported.graph.node)} "
        f"outputRange={float(output.min()):.9g}..{float(output.max()):.9g}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
