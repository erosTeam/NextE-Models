#!/usr/bin/env python3
"""Reconstruct the locked waifu2x CUNet ncnn graph and export fixed-shape ONNX."""

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

    def weights(self, count: int) -> torch.Tensor:
        tag = int.from_bytes(self.data[self.offset : self.offset + 4], "little")
        self.offset += 4
        if tag == FP16_TAG:
            byte_count = count * 2
            values = np.frombuffer(
                self.data[self.offset : self.offset + byte_count], dtype="<f2"
            ).astype(np.float32)
        elif tag == 0:
            byte_count = count * 4
            values = np.frombuffer(
                self.data[self.offset : self.offset + byte_count], dtype="<f4"
            )
        else:
            raise RuntimeError(f"unexpected ncnn weight tag 0x{tag:08x} at {self.offset - 4}")
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
            raise RuntimeError(f"unconsumed ncnn model bytes: {len(self.data) - self.offset}")


class Waifu2xCunetNoise0X2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 2, stride=2)
        self.conv4 = nn.Conv2d(64, 128, 3)
        self.conv5 = nn.Conv2d(128, 64, 3)
        self.fc6 = nn.Linear(64, 8)
        self.fc7 = nn.Linear(8, 64)
        self.deconv1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.conv8 = nn.Conv2d(64, 64, 3)
        self.deconv2 = nn.ConvTranspose2d(64, 3, 4, stride=2, padding=3)

        self.conv9 = nn.Conv2d(3, 32, 3)
        self.conv10 = nn.Conv2d(32, 64, 3)
        self.conv11 = nn.Conv2d(64, 64, 2, stride=2)
        self.conv12 = nn.Conv2d(64, 64, 3)
        self.conv13 = nn.Conv2d(64, 128, 3)
        self.fc14 = nn.Linear(128, 16)
        self.fc15 = nn.Linear(16, 128)
        self.conv16 = nn.Conv2d(128, 128, 2, stride=2)
        self.conv17 = nn.Conv2d(128, 256, 3)
        self.conv18 = nn.Conv2d(256, 128, 3)
        self.fc19 = nn.Linear(128, 16)
        self.fc20 = nn.Linear(16, 128)
        self.deconv3 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.conv21 = nn.Conv2d(128, 64, 3)
        self.conv22 = nn.Conv2d(64, 64, 3)
        self.fc23 = nn.Linear(64, 8)
        self.fc24 = nn.Linear(8, 64)
        self.deconv4 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.conv25 = nn.Conv2d(64, 64, 3)
        self.conv26 = nn.Conv2d(64, 3, 3)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

    def activate(self, value: torch.Tensor) -> torch.Tensor:
        return self.leaky_relu(value)

    @staticmethod
    def squeeze_excite(
        value: torch.Tensor,
        reduction: nn.Linear,
        expansion: nn.Linear,
    ) -> torch.Tensor:
        scale = torch.mean(value, dim=(2, 3))
        scale = torch.relu(reduction(scale))
        scale = torch.sigmoid(expansion(scale)).unsqueeze(2).unsqueeze(3)
        return value * scale

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        first1 = self.activate(self.conv1(value))
        first2 = self.activate(self.conv2(first1))
        first3 = self.activate(self.conv3(first2))
        first4 = self.activate(self.conv4(first3))
        first5 = self.activate(self.conv5(first4))
        first5 = self.squeeze_excite(first5, self.fc6, self.fc7)
        first_up = self.activate(self.deconv1(first5))
        first_skip = first2[:, :, 4:-4, 4:-4]
        first8 = self.activate(self.conv8(first_skip + first_up))
        first_output = self.deconv2(first8)

        second9 = self.activate(self.conv9(first_output))
        second10 = self.activate(self.conv10(second9))
        second11 = self.activate(self.conv11(second10))
        second12 = self.activate(self.conv12(second11))
        second13 = self.activate(self.conv13(second12))
        second13 = self.squeeze_excite(second13, self.fc14, self.fc15)
        second16 = self.activate(self.conv16(second13))
        second17 = self.activate(self.conv17(second16))
        second18 = self.activate(self.conv18(second17))
        second18 = self.squeeze_excite(second18, self.fc19, self.fc20)
        second_up3 = self.activate(self.deconv3(second18))
        second_skip13 = second13[:, :, 4:-4, 4:-4]
        second21 = self.activate(self.conv21(second_skip13 + second_up3))
        second22 = self.activate(self.conv22(second21))
        second22 = self.squeeze_excite(second22, self.fc23, self.fc24)
        second_up4 = self.activate(self.deconv4(second22))
        second_skip10 = second10[:, :, 16:-16, 16:-16]
        second25 = self.activate(self.conv25(second_skip10 + second_up4))
        second26 = self.conv26(second25)
        first_crop = first_output[:, :, 20:-20, 20:-20]
        return first_crop + second26

    @staticmethod
    def load_conv(reader: WeightReader, layer: nn.Conv2d) -> None:
        with torch.no_grad():
            layer.weight.copy_(reader.weights(layer.weight.numel()).reshape(layer.weight.shape))
            layer.bias.copy_(reader.fp32(layer.out_channels))

    @staticmethod
    def load_deconv(reader: WeightReader, layer: nn.ConvTranspose2d) -> None:
        with torch.no_grad():
            output_channels = layer.out_channels
            input_channels = layer.in_channels
            height = layer.kernel_size[0]
            width = layer.kernel_size[1]
            layer.weight.copy_(
                reader.weights(layer.weight.numel())
                .reshape(output_channels, input_channels, height, width)
                .permute(1, 0, 2, 3)
            )
            layer.bias.copy_(reader.fp32(output_channels))

    @staticmethod
    def load_linear(reader: WeightReader, layer: nn.Linear) -> None:
        with torch.no_grad():
            layer.weight.copy_(reader.weights(layer.weight.numel()).reshape(layer.weight.shape))
            layer.bias.copy_(reader.fp32(layer.out_features))

    def load_ncnn_weights(self, path: Path) -> None:
        reader = WeightReader(path)
        operations = (
            (self.load_conv, self.conv1),
            (self.load_conv, self.conv2),
            (self.load_conv, self.conv3),
            (self.load_conv, self.conv4),
            (self.load_conv, self.conv5),
            (self.load_linear, self.fc6),
            (self.load_linear, self.fc7),
            (self.load_deconv, self.deconv1),
            (self.load_conv, self.conv8),
            (self.load_deconv, self.deconv2),
            (self.load_conv, self.conv9),
            (self.load_conv, self.conv10),
            (self.load_conv, self.conv11),
            (self.load_conv, self.conv12),
            (self.load_conv, self.conv13),
            (self.load_linear, self.fc14),
            (self.load_linear, self.fc15),
            (self.load_conv, self.conv16),
            (self.load_conv, self.conv17),
            (self.load_conv, self.conv18),
            (self.load_linear, self.fc19),
            (self.load_linear, self.fc20),
            (self.load_deconv, self.deconv3),
            (self.load_conv, self.conv21),
            (self.load_conv, self.conv22),
            (self.load_linear, self.fc23),
            (self.load_linear, self.fc24),
            (self.load_deconv, self.deconv4),
            (self.load_conv, self.conv25),
            (self.load_conv, self.conv26),
        )
        for loader, layer in operations:
            loader(reader, layer)
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
        default=root / "models/waifu2x-cunet-noise0-x2/source.lock.json",
    )
    args = parser.parse_args()

    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    verify_locked_file(args.parameter, lock["parameter"], "ncnn parameter")
    verify_locked_file(args.weights, lock["weights"], "ncnn weights")
    parameter_lines = args.parameter.read_text(encoding="utf-8").splitlines()
    if parameter_lines[:2] != ["7767517", "59 71"]:
        raise RuntimeError("unexpected ncnn graph header")

    model = Waifu2xCunetNoise0X2().eval()
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
        raise RuntimeError(f"unexpected output shape {tuple(output.shape)}, expected {expected_output}")
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
