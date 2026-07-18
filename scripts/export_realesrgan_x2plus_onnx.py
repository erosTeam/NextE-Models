#!/usr/bin/env python3
"""Export the pinned Real-ESRGAN x2plus checkpoint to fixed-shape ONNX.

The RRDBNet structure follows the BSD-3-Clause Real-ESRGAN / BasicSR implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import onnx
import torch
from torch import nn
from torch.nn import functional as functional


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value1 = self.lrelu(self.conv1(value))
        value2 = self.lrelu(self.conv2(torch.cat((value, value1), 1)))
        value3 = self.lrelu(self.conv3(torch.cat((value, value1, value2), 1)))
        value4 = self.lrelu(self.conv4(torch.cat((value, value1, value2, value3), 1)))
        value5 = self.conv5(torch.cat((value, value1, value2, value3, value4), 1))
        return value5 * 0.2 + value


class RRDB(nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = self.rdb1(value)
        output = self.rdb2(output)
        output = self.rdb3(output)
        return output * 0.2 + value


class RRDBNetX2(nn.Module):
    """x2 graph with PixelUnshuffle fused into the first convolution."""

    def __init__(self) -> None:
        super().__init__()
        self.conv_first = nn.Conv2d(3, 64, 6, 2, 2)
        self.body = nn.Sequential(*[RRDB() for _ in range(23)])
        self.conv_body = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        feature = self.conv_first(value)
        feature = feature + self.conv_body(self.body(feature))
        feature = self.lrelu(
            self.conv_up1(functional.interpolate(feature, scale_factor=2, mode="nearest"))
        )
        feature = self.lrelu(
            self.conv_up2(functional.interpolate(feature, scale_factor=2, mode="nearest"))
        )
        return self.conv_last(self.lrelu(self.conv_hr(feature)))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_lock(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fuse_pixel_unshuffle_weight(weight: torch.Tensor) -> torch.Tensor:
    if tuple(weight.shape) != (64, 12, 3, 3):
        raise RuntimeError(f"unexpected conv_first weight shape: {tuple(weight.shape)}")
    fused = torch.empty(64, 3, 6, 6, dtype=weight.dtype)
    for channel in range(3):
        for row in range(2):
            for column in range(2):
                fused[:, channel, row::2, column::2] = weight[
                    :, channel * 4 + row * 2 + column, :, :
                ]
    return fused


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--lock",
        type=Path,
        default=root / "models/realesrgan-x2plus/source.lock.json",
    )
    parser.add_argument("--input-size", type=int, default=180)
    args = parser.parse_args()

    lock = load_lock(args.lock)
    contract = lock["runtimeContract"]
    if args.input_size != int(contract["inputShape"][2]):
        raise RuntimeError(
            f"input size {args.input_size} does not match the locked runtime contract"
        )
    checkpoint = args.checkpoint.resolve()
    expected = lock["checkpoint"]
    if checkpoint.stat().st_size != int(expected["bytes"]) or sha256(checkpoint) != expected["sha256"]:
        raise RuntimeError("checkpoint does not match source.lock.json")

    try:
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        loaded = torch.load(checkpoint, map_location="cpu")
    state = dict(loaded.get("params_ema", loaded.get("params", loaded)))
    original_weight = state["conv_first.weight"]
    state["conv_first.weight"] = fuse_pixel_unshuffle_weight(original_weight)

    model = RRDBNetX2().eval()
    model.load_state_dict(state, strict=True)
    torch.manual_seed(7)
    sample = torch.rand(1, 3, args.input_size, args.input_size)
    with torch.no_grad():
        original_first = functional.conv2d(
            functional.pixel_unshuffle(sample, 2),
            original_weight,
            state["conv_first.bias"],
            padding=1,
        )
        fused_first = model.conv_first(sample)
        max_difference = float((original_first - fused_first).abs().max().item())
        output = model(sample)
    if max_difference > 1e-5:
        raise RuntimeError(f"PixelUnshuffle fusion mismatch: {max_difference}")
    expected_output = tuple(int(value) for value in contract["outputShape"])
    if tuple(output.shape) != expected_output:
        raise RuntimeError(
            f"unexpected output shape {tuple(output.shape)}, expected {expected_output}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        sample,
        args.output,
        export_params=True,
        opset_version=int(lock["export"]["opset"]),
        do_constant_folding=True,
        input_names=[str(contract["inputName"])],
        output_names=[str(contract["outputName"])],
    )
    exported = onnx.load(args.output)
    onnx.checker.check_model(exported)
    print(
        f"exported={args.output.resolve()} bytes={args.output.stat().st_size} "
        f"sha256={sha256(args.output)} nodes={len(exported.graph.node)} "
        f"fusionMaxDifference={max_difference:.9g}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
