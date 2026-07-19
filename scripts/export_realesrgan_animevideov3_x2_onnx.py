#!/usr/bin/env python3
"""Export the pinned Real-ESRGAN animevideov3 checkpoint as a fixed-shape x2 graph."""

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


class AnimeVideoV3X2(nn.Module):
    """Official x4 SRVGG graph followed by the upstream x2 bilinear resize."""

    def __init__(self) -> None:
        super().__init__()
        body: list[nn.Module] = [nn.Conv2d(3, 64, 3, 1, 1), nn.PReLU(64)]
        for _ in range(16):
            body.extend([nn.Conv2d(64, 64, 3, 1, 1), nn.PReLU(64)])
        body.append(nn.Conv2d(64, 48, 3, 1, 1))
        self.body = nn.ModuleList(body)
        self.shuffle = nn.PixelShuffle(4)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = value
        for layer in self.body:
            output = layer(output)
        output = self.shuffle(output)
        output = output + functional.interpolate(value, scale_factor=4, mode="nearest")
        return functional.interpolate(
            output,
            scale_factor=0.5,
            mode="bilinear",
            align_corners=False,
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--lock",
        type=Path,
        default=root / "models/realesrgan-animevideov3-x2/source.lock.json",
    )
    args = parser.parse_args()

    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    contract = lock["runtimeContract"]
    checkpoint = args.checkpoint.resolve()
    expected = lock["checkpoint"]
    if checkpoint.stat().st_size != int(expected["bytes"]) or sha256(checkpoint) != expected["sha256"]:
        raise RuntimeError("checkpoint does not match source.lock.json")

    try:
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        loaded = torch.load(checkpoint, map_location="cpu")
    state = dict(loaded.get(str(expected["stateDictionary"]), loaded))
    model = AnimeVideoV3X2().eval()
    model.load_state_dict(state, strict=True)
    input_shape = tuple(int(value) for value in contract["inputShape"])
    sample = torch.zeros(input_shape, dtype=torch.float32)
    with torch.no_grad():
        output = model(sample)
    expected_output = tuple(int(value) for value in contract["outputShape"])
    if tuple(output.shape) != expected_output:
        raise RuntimeError(f"unexpected output shape {tuple(output.shape)}, expected {expected_output}")

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
        f"sha256={sha256(args.output)} nodes={len(exported.graph.node)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
