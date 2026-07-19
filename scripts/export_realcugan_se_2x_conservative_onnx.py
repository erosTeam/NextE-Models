#!/usr/bin/env python3
"""Export the pinned Real-CUGAN SE 2x conservative tile graph to fixed-shape ONNX."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import onnx
import torch
from torch import nn


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected_bytes: int, expected_sha256: str, label: str) -> None:
    if path.stat().st_size != expected_bytes or sha256(path) != expected_sha256:
        raise RuntimeError(f"{label} does not match source.lock.json")


def load_upcunet(source: Path) -> type[nn.Module]:
    spec = importlib.util.spec_from_file_location("realcugan_upcunet_v3", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load the pinned Real-CUGAN source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.UpCunet2x


class RealCuganTile(nn.Module):
    """The already-padded tile core used by NextE's current ncnn models-se graph."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.unet1 = model.unet1
        self.unet2 = model.unet2

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        first = self.unet1(value)
        second = self.unet2(first, 1.0)
        return second + first[:, :, 20:-20, 20:-20]


class FixedAveragePoolSE(nn.Module):
    """SE gate with an explicit fixed AveragePool operator for NNRT lowering."""

    def __init__(self, source: nn.Module, feature_size: int) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=feature_size)
        self.conv1 = source.conv1
        self.conv2 = source.conv2

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        gate = self.pool(value)
        gate = torch.relu(self.conv1(gate))
        gate = torch.sigmoid(self.conv2(gate))
        return value * gate


class SliceProbe(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value[:, :, 4:-4, 4:-4]


def export_probe(module: nn.Module, shape: tuple[int, ...], output: Path, opset: int) -> None:
    sample = torch.linspace(0.0, 1.0, steps=int(np.prod(shape)), dtype=torch.float32).reshape(shape)
    module = module.eval()
    with torch.no_grad():
        result = module(sample)
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError(f"probe {output.stem} produced non-finite output")
    torch.onnx.export(
        module,
        sample,
        output,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )
    onnx.checker.check_model(onnx.load(output))
    print(f"probe={output.name} shape={shape} bytes={output.stat().st_size} sha256={sha256(output)}")


def replace_se_means(model: nn.Module) -> None:
    # Fixed feature sizes derive from the locked 178 px input contract. AveragePool is
    # mathematically equivalent to ReduceMean here and maps through a distinct NNRT operator path.
    replacements = (
        (model.unet1.conv2, 83),
        (model.unet2.conv2, 156),
        (model.unet2.conv3, 74),
        (model.unet2.conv4, 144),
    )
    for block, feature_size in replacements:
        block.seblock = FixedAveragePoolSE(block.seblock, feature_size)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--probe-directory", type=Path)
    parser.add_argument(
        "--lock",
        type=Path,
        default=root / "models/realcugan-se-2x-conservative/source.lock.json",
    )
    args = parser.parse_args()

    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    checkpoint_lock = lock["checkpoint"]
    verify_file(
        args.checkpoint,
        int(checkpoint_lock["memberBytes"]),
        str(checkpoint_lock["memberSha256"]),
        "Real-CUGAN checkpoint",
    )
    source_lock = lock["source"]
    verify_file(
        args.source,
        int(source_lock["bytes"]),
        str(source_lock["sha256"]),
        "Real-CUGAN source",
    )

    model = load_upcunet(args.source)().eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    contract = lock["runtimeContract"]
    sample = torch.linspace(
        0.0,
        1.0,
        steps=int(np.prod(contract["inputShape"])),
        dtype=torch.float32,
    ).reshape(contract["inputShape"])
    tile = RealCuganTile(model).eval()
    with torch.no_grad():
        reference_output = tile(sample)
    replace_se_means(model)
    tile = RealCuganTile(model).eval()
    with torch.no_grad():
        output = tile(sample)
    maximum_rewrite_error = float(torch.max(torch.abs(output - reference_output)))
    # Torch's threaded ReduceMean and AvgPool accumulation order differs by a few ULPs here.
    # 4e-6 is about 0.001 of one 8-bit output step and remains far tighter than Reader QA.
    if maximum_rewrite_error > 4.0e-6:
        raise RuntimeError(
            f"fixed AveragePool SE rewrite changed output by {maximum_rewrite_error}"
        )
    expected_shape = tuple(int(value) for value in contract["outputShape"])
    if tuple(output.shape) != expected_shape:
        raise RuntimeError(f"unexpected output shape {tuple(output.shape)}, expected {expected_shape}")
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("Real-CUGAN tile graph produced non-finite output")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        tile,
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
        f"outputRange={float(output.min()):.9g}..{float(output.max()):.9g} "
        f"maximumRewriteError={maximum_rewrite_error:.9g}"
    )
    if args.probe_directory is not None:
        args.probe_directory.mkdir(parents=True, exist_ok=True)
        opset = int(lock["export"]["opset"])
        probes: tuple[tuple[str, nn.Module, tuple[int, ...]], ...] = (
            ("slice", SliceProbe(), (1, 3, 178, 178)),
            ("se83", model.unet1.conv2.seblock, (1, 64, 83, 83)),
            ("stride2-conv", model.unet1.conv1_down, (1, 64, 174, 174)),
            ("deconv2", model.unet1.conv2_up, (1, 64, 83, 83)),
            ("deconv4", model.unet1.conv_bottom, (1, 64, 164, 164)),
            ("unet1", model.unet1, (1, 3, 178, 178)),
            ("unet2", model.unet2, (1, 3, 324, 324)),
        )
        for name, module, shape in probes:
            export_probe(module, shape, args.probe_directory / f"{name}.onnx", opset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
