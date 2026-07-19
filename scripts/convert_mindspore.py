#!/usr/bin/env python3
"""Convert the locked ONNX graph to a MindSpore Lite candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


EXPECTED_CALIBRATION_BYTES = 3 * 180 * 180 * 4
FULL_QUANT_MODES = {
    "full-int8-max-min": "full-int8-max-min.cfg.in",
    "full-int8-kl": "full-int8-kl.cfg.in",
    "full-int8-removal-outlier": "full-int8-removal-outlier.cfg.in",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calibration_files(directory: Path) -> list[Path]:
    buffers = directory / "input"
    if not buffers.is_dir():
        raise RuntimeError("calibration input/ directory is missing")
    files = sorted(buffers.glob("*.bin"))
    if not 100 <= len(files) <= 500:
        raise RuntimeError(
            f"full INT8 requires 100..500 calibration buffers, found {len(files)}"
        )
    invalid = [path for path in files if path.stat().st_size != EXPECTED_CALIBRATION_BYTES]
    if invalid:
        raise RuntimeError(f"invalid calibration buffer size: {invalid[0]}")
    manifest = directory / "calibration-manifest.json"
    if not manifest.is_file():
        raise RuntimeError("calibration-manifest.json is missing")
    parsed = json.loads(manifest.read_text(encoding="utf-8"))
    if int(parsed.get("sampleCount", -1)) != len(files):
        raise RuntimeError("calibration manifest count does not match the binary buffers")
    return files


def render_quant_config(
    root: Path,
    mode: str,
    calibration: Path | None,
    target: Path,
) -> Path | None:
    config_dir = root / "models/realesrgan-x2plus/config"
    if mode == "weight-int8":
        return config_dir / "weight-int8.cfg"
    template_name = FULL_QUANT_MODES.get(mode)
    if template_name is None:
        return None
    if calibration is None:
        raise RuntimeError(f"--calibration is required for {mode}")
    calibration = calibration.resolve()
    files = calibration_files(calibration)
    template = (config_dir / template_name).read_text(encoding="utf-8")
    rendered = template.replace("{CALIBRATE_PATH}", str(calibration / "input")).replace(
        "{CALIBRATE_SIZE}", str(len(files))
    )
    target.write_text(rendered, encoding="utf-8")
    return target


def converter_environment(converter: Path) -> dict[str, str]:
    environment = dict(os.environ)
    if sys.platform != "linux" or len(converter.parents) < 4:
        return environment
    sdk_root = converter.parents[3]
    library_dirs = [
        sdk_root / "tools/converter/lib",
        sdk_root / "tools/converter/third_party/glog/lib",
        sdk_root / "runtime/lib",
        sdk_root / "runtime/third_party/glog",
        sdk_root / "runtime/third_party/libjpeg-turbo/lib",
        sdk_root / "runtime/third_party/securec",
    ]
    existing = environment.get("LD_LIBRARY_PATH", "")
    values = [str(path) for path in library_dirs if path.is_dir()]
    if existing:
        values.append(existing)
    environment["LD_LIBRARY_PATH"] = ":".join(values)
    return environment


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--converter", required=True, type=Path)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--input-name", default="input")
    parser.add_argument("--input-shape", default="1,3,180,180")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["fp16", "weight-int8", *FULL_QUANT_MODES.keys()],
    )
    parser.add_argument("--calibration", type=Path)
    args = parser.parse_args()

    converter = args.converter.resolve()
    onnx_path = args.onnx.resolve()
    output = args.output.resolve()
    input_shape = [int(value) for value in args.input_shape.split(",")]
    if len(input_shape) != 4 or any(value <= 0 for value in input_shape):
        raise RuntimeError("--input-shape must contain four positive dimensions")
    if not args.input_name:
        raise RuntimeError("--input-name must not be empty")
    if not converter.is_file():
        raise RuntimeError(f"converter does not exist: {converter}")
    if not onnx_path.is_file():
        raise RuntimeError(f"ONNX model does not exist: {onnx_path}")
    if output.suffix != ".ms":
        raise RuntimeError("--output must end in .ms")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    output_base = output.with_suffix("")

    with tempfile.TemporaryDirectory(prefix="nexte-model-convert-") as temp_name:
        quant_config = render_quant_config(
            root,
            args.mode,
            args.calibration,
            Path(temp_name) / "quant.cfg",
        )
        command = [
            str(converter),
            "--fmk=ONNX",
            f"--modelFile={onnx_path}",
            f"--outputFile={output_base}",
            "--saveType=MINDIR_LITE",
            f"--inputShape={args.input_name}:{','.join(str(value) for value in input_shape)}",
            "--inputDataFormat=NCHW",
            "--outputDataFormat=NCHW",
        ]
        if args.mode == "fp16":
            command.append("--fp16=on")
        if quant_config is not None:
            command.append(f"--configFile={quant_config.resolve()}")
        print("running:", " ".join(command))
        subprocess.run(command, check=True, env=converter_environment(converter))

    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"converter did not create the expected model: {output}")
    report = {
        "schemaVersion": 1,
        "status": "candidate",
        "mode": args.mode,
        "sourceOnnx": {
            "bytes": onnx_path.stat().st_size,
            "sha256": sha256(onnx_path),
        },
        "runtimeInput": {
            "name": args.input_name,
            "shape": input_shape,
            "dataType": "FLOAT32",
            "format": "NCHW",
        },
        "artifact": {
            "fileName": output.name,
            "bytes": output.stat().st_size,
            "sha256": sha256(output),
        },
        "runtimeValidationRequired": True,
        "qualityValidationRequired": True,
    }
    report_path = output.with_name(output.name + ".candidate.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"candidate report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
