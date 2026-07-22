#!/usr/bin/env python3
"""Convert the pinned PP-OCRv5 mobile recognizer to deterministic ncnn FP32 assets."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess

import yaml


EXPECTED_CHARACTERS = 18383


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paddle2onnx", type=Path, required=True)
    parser.add_argument("--pnnx", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "ppocrv5_mobile_rec.onnx"
    param_path = output_dir / "ppocrv5_mobile_rec.ncnn.param"
    bin_path = output_dir / "ppocrv5_mobile_rec.ncnn.bin"
    dictionary_path = output_dir / "ppocrv5_mobile_rec.dict.txt"

    subprocess.run(
        [
            str(args.paddle2onnx),
            "--model_dir", str(model_dir),
            "--model_filename", "inference.json",
            "--params_filename", "inference.pdiparams",
            "--save_file", str(onnx_path),
            "--opset_version", "11",
            "--enable_onnx_checker", "True",
            "--optimize_tool", "onnxoptimizer",
        ],
        check=True,
    )
    subprocess.run(
        [
            str(args.pnnx),
            str(onnx_path),
            "inputshape=[1,3,48,320]",
            "fp16=0",
            f"ncnnparam={param_path}",
            f"ncnnbin={bin_path}",
            f"pnnxparam={output_dir / 'ppocrv5_mobile_rec.pnnx.param'}",
            f"pnnxbin={output_dir / 'ppocrv5_mobile_rec.pnnx.bin'}",
            f"pnnxpy={output_dir / 'ppocrv5_mobile_rec_pnnx.py'}",
            f"pnnxonnx={output_dir / 'ppocrv5_mobile_rec.pnnx.onnx'}",
            f"ncnnpy={output_dir / 'ppocrv5_mobile_rec_ncnn.py'}",
        ],
        check=True,
    )

    config = yaml.safe_load((model_dir / "inference.yml").read_text(encoding="utf-8"))
    characters = config["PostProcess"]["character_dict"]
    if len(characters) != EXPECTED_CHARACTERS:
        raise RuntimeError(f"unexpected PP-OCRv5 dictionary size: {len(characters)}")
    dictionary_path.write_text(
        "".join(f"{character}\n" for character in characters),
        encoding="utf-8",
    )

    for path in (onnx_path, param_path, bin_path, dictionary_path):
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"conversion did not produce {path.name}")
        print(f"generated {path.name} bytes={path.stat().st_size} sha256={sha256(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
