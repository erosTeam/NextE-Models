#!/usr/bin/env python3
"""Validate repository locks, candidates, and publication boundaries."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys


SHA256 = re.compile(r"^[0-9a-f]{64}$")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_artifact(artifact: dict, label: str) -> None:
    require(int(artifact["bytes"]) > 0, f"{label}: artifact bytes must be positive")
    require(bool(SHA256.fullmatch(artifact["sha256"])), f"{label}: invalid SHA-256")
    require(bool(artifact["fileName"]), f"{label}: artifact file name is missing")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    lock = load(root / "models/realesrgan-x2plus/source.lock.json")
    candidate = load(root / "models/realesrgan-x2plus/candidates/fp16-baseline.json")
    matrix = load(
        root / "models/realesrgan-x2plus/experiments/fp16-device-matrix-20260719.json"
    )
    experiment = load(root / "models/realesrgan-x2plus/experiments/weight-int8-device-103.json")
    manifest = load(root / "manifests/models-v1.json")

    require(lock["upstream"]["license"] == "BSD-3-Clause", "unexpected upstream license")
    for label in ("checkpoint", "converter"):
        entry = lock[label]
        require(int(entry["bytes"]) > 0, f"{label}: bytes must be positive")
        require(bool(SHA256.fullmatch(entry["sha256"])), f"{label}: invalid SHA-256")
        require(str(entry["url"]).startswith("https://"), f"{label}: HTTPS URL required")
    contract = lock["runtimeContract"]
    require(contract["inputShape"] == [1, 3, 180, 180], "locked input shape changed")
    require(contract["outputShape"] == [1, 3, 360, 360], "locked output shape changed")
    require(contract["inputFormat"] == "NCHW", "locked input format changed")
    require(contract["inputDataType"] == "FLOAT32", "locked input type changed")

    validate_artifact(candidate["artifact"], "fp16 candidate")
    validate_artifact(matrix["artifact"], "fp16 device matrix")
    validate_artifact(experiment["artifact"], "weight INT8 experiment")
    require(candidate["status"] == "candidate", "baseline must remain a candidate before release")
    require(candidate["release"] is None, "candidate must not claim a release")
    require(
        matrix["artifact"] == candidate["artifact"],
        "FP16 device matrix artifact does not match the candidate",
    )
    matrix_devices = matrix["devices"]
    require(
        {device["deviceSelector"] for device in matrix_devices} == {"103", "197", "237"},
        "FP16 device matrix must cover selectors 103, 197, and 237",
    )
    for device in matrix_devices:
        label = f"FP16 device {device['deviceSelector']}"
        require(device["passed"] is True, f"{label}: benchmark did not pass")
        require(
            str(device["selectedAccelerator"]).startswith("NPU_"),
            f"{label}: selected accelerator is not an enumerated NPU",
        )
        require(int(device["totalPredictionMs"]) > 0, f"{label}: invalid prediction time")
    require(
        experiment["status"] == "rejected_for_performance",
        "weight INT8 decision must remain explicit",
    )

    entries = manifest.get("models", [])
    require(entries, "model manifest is empty")
    ids: set[str] = set()
    for entry in entries:
        require(entry["id"] not in ids, f"duplicate model id: {entry['id']}")
        ids.add(entry["id"])
        validate_artifact(entry["artifact"], entry["id"])
        status = entry["status"]
        urls = entry["artifact"].get("urls", [])
        if status == "published":
            require(urls, f"{entry['id']}: published model has no download URL")
            require(
                all(url.startswith("https://github.com/erosTeam/NextE-Models/releases/download/") for url in urls),
                f"{entry['id']}: published URL must point to an immutable repository release",
            )
        else:
            require(status == "candidate", f"{entry['id']}: unsupported status {status}")
            require(not urls, f"{entry['id']}: candidate must not expose download URLs")

    fp16_entry = next(entry for entry in entries if entry["id"] == candidate["modelId"])
    require(fp16_entry["artifact"] == {**candidate["artifact"], "urls": []}, "FP16 metadata drift")

    ignores = (root / ".gitignore").read_text(encoding="utf-8")
    for private_path in ("calibration-data/", "evaluation-data/", "private-data/"):
        require(private_path in ignores, f"privacy ignore missing: {private_path}")
    notices = (root / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    require("BSD 3-Clause" in notices, "third-party license notice is missing")

    tracked = subprocess.run(
        ["git", "ls-files"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    forbidden_suffixes = (".pth", ".pt", ".onnx", ".ms", ".bin")
    forbidden = [path for path in tracked if path.endswith(forbidden_suffixes)]
    require(not forbidden, f"generated model data is tracked: {forbidden}")
    for path in tracked:
        full = root / path
        require(full.stat().st_size < 5 * 1024 * 1024, f"large file belongs in Release: {path}")

    print(f"repository validation passed: models={len(entries)} trackedFiles={len(tracked)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyError, RuntimeError, ValueError) as error:
        print(f"repository validation failed: {error}", file=sys.stderr)
        sys.exit(1)
