#!/usr/bin/env python3
"""Validate repository locks, candidates, and publication boundaries."""

from __future__ import annotations

import hashlib
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


def validate_vendored_source(root: Path, source: dict, label: str) -> None:
    path = root / source["vendoredPath"]
    require(path.is_file(), f"{label}: vendored source is missing")
    require(path.stat().st_size == int(source["bytes"]), f"{label}: vendored size drift")
    require(
        hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"],
        f"{label}: vendored SHA-256 drift",
    )


def validate_device_coverage(required_selectors: list[str], devices: list[dict]) -> None:
    matrix_selectors = [str(device["deviceSelector"]) for device in devices]
    require(
        len(matrix_selectors) == len(set(matrix_selectors)),
        "FP16 device matrix contains duplicate selectors",
    )
    missing = sorted(set(required_selectors) - set(matrix_selectors))
    require(
        not missing,
        f"FP16 device matrix is missing candidate selectors: {', '.join(missing)}",
    )


def validate_comic_models(root: Path) -> int:
    manifest = load(root / "manifests/comic-translation-models-v1.json")
    lock = load(root / "models/ysgyolo-1.2-os1/source.lock.json")
    candidate = load(
        root / "models/ysgyolo-1.2-os1/candidates/ncnn-fp32-v1.json"
    )
    evidence = load(
        root / "models/ysgyolo-1.2-os1/evidence/device-237-20260722.json"
    )
    ppocr_lock = load(root / "models/ppocrv5-mobile-rec/source.lock.json")
    ppocr_candidate = load(
        root / "models/ppocrv5-mobile-rec/candidates/ncnn-fp32-v1.json"
    )
    ppocr_evidence = load(
        root / "models/ppocrv5-mobile-rec/evidence/device-237-20260722.json"
    )
    aot_lock = load(root / "models/aot-manga-inpainting-256/source.lock.json")
    aot_candidate = load(
        root / "models/aot-manga-inpainting-256/candidates/ncnn-fp32-v1.json"
    )
    aot_evidence = load(
        root / "models/aot-manga-inpainting-256/evidence/device-237-20260722.json"
    )
    text_mask_lock = load(
        root / "models/comic-text-mask-ctd-1024/source.lock.json"
    )
    text_mask_candidate = load(
        root / "models/comic-text-mask-ctd-1024/candidates/ncnn-fp16-v1.json"
    )
    text_mask_evidence = load(
        root / "models/comic-text-mask-ctd-1024/evidence/device-237-20260722.json"
    )

    disposition = lock["licenseDisposition"]
    require(
        lock["upstream"]["authorDeclaredLicense"] == "MIT",
        "YSGYolo author-declared license evidence changed",
    )
    require(
        disposition["effectiveDistributionLicense"] == "AGPL-3.0-only",
        "YSGYolo must retain its conservative artifact license",
    )
    for label in ("checkpoint", "sourceOnnx"):
        source = lock[label]
        validate_artifact(source, f"YSGYolo {label}")
        require(str(source["url"]).startswith("https://"), f"{label}: HTTPS URL required")
    validate_vendored_source(root, lock["checkpoint"], "YSGYolo checkpoint")
    validate_artifact(lock["licenseText"], "YSGYolo license text")
    require(
        str(lock["licenseText"]["url"]).startswith("https://www.gnu.org/"),
        "YSGYolo license text must use the GNU source",
    )
    validate_vendored_source(root, lock["licenseText"], "YSGYolo license text")
    contract = lock["runtimeContract"]
    require(contract["inputShape"] == [1, 3, 640, 640], "YSGYolo input shape changed")
    require(contract["outputShape"] == [11, 8400], "YSGYolo output shape changed")
    require(contract["inputFormat"] == "NCHW", "YSGYolo input format changed")
    require(
        all(value is False for value in contract["ncnnOptions"].values()),
        "YSGYolo candidate must keep FP16, packing, and Vulkan disabled",
    )

    require(candidate["status"] == "device_validated", "YSGYolo device gate is missing")
    require(candidate["license"] == "AGPL-3.0-only", "YSGYolo candidate license drift")
    require(candidate["numericalParity"]["passed"] is True, "YSGYolo parity failed")
    require(evidence["result"]["passed"] is True, "YSGYolo device evidence failed")
    require(evidence["device"]["deviceSelector"] == "237", "YSGYolo device selector changed")
    require(evidence["result"]["regions"] > 0, "YSGYolo device result is empty")

    candidate_artifacts = candidate["artifacts"]
    for artifact in candidate_artifacts:
        validate_artifact(artifact, f"YSGYolo candidate {artifact['fileName']}")

    require(
        ppocr_lock["upstream"]["license"] == "Apache-2.0",
        "PP-OCRv5 upstream license drift",
    )
    require(
        ppocr_lock["upstream"]["revision"] ==
        "682f20538d8c086cb2128e5cfac775e6c4904e85",
        "PP-OCRv5 source revision drift",
    )
    for label in ("inferenceJson", "inferenceParams", "inferenceYml"):
        source = ppocr_lock[label]
        validate_artifact(source, f"PP-OCRv5 {label}")
        require(
            str(source["url"]).startswith(
                "https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec/resolve/"
            ),
            f"PP-OCRv5 {label}: immutable official source required",
        )
    validate_artifact(ppocr_lock["licenseText"], "PP-OCRv5 license text")
    validate_vendored_source(root, ppocr_lock["licenseText"], "PP-OCRv5 license text")
    ppocr_contract = ppocr_lock["runtimeContract"]
    require(
        ppocr_contract["inputShape"] == [1, 3, 48, 320],
        "PP-OCRv5 input shape changed",
    )
    require(
        ppocr_contract["outputShape"] == [40, 18385],
        "PP-OCRv5 output shape changed",
    )
    require(
        all(value is False for value in ppocr_contract["ncnnOptions"].values()),
        "PP-OCRv5 candidate must keep FP16, packing, and Vulkan disabled",
    )
    require(
        ppocr_candidate["status"] == "device_validated",
        "PP-OCRv5 device gate is missing",
    )
    require(
        ppocr_candidate["license"] == "Apache-2.0",
        "PP-OCRv5 candidate license drift",
    )
    require(
        ppocr_candidate["numericalParity"]["passed"] is True,
        "PP-OCRv5 parity failed",
    )
    require(
        ppocr_candidate["fixtureQuality"]["caseCount"] == 12,
        "PP-OCRv5 fixture coverage changed",
    )
    require(
        ppocr_evidence["result"]["passed"] is True,
        "PP-OCRv5 device evidence failed",
    )
    require(
        ppocr_evidence["device"]["deviceSelector"] == "237",
        "PP-OCRv5 device selector changed",
    )
    ppocr_candidate_artifacts = ppocr_candidate["artifacts"]
    for artifact in ppocr_candidate_artifacts:
        validate_artifact(artifact, f"PP-OCRv5 candidate {artifact['fileName']}")

    require(
        aot_lock["upstream"]["integrationRevision"] ==
        "95227a2bb0fd306cd4f0c104d57284026f991b3a",
        "AOT manga integration revision drift",
    )
    require(
        aot_lock["upstream"]["architectureRevision"] ==
        "2cd1afd8fdfabb101c678f6062d14bc7d302509e",
        "AOT-GAN architecture revision drift",
    )
    require(
        aot_lock["licenseDisposition"]["effectiveDistributionLicense"] ==
        "GPL-3.0-only",
        "AOT manga artifact must retain GPL-3.0-only distribution",
    )
    component_licenses = {
        item["license"]
        for item in aot_lock["licenseDisposition"]["componentLicenses"]
    }
    require(
        component_licenses == {"GPL-3.0-only", "Apache-2.0"},
        "AOT manga component license provenance drift",
    )
    for label in (
        "checkpoint",
        "architectureSource",
        "gplLicenseText",
        "apacheLicenseText",
    ):
        source = aot_lock[label]
        validate_artifact(source, f"AOT manga {label}")
        require(str(source["url"]).startswith("https://"), f"AOT manga {label}: HTTPS required")
    aot_contract = aot_lock["runtimeContract"]
    require(
        aot_contract["imageInputShape"] == [1, 3, 256, 256],
        "AOT manga image input shape changed",
    )
    require(
        aot_contract["maskInputShape"] == [1, 1, 256, 256],
        "AOT manga mask input shape changed",
    )
    require(
        aot_contract["outputShape"] == [1, 3, 256, 256],
        "AOT manga output shape changed",
    )
    require(
        all(value is False for value in aot_contract["ncnnOptions"].values()),
        "AOT manga candidate must keep FP16, packing, and Vulkan disabled",
    )
    require(
        aot_candidate["status"] == "device_validated",
        "AOT manga device gate is missing",
    )
    require(
        aot_candidate["license"] == "GPL-3.0-only",
        "AOT manga candidate license drift",
    )
    require(
        aot_candidate["pixelContract"]["passed"] is True,
        "AOT manga pixel contract failed",
    )
    require(
        aot_candidate["pixelContract"]["unmaskedRgbMaximumDelta"] == 0 and
        aot_candidate["pixelContract"]["alphaMaximumDelta"] == 0,
        "AOT manga source preservation drift",
    )
    require(
        aot_evidence["result"]["passed"] is True,
        "AOT manga device evidence failed",
    )
    require(
        aot_evidence["device"]["deviceSelector"] == "237",
        "AOT manga device selector changed",
    )
    require(
        int(aot_evidence["result"]["warm"]["inferenceMs"]) > 0,
        "AOT manga device inference time is missing",
    )
    aot_candidate_artifacts = aot_candidate["artifacts"]
    for artifact in aot_candidate_artifacts:
        validate_artifact(artifact, f"AOT manga candidate {artifact['fileName']}")

    require(
        text_mask_lock["upstream"]["detectorRevision"] ==
        "440b978563c71b758e31aaa315d100faba1efa2f",
        "CTD source revision drift",
    )
    require(
        text_mask_lock["licenseDisposition"]["effectiveDistributionLicense"] ==
        "GPL-3.0-only",
        "CTD mask artifact must retain GPL-3.0-only distribution",
    )
    for label in ("sourceOnnx", "licenseText"):
        source = text_mask_lock[label]
        validate_artifact(source, f"CTD mask {label}")
        require(str(source["url"]).startswith("https://"), f"CTD mask {label}: HTTPS required")
    text_mask_contract = text_mask_lock["runtimeContract"]
    require(
        text_mask_contract["inputShape"] == [1, 3, 1024, 1024],
        "CTD mask input shape changed",
    )
    require(
        text_mask_contract["outputShape"] == [1, 1, 1024, 1024],
        "CTD mask output shape changed",
    )
    require(
        text_mask_contract["maskThreshold"] == 0.3,
        "CTD mask threshold changed",
    )
    require(
        text_mask_contract["ncnnOptions"] == {
            "usePackingLayout": True,
            "useFp16Packed": True,
            "useFp16Storage": True,
            "useFp16Arithmetic": True,
            "useVulkanCompute": False,
        },
        "CTD mask ncnn runtime options changed",
    )
    require(
        text_mask_candidate["status"] == "device_validated",
        "CTD mask device gate is missing",
    )
    require(
        text_mask_candidate["license"] == "GPL-3.0-only",
        "CTD mask candidate license drift",
    )
    require(
        text_mask_candidate["numericalParity"]["passed"] is True,
        "CTD mask parity failed",
    )
    require(
        text_mask_evidence["result"]["passed"] is True,
        "CTD mask device evidence failed",
    )
    require(
        text_mask_evidence["device"]["deviceSelector"] == "237",
        "CTD mask device selector changed",
    )
    require(
        int(text_mask_evidence["result"]["warm"]["inferenceMs"]) > 0 and
        int(text_mask_evidence["result"]["maskedPixels"]) > 0,
        "CTD mask device output is missing",
    )
    text_mask_candidate_artifacts = text_mask_candidate["artifacts"]
    for artifact in text_mask_candidate_artifacts:
        validate_artifact(artifact, f"CTD mask candidate {artifact['fileName']}")
    entries = manifest.get("models", [])
    require(entries, "comic model manifest is empty")
    release_tag = manifest.get("releaseTag", "")
    require(release_tag.startswith("model-pack-v"), "comic release tag is invalid")
    ids: set[str] = set()
    for entry in entries:
        require(entry["id"] not in ids, f"duplicate comic model id: {entry['id']}")
        ids.add(entry["id"])
        require(entry["role"], f"{entry['id']}: model role is missing")
        for artifact in entry["artifacts"]:
            validate_artifact(artifact, f"{entry['id']} {artifact['fileName']}")
            urls = artifact.get("urls", [])
            if entry["status"] == "published":
                require(urls, f"{entry['id']}: published artifact has no URL")
                require(
                    all(url.startswith(
                        "https://github.com/erosTeam/NextE-Models/releases/download/"
                    ) for url in urls),
                    f"{entry['id']}: published URL must point to an immutable release",
                )
                expected_url = (
                    "https://github.com/erosTeam/NextE-Models/releases/download/"
                    f"{release_tag}/{artifact['fileName']}"
                )
                require(urls == [expected_url], f"{entry['id']}: release URL drift")
            else:
                require(entry["status"] == "candidate", f"unsupported status: {entry['status']}")
                require(not urls, f"{entry['id']}: candidate must not expose URLs")
        if entry["status"] == "published":
            corresponding_source = entry.get("correspondingSource", [])
            require(corresponding_source, f"{entry['id']}: corresponding source is missing")
            source_names: set[str] = set()
            for item in corresponding_source:
                source = item["source"]
                artifact = item["artifact"]
                validate_artifact(source, f"{entry['id']} corresponding source")
                validate_artifact(artifact, f"{entry['id']} source artifact")
                require(
                    str(source["url"]).startswith("https://"),
                    f"{entry['id']}: corresponding source URL must use HTTPS",
                )
                require(artifact["fileName"] not in source_names, "duplicate source artifact")
                source_names.add(artifact["fileName"])
                require(
                    source["bytes"] == artifact["bytes"] and
                    source["sha256"] == artifact["sha256"],
                    f"{entry['id']}: corresponding source metadata drift",
                )
    ysg = next(entry for entry in entries if entry["id"] == candidate["modelId"])
    require(ysg["license"] == candidate["license"], "YSGYolo license metadata drift")
    require(
        candidate["release"] is not None and
        candidate["release"]["tag"] == release_tag,
        "YSGYolo candidate release metadata drift",
    )
    require(
        [
            {key: value for key, value in artifact.items() if key != "urls"}
            for artifact in ysg["artifacts"]
        ] == candidate_artifacts,
        "YSGYolo artifact metadata drift",
    )
    ppocr = next(
        entry for entry in entries if entry["id"] == ppocr_candidate["modelId"]
    )
    require(
        ppocr["license"] == ppocr_candidate["license"],
        "PP-OCRv5 license metadata drift",
    )
    require(
        ppocr_candidate["release"] is not None and
        ppocr_candidate["release"]["tag"] == release_tag,
        "PP-OCRv5 candidate release metadata drift",
    )
    require(
        [
            {key: value for key, value in artifact.items() if key != "urls"}
            for artifact in ppocr["artifacts"]
        ] == ppocr_candidate_artifacts,
        "PP-OCRv5 artifact metadata drift",
    )
    aot = next(
        entry for entry in entries if entry["id"] == aot_candidate["modelId"]
    )
    require(
        aot["license"] == aot_candidate["license"],
        "AOT manga license metadata drift",
    )
    require(
        aot_candidate["release"] is not None and
        aot_candidate["release"]["tag"] == release_tag,
        "AOT manga candidate release metadata drift",
    )
    require(
        [
            {key: value for key, value in artifact.items() if key != "urls"}
            for artifact in aot["artifacts"]
        ] == aot_candidate_artifacts,
        "AOT manga artifact metadata drift",
    )
    text_mask = next(
        entry for entry in entries if entry["id"] == text_mask_candidate["modelId"]
    )
    require(
        text_mask["license"] == text_mask_candidate["license"],
        "CTD mask license metadata drift",
    )
    require(
        text_mask_candidate["release"] is not None and
        text_mask_candidate["release"]["tag"] == release_tag,
        "CTD mask candidate release metadata drift",
    )
    require(
        [
            {key: value for key, value in artifact.items() if key != "urls"}
            for artifact in text_mask["artifacts"]
        ] == text_mask_candidate_artifacts,
        "CTD mask artifact metadata drift",
    )
    return len(entries)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    lock = load(root / "models/realesrgan-x2plus/source.lock.json")
    candidate = load(root / "models/realesrgan-x2plus/candidates/fp16-baseline.json")
    matrix = load(
        root / "models/realesrgan-x2plus/experiments/fp16-device-matrix-20260719.json"
    )
    reader_matrix = load(
        root / "models/realesrgan-x2plus/experiments/fp16-reader-device-matrix-20260719.json"
    )
    quality = load(
        root / "models/realesrgan-x2plus/experiments/fp16-quality-validation-20260719.json"
    )
    experiment = load(root / "models/realesrgan-x2plus/experiments/weight-int8-device-103.json")
    manifest = load(root / "manifests/models-v1.json")
    comic_model_count = validate_comic_models(root)
    comic_manifest = load(root / "manifests/comic-translation-models-v1.json")
    runtime_manifest = load(root / "manifests/ncnn-runtime-assets-v1.json")
    require(
        comic_manifest["releaseTag"] == manifest["releaseTag"],
        "model and comic manifests use different release tags",
    )
    waifu_lock = load(root / "models/waifu2x-photo-noise0-x2/source.lock.json")
    waifu_art_lock = load(root / "models/waifu2x-art-noise0-x2/source.lock.json")
    waifu_art_candidate = load(
        root / "models/waifu2x-art-noise0-x2/candidates/fp16-baseline.json"
    )
    waifu_art_reader_matrix = load(
        root
        / "models/waifu2x-art-noise0-x2/experiments/fp16-reader-equivalence-device-matrix-20260719.json"
    )
    waifu_cunet_lock = load(root / "models/waifu2x-cunet-noise0-x2/source.lock.json")
    waifu_cunet_candidate = load(
        root / "models/waifu2x-cunet-noise0-x2/candidates/fp16-baseline.json"
    )
    waifu_cunet_policy = load(
        root / "models/waifu2x-cunet-noise0-x2/experiments/fp16-reader-device-policy-20260719.json"
    )
    waifu_candidate = load(
        root / "models/waifu2x-photo-noise0-x2/candidates/fp16-baseline.json"
    )
    waifu_matrix = load(
        root / "models/waifu2x-photo-noise0-x2/experiments/fp16-nnrt-device-matrix-20260719.json"
    )
    waifu_reader_matrix = load(
        root
        / "models/waifu2x-photo-noise0-x2/experiments/fp16-reader-equivalence-device-matrix-20260719.json"
    )
    waifu_quality = load(
        root
        / "models/waifu2x-photo-noise0-x2/experiments/fp16-quality-equivalence-20260719.json"
    )
    espcn_lock = load(root / "models/espcn-x2/source.lock.json")
    espcn_candidate = load(root / "models/espcn-x2/candidates/fp16-baseline.json")
    espcn_raw_matrix = load(
        root / "models/espcn-x2/experiments/fp16-nnrt-device-matrix-20260719.json"
    )
    espcn_reader_matrix = load(
        root
        / "models/espcn-x2/experiments/fp16-reader-equivalence-device-matrix-20260719.json"
    )
    realcugan_lock = load(
        root / "models/realcugan-se-2x-conservative/source.lock.json"
    )
    realcugan_candidate = load(
        root / "models/realcugan-se-2x-conservative/candidates/fp16-baseline.json"
    )
    realcugan_reader_matrix = load(
        root
        / "models/realcugan-se-2x-conservative/experiments/fp16-reader-equivalence-device-matrix-20260719.json"
    )

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

    require(waifu_lock["upstream"]["license"] == "MIT", "unexpected waifu2x license")
    for label in ("parameter", "weights", "converter"):
        entry = waifu_lock[label]
        require(int(entry["bytes"]) > 0, f"waifu2x {label}: bytes must be positive")
        require(bool(SHA256.fullmatch(entry["sha256"])), f"waifu2x {label}: invalid SHA-256")
        require(str(entry["url"]).startswith("https://"), f"waifu2x {label}: HTTPS URL required")
    waifu_contract = waifu_lock["runtimeContract"]
    require(waifu_contract["inputShape"] == [1, 3, 156, 156], "waifu2x input shape changed")
    require(waifu_contract["outputShape"] == [1, 3, 284, 284], "waifu2x output shape changed")
    require(waifu_art_lock["upstream"]["license"] == "MIT", "unexpected waifu2x art license")
    for label in ("parameter", "weights", "converter"):
        entry = waifu_art_lock[label]
        require(int(entry["bytes"]) > 0, f"waifu2x art {label}: bytes must be positive")
        require(bool(SHA256.fullmatch(entry["sha256"])), f"waifu2x art {label}: invalid SHA-256")
        require(str(entry["url"]).startswith("https://"), f"waifu2x art {label}: HTTPS URL required")
    waifu_art_contract = waifu_art_lock["runtimeContract"]
    require(waifu_art_contract["inputShape"] == [1, 3, 156, 156], "waifu2x art input shape changed")
    require(waifu_art_contract["outputShape"] == [1, 3, 284, 284], "waifu2x art output shape changed")
    require(waifu_art_contract["tileSize"] == 142, "waifu2x art tile size changed")
    require(waifu_art_contract["prepadding"] == 7, "waifu2x art prepadding changed")
    require(waifu_cunet_lock["upstream"]["license"] == "MIT", "unexpected waifu2x CUNet license")
    for label in ("parameter", "weights", "converter"):
        entry = waifu_cunet_lock[label]
        require(int(entry["bytes"]) > 0, f"waifu2x CUNet {label}: bytes must be positive")
        require(bool(SHA256.fullmatch(entry["sha256"])), f"waifu2x CUNet {label}: invalid SHA-256")
        require(str(entry["url"]).startswith("https://"), f"waifu2x CUNet {label}: HTTPS URL required")
    waifu_cunet_contract = waifu_cunet_lock["runtimeContract"]
    require(waifu_cunet_contract["inputShape"] == [1, 3, 164, 164], "waifu2x CUNet input shape changed")
    require(waifu_cunet_contract["outputShape"] == [1, 3, 256, 256], "waifu2x CUNet output shape changed")
    require(waifu_cunet_contract["tileSize"] == 128, "waifu2x CUNet tile size changed")
    require(waifu_cunet_contract["prepadding"] == 18, "waifu2x CUNet prepadding changed")
    validate_artifact(waifu_cunet_candidate["artifact"], "waifu2x CUNet FP16 candidate")
    validate_artifact(waifu_cunet_policy["artifact"], "waifu2x CUNet device policy")
    require(
        waifu_cunet_policy["artifact"] == waifu_cunet_candidate["artifact"],
        "waifu2x CUNet policy artifact drift",
    )
    validate_device_coverage(
        waifu_cunet_candidate["deviceEvidence"]["deviceSelectors"],
        waifu_cunet_policy["devices"],
    )
    for device in waifu_cunet_policy["devices"]:
        label = f"waifu2x CUNet device {device['deviceSelector']}"
        require(device["passed"] is True, f"{label}: device policy did not pass")
        require(
            str(device["selectedAccelerator"]).startswith("NPU_"),
            f"{label}: selected accelerator is not an enumerated NPU",
        )
        require(
            device["runtimeDecision"] in {"nnrt", "vulkan_fallback"},
            f"{label}: unsupported runtime decision",
        )
    cunet_nnrt_devices = [
        device for device in waifu_cunet_policy["devices"]
        if device["runtimeDecision"] == "nnrt"
    ]
    require(cunet_nnrt_devices, "waifu2x CUNet has no validated NNRT device")
    for device in cunet_nnrt_devices:
        label = f"waifu2x CUNet NNRT device {device['deviceSelector']}"
        require(float(device["meanAbsoluteError"]) < 0.1, f"{label}: mean output error too large")
        require(int(device["samplesOverSixtyFour"]) == 0, f"{label}: severe output error")
    validate_artifact(waifu_art_candidate["artifact"], "waifu2x art FP16 candidate")
    validate_artifact(waifu_art_reader_matrix["artifact"], "waifu2x art Reader matrix")
    require(
        waifu_art_reader_matrix["artifact"] == waifu_art_candidate["artifact"],
        "waifu2x art Reader evidence artifact drift",
    )
    validate_device_coverage(
        waifu_art_candidate["deviceEvidence"]["deviceSelectors"],
        waifu_art_reader_matrix["devices"],
    )
    for device in waifu_art_reader_matrix["devices"]:
        label = f"waifu2x art Reader device {device['deviceSelector']}"
        require(device["passed"] is True, f"{label}: Reader benchmark did not pass")
        require(
            str(device["selectedAccelerator"]).startswith("NPU_"),
            f"{label}: selected accelerator is not an enumerated NPU",
        )
        require(int(device["nnrtProcessElapsedMs"]) > 0, f"{label}: invalid NNRT process time")
        require(int(device["ncnnProcessElapsedMs"]) > 0, f"{label}: invalid ncnn process time")
        require(float(device["meanAbsoluteError"]) < 0.4, f"{label}: mean output error too large")
        require(int(device["maximumAbsoluteError"]) <= 2, f"{label}: maximum output error too large")
        require(int(device["samplesOverFour"]) == 0, f"{label}: severe output error")
        require(device["applicationBackend"] == "nnrt", f"{label}: AUTO did not select NNRT")
        require(int(device["applicationProcessElapsedMs"]) > 0, f"{label}: invalid application time")
        require(int(device["eventLoopDelayP95Ms"]) <= 2, f"{label}: event loop P95 too large")
    validate_artifact(waifu_candidate["artifact"], "waifu2x FP16 candidate")
    validate_artifact(waifu_matrix["artifact"], "waifu2x FP16 device matrix")
    validate_artifact(waifu_reader_matrix["artifact"], "waifu2x FP16 Reader matrix")
    validate_artifact(waifu_quality["artifact"], "waifu2x FP16 quality equivalence")
    require(
        waifu_matrix["status"] == "superseded",
        "incorrect waifu2x raw candidate must remain superseded",
    )
    require(
        waifu_reader_matrix["artifact"] == waifu_candidate["artifact"],
        "waifu2x FP16 Reader evidence artifact drift",
    )
    validate_device_coverage(
        waifu_candidate["deviceEvidence"]["deviceSelectors"],
        waifu_reader_matrix["devices"],
    )
    for device in waifu_reader_matrix["devices"]:
        label = f"waifu2x FP16 Reader device {device['deviceSelector']}"
        require(device["passed"] is True, f"{label}: Reader benchmark did not pass")
        require(
            str(device["selectedAccelerator"]).startswith("NPU_"),
            f"{label}: selected accelerator is not an enumerated NPU",
        )
        require(int(device["nnrtProcessElapsedMs"]) > 0, f"{label}: invalid NNRT process time")
        require(int(device["ncnnProcessElapsedMs"]) > 0, f"{label}: invalid ncnn process time")
        require(float(device["meanAbsoluteError"]) < 0.3, f"{label}: mean output error too large")
        require(int(device["maximumAbsoluteError"]) <= 4, f"{label}: maximum output error too large")
    require(
        waifu_quality["artifact"] == waifu_candidate["artifact"],
        "waifu2x quality artifact does not match the candidate",
    )
    require(
        waifu_quality["status"] == "quality_equivalence_validated",
        "waifu2x quality-equivalence validation has not passed",
    )

    require(
        espcn_lock["upstream"]["licenseStatus"] == "verified_for_redistribution",
        "ESPCN redistribution license has not been verified",
    )
    require(espcn_lock["upstream"]["distributionLicense"] == "MIT", "unexpected Hailo license")
    require(espcn_lock["upstream"]["modelLicense"] == "Apache-2.0", "unexpected ESPCN license")
    require(
        (root / "licenses/ESPCN-PyTorch-Apache-2.0.txt").is_file(),
        "ESPCN Apache-2.0 license copy is missing",
    )
    espcn_contract = espcn_lock["runtimeContract"]
    require(espcn_contract["inputShape"] == [1, 1, 180, 180], "ESPCN input shape changed")
    require(espcn_contract["outputShape"] == [1, 1, 360, 360], "ESPCN output shape changed")
    validate_artifact(espcn_candidate["artifact"], "ESPCN FP16 candidate")
    validate_artifact(espcn_raw_matrix["artifact"], "ESPCN raw NNRT matrix")
    validate_artifact(espcn_reader_matrix["artifact"], "ESPCN Reader matrix")
    require(
        espcn_reader_matrix["artifact"] == espcn_candidate["artifact"],
        "ESPCN Reader evidence artifact drift",
    )
    validate_device_coverage(
        espcn_candidate["deviceEvidence"]["deviceSelectors"],
        espcn_reader_matrix["devices"],
    )
    for device in espcn_reader_matrix["devices"]:
        label = f"ESPCN Reader device {device['deviceSelector']}"
        require(device["passed"] is True, f"{label}: Reader benchmark did not pass")
        require(
            str(device["selectedAccelerator"]).startswith("NPU_"),
            f"{label}: selected accelerator is not an enumerated NPU",
        )
        require(int(device["applicationProcessElapsedMs"]) > 0, f"{label}: invalid time")
        require(float(device["meanAbsoluteError"]) < 0.3, f"{label}: mean output error too large")
        require(int(device["maximumAbsoluteError"]) <= 2, f"{label}: maximum output error too large")
        require(int(device["eventLoopDelayP95Ms"]) <= 2, f"{label}: event loop P95 too large")

    require(
        realcugan_lock["upstream"]["licenseStatus"] == "verified_for_redistribution",
        "Real-CUGAN redistribution license has not been verified",
    )
    require(realcugan_lock["upstream"]["license"] == "MIT", "unexpected Real-CUGAN license")
    require(
        (root / "licenses/Real-CUGAN-MIT.txt").is_file(),
        "Real-CUGAN MIT license copy is missing",
    )
    for label in ("checkpoint", "source", "converter"):
        entry = realcugan_lock[label]
        require(int(entry["bytes"]) > 0, f"Real-CUGAN {label}: bytes must be positive")
        require(bool(SHA256.fullmatch(entry["sha256"])), f"Real-CUGAN {label}: invalid SHA-256")
        require(str(entry["url"]).startswith("https://"), f"Real-CUGAN {label}: HTTPS URL required")
    realcugan_contract = realcugan_lock["runtimeContract"]
    require(realcugan_contract["inputShape"] == [1, 3, 164, 164], "Real-CUGAN input shape changed")
    require(realcugan_contract["outputShape"] == [1, 3, 256, 256], "Real-CUGAN output shape changed")
    require(realcugan_contract["tileSize"] == 128, "Real-CUGAN tile size changed")
    require(realcugan_contract["seScope"] == "per-tile", "Real-CUGAN SE scope changed")
    validate_artifact(realcugan_candidate["artifact"], "Real-CUGAN FP16 candidate")
    validate_artifact(realcugan_reader_matrix["artifact"], "Real-CUGAN Reader matrix")
    require(
        realcugan_reader_matrix["artifact"] == realcugan_candidate["artifact"],
        "Real-CUGAN Reader evidence artifact drift",
    )
    validate_device_coverage(
        realcugan_candidate["deviceEvidence"]["deviceSelectors"],
        realcugan_reader_matrix["devices"],
    )
    for device in realcugan_reader_matrix["devices"]:
        label = f"Real-CUGAN Reader device {device['deviceSelector']}"
        require(device["passed"] is True, f"{label}: Reader benchmark did not pass")
        require(
            str(device["selectedAccelerator"]).startswith("NPU_"),
            f"{label}: selected accelerator is not an enumerated NPU",
        )
        require(int(device["nnrtProcessElapsedMs"]) > 0, f"{label}: invalid NNRT time")
        require(int(device["ncnnProcessElapsedMs"]) > 0, f"{label}: invalid ncnn time")
        require(float(device["meanAbsoluteError"]) < 0.2, f"{label}: mean output error too large")
        require(int(device["samplesOverSixtyFour"]) == 0, f"{label}: severe output error")
        require(device["applicationBackend"] == "nnrt", f"{label}: AUTO did not select NNRT")
        require(int(device["applicationProcessElapsedMs"]) > 0, f"{label}: invalid application time")
        require(int(device["eventLoopDelayP95Ms"]) <= 2, f"{label}: event loop P95 too large")

    validate_artifact(candidate["artifact"], "fp16 candidate")
    validate_artifact(matrix["artifact"], "fp16 device matrix")
    validate_artifact(reader_matrix["artifact"], "fp16 Reader device matrix")
    validate_artifact(quality["artifact"], "fp16 quality validation")
    validate_artifact(experiment["artifact"], "weight INT8 experiment")
    require(candidate["status"] in {"candidate", "published"}, "unsupported baseline status")
    require(
        matrix["artifact"] == candidate["artifact"],
        "FP16 device matrix artifact does not match the candidate",
    )
    matrix_devices = matrix["devices"]
    validate_device_coverage(
        candidate["deviceEvidence"]["deviceSelectors"],
        matrix_devices,
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
        reader_matrix["artifact"] == candidate["artifact"],
        "FP16 Reader device matrix artifact does not match the candidate",
    )
    validate_device_coverage(candidate["deviceEvidence"]["deviceSelectors"], reader_matrix["devices"])
    for device in reader_matrix["devices"]:
        require(device["passed"] is True, f"Reader device {device['deviceSelector']}: benchmark did not pass")
        require(int(device["processElapsedMs"]) > 0, f"Reader device {device['deviceSelector']}: invalid time")
    require(quality["artifact"] == candidate["artifact"], "quality artifact does not match the candidate")
    require(quality["status"] == "quality_validated", "quality validation has not passed")
    require(quality["evaluation"]["sourceTracked"] is False, "quality input must remain local-only")
    require(
        experiment["status"] == "rejected_for_performance",
        "weight INT8 decision must remain explicit",
    )

    entries = manifest.get("models", [])
    require(entries, "model manifest is empty")
    model_release_tag = manifest["releaseTag"]
    require(model_release_tag == "model-pack-v1.1.6", "unexpected model release tag")
    ids: set[str] = set()
    for entry in entries:
        require(entry["id"] not in ids, f"duplicate model id: {entry['id']}")
        ids.add(entry["id"])
        validate_artifact(entry["artifact"], entry["id"])
        status = entry["status"]
        urls = entry["artifact"].get("urls", [])
        if status == "published":
            require(urls, f"{entry['id']}: published model has no download URL")
            source_urls = entry["artifact"].get("sourceUrls", [])
            require(source_urls, f"{entry['id']}: published model has no pack source URL")
            require(
                all(url.startswith("https://github.com/erosTeam/NextE-Models/releases/download/") for url in urls),
                f"{entry['id']}: published URL must point to an immutable repository release",
            )
            expected_url = (
                "https://github.com/erosTeam/NextE-Models/releases/download/"
                f"{model_release_tag}/{entry['artifact']['fileName']}"
            )
            require(urls == [expected_url], f"{entry['id']}: unified release URL drift")
        else:
            require(status == "candidate", f"{entry['id']}: unsupported status {status}")
            require(not urls, f"{entry['id']}: candidate must not expose download URLs")

    fp16_entry = next(entry for entry in entries if entry["id"] == candidate["modelId"])
    require(
        {key: value for key, value in fp16_entry["artifact"].items() if key not in {"urls", "sourceUrls"}} == candidate["artifact"],
        "FP16 metadata drift",
    )
    if candidate["status"] == "published":
        require(candidate["deviceEvidence"]["endToEndReaderValidated"] is True, "Reader validation missing")
        require(candidate["deviceEvidence"]["qualityValidated"] is True, "quality validation missing")
        require(candidate["release"] is not None, "published candidate has no release")
        require(fp16_entry["status"] == "published", "manifest publication state drift")
        require(bool(fp16_entry["artifact"]["urls"]), "published artifact URL missing")
    else:
        require(candidate["release"] is None, "candidate must not claim a release")
        require(fp16_entry["status"] == "candidate", "manifest candidate state drift")

    waifu_entry = next(entry for entry in entries if entry["id"] == waifu_candidate["modelId"])
    require(
        {key: value for key, value in waifu_entry["artifact"].items() if key not in {"urls", "sourceUrls"}}
        == waifu_candidate["artifact"],
        "waifu2x metadata drift",
    )
    require(waifu_candidate["status"] == "published", "waifu2x publication state drift")
    require(waifu_entry["status"] == "published", "waifu2x manifest publication drift")
    require(waifu_candidate["deviceEvidence"]["endToEndReaderValidated"] is True, "waifu2x Reader validation missing")
    require(waifu_candidate["deviceEvidence"]["qualityValidated"] is True, "waifu2x quality equivalence missing")
    require(waifu_candidate["release"] is not None, "published waifu2x candidate has no release")
    require(bool(waifu_entry["artifact"]["urls"]), "published waifu2x artifact URL missing")

    waifu_art_entry = next(
        entry for entry in entries if entry["id"] == waifu_art_candidate["modelId"]
    )
    require(
        {key: value for key, value in waifu_art_entry["artifact"].items() if key not in {"urls", "sourceUrls"}}
        == waifu_art_candidate["artifact"],
        "waifu2x art metadata drift",
    )
    require(waifu_art_candidate["status"] == "published", "waifu2x art publication state drift")
    require(waifu_art_entry["status"] == "published", "waifu2x art manifest publication drift")
    require(
        waifu_art_candidate["deviceEvidence"]["endToEndReaderValidated"] is True,
        "waifu2x art Reader validation missing",
    )
    require(
        waifu_art_candidate["deviceEvidence"]["qualityValidated"] is True,
        "waifu2x art quality equivalence missing",
    )
    require(waifu_art_candidate["release"] is not None, "published waifu2x art candidate has no release")
    require(bool(waifu_art_entry["artifact"]["urls"]), "published waifu2x art artifact URL missing")

    waifu_cunet_entry = next(
        entry for entry in entries if entry["id"] == waifu_cunet_candidate["modelId"]
    )
    require(
        {key: value for key, value in waifu_cunet_entry["artifact"].items() if key not in {"urls", "sourceUrls"}}
        == waifu_cunet_candidate["artifact"],
        "waifu2x CUNet metadata drift",
    )
    require(waifu_cunet_candidate["status"] == "published", "waifu2x CUNet publication state drift")
    require(waifu_cunet_entry["status"] == "published", "waifu2x CUNet manifest publication drift")
    require(waifu_cunet_candidate["runtimeValidationRequired"] is True, "waifu2x CUNet runtime guard missing")
    require(waifu_cunet_candidate["release"] is not None, "published waifu2x CUNet candidate has no release")
    require(bool(waifu_cunet_entry["artifact"]["urls"]), "published waifu2x CUNet artifact URL missing")

    espcn_entry = next(entry for entry in entries if entry["id"] == espcn_candidate["modelId"])
    require(
        {key: value for key, value in espcn_entry["artifact"].items() if key not in {"urls", "sourceUrls"}}
        == espcn_candidate["artifact"],
        "ESPCN metadata drift",
    )
    require(espcn_candidate["status"] == "published", "ESPCN publication state drift")
    require(espcn_entry["status"] == "published", "ESPCN manifest publication drift")
    require(espcn_candidate["deviceEvidence"]["endToEndReaderValidated"] is True, "ESPCN Reader validation missing")
    require(espcn_candidate["deviceEvidence"]["qualityValidated"] is True, "ESPCN quality validation missing")
    require(espcn_candidate["release"] is not None, "published ESPCN candidate has no release")
    require(bool(espcn_entry["artifact"]["urls"]), "published ESPCN artifact URL missing")

    realcugan_entry = next(
        entry for entry in entries if entry["id"] == realcugan_candidate["modelId"]
    )
    require(
        {key: value for key, value in realcugan_entry["artifact"].items() if key not in {"urls", "sourceUrls"}}
        == realcugan_candidate["artifact"],
        "Real-CUGAN metadata drift",
    )
    require(realcugan_candidate["status"] == "published", "Real-CUGAN publication state drift")
    require(realcugan_entry["status"] == "published", "Real-CUGAN manifest publication drift")
    require(
        realcugan_candidate["deviceEvidence"]["endToEndReaderValidated"] is True,
        "Real-CUGAN Reader validation missing",
    )
    require(
        realcugan_candidate["deviceEvidence"]["qualityValidated"] is True,
        "Real-CUGAN quality validation missing",
    )
    require(realcugan_candidate["release"] is not None, "published Real-CUGAN candidate has no release")
    require(bool(realcugan_entry["artifact"]["urls"]), "published Real-CUGAN artifact URL missing")

    require(runtime_manifest["schemaVersion"] == 1, "runtime asset schema changed")
    require(runtime_manifest["status"] == "published", "runtime assets are not published")
    release_tag = runtime_manifest["releaseTag"]
    require(release_tag == model_release_tag, "model and runtime release tags differ")
    runtime_ids: set[str] = set()
    runtime_names: set[str] = set()
    for entry in runtime_manifest["assets"]:
        asset_id = entry["id"]
        require(asset_id not in runtime_ids, f"duplicate runtime asset id: {asset_id}")
        runtime_ids.add(asset_id)
        require(
            entry["license"] in {"MIT", "BSD-3-Clause"},
            f"{asset_id}: unexpected runtime asset license",
        )
        require(str(entry["upstream"]).startswith("https://github.com/"), f"{asset_id}: invalid upstream")
        source = entry["source"]
        artifact = entry["artifact"]
        validate_artifact(source, f"{asset_id} source")
        validate_artifact(artifact, f"{asset_id} artifact")
        require(str(source["url"]).startswith("https://"), f"{asset_id}: HTTPS source required")
        require(source["bytes"] == artifact["bytes"], f"{asset_id}: byte count drift")
        require(source["sha256"] == artifact["sha256"], f"{asset_id}: SHA-256 drift")
        artifact_name = artifact["fileName"]
        require(artifact_name not in runtime_names, f"duplicate runtime artifact: {artifact_name}")
        runtime_names.add(artifact_name)
        expected_url = (
            "https://github.com/erosTeam/NextE-Models/releases/download/"
            f"{release_tag}/{artifact_name}"
        )
        require(artifact["urls"] == [expected_url], f"{asset_id}: release URL drift")
    required_external_runtime_ids = {
        "realesrgan_animevideov3_x2_param",
        "realesrgan_animevideov3_x2_model",
        "realesrgan_x2plus_source_param",
        "realesrgan_x2plus_source_model",
    }
    require(
        required_external_runtime_ids <= runtime_ids,
        "unified model pack is missing selectable Real-ESRGAN runtime assets",
    )

    ignores = (root / ".gitignore").read_text(encoding="utf-8")
    for private_path in ("calibration-data/", "evaluation-data/", "private-data/"):
        require(private_path in ignores, f"privacy ignore missing: {private_path}")
    notices = (root / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    require("BSD 3-Clause" in notices, "third-party license notice is missing")
    for notice in (
        "waifu2x-ncnn-vulkan",
        "realcugan-ncnn-vulkan",
        "Hailo Model Zoo",
        "ESPCN-PyTorch",
        "Apache-2.0",
        "Real-CUGAN architecture and checkpoint",
    ):
        require(notice in notices, f"third-party notice is missing: {notice}")
    require("YSGYolo" in notices, "YSGYolo notice is missing")
    require("AGPL-3.0-only" in notices, "YSGYolo effective license is missing")
    require("PP-OCRv5" in notices, "PP-OCRv5 notice is missing")
    require("PaddleOCR" in notices, "PaddleOCR notice is missing")
    require("AOT Manga Inpainting" in notices, "AOT manga notice is missing")
    require("GPL-3.0-only" in notices, "AOT manga effective license is missing")
    require("AOT-GAN" in notices and "Apache-2.0" in notices,
            "AOT-GAN architecture provenance is missing")

    tracked = subprocess.run(
        ["git", "ls-files"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    vendored_release_sources = {
        "models/ysgyolo-1.2-os1/sources/ysgyolo_1.2_OS1.0.pt",
    }
    forbidden_suffixes = (".pth", ".pt", ".onnx", ".ms", ".bin")
    forbidden = [
        path for path in tracked
        if path.endswith(forbidden_suffixes) and path not in vendored_release_sources
    ]
    require(not forbidden, f"generated model data is tracked: {forbidden}")
    for path in tracked:
        full = root / path
        if not full.exists():
            continue
        require(
            path in vendored_release_sources or full.stat().st_size < 5 * 1024 * 1024,
            f"large file belongs in Release: {path}",
        )

    print(
        f"repository validation passed: models={len(entries)} "
        f"runtimeAssets={len(runtime_ids)} comicModels={comic_model_count} "
        f"trackedFiles={len(tracked)}"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyError, RuntimeError, ValueError) as error:
        print(f"repository validation failed: {error}", file=sys.stderr)
        sys.exit(1)
