from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_runtime_release_assets",
    ROOT / "scripts/prepare_runtime_release_assets.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PrepareRuntimeReleaseAssetsTest(unittest.TestCase):
    def test_fetch_asset_renames_and_verifies_source(self) -> None:
        payload = b"ncnn-runtime-asset"
        digest = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            source_path = temp / "source.bin"
            source_path.write_bytes(payload)
            source = {
                "fileName": "source.bin",
                "url": source_path.as_uri(),
                "bytes": len(payload),
                "sha256": digest,
            }
            artifact = {
                "fileName": "published.bin",
                "bytes": len(payload),
                "sha256": digest,
            }
            output = MODULE.fetch_asset(source, artifact, temp / "output")
            self.assertEqual(output.name, "published.bin")
            self.assertEqual(output.read_bytes(), payload)

    def test_fetch_asset_rejects_hash_drift(self) -> None:
        payload = b"ncnn-runtime-asset"
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            source_path = temp / "source.bin"
            source_path.write_bytes(payload)
            source = {
                "fileName": "source.bin",
                "url": source_path.as_uri(),
                "bytes": len(payload),
                "sha256": "0" * 64,
            }
            artifact = {
                "fileName": "published.bin",
                "bytes": len(payload),
                "sha256": "0" * 64,
            }
            with self.assertRaisesRegex(RuntimeError, "source SHA-256 mismatch"):
                MODULE.fetch_asset(source, artifact, temp / "output")


if __name__ == "__main__":
    unittest.main()
