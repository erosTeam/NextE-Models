from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_model_pack_release_assets",
    ROOT / "scripts/prepare_model_pack_release_assets.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(ROOT / "scripts"))
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PrepareModelPackReleaseAssetsTest(unittest.TestCase):
    def test_collects_published_model_and_runtime_asset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            model_payload = b"model"
            runtime_payload = b"runtime"
            model_source = temp / "model-source.ms"
            runtime_source = temp / "runtime-source.bin"
            model_source.write_bytes(model_payload)
            runtime_source.write_bytes(runtime_payload)
            model_digest = hashlib.sha256(model_payload).hexdigest()
            runtime_digest = hashlib.sha256(runtime_payload).hexdigest()
            models = {
                "releaseTag": "model-pack-v1.0.0",
                "models": [
                    {
                        "status": "published",
                        "artifact": {
                            "fileName": "model.ms",
                            "bytes": len(model_payload),
                            "sha256": model_digest,
                            "sourceUrls": [model_source.as_uri()],
                        },
                    },
                    {"status": "candidate", "artifact": {"fileName": "skip.ms"}},
                ],
            }
            runtime = {
                "releaseTag": "model-pack-v1.0.0",
                "assets": [
                    {
                        "source": {
                            "fileName": "runtime-source.bin",
                            "url": runtime_source.as_uri(),
                            "bytes": len(runtime_payload),
                            "sha256": runtime_digest,
                        },
                        "artifact": {
                            "fileName": "runtime.bin",
                            "bytes": len(runtime_payload),
                            "sha256": runtime_digest,
                        },
                    }
                ],
            }
            outputs = MODULE.prepare_pack(models, runtime, temp / "output")
            self.assertEqual([output.name for output in outputs], ["model.ms", "runtime.bin"])

    def test_rejects_release_tag_drift(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "different release tags"):
            MODULE.prepare_pack(
                {"releaseTag": "model-pack-v1.0.0", "models": []},
                {"releaseTag": "model-pack-v2.0.0", "assets": []},
                Path("unused"),
            )


if __name__ == "__main__":
    unittest.main()
