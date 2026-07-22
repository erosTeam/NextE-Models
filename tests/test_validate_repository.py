from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_repository", ROOT / "scripts/validate_repository.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ValidateDeviceCoverageTest(unittest.TestCase):
    def test_allows_additional_validated_devices(self) -> None:
        MODULE.validate_device_coverage(
            ["103", "197", "237"],
            [
                {"deviceSelector": "103"},
                {"deviceSelector": "197"},
                {"deviceSelector": "237"},
                {"deviceSelector": "200"},
            ],
        )

    def test_rejects_missing_candidate_device(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "missing candidate selectors: 200"):
            MODULE.validate_device_coverage(
                ["103", "200"],
                [{"deviceSelector": "103"}],
            )


class ValidateComicModelsTest(unittest.TestCase):
    def test_repository_comic_model_metadata_is_consistent(self) -> None:
        self.assertEqual(MODULE.validate_comic_models(ROOT), 4)


if __name__ == "__main__":
    unittest.main()
