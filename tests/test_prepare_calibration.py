from __future__ import annotations

from array import array
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_calibration", ROOT / "scripts/prepare_calibration.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PrepareCalibrationTest(unittest.TestCase):
    def test_tile_matches_clamped_rgb_nchw_contract(self) -> None:
        image = Image.new("RGB", (2, 2))
        image.putdata(
            [
                (255, 0, 0),
                (0, 255, 0),
                (0, 0, 255),
                (255, 255, 255),
            ]
        )
        tile = MODULE.build_runtime_tile(image, 0, 0)
        with tempfile.TemporaryDirectory() as temp_name:
            output = Path(temp_name) / "tile.bin"
            MODULE.write_nchw_float32(tile, output)
            values = array("f")
            values.frombytes(output.read_bytes())
        plane = MODULE.INPUT_SIZE * MODULE.INPUT_SIZE
        self.assertEqual(len(values) * 4, MODULE.EXPECTED_BYTES)
        self.assertEqual(values[0], 1.0)
        self.assertEqual(values[plane], 0.0)
        self.assertEqual(values[plane * 2], 0.0)
        origin = MODULE.PREPADDING * MODULE.INPUT_SIZE + MODULE.PREPADDING
        self.assertEqual(values[origin], 1.0)
        self.assertEqual(values[origin + 1], 0.0)
        self.assertEqual(values[plane + origin + 1], 1.0)
        last = plane - 1
        self.assertEqual(values[last], 1.0)
        self.assertEqual(values[plane + last], 1.0)
        self.assertEqual(values[plane * 2 + last], 1.0)
        tile.close()
        image.close()

    def test_last_partial_tile_replicates_page_edge(self) -> None:
        image = Image.new("RGB", (161, 161), (8, 16, 32))
        image.putpixel((160, 160), (64, 128, 255))
        tile = MODULE.build_runtime_tile(image, 160, 160)
        self.assertEqual(tile.getpixel((10, 10)), (64, 128, 255))
        self.assertEqual(tile.getpixel((179, 179)), (64, 128, 255))
        tile.close()
        image.close()

    def test_cli_keeps_converter_input_directory_binary_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            source_dir = temp / "source"
            source_dir.mkdir()
            page = source_dir / "reader-cache.img"
            image = Image.new("RGB", (32, 48), (12, 34, 56))
            image.save(page, format="WEBP")
            image.close()
            output = temp / "calibration"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/prepare_calibration.py"),
                    "--input",
                    str(source_dir),
                    "--output",
                    str(output),
                    "--samples",
                    "1",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            manifest = json.loads(
                (output / "calibration-manifest.json").read_text(encoding="utf-8")
            )
            buffers = list((output / "input").iterdir())
            self.assertEqual(manifest["sampleCount"], 1)
            self.assertEqual(len(buffers), 1)
            self.assertEqual(buffers[0].suffix, ".bin")
            self.assertEqual(buffers[0].stat().st_size, MODULE.EXPECTED_BYTES)


if __name__ == "__main__":
    unittest.main()
