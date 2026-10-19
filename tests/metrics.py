# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from engine.metrics import Protocol, pixels
from assess import grouped, main

class Calculator:
    def __init__(self):
        self.calls = []

    def calculate_mse(self, origin, edited, first, second):
        a = np.asarray(origin, dtype=np.float32) / 255
        b = np.asarray(edited, dtype=np.float32) / 255
        return float(np.mean((a * first - b * second) ** 2))

    def calculate_psnr(self, *args):
        return -10 * np.log10(self.calculate_mse(*args))

    def calculate_lpips(self, *args):
        self.calls.append(("lpips", args[2].copy()))
        return .1

    def calculate_ssim(self, *args):
        return .9

    def calculate_structure_distance(self, *args):
        self.calls.append(("structure", len(args)))
        return .02

    def calculate_clip_similarity(self, image, text, mask=None):
        self.calls.append(("clip", mask))
        return 25. if mask is None else 22.

class MetricTests(unittest.TestCase):
    def setUp(self):
        self.origin = Image.new("RGB", (4, 4), 0)
        self.edited = Image.new("RGB", (4, 4), (255, 255, 255))
        values = np.zeros((4, 4), dtype=np.uint8)
        values[:2] = 255
        self.mask = Image.fromarray(values)

    def testdenominator(self):
        raw, status = pixels(self.origin, self.edited, self.mask)
        self.assertEqual(raw["mse"], .5)
        self.assertAlmostEqual(raw["psnr"], 3.0102999566)
        self.assertEqual(status["psnr"], "finite")

    def testemptyandidentity(self):
        empty, status = pixels(self.origin, self.origin, Image.new("L", (4, 4), 255))
        self.assertIsNone(empty["mse"])
        self.assertEqual(status["psnr"], "empty protected region")
        identity, status = pixels(self.origin, self.origin, self.mask)
        self.assertEqual(identity["mse"], 0)
        self.assertIsNone(identity["psnr"])
        self.assertEqual(status["psnr"], "positive infinity")
        json.dumps({"raw": identity, "status": status}, allow_nan=False)

    def testmissingmask(self):
        raw, status = pixels(self.origin, self.edited, None)
        self.assertIsNone(raw["mse"])
        self.assertEqual(status["psnr"], "missing mask")

    def testbinary(self):
        with self.assertRaises(ValueError):
            pixels(self.origin, self.edited, Image.new("L", (4, 4), 128))
        with self.assertRaises(ValueError):
            pixels(self.origin, Image.new("RGB", (5, 4)), self.mask)

    def testprotocolrouting(self):
        calculator = Calculator()
        report = Protocol(calculator).evaluate(self.origin, self.edited, self.mask, "target")
        self.assertEqual(report["table"]["mse"], 500.)
        self.assertEqual(report["table"]["lpips"], 100.)
        self.assertEqual(report["table"]["ssim"], 90.)
        self.assertEqual(report["table"]["structure"], 20.)
        self.assertEqual(report["table"]["whole"], 25.)
        self.assertEqual(report["table"]["edited"], 22.)
        self.assertIn(("structure", 2), calculator.calls)
        masks = [mask for name, mask in calculator.calls if name == "clip" and mask is not None]
        self.assertEqual(masks[0].shape, (4, 4, 3))
        self.assertEqual(float(masks[0][:2].mean()), 1.)

    def testpixelonly(self):
        result = Protocol.load().evaluate(self.origin, self.edited, self.mask, "target")
        self.assertFalse(result["backend"]["complete"])
        self.assertIsNone(result["raw"]["lpips"])

    def testrejectbackendmismatch(self):
        calculator = Calculator()
        with patch.object(calculator, "calculate_mse", return_value=1.):
            with self.assertRaises(RuntimeError):
                Protocol(calculator).evaluate(self.origin, self.edited, self.mask, "target")

    def testimagegroupedmean(self):
        rows = [{"key": "a", "raw": {"mse": 1.}}, {"key": "a", "raw": {"mse": 3.}}, {"key": "b", "raw": {"mse": 10.}}]
        self.assertEqual(grouped(rows, "raw")["mse"], {"mean": 6., "count": 2})

    def testinfiniteaggregation(self):
        rows = [{"key": "a", "raw": {"psnr": None}, "status": {"psnr": "positive infinity"}}, {"key": "a", "raw": {"psnr": 30.}}, {"key": "b", "raw": {"psnr": 40.}}]
        result = grouped(rows, "raw")["psnr"]
        self.assertIsNone(result["mean"])
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["infinite"], 1)

    def testassessmentcli(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            root = Path(directory).resolve().relative_to(Path.cwd())
            (root / "images").mkdir()
            Image.new("RGB", (512, 512)).save(root / "images" / "source.png")
            Image.new("RGB", (512, 512), (32, 32, 32)).save(root / "result.png")
            mapping = {"A": {"image": "source.png", "source": "source", "target": "target", "mask": [0, 512]}}
            (root / "mapping.json").write_text(json.dumps(mapping), encoding="utf-8")
            (root / "summary.json").write_text(json.dumps({"rows": [{"key": "a", "seed": 1, "image": "result.png"}]}), encoding="utf-8")
            args = ["assess.py", "--input", str(root / "summary.json"), "--dataset", str(root), "--mapping", "mapping.json", "--images", "images", "--output", str(root / "assessment.json")]
            with patch("sys.argv", args), patch("builtins.print"):
                main()
            result = json.loads((root / "assessment.json").read_text(encoding="utf-8"))
            self.assertEqual(result["raw"]["mse"]["count"], 1)
            self.assertFalse(result["backend"]["complete"])
            self.assertIsNone(result["rows"][0]["raw"]["lpips"])

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
