# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from __future__ import annotations
import contextlib
import io
import json
import os
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
from data import Record, records
from engine.model import EditConfig, Editor
from engine.spectral import Sampler, controls, describe, grid, predictions, profiles, spectrum, validate
from engine.timing import collect, summarize
from search import main
from split import create
from test import StubScheduler, StubText, StubTokenizer, StubUNet, StubVAE
from timing import Judge, row

def editor():
    return Editor(StubUNet(), StubScheduler(), StubVAE(), StubTokenizer(), StubText(), "cpu", torch.float32, 64)

class SpectralTest(unittest.TestCase):
    def testgrid(self):
        configs = grid([0.6, 0.9])
        pairs = [(value["lowtime"], value["hightime"]) for value in configs.values() if value.get("operator") == "bands"]
        self.assertEqual(len(configs), 10)
        self.assertIn((0.75, 0.75), pairs)
        self.assertIn((0.6, 0.9), pairs)
        self.assertIn((0.9, 0.6), pairs)
        self.assertEqual(len(grid([0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])), 50)
        for times in ([], [0.0], [float("nan")], [float("inf")], [True]):
            with self.assertRaises(ValueError):
                grid(times)
        with self.assertRaises(ValueError):
            grid([0.5], pivot=0.0)

    def testconfig(self):
        value = next(value for value in grid([0.6]).values() if value.get("operator") == "bands")
        self.assertEqual(validate(value).method, "base")
        self.assertEqual(describe(value)["lowtime"], 0.6)
        self.assertNotIn("start", describe(value))
        for change in ({"lowtime": 0.0}, {"hightime": float("nan")}, {"edit": asdict(EditConfig(method="thread"))}, {"extra": 0}):
            with self.assertRaises(ValueError):
                validate({**value, **change})

    def testnative(self):
        model = editor()
        generator = np.random.default_rng(3)
        image = Image.fromarray(generator.integers(0, 256, (64, 64, 3), dtype=np.uint8))
        config = EditConfig(method="base", noise=2)
        expected = model.edit(image, "a cat", "a dog", config, seed=3)
        sampler = Sampler(model, image, "a cat", "a dog", 3, 2)
        result = sampler.render(asdict(config))
        self.assertTrue(torch.equal(result.latent, expected.latent))
        self.assertEqual(result.report["samples"], 9)
        self.assertEqual(result.report["calls"], 2)
        cached = sampler.render(asdict(config))
        self.assertTrue(torch.equal(cached.latent, expected.latent))
        self.assertEqual(cached.report["samples"], 1)

    def testsametime(self):
        model = editor()
        image = Image.new("RGB", (64, 64), (30, 80, 140))
        for cleanup in (False, True):
            expected = model.edit(image, "cat", "dog", EditConfig(method="base", start=0.75, delta=0.0, cleanup=cleanup), 4)
            sampler = Sampler(model, image, "cat", "dog", 4, 1)
            config = next(value for value in grid([0.75], cleanup=cleanup).values() if value.get("operator") == "bands")
            result = sampler.render(config)
            self.assertTrue(torch.allclose(result.latent, expected.latent, atol=1e-5))
            self.assertEqual(result.report["requiredsamples"], 2 + int(cleanup))

    def testassembly(self):
        sampler = Sampler(editor(), Image.new("RGB", (64, 64)), "source", "target", 1, 1)
        anchor = sampler.anchor[None]
        first, second = torch.randn_like(anchor), torch.randn_like(anchor)
        sampler.remember(0.6, anchor + first, anchor)
        sampler.remember(0.9, anchor + second, anchor)
        value = {"operator": "bands", "edit": asdict(EditConfig(method="base", cleanup=False)), "lowtime": 0.6, "hightime": 0.9}
        result = sampler.render(value)
        expected = sampler.router.low(first[0]) + sampler.router.high(second[0])
        self.assertTrue(torch.allclose(result.maps["field"], expected, atol=1e-6))
        self.assertTrue(torch.allclose(result.latent, sampler.anchor + expected, atol=1e-6))
        self.assertEqual(result.report["samples"], 0)

    def testdiagnostics(self):
        sampler = Sampler(editor(), Image.new("RGB", (64, 64)), "source", "target", 1, 1)
        anchor = sampler.anchor[None]
        error = torch.full_like(anchor, 0.5)
        change = torch.zeros_like(anchor)
        change[..., ::2] = 0.2
        change[..., 1::2] = -0.2
        sampler.remember(0.75, anchor + error + change, anchor + error)
        before = sampler.editor.calls
        values = sampler.diagnose(0.75, Image.new("L", (64, 64), 255))
        self.assertEqual(sampler.editor.calls, before)
        self.assertAlmostEqual(values["low"]["reconstruction"], 0.25, places=5)
        self.assertAlmostEqual(values["low"]["correction"], 0.0, places=5)
        self.assertGreater(values["high"]["correction"], 0.03)
        self.assertIsNone(values["high"]["protected"])
        self.assertIsNone(values["high"]["variance"])
        with self.assertRaises(ValueError):
            sampler.diagnose(0.6, Image.new("L", (64, 64), 255))

    def teststripes(self):
        original = Image.new("RGB", (8, 8), (60, 90, 160))
        pixels = np.asarray(original).copy()
        pixels[:4, ::2] = 255
        pixels[:4, 1::2] = 0
        region = Image.fromarray(np.pad(np.full((4, 8), 255, dtype=np.uint8), ((0, 4), (0, 0))))
        scores = spectrum(original, Image.fromarray(pixels), region)
        self.assertEqual(scores["high"]["original"], 0.0)
        self.assertGreater(scores["high"]["edit"], 0.0)
        self.assertEqual(scores["high"]["protected"], 0.0)
        self.assertEqual(scores["low"]["protected"], 0.0)
        self.assertEqual(scores["protectarea"], 0.5)

    def testboundary(self):
        original = Image.new("RGB", (7, 5))
        region = Image.new("L", original.size)
        region.putpixel((1, 1), 255)
        target = original.copy()
        target.putpixel((1, 1), (255, 255, 255))
        scores = spectrum(original, target, region)
        self.assertEqual(scores["editarea"], 0.0)
        self.assertIsNone(scores["high"]["edit"])
        self.assertEqual(scores["high"]["protected"], 0.0)
        full = spectrum(original, target, Image.new("L", original.size, 255))
        self.assertIsNone(full["low"]["protected"])

    def testbudget(self):
        base, good, bad = row("native"), row("good", gain=1.0), row("bad", gain=2.0)
        for value in (base, good, bad):
            value["frequency"] = {"space": "rgb", "low": {"protected": 0.01}, "high": {"protected": 0.001}}
        bad["frequency"]["high"]["protected"] = 0.01
        report = summarize([base, good, bad], {"native": {}, "good": {}, "bad": {}}, bandbudget=1.0, repeats=20)
        self.assertEqual(report["selected"], "good")
        self.assertFalse(report["table"][2]["feasible"])
        self.assertAlmostEqual(report["table"][2]["highdrop"], 10.0, places=6)
        with self.assertRaises(ValueError):
            summarize([row("native")], {"native": {}}, bandbudget=1.0)
        for value in (base, good, bad):
            for band in ("low", "high"):
                value["frequency"][band]["protected"] = None
        self.assertEqual(summarize([base, good, bad], {"native": {}, "good": {}, "bad": {}}, bandbudget=1.0, repeats=20)["selected"], "bad")

    def testinteraction(self):
        configs = grid([0.75, 0.9])
        gains = {(0.75, 0.75): 0.0, (0.9, 0.75): 2.0, (0.75, 0.9): 3.0, (0.9, 0.9): 1.0}
        rows = [row("native")]
        for name, config in configs.items():
            if name != "native":
                rows.append(row(name, gain=gains[(config["lowtime"], config["hightime"]) ]))
        report = summarize(rows, configs, repeats=20)
        values = profiles(report, 0.75)
        self.assertEqual(values["lowchoice"]["lowtime"], 0.9)
        self.assertEqual(values["highchoice"]["hightime"], 0.9)
        self.assertNotEqual(values["independent"]["id"], report["selected"])
        coupled = next(value for value in values["matrix"] if value["lowtime"] == value["hightime"] == 0.9)
        self.assertEqual(coupled["semanticinteraction"], -4.0)
        self.assertIn(values["independent"]["id"], controls(report, 0.75))
        self.assertIsNone(profiles(report, 0.75, selecting=False)["lowchoice"])

    def testcollect(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.png"
            Image.new("RGB", (64, 64), (40, 60, 80)).save(path)
            item = Record("a", path, Path("source.png"), "cat", "dog", "change", mask=(0, 2048), masksize=(64, 64))
            configs = grid([0.6, 0.75], cleanup=False)
            model = editor()
            rows = list(collect(model, Judge(), [item], configs, [1], {"a": "group"}, scan="bands"))
            self.assertEqual(len(rows), 5)
            self.assertEqual(sum(row["samples"] for row in rows), 6)
            self.assertEqual(sum(row["calls"] for row in rows), 2)
            self.assertEqual(len(predictions(rows)), 3)
            completed = {(value["key"], value["seed"], value["config"]) for value in rows[:2]}
            resumed = list(collect(model, Judge(), [item], configs, [1], {"a": "group"}, completed=completed, scan="bands"))
            for original, current in zip(rows[2:], resumed):
                self.assertEqual(original["probes"], current["probes"])
                self.assertEqual(original["semantic"], current["semantic"])
            self.assertEqual(predictions(rows[:2] + resumed), predictions(rows))

    def testworkflow(self):
        origin = Path.cwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder)
                Path("pie/images").mkdir(parents=True)
                mapping = {}
                for index in range(6):
                    Image.new("RGB", (64, 64), (index + 30, 20, 10)).save(f"pie/images/{index}.png")
                    mapping[str(index)] = {"image": f"{index}.png", "source": "cat", "target": "dog", "mask": [0, 2048], "masksize": [64, 64]}
                Path("pie/map.json").write_text(json.dumps(mapping), encoding="utf-8")
                Path("split.json").write_text(json.dumps(create(records("pie", "map.json", "images"), ratios=(0.34, 0.33, 0.33))), encoding="utf-8")
                shared = ["search.py", "--dataset", "pie", "--mapping", "map.json", "--images", "images", "--manifest", "split.json", "--seeds", "1"]
                fit = shared + ["--scan", "bands", "--size", "64", "--times", "0.6", "0.75", "--output", "fit"]
                with patch("engine.model.Editor.load", side_effect=lambda *args: editor()), patch("engine.clip.ClipJudge.load", return_value=Judge()), contextlib.redirect_stdout(io.StringIO()):
                    with patch("sys.argv", fit + ["--dry"]):
                        main()
                    self.assertFalse(Path("fit").exists())
                    with patch("sys.argv", fit):
                        main()
                    with patch("sys.argv", fit + ["--resume"]):
                        main()
                    with patch("sys.argv", shared + ["--phase", "test", "--recipe", "fit/recipe.json", "--output", "test"]):
                        main()
                    with patch("sys.argv", fit + ["--deltas", "0.15"]), self.assertRaises(ValueError):
                        main()
                recipe = json.loads(Path("fit/recipe.json").read_text())
                tested = json.loads(Path("test/summary.json").read_text())
                self.assertEqual(tested["selected"], recipe["selected"])
                self.assertTrue(tested["locked"])
                self.assertIsNone(tested["bands"]["lowchoice"])
                self.assertTrue(tested["predictions"])
                self.assertEqual(recipe["settings"]["bandbudget"], 1.0)
            finally:
                os.chdir(origin)

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
