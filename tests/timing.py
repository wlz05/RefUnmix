# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from __future__ import annotations
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
from data import Record, records
from engine.model import EditConfig, Editor
from engine.timing import Metric, collect, grid, interval, subset, summarize
from search import fingerprint, main, pending
from split import create, identities
from test import StubScheduler, StubText, StubTokenizer, StubUNet, StubVAE

class Judge:
    device = "cpu"

    def texts(self, values):
        return torch.tensor([[1.0, 0.0] for value in values])

    def images(self, images):
        levels = torch.tensor([np.asarray(image, dtype=np.float32).mean() / 255.0 for image in images])
        return torch.stack((levels, (1.0 - levels.square()).clamp_min(0.0).sqrt()), dim=1)

class FakeEditor:
    device = "cpu"
    size = 8
    calls = 0
    samples = 0

    def edit(self, image, source, target, config, seed, **kwargs):
        if kwargs:
            raise AssertionError("Evaluation annotations must not enter inference.")
        self.calls = 1 + int(config.cleanup)
        self.samples = (2 if config.delta == 0 else 4) + int(config.cleanup)
        pixels = np.asarray(image.resize((8, 8)), dtype=np.int16).copy()
        pixels[:4] = np.clip(pixels[:4] + round(20 * config.start), 0, 255)
        return SimpleNamespace(images=[Image.fromarray(pixels.astype(np.uint8))], report={"calls": self.calls, "samples": self.samples})

def row(name, key="a", seed=1, gain=0.0, drop=0.0, group=None):
    return {"config": name, "key": key, "seed": seed, "group": group or key, "category": "object", "semantic": 20.0 + gain, "global": 20.0 + gain, "local": 20.0 + gain, "mse": 0.01, "psnr": 20.0 - drop, "seconds": 0.1, "calls": 2, "samples": 5}

class TimingTest(unittest.TestCase):
    def testgrid(self):
        configs, skipped = grid([0.9, 0.75, 0.9], [0.15, 0.0, 0.15])
        self.assertEqual(len(configs), 4)
        self.assertFalse(skipped)
        self.assertEqual(configs["native"]["delta"], 0.15)
        self.assertTrue(any(value["start"] == 0.75 and value["delta"] == 0 for value in configs.values()))
        defaults, _ = grid([0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95], [0.0, 0.05, 0.10, 0.15])
        self.assertEqual(len(defaults), 28)

    def testinvalid(self):
        for starts, deltas in (([], [0]), ([float("nan")], [0]), ([1.1], [0]), ([0.0], [0.15]), ([0.7], [float("inf")]), ([0.7], [0.8])):
            with self.assertRaises(ValueError):
                grid(starts, deltas)

    def testsubset(self):
        items = [SimpleNamespace(key=str(index), category=str(index // 5)) for index in range(15)]
        selected = subset(items, 6)
        self.assertEqual(len(selected), 6)
        self.assertEqual({item.category for item in selected}, {"0", "1", "2"})
        self.assertEqual([item.key for item in selected], [item.key for item in subset(items, 6)])

    def testselection(self):
        configs = {"native": {}, "good": {}, "bad": {}}
        rows = [row("native"), row("good", gain=1.0, drop=0.5), row("bad", gain=5.0, drop=2.0)]
        report = summarize(rows, configs, budget=1.0, risk=0.0, repeats=20)
        self.assertEqual(report["selected"], "good")
        self.assertFalse(report["table"][2]["feasible"])
        self.assertIsNone(report["table"][1]["interval"])
        report = summarize(rows, configs, locked="bad", repeats=20)
        self.assertEqual(report["selected"], "bad")
        self.assertTrue(report["locked"])

    def testfallback(self):
        report = summarize([row("native"), row("worse", gain=-0.1)], {"native": {}, "worse": {}}, repeats=20)
        self.assertEqual(report["selected"], "native")
        report = summarize([row("native"), row("equal")], {"native": {}, "equal": {}}, repeats=20)
        self.assertEqual(report["selected"], "native")

    def testcoverage(self):
        configs = {"native": {}, "good": {}}
        for rows in ([row("native")], [row("native"), row("good"), row("good")], [row("native"), row("good", gain=float("nan"))]):
            with self.assertRaises(ValueError):
                summarize(rows, configs, repeats=20)

    def testrisk(self):
        rows = [row(name, key=key, gain=1.0 if name == "trial" else 0.0, drop=1.5 if name == "trial" and key == "a" else 0.0) for name in ("native", "trial") for key in ("a", "b")]
        report = summarize(rows, {"native": {}, "trial": {}}, budget=1.0, risk=0.1, repeats=20)
        self.assertEqual(report["selected"], "native")
        self.assertEqual(report["table"][1]["violation"], 0.5)

    def testgroups(self):
        self.assertIsNone(interval([1.0, 2.0], ["same", "same"], repeats=20))
        self.assertEqual(interval([2.0, 2.0], ["a", "b"], repeats=20), [2.0, 2.0])
        rows = [row(name, key=key, seed=seed, gain=1.0 if name == "trial" else 0.0) for name in ("native", "trial") for key in ("a", "b") for seed in (1, 2)]
        report = summarize(rows, {"native": {}, "trial": {}}, repeats=20)
        self.assertEqual(report["table"][1]["records"], 2)
        self.assertEqual(report["table"][1]["interval"], [1.0, 1.0])

    def testmetric(self):
        original = Image.new("RGB", (8, 8), (50, 50, 50))
        region = Image.fromarray(np.pad(np.full((4, 8), 255, dtype=np.uint8), ((0, 4), (0, 0))))
        metric = Metric(Judge(), original, "a target", region)
        unchanged = metric(original)
        self.assertEqual(unchanged["mse"], 0.0)
        self.assertEqual(unchanged["psnr"], 120.0)
        changed = np.asarray(original).copy()
        changed[:4] = 100
        scores = metric(Image.fromarray(changed))
        self.assertGreater(scores["semantic"], unchanged["semantic"])
        self.assertEqual(scores["mse"], 0.0)
        changed[4:] = 100
        self.assertGreater(metric(Image.fromarray(changed))["mse"], 0.0)
        full = Metric(Judge(), original, "a target", Image.new("L", (8, 8), 255))(original)
        self.assertIsNone(full["psnr"])
        with self.assertRaises(ValueError):
            Metric(Judge(), original, "a target", Image.new("L", (8, 8)))

    def testcollect(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.png"
            Image.new("RGB", (8, 8), (50, 50, 50)).save(path)
            item = Record("a", path, Path("source.png"), "source", "target", "change", mask=(0, 32), masksize=(8, 8))
            configs, _ = grid([0.75], [0.0])
            rows = list(collect(FakeEditor(), Judge(), [item], configs, [1, 2], {"a": "group"}))
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(value["mse"] == 0.0 for value in rows))
            completed = {(value["key"], value["seed"], value["config"]) for value in rows}
            self.assertEqual(list(collect(FakeEditor(), Judge(), [item], configs, [1, 2], {"a": "group"}, completed=completed)), [])

    def testshared(self):
        editor = Editor(StubUNet(), StubScheduler(), StubVAE(), StubTokenizer(), StubText(), "cpu", torch.float32, 64)
        anchor = torch.randn(1, 4, 8, 8)
        source, target = editor.prompt("a cat"), editor.prompt("a dog")
        noises = [torch.randn_like(anchor), torch.randn_like(anchor)]
        probe = editor.probe(anchor, source, target, noises, 0.75, 0.0)
        self.assertEqual(editor.samples, 4)
        self.assertTrue(torch.equal(probe.current, probe.previous))
        self.assertTrue(torch.allclose(probe.fields, probe.current))
        editor.samples = 0
        probe = editor.probe(anchor, source, target, noises, 0.75, 0.000001)
        self.assertEqual(editor.samples, 4)
        self.assertTrue(torch.equal(probe.current, probe.previous))

    def testrefinement(self):
        configs, _ = grid([0.20, 0.30], [0.0, 0.15], end=0.30)
        self.assertEqual(len(configs), 5)
        editor = Editor(StubUNet(), StubScheduler(), StubVAE(), StubTokenizer(), StubText(), "cpu", torch.float32, 64)
        for time in (0.20, 0.30):
            result = editor.edit(Image.new("RGB", (64, 64)), "a cat", "a dog", EditConfig(method="base", start=time, end=0.30))
            self.assertTrue(torch.isfinite(result.latent).all())
        for values in ({"end": -0.1}, {"end": 1.1}, {"start": 0.0}):
            with self.assertRaises(ValueError):
                EditConfig(**values).validate()

    def testresume(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = {"keys": ["a"], "seeds": [1], "configs": {"native": {}}}
            self.assertEqual(pending(root, plan, False), ([], set()))
            with self.assertRaises(FileExistsError):
                pending(root, plan, False)
            (root / "rows.jsonl").write_text(json.dumps(row("native")) + "\n", encoding="utf-8")
            rows, completed = pending(root, plan, True)
            self.assertEqual(len(rows), 1)
            self.assertIn(("a", 1, "native"), completed)
            with self.assertRaises(FileExistsError):
                pending(root, {**plan, "seeds": [2]}, True)

    def testworkflow(self):
        origin = Path.cwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder)
                Path("pie/images").mkdir(parents=True)
                mapping = {}
                for index in range(9):
                    Image.new("RGB", (8, 8), (index + 30, 20, 10)).save(f"pie/images/{index}.png")
                    mapping[str(index)] = {"image": f"{index}.png", "source": "source", "target": "target", "mask": [0, 32], "masksize": [8, 8], "category": str(index // 3)}
                Path("pie/map.json").write_text(json.dumps(mapping), encoding="utf-8")
                items = records("pie", "map.json", "images")
                Path("split.json").write_text(json.dumps(create(items, ratios=(0.34, 0.33, 0.33))), encoding="utf-8")
                signature = fingerprint(items, identities(items))
                self.assertEqual(len(signature), 64)
                shared = ["search.py", "--dataset", "pie", "--mapping", "map.json", "--images", "images", "--manifest", "split.json", "--seeds", "1"]
                fit = shared + ["--size", "8", "--starts", "0.75", "--deltas", "0", "--output", "fit"]
                with patch("engine.model.Editor.load", return_value=FakeEditor()), patch("engine.clip.ClipJudge.load", return_value=Judge()), contextlib.redirect_stdout(io.StringIO()):
                    with patch("sys.argv", fit + ["--dry"]):
                        main()
                    self.assertFalse(Path("fit").exists())
                    with patch("sys.argv", fit):
                        main()
                    with patch("sys.argv", fit + ["--resume"]):
                        main()
                    with patch("sys.argv", shared + ["--phase", "test", "--recipe", "fit/recipe.json", "--output", "test"]):
                        main()
                    with patch("sys.argv", shared + ["--phase", "test", "--recipe", "fit/recipe.json", "--starts", "0.8"]), self.assertRaises(ValueError):
                        main()
                selected = json.loads(Path("fit/recipe.json").read_text())["selected"]
                tested = json.loads(Path("test/summary.json").read_text())
                self.assertEqual(tested["selected"], selected)
                self.assertTrue(tested["locked"])
                self.assertLessEqual(len(tested["table"]), 2)
            finally:
                os.chdir(origin)

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu 2026
# Email: wlzchina05@gmail.com
