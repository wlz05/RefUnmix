# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from __future__ import annotations
import contextlib
import csv
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from data import records
from engine.experiment import analyze, candidates, export, paired, point, roles
from engine.timing import summarize
from search import main
from split import create
from timing import FakeEditor, Judge, row

def samples(configs, gains=None, drops=None):
    gains, drops = gains or {}, drops or {}
    return [row(name, key=key, seed=seed, gain=gains.get(name, 0.0), drop=drops.get(name, 0.0)) for name in configs for key in ("a", "b") for seed in (1, 2)]

class ExperimentTest(unittest.TestCase):
    def testgrid(self):
        configs, skipped = candidates()
        self.assertEqual(len(configs), 24)
        self.assertFalse(skipped)
        table = [{"id": name, "config": config} for name, config in configs.items()]
        self.assertEqual(sum(value["delta"] == 0 for value in configs.values()), 14)
        for config in configs.values():
            self.assertEqual(config["method"], "base")
            self.assertEqual(config["end"], 0.30)
            self.assertEqual(config["scale"], 1.0)
            self.assertIsNotNone(point(table, config["start"]))
            self.assertIsNotNone(point(table, config["start"] - config["delta"]))
        self.assertEqual(configs["native"]["start"], 0.9)
        self.assertEqual(configs["native"]["delta"], 0.15)

    def testcustom(self):
        configs, skipped = candidates([0.2, 0.2, 0.7], [0.0, 0.3], end=0.4, cleanup=False, noise=2)
        self.assertEqual(skipped, [{"start": 0.2, "delta": 0.3}])
        for config in configs.values():
            self.assertFalse(config["cleanup"])
            self.assertEqual(config["noise"], 2)
            self.assertEqual(config["end"], 0.4)
        for starts, deltas in (([], [0]), ([0.2], []), ([float("nan")], [0]), ([0.1], [0.2])):
            with self.assertRaises(ValueError):
                candidates(starts, deltas)

    def testroles(self):
        configs, _ = candidates([0.75, 0.8], [0.0, 0.15])
        single = next(name for name, config in configs.items() if config["start"] == 0.8 and config["delta"] == 0)
        chord = next(name for name, config in configs.items() if config["start"] == 0.8 and config["delta"] > 0)
        rows = samples(configs, {single: 2.0, chord: 3.0})
        report = summarize(rows, configs, repeats=20)
        chosen = roles(report)
        self.assertEqual(chosen["single"], single)
        self.assertEqual(chosen["chord"], chord)
        self.assertEqual(chosen["current"], single)
        self.assertAlmostEqual(configs[chosen["previous"]]["start"], 0.65)
        self.assertEqual(configs[chosen["anchor"]]["start"], 0.75)
        report = summarize(samples(configs, {single: 2.0, chord: 3.0}, {chord: 2.0}), configs, repeats=20)
        self.assertEqual(roles(report)["chord"], "native")

    def testfallback(self):
        configs, _ = candidates([0.75], [0.0])
        report = summarize(samples(configs), configs, repeats=20)
        chosen = roles(report)
        self.assertEqual(chosen["single"], chosen["anchor"])
        self.assertEqual(chosen["chord"], "native")
        drops = {name: 2.0 for name, config in configs.items() if config["delta"] == 0}
        rows = samples(configs, drops=drops)
        report = summarize(rows, configs, repeats=20)
        study = analyze(report, rows, repeats=20)
        self.assertIsNone(study["roles"]["single"])
        self.assertNotIn("chordsingle", study["comparisons"])
        self.assertIn("anchornative", study["comparisons"])

    def testpaired(self):
        rows = [row(name, key=key, seed=seed, gain=(2 * seed - 1) if name == "trial" else 0.0) for name in ("native", "trial") for key in ("a", "b") for seed in (1, 2)]
        result = paired(rows, "trial", "native", repeats=20)
        self.assertEqual(result["metrics"]["semantic"], {"mean": 2.0, "interval": [2.0, 2.0], "records": 2})
        for value in rows:
            value["group"] = "same"
            value["psnr"] = value["mse"] = None
        result = paired(rows, "trial", "native", repeats=20)
        self.assertIsNone(result["metrics"]["semantic"]["interval"])
        self.assertIsNone(result["metrics"]["psnr"]["mean"])
        self.assertEqual(result["metrics"]["psnr"]["records"], 0)

    def testcoverage(self):
        for rows in ([row("native")], [row("native"), row("trial"), row("trial")], [row("native"), row("trial", group="other")], [row("native"), row("trial", gain=float("inf"))]):
            with self.assertRaises(ValueError):
                paired(rows, "trial", "native", repeats=20)
        rows = [row("native"), row("trial")]
        rows[0]["psnr"] = None
        with self.assertRaises(ValueError):
            paired(rows, "trial", "native", repeats=20)

    def testfrozen(self):
        configs, _ = candidates([0.75, 0.8], [0.0, 0.15])
        single = next(name for name, config in configs.items() if config["start"] == 0.8 and config["delta"] == 0)
        chord = next(name for name, config in configs.items() if config["start"] == 0.8 and config["delta"] > 0)
        rows = samples(configs, {single: 2.0, chord: 3.0})
        study = analyze(summarize(rows, configs, repeats=20), rows, repeats=20)
        selected = study["roles"]
        controls = {name: configs[name] for name in dict.fromkeys(selected.values()) if name is not None}
        rows = samples(controls, {single: -2.0, chord: -1.0}, {single: 2.0, chord: 2.0})
        report = summarize(rows, controls, locked=chord, repeats=20)
        tested = analyze(report, rows, frozen=selected, repeats=20)
        self.assertEqual(tested["roles"], selected)
        self.assertTrue(tested["locked"])
        self.assertEqual(tested["comparisons"]["chordsingle"]["metrics"]["semantic"]["mean"], 1.0)
        self.assertEqual(tested["comparisons"]["chordnative"]["metrics"]["semantic"]["mean"], -1.0)
        for frozen in ({}, {**selected, "single": "missing"}):
            with self.assertRaises(ValueError):
                analyze(report, rows, frozen=frozen, repeats=20)

    def testexport(self):
        configs, _ = candidates([0.75], [0])
        rows = samples(configs)
        study = analyze(summarize(rows, configs, repeats=20), rows, repeats=20)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "curves.csv"
            export(path, study)
            export(path, study)
            values = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))
            self.assertEqual(len(values), len(configs))
            self.assertEqual({value["family"] for value in values}, {"single", "chord"})
            self.assertEqual(float(values[0]["lower"]), 0.0)
            study["curves"]["single"][0]["semantic"] += 1.0
            with self.assertRaises(FileExistsError):
                export(path, study)

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
                shared = ["search.py", "--dataset", "PIE", "--mapping", "MAP.JSON", "--images", "IMAGES", "--manifest", "split.json", "--seeds", "1"]
                fit = shared + ["--study", "--size", "8", "--starts", "0.60", "0.75", "--deltas", "0", "0.15", "--output", "fit"]
                with patch("engine.model.Editor.load", return_value=FakeEditor()) as model, patch("engine.clip.ClipJudge.load", return_value=Judge()), contextlib.redirect_stdout(io.StringIO()):
                    with patch("sys.argv", fit + ["--dry"]):
                        main()
                    model.assert_not_called()
                    self.assertFalse(Path("fit").exists())
                    for extra in (["--scan", "bands"], ["--method", "thread"]):
                        with patch("sys.argv", fit + extra), self.assertRaises(ValueError):
                            main()
                    with patch("sys.argv", fit):
                        main()
                    with patch("sys.argv", fit + ["--resume"]):
                        main()
                    self.assertEqual(model.call_count, 1)
                    with patch("sys.argv", shared + ["--phase", "test", "--recipe", "fit/recipe.json", "--output", "test"]):
                        main()
                    with patch("sys.argv", shared + ["--phase", "test", "--recipe", "fit/recipe.json", "--deltas", "0.1"]), self.assertRaises(ValueError):
                        main()
                    with patch("sys.argv", shared + ["--phase", "test", "--recipe", "fit/recipe.json", "--scan", "bands"]), self.assertRaises(ValueError):
                        main()
                recipe = json.loads(Path("fit/recipe.json").read_text())
                tested = json.loads(Path("test/summary.json").read_text())
                self.assertTrue(recipe["study"])
                self.assertEqual(tested["study"]["roles"], recipe["roles"])
                self.assertEqual(tested["selected"], recipe["selected"])
                self.assertTrue(tested["study"]["locked"])
                self.assertGreaterEqual(len(tested["table"]), 3)
                self.assertTrue(Path("fit/curves.csv").exists())
                self.assertTrue(Path("test/curves.csv").exists())
                self.assertFalse(set(recipe["keys"]) & set(json.loads(Path("test/plan.json").read_text())["keys"]))
            finally:
                os.chdir(origin)

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
