# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
import torch
from data import Record, clean, locate
from evaluate import identifier, metrics, regions, summary
from run import common, parser, run, times, validate
from split import create, select
from study import choose, configurations, field, frozen

class WorkflowTests(unittest.TestCase):
    def args(self, **kwargs):
        result = common(argparse.ArgumentParser()).parse_args(["--method", "projection", "--field", "chord"])
        for key, value in kwargs.items():
            setattr(result, key, value)
        return result

    def testtimes(self):
        self.assertEqual(times([.76, .72, .76]), [.72, .76])
        for values in ([], [float("nan")], [float("inf")], [0], [1.1], [-.1]):
            with self.assertRaises(ValueError):
                times(values)

    def testrelativepaths(self):
        # Reject absolute and escaping paths.
        for path in ("D:/models", "D:models", "\\\\server\\share", "/tmp/model", "../model", "a/../../model"):
            with self.assertRaises(ValueError):
                clean(path, "model")
        self.assertEqual(clean("./Models/CLIP", "model"), Path("models/clip"))

    def testvalidatebeforeloading(self):
        with self.assertRaises(ValueError):
            validate(self.args(clip="./clip", budget=float("nan")))
        for options in ({"clip": "./clip", "step": -1}, {"clip": "./clip", "method": "thread", "proposal": "chord", "times": [.1]}, {"clip": "./clip", "method": "thread", "budget": float("nan")}):
            with self.assertRaises(ValueError):
                validate(self.args(**options))
        self.assertEqual(validate(self.args(clip="./clip", backtrack=0)).backtrack, 0)

    def testprojectiondefaults(self):
        from engine.projection import Config
        args = self.args(clip="./clip")
        self.assertEqual(args.method, "projection")
        self.assertEqual(validate(args), Config())
        self.assertEqual(args.lock, "cone")
        self.assertEqual(args.basis, "frequency")
        self.assertEqual(args.calibration, "secant")
        self.assertEqual(validate(self.args(clip="./clip", calibration="fixed", updates=0, factorial=True)).updates, 0)
        self.assertEqual(validate(self.args(clip="./clip", calibration="shrink")).calibration, "shrink")
        with self.assertRaises(ValueError):
            validate(self.args(method="base", factorial=True))
        self.assertEqual(validate(self.args(clip="./clip", lock="semantic", basis="channel")).basis, "channel")

    def testprojectionzeroradius(self):
        from engine.projection import Config
        config = validate(self.args(clip="./clip", radius=0.))
        self.assertIsInstance(config, Config)
        self.assertEqual(config.radius, 0.)
        self.assertEqual(config.step, .25)

    def testprojectionindependentstep(self):
        config = validate(self.args(clip="./clip", step=.25, radius=.1))
        self.assertEqual(config.step, .25)
        self.assertEqual(config.radius, .1)
        with self.assertRaises(ValueError):
            validate(self.args(clip="./clip", method="thread", step=.25, radius=.1))

    def testprojectiongeometry(self):
        for options in ({"size": 8}, {"size": 24}, {"size": 16, "grid": 2}, {"grid": 6}, {"grid": 2, "donors": [.1, .2, .3, .4, .5, .6, .7, .8]}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                validate(self.args(clip="./clip", **options))
        self.assertEqual(validate(self.args(clip="./clip", size=16, grid=1)).grid, 1)
        self.assertEqual(validate(self.args(clip="./clip", grid=5)).grid, 5)

    def testprojectionduplicatedonors(self):
        args = self.args(clip="./clip", grid=4, donors=[.76] * 12)
        self.assertEqual(validate(args).grid, 4)
        self.assertEqual(args.donors, [.76])

    def testprojectioninactivetimes(self):
        from engine.projection import Config
        args = self.args(clip="./clip", proposal="chord", times=[.1], donors=[.72])
        self.assertIsInstance(validate(args), Config)
        self.assertEqual(args.donors, [.72])

    def testprojectioninvaliddonors(self):
        for values in ([float("nan")], [float("inf")], [0.], [-.1], [1.1]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate(self.args(clip="./clip", donors=values))
        for value in (.1, .15):
            with self.subTest(time=value), self.assertRaises(ValueError):
                validate(self.args(clip="./clip", proposal="chord", donors=[value], delta=.15))

    def testlocatecase(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            root = Path(directory)
            (root / "MIXED").mkdir()
            (root / "MIXED" / "Image.PNG").write_bytes(b"image")
            self.assertTrue(locate(root, Path("mixed/image.png")).samefile(root / "MIXED" / "Image.PNG"))

    def testparser(self):
        args = parser().parse_args(["--image", "./input.png", "--source", "Blue Shirt", "--target", "Striped Shirt", "--method", "GLOBAL", "--field", "CHORD", "--bands", "HIGH", "--times", ".72", ".76"])
        self.assertEqual(args.method, "global")
        self.assertEqual(args.bands, "high")
        self.assertEqual(args.source, "Blue Shirt")
        self.assertEqual(args.times, [.72, .76])

    def testscanconfigurations(self):
        time = configurations(self.args(scan="time", times=[.72, .76], delta=.15))
        self.assertEqual(len(time), 5)
        bands = configurations(self.args(scan="bands", times=[.72, .76]))
        self.assertEqual(len(bands), 5)
        self.assertIn({"kind": "bands", "low": .72, "high": .76, "delta": 0.}, bands)
        self.assertIn({"kind": "bands", "low": .76, "high": .72, "delta": 0.}, bands)

    def testbandfield(self):
        class Session:
            def field(self, time, delta=0):
                return torch.tensor([[[[time, -time], [-time, time]]]]) + time * 2

        result = field(Session(), {"kind": "bands", "low": .72, "high": .76})
        expected = torch.tensor([[[[2.20, .68], [.68, 2.20]]]])
        self.assertTrue(torch.allclose(result, expected))
        with self.assertRaises(ValueError):
            field(Session(), {"kind": "single", "start": .3, "delta": .4})

    def testmaskedmetrics(self):
        origin = torch.zeros(1, 3, 4, 4)
        edited = origin.clone()
        edited[:, :, :2] = 1
        mask = torch.zeros(1, 1, 4, 4)
        mask[:, :, :2] = 1
        result = metrics(edited, origin, mask)
        self.assertEqual(result["protectedmse"], 0.)
        self.assertEqual(result["editedmse"], 1.)
        self.assertAlmostEqual(result["lowmse"] + result["highmse"], result["mse"])
        empty = metrics(edited, origin, torch.ones_like(mask))
        self.assertIsNone(empty["protectedmse"])
        self.assertIsNone(empty["protectedpsnr"])
        json.dumps(empty, allow_nan=False)

    def testmaskgeometry(self):
        from PIL import Image
        record = Record("a", Path("a.png"), Path("a.png"), "a", "b", "edit", mask=(0, 4), masksize=(4, 4))
        original, fitted = regions(record, Image.new("RGB", (32, 16)), 8)
        self.assertEqual(original.size, (32, 16))
        self.assertEqual(fitted.size, (8, 8))

    def testselection(self):
        base = {"kind": "base", "start": .9, "delta": .15}
        alternate = {"kind": "single", "start": .72, "delta": 0.}
        configs = [base, alternate]
        rows = [{"key": "a", "seed": 1, "configuration": config, "selectionscore": score, "selectionloss": loss} for config, score, loss in ((base, 1., .1), (alternate, 2., .3))]
        self.assertEqual(choose(rows, configs, .01)[0], base)
        self.assertEqual(choose(rows, configs, .3)[0], alternate)
        rows[1]["key"] = "b"
        with self.assertRaises(ValueError):
            choose(rows, configs, .3)

    def testfullmaskselection(self):
        base = {"kind": "base", "start": .9, "delta": .15}
        rows = [{"key": "a", "seed": 1, "configuration": base, "selectionscore": 1., "selectionloss": 0., "protectedvalid": False}]
        selected, table = choose(rows, [base], .001)
        self.assertEqual(selected, base)
        self.assertIsNone(table[0]["loss"])
        self.assertIsNone(table[0]["feasible"])
        self.assertEqual(table[0]["protectedcount"], 0)

    def testsummarymissing(self):
        result = summary([{"metrics": {"a": 2., "b": None}}, {"metrics": {"a": 4., "b": None}}])
        self.assertEqual(result["a"], {"mean": 3., "count": 2})
        self.assertEqual(result["b"], {"mean": None, "count": 0})
        self.assertEqual(identifier("A", 1), identifier("A", 1))
        self.assertNotEqual(identifier("A", 1), identifier("A", 2))

    def testgroupedmanifest(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            root = Path(directory)
            records = []
            for index in range(4):
                image = root / (str(index) + ".png")
                Image.new("RGB", (8, 8), (index, 20, 30)).save(image)
                records.append(Record(str(index), image, image, "source", "target", "edit", group="pair" if index < 2 else str(index)))
            manifest = create(records)
            path = root / "split.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            relative = path.resolve().relative_to(Path.cwd())
            divided = {name: {item.key for item in select(records, str(relative), name)} for name in ("train", "val", "test")}
            self.assertTrue(any({"0", "1"} <= keys for keys in divided.values()))
            self.assertFalse(divided["train"] & divided["test"])

    def testfrozenrejectstestselection(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            path = Path(directory) / "selection.json"
            path.write_text(json.dumps({"schema": 1, "split": "test", "selected": {}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                frozen(argparse.Namespace(recipe=str(path.resolve().relative_to(Path.cwd()))))

    def testfrozenreference(self):
        import hashlib
        with tempfile.TemporaryDirectory(dir=".") as directory:
            root = Path(directory).resolve().relative_to(Path.cwd().resolve())
            manifest, selection = root / "manifest.json", root / "selection.json"
            manifest.write_text("{}", encoding="utf-8")
            args = self.args(manifest=str(manifest), recipe=str(selection), reference="pixel", oracle=True)
            config = {"kind": "base", "start": .9, "delta": .15}
            payload = {"schema": 1, "split": "val", "selected": config, "configurations": [config], "renderer": "pixel", "manifest": str(manifest), "manifestdigest": hashlib.sha256(manifest.read_bytes()).hexdigest(), "settings": vars(args)}
            selection.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(frozen(args), [config, config])
            args.reference = "latent"
            with self.assertRaises(ValueError):
                frozen(args)

    def testpipelineandnoop(self):
        class Session:
            def __init__(self):
                self.anchor = torch.zeros(1, 1, 4, 4)
                self.origin = torch.full((1, 3, 4, 4), .5)
                self.mask = None
                self.calls = 0

            def field(self, time, delta=0):
                self.calls += 1
                return torch.full_like(self.anchor, .01 * time)

            def render(self, field):
                return (self.origin + field).clamp(0, 1)

            def stats(self):
                return {"calls": self.calls}

        class Editor:
            def prepare(self, *args, **kwargs):
                return Session()

        class Judge:
            def __init__(self):
                self.calls = 0

            def stats(self):
                return {"evaluations": self.calls}

            def prepare(self, *args):
                def score(image):
                    self.calls += 1
                    value = float(image.mean())
                    return {"score": value, "global": value, "local": value, "target": value}

                return score

        image, report, session = run(Editor(), None, None, "a", "b", self.args(method="base"))
        self.assertEqual(image.shape, (1, 3, 4, 4))
        self.assertEqual(report["stats"]["calls"], 1)
        judge = Judge()
        image, report, session = run(Editor(), judge, None, "a", "b", self.args(clip="./clip", method="global", times=[.72, .76]))
        self.assertEqual(report["mask"], "inferred")
        self.assertGreater(report["judgestats"]["evaluations"], 0)
        image, report, session = run(Editor(), judge, None, "a", "a", self.args(clip="./clip"))
        self.assertTrue(report["noop"])
        self.assertEqual(session.calls, 0)
        self.assertTrue(torch.equal(image, session.origin))

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
