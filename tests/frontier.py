# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import argparse
import json
import unittest
from unittest.mock import patch
import numpy as np
import torch
from audit import summarize, values
from engine.frontier import analyze
from engine.leverage import analyze as leverage
from engine.limit import Exhausted
from engine.poll import Config, select
from engine.trust import Model
from run import common, validate
from trial import settings, variants
from trust import Fake, Expensive

class FrontierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def model(self, response=None, semantic=None, residual=None, active=(0, 2), anchor=None):
        response = np.eye(len(active)) if response is None else np.asarray(response, dtype=float)
        semantic = np.array([[0., 1.]]) if semantic is None else np.asarray(semantic, dtype=float)
        residual = np.zeros(len(response)) if residual is None else np.asarray(residual, dtype=float)
        anchor = np.array([1., 1., .5]) if anchor is None else np.asarray(anchor, dtype=float)
        points = [anchor]
        for axis in active:
            point = anchor.copy()
            point[axis] += .1 if anchor[axis] < (1. if axis == 2 else 2.) else -.1
            points.append(point)
        residuals = [residual + response @ (point[list(active)] - anchor[list(active)]) for point in points]
        scores = [np.ones(len(semantic)) + semantic @ (point[list(active)] - anchor[list(active)]) for point in points]
        return Model(anchor, points, residuals, scores, [0.] * len(points), active, regularization=1e-12)

    def testpositivebudget(self):
        model = self.model()
        neutral = leverage(model.response, model.semantic, model.residual, model.metric, model.active)
        self.assertAlmostEqual(neutral["neutralgain"]["component0"]["full"], 0.)
        report = analyze(model, [1.], [1.], .04, .5)
        interval = report["components"]["component0"]
        self.assertTrue(interval["verified"])
        self.assertTrue(interval["positiveidentified"])
        self.assertLessEqual(interval["incrementlower"], .2 + 1e-7)
        self.assertGreaterEqual(interval["incrementupper"], .2 - 1e-7)
        self.assertLess(interval["incrementupper"] - interval["incrementlower"], 2e-6)
        json.dumps(report, allow_nan=False)

    def testboundary(self):
        report = analyze(self.model(anchor=[1., 1., 1.]), [1.], [1.], .04, .5)
        interval = report["components"]["component0"]
        self.assertFalse(interval["positiveidentified"])
        self.assertLess(interval["incrementupper"], 2e-6)

    def testredundantbutefficient(self):
        model = self.model(response=[[1., 1.]], semantic=[[1., 1.]])
        report = analyze(model, [1.], [1.], 1., .1)
        interval = report["components"]["component0"]
        self.assertTrue(interval["positiveidentified"])
        self.assertGreater(interval["incrementlower"], .02)

    def testredundantclosedform(self):
        model = self.model(response=[[0., 0.]], semantic=[[1., 1.]])
        model.metric = np.eye(2)
        interval = analyze(model, [1.], [1.], 0., 1.)["components"]["component0"]
        expected = np.sqrt(3.) / 2. - .5
        self.assertTrue(interval["verified"])
        self.assertLessEqual(interval["incrementlower"], expected + 1e-6)
        self.assertGreaterEqual(interval["incrementupper"], expected - 1e-6)
        self.assertGreater(interval["incrementlower"], .36)
        diagnostic = leverage(model.response, model.semantic, model.residual, model.metric, model.active)
        self.assertAlmostEqual(diagnostic["jointnovelty"], 0.)

    def testnovelwithoutcapacity(self):
        model = self.model(response=np.eye(2), semantic=[[1., 0.]])
        diagnostic = leverage(model.response, model.semantic, model.residual, model.metric, model.active)
        self.assertGreater(diagnostic["jointnovelty"], 0.)
        interval = analyze(model, [1.], [1.], 0., 1.)["components"]["component0"]
        self.assertFalse(interval["positiveidentified"])
        if interval["verified"]:
            self.assertLess(interval["incrementupper"], 1e-5)

    def testjointfloors(self):
        model = self.model(response=[[0., 0.]], semantic=[[0., 1.], [0., -1.]])
        report = analyze(model, [1., 1.], [1., 1.], .04, .5)
        self.assertTrue(all(not item["positiveidentified"] for item in report["components"].values()))
        self.assertTrue(all(item["incrementupper"] < 2e-6 for item in report["components"].values()))

    def testnormalization(self):
        first = analyze(self.model(), [1.], [1.], .04, .5)["components"]["component0"]
        second = analyze(self.model(semantic=[[0., 100.]]), [1.], [100.], .04, .5)["components"]["component0"]
        self.assertAlmostEqual(first["incrementlower"], second["incrementlower"], places=5)

    def testfieldonly(self):
        report = analyze(self.model(active=(0, 1)), [1.], [1.], .04, .5)
        self.assertFalse(report["active"])
        self.assertFalse(report["components"])

    def testreferenceonly(self):
        report = analyze(self.model(response=[[1.]], semantic=[[1.]], active=(2,)), [1.], [1.], .04, .5)
        interval = report["components"]["component0"]
        self.assertEqual(interval["field"]["upper"], 0.)
        self.assertTrue(interval["positiveidentified"])

    def testunresolved(self):
        with patch("engine.frontier.solve", side_effect=RuntimeError("Unresolved synthetic solve.")):
            interval = analyze(self.model(), [1.], [1.], .04, .5)["components"]["component0"]
        self.assertFalse(interval["verified"])
        self.assertIsNone(interval["incrementlower"])
        self.assertFalse(interval["positiveidentified"])

    def testbudget(self):
        def checkpoint():
            raise Exhausted("Synthetic shared budget.")

        with self.assertRaises(Exhausted):
            analyze(self.model(), [1.], [1.], .04, .5, checkpoint=checkpoint)

    def testinvalid(self):
        for controls in ({"radius": 0.}, {"radius": 1e308}, {"radius": 1e-308}, {"damage": -1.}, {"iterations": True}, {"regularization": float("nan")}, {"scales": [0.]}, {"names": [["target"]]}, {"names": True}, {"names": "target"}):
            args = {"lower": [1.], "scales": [1.], "damage": .04, "radius": .5, **controls}
            with self.subTest(controls=controls), self.assertRaises(ValueError):
                analyze(self.model(), **args)

    def testoverflow(self):
        interval = analyze(self.model(semantic=[[0., 100.]]), [1.], [1e-320], .04, .5)["components"]["component0"]
        self.assertFalse(interval["verified"])
        self.assertIsNone(interval["incrementlower"])
        json.dumps(interval, allow_nan=False)

    def testinvalidprimal(self):
        def solved(hessian, linear, constant, constraints, bounds, iterations, checkpoint):
            return {"point": np.ones(len(linear)), "dualbound": -10., "iterations": 1, "tolerance": 1e-7}

        with patch("engine.frontier.solve", side_effect=solved):
            interval = analyze(self.model(), [1.], [1.], .04, .5)["components"]["component0"]
        self.assertFalse(interval["verified"])
        self.assertIsNone(interval["incrementlower"])

    def testzerodamage(self):
        interval = analyze(self.model(), [1.], [1.], 0., .5)["components"]["component0"]
        self.assertFalse(interval["positiveidentified"])

    def testintegration(self):
        path = Fake()
        result = select(path, lambda image: {"target": 1.}, torch.ones_like(path.session.origin[:, :1]), Expensive(), Config(frontier=True))
        initial = result.report["initialleverage"]
        self.assertIn("frontier", initial)
        self.assertTrue(initial["frontier"]["components"])
        json.dumps(result.report, allow_nan=False)
        row = {"key": "sample", "report": {"method": "path", **result.report}}
        self.assertIn("frontier:iterations", values(row))
        self.assertEqual(summarize([row])["diagnostics"]["frontier"]["recorded"], 1)
        self.assertLessEqual(result.report["damage"], result.report["rawdamage"])

    def testflags(self):
        args = common(argparse.ArgumentParser()).parse_args(["--clip", "clip", "--frontier"])
        self.assertTrue(validate(args).frontier)
        args.algorithm = "random"
        with self.assertRaises(ValueError):
            validate(args)
        with self.assertRaises(ValueError):
            Config(frontier=True, algorithm="poll").validate()
        self.assertTrue(variants("frontier")["capacity"]["frontier"])
        self.assertFalse(variants("frontier")["raw"]["frontier"])

    def testdiagnosticneutrality(self):
        first = select(Fake(), lambda image: {"target": 1.}, torch.ones(1, 1, 2, 2), Expensive(), Config(frontier=False))
        second = select(Fake(), lambda image: {"target": 1.}, torch.ones(1, 1, 2, 2), Expensive(), Config(frontier=True))
        self.assertEqual(first.report["point"], second.report["point"])
        self.assertEqual(first.report["evaluations"], second.report["evaluations"])
        self.assertTrue(torch.equal(first.image, second.image))

    def testcoverage(self):
        reports = [None, {"active": True, "fittrusted": True}, {"active": False, "fittrusted": True, "frontier": {"active": False, "components": {}}}, {"active": True, "fittrusted": False, "frontier": {"active": True, "components": {}}}, {"active": True, "fittrusted": True, "frontier": {"active": True, "components": {}}}, {"active": True, "fittrusted": True, "frontier": {"active": True, "components": {"first": {"verified": False}}}}, {"active": True, "fittrusted": True, "frontier": {"active": True, "components": {"first": {"verified": True}, "second": {"verified": False}}}}, {"active": True, "fittrusted": True, "frontier": {"active": True, "components": {"first": {"verified": True}}}}]
        rows = [{"key": str(index), "report": {"method": "path", "initialleverage": report}} for index, report in enumerate(reports)]
        coverage = summarize(rows)["diagnostics"]["frontier"]
        self.assertEqual(coverage["recorded"], 6)
        self.assertEqual(coverage["unrecorded"], 2)
        for key in ("inactive", "untrusted", "unresolved", "unverified", "partiallyverified", "fullyverified"):
            self.assertEqual(coverage[key], 1)
        self.assertEqual(coverage["fraction"], 1 / 8)

    def testcodebinding(self):
        args = common(argparse.ArgumentParser()).parse_args(["--clip", "clip"])
        first = settings(args)["codehash"]
        self.assertRegex(first, r"^[a-f0-9]{64}$")
        self.assertEqual(first, settings(args)["codehash"])
        from pathlib import Path
        original = Path.read_bytes
        with patch.object(Path, "read_bytes", lambda path: original(path) + b"# Synthetic code change.\n"):
            second = settings(args)["codehash"]
        self.assertNotEqual(first, second)

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
