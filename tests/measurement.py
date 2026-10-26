# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import argparse
import contextlib
from copy import deepcopy
import io
import json
import os
from pathlib import Path as File
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
import torch
from audit import values
from engine.field import Field, build, measure
from engine.limit import Budget
from engine.native import Native
from engine.path import Path
from run import common, run, validate
from trial import design, variants
from session import editor
from field import Native as Linear, Session

class MeasurementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def args(self, *values):
        return common(argparse.ArgumentParser()).parse_args(["--size", "16", "--prior", "off", *values])

    def testzero(self):
        session = Session()
        constructed = build(session, None, "zero")
        self.assertTrue(torch.equal(constructed.value, torch.zeros_like(session.anchor)))
        self.assertEqual(session.samples, 0)
        self.assertEqual(constructed.report["actual"], "zero")
        self.assertEqual(constructed.report["probes"], 0)

    def testnullmatrix(self):
        args = self.args("--clip", "clip")
        args.suite = "null"
        controls = variants("null", base=args)
        plan = design(args, controls)
        self.assertEqual(set(controls), {"native", "centered", "condition", "reference", "raw", "full"})
        self.assertEqual(plan["variants"]["full"]["changes"], {})
        self.assertEqual(controls["centered"]["eta"], 0.)
        self.assertEqual(controls["condition"]["eta"], 1.)
        for key in ("native", "centered", "condition", "reference"):
            self.assertEqual(controls[key]["field"], "zero")
        for key in ("native", "centered", "condition"):
            current = deepcopy(args)
            for name, value in controls[key].items():
                setattr(current, name, value)
            validate(current)

    def testnulloutputs(self):
        image = Image.new("RGB", (16, 16), (96, 128, 160))
        source, target = "source", "target long"
        session = editor().prepare(image, source, target)
        native = Native(session, path="embedding")
        zero = torch.zeros_like(session.anchor)
        targetzero, sourcezero = native.endpoint(zero), native.endpoint(zero, 1.)
        expected = {"native": session.decode(targetzero).clamp(0, 1), "centered": session.decode(session.anchor).clamp(0, 1), "condition": session.decode(session.anchor + targetzero - sourcezero).clamp(0, 1)}
        controls = variants("null")
        actual = {}
        for key in expected:
            args = self.args()
            for name, value in controls[key].items():
                setattr(args, name, value)
            actual[key], report, _ = run(editor(), None, image, source, target, args)
            self.assertTrue(torch.allclose(actual[key], expected[key], atol=1e-6))
            self.assertEqual(report["fieldnorm"], 0.)
            self.assertEqual(report["field"]["actual"], "zero")
        self.assertFalse(torch.equal(actual["centered"], session.origin))
        self.assertFalse(torch.equal(actual["condition"], actual["centered"]))

    def testzerocache(self):
        session = editor().prepare(Image.new("RGB", (16, 16)), "source", "target long")
        path = Path(Native(session), torch.zeros_like(session.anchor))
        self.assertEqual(path.cost((1., 1., 0.)), (1, 1))
        before = session.samples
        latent, _ = path.candidate((1., 1., 0.))
        self.assertEqual(session.samples - before, 1)
        self.assertTrue(torch.equal(latent, session.anchor))
        self.assertEqual(path.cost((1., 1., 0.)), (0, 0))
        self.assertIs(path.fields[(1., 1.)], path.references[0.])

    def testzeroprepaid(self):
        session = editor().prepare(Image.new("RGB", (16, 16)), "source", "target long")
        native = Native(session)
        neutral = native.endpoint(torch.zeros_like(session.anchor))
        path = Path(native, torch.zeros_like(session.anchor), neutral=neutral)
        session.limit = Budget(1)
        latent, _ = path.candidate((1., 1., 0.))
        self.assertEqual(session.limit.used, 0)
        self.assertTrue(torch.equal(latent, session.anchor))

    def testcachedrawbudget(self):
        image = Image.new("RGB", (16, 16))
        args = self.args("--method", "raw", "--maxsamples", "3", "--maxcalls", "2")

        def built(session, native, *args):
            zero = torch.zeros_like(session.anchor)
            return Field(zero, {"actual": "pullback"}, native.endpoint(zero))

        with patch("engine.field.build", side_effect=built):
            _, report, session = run(editor(), None, image, "source", "target long", args)
        self.assertEqual(session.samples, 3)
        self.assertEqual(report["selected"], "raw")
        self.assertNotIn("budgetstopped", report)

    def testcachecosts(self):
        for scale in (0., 1.):
            for mode in ("latent", "rgb", "mix", "random", "cfg"):
                session = editor().prepare(Image.new("RGB", (16, 16)), "source", "target long", scale=scale)
                path = Path(Native(session), torch.zeros_like(session.anchor), mode)
                for point in ((1., 1., 0.), (.5, 1.5, .5), (1., 1., 1.)):
                    expected = path.cost(point)
                    before = session.samples, session.calls
                    path.candidate(point)
                    self.assertEqual((session.samples - before[0], session.calls - before[1]), expected)

    def testresponse(self):
        request = torch.tensor([1., 2., 3., 4.]).reshape(1, 1, 2, 2)
        result = measure(request * 2, request, request)
        self.assertAlmostEqual(result["relativefit"], 1.)
        self.assertAlmostEqual(result["responsecosine"], 1.)
        self.assertAlmostEqual(result["relativegap"], 1.)
        self.assertEqual(result["actualfit"], result["modelgap"])
        json.dumps(result, allow_nan=False)

    def testzeroresponse(self):
        zero = torch.zeros(1, 1, 2, 2)
        result = measure(zero, torch.ones_like(zero))
        self.assertEqual(result["responsestatus"], "zero response")
        self.assertIsNone(result["responsecosine"])
        self.assertEqual(result["relativefit"], 1.)

    def testzerorequest(self):
        zero = torch.zeros(1, 1, 2, 2)
        result = measure(torch.ones_like(zero), zero)
        self.assertEqual(result["responsestatus"], "zero request")
        self.assertIsNone(result["responsecosine"])
        self.assertIsNone(result["relativefit"])
        json.dumps(result, allow_nan=False)

    def testinvalid(self):
        value = torch.ones(1, 1, 2, 2)
        for displacement, request, prediction in ((value, value[..., :1], None), (value, value, value[..., :1]), (value * float("nan"), value, None)):
            with self.assertRaises(ValueError):
                measure(displacement, request, prediction)

    def testoverflow(self):
        actual = torch.full((1, 1, 1, 1), 1e100, dtype=torch.float64)
        request = torch.full_like(actual, 1e-300)
        result = measure(actual, request)
        self.assertIsNone(result["relativefit"])
        self.assertEqual(result["actualfit"], 1e200)
        json.dumps(result, allow_nan=False)

    def testdirection(self):
        session = Session()
        native = Linear(session)
        field = build(session, native)
        before = session.samples
        field.observe(native.endpoint(field.value))
        self.assertEqual(session.samples - before, 1)
        self.assertLess(field.report["modelgap"], 1e-10)
        self.assertGreater(field.report["alignment"]["angle"], 0.)
        self.assertGreater(field.report["alignment"]["orthogonalratio"], 0.)
        self.assertEqual(field.report["rank"], 2)
        self.assertGreater(field.report["responsecosine"], .99)

    def testscalar(self):
        session = Session()
        field = build(session, Linear(session), geometry="scalar")
        self.assertAlmostEqual(field.report["alignment"]["angle"], 0., places=5)
        self.assertAlmostEqual(field.report["alignment"]["orthogonalratio"], 0., places=10)

    def testrankone(self):
        session = Session()
        field = build(session, Linear(session), delta=0.)
        self.assertEqual(field.report["rank"], 1)
        self.assertIsNone(field.report["alignment"]["singularratio"])
        self.assertAlmostEqual(field.report["alignment"]["orthogonalratio"], 0.)

    def testexternalreference(self):
        value = torch.ones(1, 1, 2, 2)
        field = Field(value, {})
        field.observe(value * 3, value, value * 2)
        self.assertEqual(field.report["actualfit"], 0.)
        self.assertEqual(field.report["responsecosine"], 1.)
        self.assertNotIn("modelgap", field.report)

    def testaudit(self):
        report = {"field": {"relativefit": .5, "responsecosine": .9, "alignment": {"angle": 10., "cosine": None, "scope": "input geometry"}}}
        result = values({"report": report})
        self.assertEqual(result["field:relativefit"], .5)
        self.assertEqual(result["field:inputangle"], 10.)
        self.assertNotIn("field:inputcosine", result)

    def testprobe(self):
        import probe
        origin = File.cwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder)
                Image.new("RGB", (16, 16), (96, 128, 160)).save("input.png")
                arguments = ["probe.py", "--image", "input.png", "--source", "source", "--target", "target long", "--size", "16", "--withfield", "--output", "probes"]
                with patch("sys.argv", arguments), patch("probe.load", return_value=(editor(), None)), contextlib.redirect_stdout(io.StringIO()):
                    probe.main()
                report = json.loads(File("probes/summary.json").read_text())
                self.assertIn("responsecosine", report["field"])
                self.assertIn("relativefit", report["field"])
                self.assertIn("controlsignature", report)
                self.assertIn("reproducibility", report)
            finally:
                os.chdir(origin)

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
