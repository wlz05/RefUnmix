# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import argparse
import json
import math
import os
from pathlib import Path as File
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from engine.field import measure, solve
from engine.native import Native
from engine.path import Path
from engine.poll import Config, select
from engine.protocol import signature
from run import common, validate
from trial import frozen, settings, variants
from session import editor
from trust import Expensive, Fake

class BoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def args(self, *values):
        return common(argparse.ArgumentParser()).parse_args(["--clip", "clip", "--size", "16", *values])

    def path(self):
        session = editor().prepare(Image.new("RGB", (16, 16)), "source", "target long")
        return Path(Native(session), session.field(.9, .15))

    def testtinynorm(self):
        request = torch.full((1, 1, 1, 1), 1e-200, dtype=torch.float64)
        report = measure(torch.zeros_like(request), request)
        self.assertEqual(report["targetnorm"], 1e-200)
        self.assertEqual(report["relativefit"], 1.)
        self.assertIsNone(report["actualfit"])
        self.assertEqual(report["fitstatus"], "underflow")

    def testtinygap(self):
        request = torch.full((1, 1, 1, 1), 1e-200, dtype=torch.float64)
        report = measure(torch.zeros_like(request), request, request)
        self.assertIsNone(report["modelgap"])
        self.assertEqual(report["gapstatus"], "underflow")
        self.assertEqual(report["relativegap"], 1.)
        json.dumps(report, allow_nan=False)

    def testrelativeoverflow(self):
        request = torch.full((1, 1, 1, 1), 1e-300, dtype=torch.float64)
        report = measure(torch.full_like(request, 1e100), request)
        self.assertGreater(report["targetnorm"], 0)
        self.assertEqual(report["relativefitstatus"], "overflow")
        self.assertIsNone(report["relativefit"])

    def testmeanunderflow(self):
        request = torch.zeros(1, 1, 2, 1024, dtype=torch.float64)
        request[..., 0, 0] = 1e-161
        report = measure(torch.zeros_like(request), request)
        self.assertGreater(report["actualfit"], 0)
        self.assertIsNone(report["actualfitmean"])
        self.assertEqual(report["fitstatus"], "underflow")

    def testgapmeanunderflow(self):
        request = torch.zeros(1, 1, 2, 1024, dtype=torch.float64)
        request[..., 0, 0] = 1e-161
        report = measure(torch.zeros_like(request), request, request)
        self.assertGreater(report["modelgap"], 0)
        self.assertIsNone(report["modelgapmean"])
        self.assertEqual(report["gapstatus"], "underflow")

    def testlargesolve(self):
        coefficients, report = solve(np.eye(2) * 1e100, np.ones(2) * 1e100, np.ones(2) * .01, radius=.1)
        self.assertTrue(math.isfinite(report["multiplier"]))
        self.assertLessEqual(math.hypot(*(coefficients - .01)), report["bound"] * (1 + 1e-10))
        self.assertGreater(report["multiplier"], 0)
        json.dumps(report, allow_nan=False)

    def testsmallball(self):
        _, report = solve(np.eye(2), np.ones(2) * 1e-200, np.ones(2) * 1e-200, radius=.1)
        self.assertGreater(report["bound"], 0)

    def testrawprior(self):
        with self.assertRaisesRegex(ValueError, "Raw outputs"):
            validate(self.args("--method", "raw", "--prior", "rec"))

    def testrawheldout(self):
        with self.assertRaisesRegex(ValueError, "Raw outputs"):
            validate(self.args("--method", "raw", "--heldout"))

    def testrawcontrol(self):
        with self.assertRaisesRegex(ValueError, "Raw outputs"):
            validate(self.args("--method", "raw", "--control"))

    def testheldoutoff(self):
        with self.assertRaisesRegex(ValueError, "enabled prior"):
            validate(self.args("--heldout", "--prior", "off"))

    def testablationraw(self):
        args = self.args("--prior", "rec", "--heldout", "--control")
        for suite in ("alignment", "fields", "core", "diagnosis"):
            for values in variants(suite, base=args).values():
                if values["method"] == "raw":
                    self.assertEqual(values["prior"], "off")
                    self.assertFalse(values["heldout"])
                    self.assertFalse(values["control"])

    def testcostnonfinite(self):
        path = self.path()
        for point in ((1., float("nan"), 0.), (1., 1., float("inf")), (True, 1., 0.)):
            with self.assertRaises(ValueError):
                path.cost(point)

    def testcostarity(self):
        path = self.path()
        for point in (None, (1., 1.), (1., 1., 0., 0.), "abc"):
            with self.assertRaises(ValueError):
                path.cost(point)

    def testcanonicalnonfinite(self):
        with self.assertRaises(ValueError):
            self.path().canonical((1., 1., float("nan")))

    def testcandidatebounds(self):
        path = self.path()
        for point in ((-.01, 1., 0.), (1., 2.01, 0.), (1., 1., 1.01), None):
            with self.assertRaises(ValueError):
                path.candidate(point)

    def testnonfiniteproposal(self):
        from engine.trust import Model

        def unresolved(*args, **kwargs):
            model = Model(*args, **kwargs)
            model.rank = 0
            return model

        path = Fake()
        with patch("engine.trust.Model", side_effect=unresolved), patch("engine.trust.geometry", return_value=(1., float("nan"), 0.)):
            with self.assertRaises(ValueError):
                select(path, lambda image: {"target": 1.}, torch.ones_like(path.session.origin[:, :1]), Expensive(), Config())

    def testconfigcontent(self):
        origin = File.cwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder)
                File("control.json").write_text('{"slacks": {}}', encoding="utf-8")
                args = self.args("--constraints", "control.json")
                before = signature(args)
                File("control.json").write_text('{"slacks": {"target": 0.1}}', encoding="utf-8")
                after = signature(args)
                self.assertNotEqual(before["sha256"], after["sha256"])
                self.assertNotEqual(before["settings"]["configurationhashes"], after["settings"]["configurationhashes"])
            finally:
                os.chdir(origin)

    def testconfigmissing(self):
        with self.assertRaises(FileNotFoundError):
            signature(self.args("--constraints", "nonexistent.json"))

    def testmanifestmutation(self):
        args = self.args()
        args.suite, args.recipe, args.manifest = "core", "recipe.json", "split.json"
        payload = {"schema": 1, "split": "val", "suite": "core", "settings": settings(args), "variants": variants("core"), "manifestdigest": "before", "digests": {}}
        with patch("trial.locate"), patch("trial.read", return_value=payload), patch("trial.digest", return_value="after"):
            with self.assertRaisesRegex(ValueError, "split manifest"):
                frozen(args, [])

    def testmanifestmissing(self):
        args = self.args()
        args.suite, args.recipe = "core", "recipe.json"
        payload = {"schema": 1, "split": "val", "suite": "core", "settings": settings(args), "variants": variants("core"), "digests": {}}
        with patch("trial.locate"), patch("trial.read", return_value=payload):
            with self.assertRaisesRegex(ValueError, "split manifest"):
                frozen(args, [])

    def testprobemode(self):
        import probe
        arguments = ["probe.py", "--image", "input.png", "--source", "source", "--target", "target", "--method", "path"]
        with patch("sys.argv", arguments), patch("probe.load") as load:
            with self.assertRaisesRegex(ValueError, "Native probes"):
                probe.main()
        load.assert_not_called()

    def testprobediscrete(self):
        import probe
        origin = File.cwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder)
                Image.new("RGB", (16, 16)).save("input.png")
                arguments = ["probe.py", "--image", "input.png", "--source", "source", "--target", "target long", "--referencepath", "discrete", "--size", "16"]
                with patch("sys.argv", arguments), patch("probe.load", return_value=(editor(), None)):
                    with self.assertRaisesRegex(ValueError, "Discrete references"):
                        probe.main()
                self.assertFalse(File("probes").exists())
            finally:
                os.chdir(origin)

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
