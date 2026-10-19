# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import argparse
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from engine.poll import Config, select
from engine.radius import bound, initialize, profile
from run import common, validate
from trial import settings, variants
from trust import Expensive, Fake

class RadiusTests(unittest.TestCase):
    def profile(self):
        return {"gradient": 2.015, "curvature": 1.130, "third": 6.053, "ratio": .25}

    def testmanuscriptnumbers(self):
        result = profile(self.profile())
        self.assertEqual(result["branch"], "gradient")
        self.assertAlmostEqual(result["uncapped"], (6 * .25 * 2.015 / 6.053) ** .5, places=10)
        self.assertAlmostEqual(result["threshold"], 2 * 6.053 * 2.015 / (3 * 1.130 ** 2), places=10)

    def testcurvaturebranch(self):
        result = bound(1., 2., 3., 1.)
        self.assertEqual(result["branch"], "curvature")
        self.assertEqual(result["radius"], 1.)
        self.assertAlmostEqual(result["uncapped"], 2.)

    def testcontinuousswitch(self):
        first = bound(1., 2., 3., .5 * (1 - 1e-8), 10.)
        second = bound(1., 2., 3., .5 * (1 + 1e-8), 10.)
        self.assertAlmostEqual(first["radius"], second["radius"], places=6)

    def testdegenerateconstants(self):
        self.assertEqual(bound(0., 0., 0., .25)["radius"], 1.)
        self.assertEqual(bound(0., 0., 1., .25)["radius"], 0.)
        self.assertEqual(bound(0., 2., 1., .25)["branch"], "curvature")
        self.assertEqual(bound(2., 0., 1., .25)["branch"], "gradient")

    def testscaleinvariance(self):
        baseline = bound(2., 1., 6., .25)
        for scale in (1e-200, 1e200):
            result = bound(2 * scale, scale, 6 * scale, .25)
            self.assertAlmostEqual(result["radius"], baseline["radius"], places=10)
            self.assertAlmostEqual(result["threshold"], baseline["threshold"], places=10)

    def testlargecoefficients(self):
        result = bound(1e308, 1e308, 1e-308, 1e308)
        self.assertEqual(result["radius"], 1.)
        json.dumps(result, allow_nan=False)

    def testmonotonicity(self):
        loose, conservative = bound(1., 2., 1., .01), bound(1., 2., 2., .01)
        self.assertLessEqual(conservative["radius"], loose["radius"])

    def testdeclaredinequality(self):
        for gradient, curvature, third, ratio in ((2., 1., 6., .25), (1., 2., 3., .5), (.1, 4., 2., .2)):
            result = bound(gradient, curvature, third, ratio, 10.)
            point = result["radius"]
            self.assertLessEqual(third * point ** 3 / 6, ratio * max(point * gradient, .5 * point ** 2 * curvature) + 1e-12)

    def testinvalidprofiles(self):
        for value in (None, {"third": 1}, {**self.profile(), "unknown": 0}, {**self.profile(), "gradient": -1}, {**self.profile(), "third": float("inf")}, {**self.profile(), "ratio": 0}, {**self.profile(), "verified": 1}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                profile(value)

    def testinitialscope(self):
        values = {"gradient": 1., "curvature": 0., "third": 100., "ratio": .01}
        step, metadata = initialize(values, .5, SimpleNamespace(affine=True), "latent", (0, 1, 2))
        self.assertLess(step, .5)
        self.assertFalse(metadata["certified"])
        with self.assertRaises(ValueError):
            initialize(values, .5, SimpleNamespace(affine=False), "latent", (2,))
        step, metadata = initialize(values, .5, SimpleNamespace(affine=False), "latent", (0, 1))
        self.assertEqual(step, .5)
        self.assertEqual(metadata["mode"], "inactive reference coordinate")

    def testsearchintegration(self):
        path = Fake()
        path.native.affine = True
        values = {"gradient": 1., "curvature": 0., "third": 100., "ratio": .01}
        result = select(path, lambda image: {"target": 1.}, torch.ones_like(path.session.origin[:, :1]), Expensive(), Config(radiusprofile=values))
        self.assertLess(result.report["radiusprofile"]["initial"], .5)
        self.assertFalse(result.report["radiusprofile"]["certified"])
        self.assertLessEqual(result.report["damage"], result.report["rawdamage"])
        json.dumps(result.report, allow_nan=False)

    def testprofileinterface(self):
        args = common(argparse.ArgumentParser()).parse_args(["--clip", "clip", "--radiusprofile", "profile.json"])
        with patch("run.jsonfile", side_effect=lambda path, default=None: self.profile() if path == "profile.json" else default):
            config = validate(args)
        self.assertEqual(config.radiusprofile, self.profile())
        with patch("trial.jsonfile", side_effect=lambda path: self.profile() if path == "profile.json" else None):
            frozen = settings(args)
        self.assertEqual(frozen["configuration"]["radiusprofile"], self.profile())

    def testsolverablation(self):
        result = variants("solver")
        self.assertFalse(result["nocertificate"]["certificate"])
        self.assertEqual(result["shortsolve"]["solversteps"], 25)

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu 2026
# Email: wlzchina05@gmail.com
