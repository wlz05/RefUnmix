# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from engine.certificate import distancebounds, identify
from engine.limit import Budget, Exhausted
from engine.poll import Config, select
from engine.quadratic import solve
from engine.trust import Model, propose
from audit import values
from run import common, validate
from trust import Expensive, Fake

class CertificateTests(unittest.TestCase):
    def certificate(self, radius, damage=.01, target=.2):
        return identify([[1]], [0], [[1]], [[1]], [target], radius, damage)

    def model(self, residual=.3, response=.4, penalty=.05, slope=.6):
        anchor = np.array([1., 1., .5])
        points = [anchor, anchor + [-1., 0, 0], anchor + [1., 0, 0]]
        return Model(anchor, points, [[residual + response * (point[0] - 1)] for point in points], [[0.]] * 3, [penalty + slope * (point[0] - 1) for point in points], (0,), regularization=1e-12)

    def testclosedformbranches(self):
        for radius, expected in ((.05, .15), (.1, .1), (.125, np.sqrt(.075 ** 2 + .025 ** 2)), (.2, .1 / np.sqrt(2))):
            with self.subTest(radius=radius):
                result = self.certificate(radius)
                self.assertTrue(result["infeasible"])
                self.assertAlmostEqual(result["margin"], expected, places=6)
                self.assertLessEqual(result["norm"], 1. + 1e-12)
                self.assertTrue(all(value >= 0 for value in result["multipliers"]))
                json.dumps(result, allow_nan=False)

    def testmonotonebounds(self):
        values = [self.certificate(value)["margin"] for value in np.linspace(.01, .25, 9)]
        self.assertTrue(all(first + 1e-7 >= second for first, second in zip(values, values[1:])))

    def testzeronevercertifiesfeasibility(self):
        result = self.certificate(1., damage=.09)
        self.assertFalse(result["infeasible"])
        self.assertEqual(result["status"], "no certificate found")
        self.assertNotIn("feasible", result)

    def testexplicitzerowitness(self):
        result = identify([[1]], [.1], [[1]], [[1]], [-.1], .5, .04)
        self.assertEqual(result["iterations"], 0)
        self.assertEqual(result["primalwitness"], [0.])
        self.assertEqual(result["margin"], 0.)
        self.assertEqual(result["distance"]["upper"], 0.)
        self.assertTrue(result["distance"]["consistent"])

    def testdistanceintervals(self):
        for radius, expected in ((.05, .15), (.125, np.sqrt(.075 ** 2 + .025 ** 2)), (.2, .1 / np.sqrt(2))):
            result = self.certificate(radius)
            self.assertTrue(result["distance"]["consistent"])
            self.assertLessEqual(result["distance"]["lower"], expected + 1e-8)
            self.assertGreaterEqual(result["distance"]["upper"], expected - 1e-8)
            self.assertAlmostEqual(result["distance"]["upper"], .2)
            self.assertFalse(result["optimalitycertified"])

    def testunknownzerodistance(self):
        result = identify([[1]], [0], [[1]], [[1]], [.2], 1., .09, iterations=1)
        self.assertEqual(result["distance"]["lower"], 0.)
        self.assertGreater(result["distance"]["upper"], 0.)
        self.assertFalse(result["optimalitycertified"])
        self.assertNotIn("primalwitness", result)

    def testindependentunits(self):
        first = self.certificate(.2)
        second = identify([[10]], [0], [[1]], [[1]], [.2], .2, 1.)
        self.assertNotAlmostEqual(first["normalizedmargin"], second["normalizedmargin"], places=3)

    def testdistanceinvalidbounds(self):
        self.assertFalse(distancebounds(.5, .1, 1.)["consistent"])
        for values in ((0., float("inf"), 1.), (-1., 1., 1.), (0., 1., 0.)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                distancebounds(*values)

    def testdistancecontradiction(self):
        with patch("engine.certificate.support", return_value={"margin": 1., "bound": -1., "requirement": 0.}):
            result = self.certificate(.2)
        self.assertFalse(result["infeasible"])
        self.assertFalse(result["distance"]["consistent"])
        self.assertIsNone(result["distance"]["lower"])
        self.assertEqual(result["margin"], 0.)
        self.assertEqual(result["status"], "unverified local distance interval")

    def testcertificateoverflow(self):
        with self.assertRaises(ValueError):
            identify([[1e300]], [0], [[1]], [[1]], [.2], 1., .01)

    def testcertificateinvalidscalar(self):
        for values in ({"radius": "1"}, {"damage": None}):
            options = {"response": [[1]], "residual": [0], "semantics": [[1]], "metric": [[1]], "beta": [.2], "radius": .5, "damage": .01}
            options.update(values)
            with self.subTest(values=values), self.assertRaises(ValueError):
                identify(**options)

    def testzeroresponse(self):
        result = identify([[0]], [0], [[0]], [[1]], [.2], 0., 0.)
        self.assertTrue(result["infeasible"])
        self.assertAlmostEqual(result["margin"], .2)

    def testnormalizationinvariance(self):
        baseline = self.certificate(.2)
        for scale in (1e-8, 10000.):
            result = identify([[scale]], [0], [[scale]], [[1]], [.2 * scale], .2, .01 * scale ** 2)
            self.assertAlmostEqual(result["normalizedmargin"], baseline["normalizedmargin"], places=6)
            self.assertAlmostEqual(result["margin"] / scale, baseline["margin"], places=6)

    def testreduceddualspace(self):
        response = np.zeros((1000, 2))
        response[:2] = np.eye(2)
        result = identify(response, np.zeros(1000), [[1., 0]], np.eye(2), [.2], 1., .01)
        self.assertTrue(result["infeasible"])
        self.assertLessEqual(result["dimension"], 4)
        self.assertEqual(len(result["dualweights"]), 3)

    def testmetricgeometry(self):
        metric = np.array([[2., .4], [.4, 1.]])
        feasible = np.array([.2, -.1])
        response = np.array([[1., .1], [.2, 1.]])
        semantics = np.array([[1., 0]])
        result = identify(response, np.zeros(2), semantics, metric, [.15], 1., float(np.square(response @ feasible).sum()) + .01)
        self.assertFalse(result["infeasible"])

    def testinvalidcontrols(self):
        for options in ({"iterations": 0}, {"iterations": True}, {"radius": -1.}, {"damage": float("nan")}, {"beta": [float("nan")]}, {"checkpoint": 3}):
            values = {"response": [[1]], "residual": [0], "semantics": [[1]], "metric": [[1]], "beta": [.2], "radius": .5, "damage": .01}
            values.update(options)
            with self.subTest(options=options), self.assertRaises(ValueError):
                identify(**values)

    def testwallbudgetcheckpoint(self):
        budget = Budget(seconds=.1, started=0.)
        with self.assertRaises(Exhausted):
            identify([[1]], [0], [[1]], [[1]], [.2], .5, .01, checkpoint=budget.require)
        with self.assertRaises(Exhausted):
            solve([[1]], [0], 0., [], [(-1., 1.)], checkpoint=budget.require)

    def testquadraticdualgap(self):
        result = solve([[1]], [-4.], 5., [], [(-1., 1.)])
        self.assertTrue(result["converged"])
        self.assertTrue(result["optimalitycertified"])
        self.assertAlmostEqual(float(result["point"][0]), 1.)
        self.assertAlmostEqual(result["objective"], 2.)
        self.assertLessEqual(result["gap"], result["tolerance"])

    def testquadraticrescaling(self):
        for scale in (1e-16, 1., 1e12):
            result = solve([[scale]], [-4. * scale], 5. * scale, [], [(-1., 1.)])
            self.assertAlmostEqual(float(result["point"][0]), 1.)
            self.assertTrue(result["optimalitycertified"])

    def testnonconvexconstraintrejection(self):
        with self.assertRaises(ValueError):
            solve([[1]], [0.], 0., [([[-1]], [0.], 0.)], [(-1., 1.)])

    def testnonfiniteproposalrejection(self):
        with patch("engine.quadratic.minimize", return_value=SimpleNamespace(x=np.array([np.nan]), success=False, nit=1, message="failed")):
            result = solve([[1]], [0.], 0., [], [(-1., 1.)])
        self.assertIsNone(result["point"])
        self.assertFalse(result["converged"])

    def testsolverexceptionfallback(self):
        with patch("engine.quadratic.solve", side_effect=np.linalg.LinAlgError("failed")):
            self.assertIsNone(propose(self.model(), .5, [-1.], [1.], 2., 1., .1, .001))
        with patch("engine.quadratic.solve", side_effect=Exhausted("deadline")), self.assertRaises(Exhausted):
            propose(self.model(), .5, [-1.], [1.], 2., 1., .1, .001)

    def testcertificateexceptionwitness(self):
        with patch("engine.certificate.minimize", side_effect=np.linalg.LinAlgError("failed")):
            result = self.certificate(.05)
        self.assertTrue(result["infeasible"])
        self.assertFalse(result["converged"])
        self.assertEqual(result["failure"], "failed")
        with patch("engine.certificate.minimize", side_effect=Exhausted("deadline")), self.assertRaises(Exhausted):
            self.certificate(.05)

    def testallclippedpieces(self):
        for slope in (.6, -.6):
            model = self.model(slope=slope)
            result = propose(model, .5, [-1.], [1.], 2., 1., .1, .001)
            grid = np.linspace(-1., 1., 20001)
            costs = (model.residual[0] + model.response[0, 0] * grid) ** 2 + np.clip(model.penalty + model.slope[0] * grid, 0., .1) + .00025 * grid ** 2
            self.assertLessEqual(result["objective"], float(costs.min()) + 1e-7)
            self.assertEqual(len(result["solver"]["pieces"]), 3)
            self.assertTrue(result["solver"]["optimalitycertified"])

    def testcrosspieceties(self):
        model = self.model(residual=-.8, response=1., penalty=0., slope=2.)
        result = propose(model, .5, [-1.], [1.], 2., 1., .64 / 1.00025, .001)
        self.assertGreaterEqual(result["solver"]["tiedpieces"], 2)
        self.assertAlmostEqual(result["point"][0], 1., places=6)

    def testrandomizedenumeration(self):
        rng = np.random.default_rng(20261010)
        grid = np.linspace(-1., 1., 20001)
        for _ in range(20):
            r, a, p, b = rng.uniform(-.7, .7), rng.uniform(.1, 1.), rng.uniform(0, .1), rng.uniform(-1., 1.)
            model = self.model(r, a, p, b)
            result = propose(model, .5, [-1.], [1.], 4., 1., .1, .001)
            costs = (model.residual[0] + model.response[0, 0] * grid) ** 2 + np.clip(model.penalty + model.slope[0] * grid, 0., .1) + .00025 * grid ** 2
            self.assertLessEqual(result["objective"], float(costs.min()) + 2e-7)

    def testsearchdiagnostics(self):
        path = Fake()
        result = select(path, lambda image: {"target": 1.}, torch.ones_like(path.session.origin[:, :1]), Expensive(), Config())
        self.assertTrue(all("certificate" in row for row in result.report["history"]))
        self.assertLessEqual(result.report["damage"], result.report["rawdamage"])
        result.report["method"] = "path"
        metrics = values({"report": result.report})
        self.assertIn("path:certificateiterations", metrics)
        self.assertIn("path:certifiedproposals", metrics)
        json.dumps(result.report, allow_nan=False)

    def testsolverflags(self):
        args = common(argparse.ArgumentParser()).parse_args(["--clip", "clip", "--solversteps", "100", "--certificatestep", "50", "--no-certificate"])
        config = validate(args)
        self.assertEqual(config.solversteps, 100)
        self.assertEqual(config.certificatestep, 50)
        self.assertFalse(config.certificate)
        for options in ({"solversteps": 0}, {"certificatestep": True}, {"certificate": 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                Config(**options).validate()

    def testsourceconventions(self):
        root = Path(__file__).resolve().parents[1]
        for source in root.rglob("*.py"):
            self.assertRegex(source.name, r"^[a-z]+\.py$")
            text = source.read_text(encoding="utf-8")
            self.assertTrue(text.isascii(), str(source))
            lines = text.splitlines()
            self.assertGreaterEqual(len(lines), 6, str(source))
            self.assertEqual(lines[:3], ["# -*- coding: utf-8 -*-", "# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing", ""], str(source))
            self.assertEqual(lines[-3:], ["", "# Author: Weiliang Zhu", "# Email: wlzchina05@gmail.com"], str(source))
            self.assertNotIn("\n\n\n", text, str(source))
            compile(source.read_bytes(), str(source), "exec")

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
