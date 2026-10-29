# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import json
import unittest
import numpy as np
import torch
from engine.curvature import Guard, fallback
from engine.poll import Config, select
from engine.trust import Model, propose
from trust import Expensive, Fake
from trial import variants
from evaluate import costs

class CurvatureTests(unittest.TestCase):
    def model(self, semantic=lambda value: value, initial=None, residual=lambda value: np.array([.2 + value])):
        anchor = np.array([1., 1., .5])
        points = [anchor, anchor + [.5, 0, 0], anchor - [.5, 0, 0]]
        return Model(anchor, points, [residual(point[0] - 1) for point in points], [[semantic(point[0] - 1)] for point in points], [0.] * 3, (0,), regularization=1e-10, initial=initial), points

    def testaffineresponses(self):
        model, _ = self.model()
        result = Guard().inspect(model, [1.], 1., .001)
        self.assertLess(result["semanticerror"], 1e-8)
        self.assertFalse(result["fallback"])
        self.assertAlmostEqual(result["damping"], .001)

    def teststationarycurvature(self):
        model, _ = self.model(semantic=lambda value: 1 - 4 * value ** 2)
        result = Guard().inspect(model, [1.], 1., .001)
        self.assertTrue(result["flat"])
        self.assertIn("semantic residual", result["reasons"])
        self.assertTrue(result["fallback"])
        self.assertGreater(result["damping"], .001)

    def testactualsignreversal(self):
        model, _ = self.model(semantic=lambda value: -value, initial=np.array([[1.], [1.]]))
        result = Guard().inspect(model, [1.], 1., .001)
        self.assertEqual(result["signreversals"], 2)
        self.assertTrue(result["fallback"])

    def testinnovationisnotsignproof(self):
        model, _ = self.model(semantic=lambda value: .5 * value, initial=np.array([[1.], [1.]]))
        result = Guard().inspect(model, [1.], 1., .001)
        self.assertGreater(model.metadata()["innovation"], 0)
        self.assertEqual(result["signreversals"], 0)
        self.assertFalse(result["fallback"])

    def testhessiancondition(self):
        anchor = np.array([1., 1., .5])
        points = [anchor, anchor + [.5, 0, 0], anchor - [.5, 0, 0], anchor + [0, .5, 0], anchor - [0, .5, 0]]
        residuals = [np.array([1e4 * (point[0] - 1) + .2, 1e-6 * (point[1] - 1) + .2]) for point in points]
        model = Model(anchor, points, residuals, [[1.]] * 5, [0.] * 5, (0, 1), regularization=1e-12)
        result = Guard(condition=100.).inspect(model, [1.], 1., .001)
        self.assertLessEqual(result["condition"], 100. + 1e-8)
        self.assertGreater(result["damping"], .001)

    def testobservedmodelbreak(self):
        model, _ = self.model()
        guard = Guard()
        result = guard.observe(model, np.array([.4]), np.array([.6]), [-.24], [1.], 1.)
        self.assertTrue(result["alert"])
        self.assertGreater(result["semanticerror"], .5)
        self.assertTrue(guard.pending)
        guard.accepted()
        self.assertFalse(guard.pending)

    def testreversefinitequery(self):
        model, points = self.model()
        point = fallback(model, points, .5, preferred=np.array([.4]))
        self.assertIsNotNone(point)
        self.assertLess(point[0], model.anchor[0])
        move = np.asarray(point)[model.active] - model.anchor[model.active]
        self.assertLessEqual(np.linalg.norm(move * model.root), .5 + 1e-12)

    def testboxandduplicates(self):
        model, points = self.model()
        model.anchor[0] = 0
        points.append(model.anchor.copy())
        point = fallback(model, points, .5, preferred=np.array([1.]))
        self.assertIsNotNone(point)
        self.assertTrue(0 <= point[0] <= 2)
        points.extend([point, (2., 1., .5), (.5, 1., .5)])
        self.assertIsNone(fallback(model, points, .5))

    def testdisabledablation(self):
        model, _ = self.model(semantic=lambda value: -4 * value ** 2)
        result = Guard(enabled=False).inspect(model, [1.], 1., .001)
        self.assertFalse(result["fallback"])
        self.assertEqual(result["damping"], .001)

    def testdampingchangesproposal(self):
        model, _ = self.model()
        first = propose(model, .5, [-1.], [1.], 1., 1., .1, .001)
        second = propose(model, .5, [-1.], [1.], 1., 1., .1, 10.)
        self.assertLess(np.linalg.norm(second["move"]), np.linalg.norm(first["move"]))

    def testinvalidcontrols(self):
        for options in ({"curvature": 0}, {"hessian": 1}, {"damping": 1}, {"monitor": 1}, {"critical": float("nan")}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                Config(**options).validate()
        for options in ({"condition": 1}, {"growth": 1}, {"enabled": 1}):
            with self.assertRaises(ValueError):
                Guard(**options)

    def testcostablations(self):
        light = variants("cost")["light"]
        self.assertEqual((light["candidates"], light["axes"], light["prior"]), (3, "path", "off"))
        self.assertFalse(variants("search")["unguarded"]["monitor"])

    def testgroupedcosts(self):
        rows = [{"key": "a", "seconds": 1., "report": {"stats": {"samples": 10}}}, {"key": "a", "seconds": 3., "report": {"stats": {"samples": 20}}}, {"key": "b", "seconds": 8., "report": {"stats": {"samples": 30}}}]
        result = costs(rows)["metrics"]
        self.assertEqual(result["seconds"]["mean"], 5.)
        self.assertEqual(result["seconds"]["sources"], 2)
        self.assertEqual(result["samples"]["mean"], 22.5)
        self.assertNotIn("vram", result)

    def testmissingcosts(self):
        self.assertEqual(costs([])["metrics"], {})
        result = costs([{"seconds": float("nan")}, {"seconds": True}, {"seconds": -1}])
        self.assertEqual(result["metrics"], {})

    def testcurvedsearchbudget(self):
        class Curved(Fake):
            def candidate(self, point):
                latent, image = super().candidate(point)
                image[:, 0] = .2 + .05 * (point[0] - 1)
                image[:, 1] = .5 - 2 * (point[0] - 1) ** 2
                return latent, image

        path = Curved()
        result = select(path, lambda image: {"target": float(image[:, 1].mean())}, torch.ones_like(path.session.origin[:, :1]), Expensive(), Config(candidates=18, curvature=.005))
        self.assertLessEqual(len(result.report["candidates"]), 18)
        self.assertLessEqual(result.report["damage"], result.report["rawdamage"])
        self.assertTrue(any(row["kind"] == "fallback" for row in result.report["history"]))
        self.assertGreater(result.report["curvature"]["alerts"], 0)
        self.assertFalse(result.report["termination"]["stationaritycertified"])
        json.dumps(result.report, allow_nan=False)

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
