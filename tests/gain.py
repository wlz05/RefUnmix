# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import argparse
import json
import unittest
from unittest.mock import patch
import numpy as np
import torch
from audit import summarize, values
from engine.leverage import analyze, probe
from engine.poll import Config, select
from engine.trust import Model
from run import common, validate
from trial import variants
from trust import Fake, Expensive

class GainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def analyze(self, response, semantic, residual=None, metric=None, active=(0, 2), **kwargs):
        response = np.asarray(response, dtype=float)
        return analyze(response, semantic, np.zeros(len(response)) if residual is None else residual, np.eye(len(active)) if metric is None else metric, active, **kwargs)

    def testredundant(self):
        result = self.analyze([[1., 1.]], [[1., 1.]])
        self.assertLess(result["jointnovelty"], 1e-12)
        self.assertLess(result["protectionresidual"], 1e-12)
        self.assertAlmostEqual(result["compensatedsemantic"]["component0"], 0.)
        self.assertAlmostEqual(result["neutralgain"]["component0"]["increment"], 0.)

    def testuseful(self):
        result = self.analyze([[1., 1.]], [[0., 1.]])
        self.assertGreater(result["jointnovelty"], .5)
        np.testing.assert_allclose(result["direction"], [-1., 1.])
        self.assertAlmostEqual(result["compensatedsemantic"]["component0"], 1.)
        self.assertAlmostEqual(result["neutralgain"]["component0"]["full"], 1 / np.sqrt(2))
        self.assertAlmostEqual(result["neutralgain"]["component0"]["field"], 0.)

    def testnewbutdamaging(self):
        result = self.analyze([[1., 0.], [0., 1.]], [[0., 1.]])
        self.assertAlmostEqual(result["jointnovelty"], 1.)
        self.assertAlmostEqual(result["protectionuncancellable"], 1.)
        self.assertAlmostEqual(result["damagecurvature"], 1.)
        self.assertAlmostEqual(result["neutralgain"]["component0"]["full"], 0.)

    def testdamagecross(self):
        result = self.analyze([[1., 0.], [0., 1.]], [[0., 1.]], residual=[0., -.2])
        self.assertAlmostEqual(result["damageslope"], -.4)
        delta = .1
        self.assertLess(result["damageslope"] * delta + result["damagecurvature"] * delta ** 2, 0.)

    def testzeroreference(self):
        result = self.analyze([[1., 0.]], [[1., 0.]])
        self.assertEqual(result["jointsignal"], 0.)
        self.assertEqual(result["jointnovelty"], 0.)
        self.assertEqual(result["protectionuncancellable"], 0.)

    def testfieldonly(self):
        result = self.analyze([[1., 2.]], [[0., 1.]], active=(0, 1))
        self.assertFalse(result["active"])
        self.assertNotIn("direction", result)

    def testreferenceonly(self):
        result = self.analyze([[0.]], [[1.]], active=(2,))
        self.assertEqual(result["fieldrank"], 0)
        self.assertEqual(result["jointnovelty"], 1.)
        self.assertEqual(result["neutralgain"]["component0"]["full"], 1.)

    def testmetriccompensation(self):
        response = [[1., 1., 1.]]
        result = self.analyze(response, [[0., 0., 1.]], active=(0, 1, 2), metric=np.diag([1., 4., 1.]))
        np.testing.assert_allclose(result["direction"], [-.8, -.2, 1.], atol=1e-12)
        self.assertAlmostEqual(float((np.asarray(response) @ np.asarray(result["direction"]))[0]), 0.)

    def testaxisordering(self):
        result = self.analyze([[1., 1.]], [[1., 0.]], active=(2, 0))
        np.testing.assert_allclose(result["direction"], [1., -1.])
        self.assertAlmostEqual(result["compensatedsemantic"]["component0"], 1.)

    def testbasisinvariance(self):
        response = np.array([[1., 0., 1.], [0., 1., 1.]])
        semantic = np.array([[1., -1., 2.]])
        transform = np.array([[1., .2], [-.3, 2.]])
        changed = response.copy()
        changed[:, :2] = response[:, :2] @ transform
        condition = semantic.copy()
        condition[:, :2] = semantic[:, :2] @ transform
        metric = np.eye(3)
        metric[:2, :2] = transform.T @ transform
        first = self.analyze(response, semantic, active=(0, 1, 2))
        second = self.analyze(changed, condition, active=(0, 1, 2), metric=metric)
        self.assertAlmostEqual(first["jointnovelty"], second["jointnovelty"])
        self.assertAlmostEqual(first["directionnorm"], second["directionnorm"])
        self.assertAlmostEqual(first["neutralgain"]["component0"]["full"], second["neutralgain"]["component0"]["full"])

    def testsemanticnormalization(self):
        first = self.analyze([[1., 1.]], [[0., 1.]], names=["target"])
        second = self.analyze([[1., 1.]], [[0., 100.]], names=["target"], scales=[100.])
        self.assertAlmostEqual(first["jointnovelty"], second["jointnovelty"])
        self.assertAlmostEqual(first["neutralgain"]["target"]["full"], second["neutralgain"]["target"]["full"])

    def testinvalid(self):
        for kwargs in ({"tolerance": 0.}, {"damagescale": 0.}, {"scales": [0.]}, {"metric": [[1., 1.], [1., 1.]]}, {"metric": [[1., .1], [.1, 1.]]}, {"metric": [[1., 1e-9], [0., 1.]]}, {"active": (0, 0)}, {"names": ["target", "extra"]}, {"names": [["target"]]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.analyze([[1., 1.]], [[0., 1.]], **kwargs)
        with self.assertRaises(ValueError):
            self.analyze([[1., float("nan")]], [[0., 1.]])

    def model(self):
        anchor = np.array([1., 1., .5])
        points = np.array([anchor, anchor + [.2, 0, 0], anchor - [.2, 0, 0], anchor + [0, .2, 0], anchor - [0, .2, 0], anchor + [0, 0, .2], anchor - [0, 0, .2]])
        response = np.array([[1., 0., 1.], [0., 0., 1.]])
        semantic = np.array([[0., 0., 1.]])
        residuals = [np.array([.2, .1]) + response @ (point - anchor) for point in points]
        scores = [np.array([1.]) + semantic @ (point - anchor) for point in points]
        return Model(anchor, points, residuals, scores, [0.] * len(points), (0, 1, 2), regularization=1e-12), points

    def testprobe(self):
        model, points = self.model()
        diagnosis = analyze(model.response, model.semantic, model.residual, model.metric, model.active)
        diagnosis.update({"geometryresolved": True, "fittrusted": True})
        point = probe(model, diagnosis, points, .25, np.array([.5]), np.ones(1), .05, 1.)
        self.assertIsNotNone(point)
        move = np.asarray(point)[model.active] - model.anchor[model.active]
        damage, semantic, _ = model.prediction(move, 0.)
        self.assertLess(damage, model.damage)
        self.assertGreaterEqual(semantic[0], .5)
        self.assertLessEqual(move @ model.metric @ move, .25 ** 2 + 1e-12)

    def testuntrustedprobe(self):
        model, points = self.model()
        for diagnosis in ({"active": False}, {"active": True, "geometryresolved": False}, {"active": True, "geometryresolved": True, "fittrusted": False}):
            self.assertIsNone(probe(model, diagnosis, points, .25, [.5], [1.], .05, 1.))

    def testprobeconstraint(self):
        model, points = self.model()
        diagnosis = analyze(model.response, model.semantic, model.residual, model.metric, model.active)
        diagnosis.update({"geometryresolved": True, "fittrusted": True})
        self.assertIsNone(probe(model, diagnosis, points, .25, [1.], [1.], .05, 1.))

    def testnarrowcompensation(self):
        anchor = np.array([1., 1., .5])
        points = [anchor, anchor + [.2, 0, 0], anchor + [0, 0, .2]]
        response = np.array([[1., 1.], [0., 1.]])
        residuals = [np.array([.2, -.01]) + response @ (point[[0, 2]] - anchor[[0, 2]]) for point in points]
        model = Model(anchor, points, residuals, [[1.]] * len(points), [0.] * len(points), (0, 2), regularization=1e-12)
        diagnosis = analyze(model.response, model.semantic, model.residual, model.metric, model.active)
        diagnosis.update({"geometryresolved": True, "fittrusted": True})
        point = probe(model, diagnosis, points, .5, [1.], [1.], model.damage, 1.)
        self.assertIsNotNone(point)
        self.assertTrue(.005 < point[2] - .5 < .02)
        move = np.asarray(point)[model.active] - anchor[model.active]
        self.assertLess(model.prediction(move, 0.)[0], model.damage)

    def testreportintegration(self):
        path = Fake()
        result = select(path, lambda image: {"target": 1.}, torch.ones_like(path.session.origin[:, :1]), Expensive(), Config())
        models = [row for row in result.report["history"] if "model" in row]
        self.assertTrue(models)
        self.assertTrue(all("leverage" in row for row in models))
        self.assertTrue(all(not row["leverage"]["enabled"] for row in models))
        self.assertEqual(result.report["initialleverage"]["stage"], "first fitted model after frozen initial probes")
        json.dumps(result.report, allow_nan=False)
        audited = values({"report": {"method": "path", **result.report}})
        self.assertIn("leverage:queries", audited)

    def testforcedfallbacksafe(self):
        path = Fake()
        with patch("engine.trust.propose", return_value=None):
            result = select(path, lambda image: {"target": 1.}, torch.ones_like(path.session.origin[:, :1]), Expensive(), Config(compensation=True))
        self.assertLessEqual(len(result.report["candidates"]), 12)
        self.assertLessEqual(result.report["damage"], result.report["rawdamage"])

    def testcompensationquery(self):
        class Linear(Fake):
            def candidate(self, point):
                low, high, eta = point
                image = torch.full_like(self.session.origin, .2)
                image[:, :, 0] = .2 + .1 * (low - 1) + .1 * eta
                image[:, :, 1] = .3 - .1 * eta
                image[:, :, 2] = .2 + .1 * (high - 1)
                return image.clone(), image

        path = Linear()
        with patch("engine.trust.propose", return_value=None):
            result = select(path, lambda image: {"target": 1.}, torch.ones_like(path.session.origin[:, :1]), Expensive(), Config(compensation=True))
        queries = [row for row in result.report["history"] if row.get("kind") == "compensation"]
        self.assertTrue(queries)
        self.assertTrue(all(row["leverage"]["fittrusted"] for row in queries))
        self.assertLessEqual(len(result.report["candidates"]), 12)
        self.assertLessEqual(result.report["damage"], result.report["rawdamage"])
        json.dumps(result.report, allow_nan=False)

    def testdiagnosticcoverage(self):
        trusted = {"active": True, "geometryresolved": True, "fittrusted": True}
        rows = [{"key": str(index), "report": {"method": "path", "initialleverage": record}} for index, record in enumerate((None, {}, {"active": False}, {"active": True, "fittrusted": False}, trusted))]
        coverage = summarize(rows)["diagnostics"]["initialleverage"]
        self.assertEqual((coverage["pathrows"], coverage["missing"], coverage["inactive"], coverage["untrusted"], coverage["usable"]), (5, 1, 2, 1, 1))
        self.assertEqual(coverage["fraction"], .2)
        self.assertIsNone(summarize([])["diagnostics"]["initialleverage"]["fraction"])

    def testflags(self):
        args = common(argparse.ArgumentParser()).parse_args(["--clip", "clip", "--compensation"])
        self.assertTrue(validate(args).compensation)
        args.method = "raw"
        with self.assertRaises(ValueError):
            validate(args)
        args.method, args.algorithm = "path", "poll"
        with self.assertRaises(ValueError):
            validate(args)
        with self.assertRaises(ValueError):
            Config(compensation=True, algorithm="random").validate()
        self.assertTrue(variants("leverage")["compensated"]["compensation"])
        self.assertFalse(variants("leverage")["raw"]["compensation"])

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu 2026
# Email: wlzchina05@gmail.com
