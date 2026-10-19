# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import argparse
import json
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from engine.geometry import response
from engine.poll import Config, select
from engine.release import directions, propose
from engine.trust import Model
from run import common, validate
from trial import variants
from trust import Fake

class GeometryTests(unittest.TestCase):
    def model(self):
        return SimpleNamespace(anchor=np.array([1., 1., .5]), active=np.array([0, 2]), response=np.array([[1., 1.]]), latentresponse=np.eye(2), semantic=np.array([[0., 1.]]), residual=np.array([.2]), scores=np.array([0.]), metric=np.eye(2))

    def proposed(self, geometry):
        model = self.model()
        return propose(model, .5, [-1.], float(model.residual @ model.residual), [0.], [1.], [0], .001, geometry=geometry)

    def testoutputgeometry(self):
        rows = directions(self.model(), .001, geometry="output")
        np.testing.assert_allclose(rows[0]["direction"], [-1., 1.])
        self.assertAlmostEqual(rows[0]["protectednorm"], 0.)

    def testlatentgeometry(self):
        row = directions(self.model(), .001, geometry="latent")[0]
        np.testing.assert_allclose(row["direction"], [0., 1.])
        self.assertAlmostEqual(row["protectednorm"], 1.)

    def testattenuation(self):
        rows = directions(self.model(), .001, geometry="attenuate")
        self.assertEqual(len(rows), 1)
        np.testing.assert_allclose(rows[0]["direction"], [0., 1.])

    def testprotectedbudget(self):
        self.assertIsNotNone(self.proposed("output"))
        self.assertIsNone(self.proposed("latent"))
        self.assertIsNone(self.proposed("attenuate"))

    def testinvalidgeometry(self):
        with self.assertRaises(ValueError):
            directions(self.model(), .01, geometry="unknown")
        model = self.model()
        model.latentresponse = np.full((2, 2), np.nan)
        with self.assertRaises(ValueError):
            directions(model, .01, geometry="latent")
        del model.latentresponse
        with self.assertRaises(ValueError):
            directions(model, .01, geometry="latent")

    def testpairedfit(self):
        points = np.array([[1., 1., .5], [1.5, 1., .5], [1., 1., 1.]])
        values = points[:, [0, 2]]
        model = Model(points[0], points, [[.2], [.7], [.7]], [[0.], [0.], [.5]], [0., 0., 0.], (0, 2))
        fitted, metadata = response(model, points, values, 1e-12)
        np.testing.assert_allclose(fitted, np.eye(2), atol=1e-10)
        self.assertEqual(metadata["neighbors"], model.indices.tolist())
        self.assertEqual(metadata["queries"], 0)
        reused, _ = response(model, points, values, 1e-12, fitted)
        np.testing.assert_allclose(reused, fitted, atol=1e-10)

    def testfitvalidation(self):
        model = self.model()
        model.indices, model.directions = np.array([0]), np.empty((2, 0))
        for points, values in (([[0., 0., 0.]], [[1.]]), ([[1., 1., .5]], [[np.inf]]), ([[1., 1., .5]], [[]])):
            with self.subTest(points=points), self.assertRaises(ValueError):
                response(model, points, values)
        model.indices = np.array([4])
        with self.assertRaises(ValueError):
            response(model, [[1., 1., .5]], [[1.]])

    def testconfiguration(self):
        args = common(argparse.ArgumentParser()).parse_args(["--clip", "clip", "--geometry", "LATENT"])
        self.assertEqual(validate(args).geometry, "latent")
        for changes in ({"geometry": "invalid"}, {"geometry": "latent", "release": False}, {"geometry": "attenuate", "selection": "protect"}, {"geometry": "latent", "algorithm": "random"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                Config(**changes).validate()

    def testvariants(self):
        controls = variants("geometry")
        self.assertEqual(controls["output"]["geometry"], "output")
        self.assertEqual(controls["latent"]["geometry"], "latent")
        self.assertEqual(controls["attenuated"]["geometry"], "attenuate")
        self.assertFalse(controls["joint"]["release"])
        for name in ("output", "latent", "attenuated"):
            self.assertNotIn("candidates", controls[name])
            self.assertNotIn("axes", controls[name])

    def testintegration(self):
        class Linear(Fake):
            def raw(self):
                image = torch.zeros_like(self.session.origin)
                image[:, :, 0] = .2
                return image.clone(), image

            def candidate(self, point):
                low, high, eta = point
                image = torch.zeros_like(self.session.origin)
                image[:, :, 0] = .2 + .1 * (low - 1 + eta)
                image[:, :, 1] = .4 * eta
                latent = torch.zeros_like(image)
                latent[:, :, 0], latent[:, :, 1] = .1 * low, .4 * eta
                return latent, image

        class Prior:
            mode = "off"

            def __call__(self, latent):
                return 0.

            def metadata(self):
                return {}

        for geometry in ("output", "latent", "attenuate"):
            path = Linear()
            weight = torch.zeros_like(path.session.origin[:, :1])
            weight[:, :, 0] = 1.
            result = select(path, lambda image: {"target": float(image[:, :, 1].mean())}, weight, Prior(), Config(geometry=geometry, secant=1e-12))
            self.assertLessEqual(result.report["damage"], result.report["rawdamage"])
            self.assertLessEqual(len(result.report["candidates"]), 12)
            self.assertEqual(result.report["release"]["geometry"], geometry)
            if geometry == "latent":
                self.assertTrue(any("latentgeometry" in row for row in result.report["history"]))
            json.dumps(result.report, allow_nan=False)

    def testentry(self):
        import main
        from run import main as entry
        self.assertIs(main.main, entry)

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu 2026
# Email: wlzchina05@gmail.com
