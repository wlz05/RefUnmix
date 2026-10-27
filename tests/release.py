# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import argparse
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from audit import values
from engine.limit import Exhausted
from engine.poll import Config, select
from engine.release import directions, interval, propose
from run import common, validate
from trial import variants
from trust import Fake

class ReleaseTests(unittest.TestCase):
    def model(self, response=((1., 1.),), semantic=((0., 1.),), residual=(.2,), scores=(0.,), active=(0, 2), metric=None):
        return SimpleNamespace(anchor=np.array([1., 1., .5]), active=np.asarray(active), response=np.asarray(response, dtype=float), semantic=np.asarray(semantic, dtype=float), residual=np.asarray(residual, dtype=float), scores=np.asarray(scores, dtype=float), metric=np.eye(len(active)) if metric is None else np.asarray(metric, dtype=float))

    def proposed(self, model=None, **changes):
        model = model or self.model()
        options = dict(radius=.5, lower=[-1.] * len(model.scores), damage=float(model.residual @ model.residual), reference=[0.] * len(model.scores), scales=[1.] * len(model.scores), indices=list(range(len(model.scores))), damping=.001)
        options.update(changes)
        return propose(model, **options)

    def testexactcompensation(self):
        row = directions(self.model(), .001)[0]
        np.testing.assert_allclose(row["direction"], [-1., 1.])
        np.testing.assert_allclose(row["protected"], [0.])
        np.testing.assert_allclose(row["retained"], [1.])

    def testdampedcompensation(self):
        rows = directions(self.model(), .1)
        self.assertEqual(len(rows), 3)
        self.assertLess(abs(rows[-1]["direction"][0]), abs(rows[0]["direction"][0]))
        self.assertGreater(rows[-1]["protected"][0], rows[0]["protected"][0])

    def testmetric(self):
        model = self.model(response=((1., 1., 1.),), semantic=((0., 0., 1.),), active=(0, 1, 2), metric=np.diag([1., 4., 1.]))
        np.testing.assert_allclose(directions(model, .1)[0]["direction"], [-.8, -.2, 1.])

    def testordering(self):
        model = self.model(semantic=((1., 0.),), active=(2, 0))
        np.testing.assert_allclose(directions(model, .01)[0]["direction"], [1., -1.])

    def testfieldonly(self):
        self.assertEqual(directions(self.model(active=(0, 1)), .01), [])
        self.assertIsNone(self.proposed(self.model(active=(0, 1))))

    def testreferenceonly(self):
        model = self.model(response=((0.,),), semantic=((1.,),), residual=(0.,), active=(2,))
        self.assertAlmostEqual(self.proposed(model, damage=0.)["point"][2], 1.)

    def teststrictgain(self):
        result = self.proposed()
        self.assertIsNotNone(result)
        self.assertGreater(result["solver"]["gain"], .35)
        self.assertAlmostEqual(result["solver"]["predicteddamage"], .04)
        self.assertEqual(result["solver"]["operator"], "release")
        json.dumps(result["solver"], allow_nan=False)

    def testredundancy(self):
        model = self.model(semantic=((1., 1.),), residual=(0.,))
        self.assertIsNone(self.proposed(model, damage=0.))

    def testuncancellable(self):
        model = self.model(response=((1., 0.), (0., 1.)), residual=(0., 0.))
        self.assertIsNone(self.proposed(model, damage=0.))

    def testexistingbias(self):
        model = self.model(response=((1., 0.), (0., 1.)), residual=(0., -.1))
        result = self.proposed(model)
        self.assertAlmostEqual(result["solver"]["amplitude"], .2)
        self.assertLessEqual(result["solver"]["predicteddamage"], .01 + 1e-12)
        self.assertEqual(np.linalg.matrix_rank(model.response), len(model.active))
        self.assertGreater(result["solver"]["protectednorm"], 0.)
        self.assertLess(result["solver"]["damagecross"], 0.)
        self.assertAlmostEqual(result["solver"]["repaircredit"], result["solver"]["damagequadratic"])
        self.assertAlmostEqual(result["solver"]["predicteddamage"], model.residual @ model.residual + result["solver"]["damagecross"] + result["solver"]["damagequadratic"])

    def testrepaircapacity(self):
        model = self.model(response=((2., 1.), (0., 1.)), semantic=((1., 2.),), residual=(0., -.1))
        joint = self.proposed(model)
        reference = self.model(response=((1.,), (1.,)), semantic=((2.,),), residual=(0., -.1), active=(2,))
        alone = self.proposed(reference)
        self.assertAlmostEqual(alone["solver"]["gain"], .2)
        self.assertGreater(joint["solver"]["gain"], .29)
        self.assertGreater(joint["solver"]["gain"], alone["solver"]["gain"])
        self.assertGreater(joint["solver"]["repaircredit"], 0.)
        self.assertGreater(joint["solver"]["protectednorm"], 0.)
        self.assertLessEqual(joint["solver"]["predicteddamage"], .01 + 1e-12)
        self.assertLess(np.linalg.norm(np.eye(2) - np.linalg.pinv(model.response) @ model.response), 1e-12)

    def testsemanticfloor(self):
        model = self.model(semantic=((0., 1.), (0., -1.)), scores=(0., 0.))
        result = self.proposed(model, lower=[-1., -.03], indices=[0])
        self.assertAlmostEqual(result["solver"]["amplitude"], .03)

    def testcommonbottleneck(self):
        model = self.model(response=((0., 0.),), semantic=((0., 1.), (0., -1.)), residual=(0.,), scores=(0., .2))
        result = self.proposed(model, damage=0.)
        self.assertAlmostEqual(result["solver"]["amplitude"], .1)
        self.assertAlmostEqual(result["solver"]["gain"], .1)

    def testreverse(self):
        model = self.model(semantic=((0., -1.),))
        self.assertLess(self.proposed(model)["solver"]["amplitude"], 0.)

    def testboundary(self):
        model = self.model()
        model.anchor[2] = 1.
        self.assertIsNone(self.proposed(model))

    def testtinyinterval(self):
        model = self.model(response=((0., 1.),), residual=(-1e-6,))
        result = self.proposed(model, damping=1e-9)
        self.assertAlmostEqual(result["solver"]["amplitude"], 2e-6)

    def testemptyinterval(self):
        self.assertIsNone(interval(self.model(), [-1., 1.], .5, [1.], .04))

    def testinvalid(self):
        for changes in ({"damping": 0.}, {"damping": np.nan}, {"damping": 1e308}, {"loss": -1.}, {"radius": 0.}, {"radius": 1e-300}, {"radius": 1e300}, {"damage": -1.}, {"lower": [np.inf]}, {"scales": [0.]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.proposed(**changes)
        model = self.model(metric=[[1., .1], [.1, 1.]])
        with self.assertRaises(ValueError):
            directions(model, .01)
        model = self.model(active=(0, 0))
        with self.assertRaises(ValueError):
            directions(model, .01)

    def testgainthreshold(self):
        self.assertIsNone(self.proposed(loss=1.))

    def testregularizedoptimum(self):
        model = self.model(response=((0., 0.),), residual=(0.,), scores=(0.,))
        result = self.proposed(model, damping=10., damage=0.)
        self.assertAlmostEqual(result["solver"]["amplitude"], .05)

    def testbudget(self):
        def stopped():
            raise Exhausted("Synthetic release time budget.")

        with self.assertRaises(Exhausted):
            self.proposed(checkpoint=stopped)

    def testflags(self):
        args = common(argparse.ArgumentParser()).parse_args(["--clip", "clip", "--no-release"])
        self.assertFalse(validate(args).release)
        self.assertTrue(Config().release)
        with self.assertRaises(ValueError):
            Config(release=1).validate()
        self.assertTrue(variants("release")["released"]["release"])
        self.assertFalse(variants("release")["joint"]["release"])

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
                return image.clone(), image

        class Prior:
            mode = "off"

            def __call__(self, latent):
                return 0.

            def metadata(self):
                return {}

        path = Linear()
        mask = torch.zeros_like(path.session.origin[:, :1])
        mask[:, :, 0] = 1.
        with patch("engine.trust.propose", return_value=None):
            result = select(path, lambda image: {"target": float(image[:, :, 1].mean())}, mask, Prior(), Config(secant=1e-12))
        rows = [row for row in result.report["history"] if row.get("kind") == "release"]
        self.assertTrue(rows)
        self.assertTrue(any(row.get("accepted") for row in rows))
        for row in rows:
            checked = row["verification"]
            self.assertTrue(np.isfinite([checked[key] for key in ("predictedgain", "actualgain", "predicteddamagechange", "actualdamagechange")]).all())
            self.assertIsInstance(checked["feasible"], bool)
            center = next(candidate for candidate in result.report["candidates"] if candidate["point"] == row["center"])
            self.assertAlmostEqual(checked["actualdamagechange"], row["actual"] - center["damage"])
            self.assertAlmostEqual(checked["predicteddamagechange"], row["predicted"] - center["damage"])
            self.assertAlmostEqual(checked["predictedgain"], checked["actualgain"], places=5)
        self.assertLessEqual(result.report["damage"], result.report["rawdamage"])
        self.assertLessEqual(len(result.report["candidates"]), 12)
        self.assertGreater(result.report["attainment"]["selectedgain"], .2)
        self.assertGreater(values({"report": {"method": "path", **result.report}})["release:queries"], 0)
        self.assertIsNotNone(values({"report": {"method": "path", **result.report}})["release:verifiedactualgain"])
        json.dumps(result.report, allow_nan=False)

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
