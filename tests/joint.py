# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import argparse
import json
import unittest
import torch
from engine.joint import Config, Surface, coefficients, eligible, select
from engine.response import Protection
from run import common, jointconfig, run

class JointTests(unittest.TestCase):
    def setUp(self):
        self.origin = torch.full((1, 3, 4, 4), .3)
        self.mask = torch.zeros(1, 1, 4, 4)
        self.mask[..., :2, :] = 1
        self.low = torch.ones(1, 1, 2, 2)
        self.high = torch.tensor([[[[1., -1.], [-1., 1.]]]])
        self.base = self.low + self.high

    def render(self, field):
        a, b = float(field.mean()), float((field * self.high).mean())
        pixels = self.origin.clone()
        pixels[..., :2, :] += .05 * a
        pixels[..., 2:, :] += .03 * a + .06 * b + .02 * a * b
        return pixels

    def score(self, pixels):
        return {"score": float(pixels[..., :2, :].mean()), "local": float(pixels[..., :2, :].mean())}

    def surface(self, nonlinear=True):
        zero = self.origin.clone()
        low = torch.full_like(zero, .04)
        high = torch.full_like(zero, .05)
        cross = torch.full_like(zero, .03 if nonlinear else 0)
        images = [zero, zero + low, zero + high, zero + low + high + cross]
        return Surface(images, [{"score": value} for value in (0., 1., 2., 4. if nonlinear else 3.)], Protection(self.origin, self.mask)), images

    def testfactorialidentity(self):
        surface, images = self.surface()
        self.assertTrue(torch.allclose(surface.pixels[3], torch.full_like(self.origin, .03), atol=1e-7))
        for point, image in zip(((0, 0), (1, 0), (0, 1), (1, 1)), images):
            self.assertTrue(torch.allclose(surface.image(*point), image, atol=1e-7))
            self.assertAlmostEqual(surface.predict(*point)["damage"], surface.protection.energy(image), places=7)

    def testdamageidentity(self):
        surface, images = self.surface()
        values = [surface.protection.energy(image) for image in images]
        expected = values[3] - values[1] - values[2] + values[0]
        report = surface.report()
        self.assertAlmostEqual(report["damageinteraction"], expected, places=7)
        self.assertAlmostEqual(report["damageinteraction"], report["damageoverlap"] + report["damagenonlinear"], places=7)
        self.assertAlmostEqual(surface.predict(.3, .7)["damage"], surface.protection.energy(surface.image(.3, .7)), places=7)

    def testoverlapwithoutnonlinearity(self):
        surface, _ = self.surface(False)
        report = surface.report()
        self.assertAlmostEqual(report["interactionenergy"], 0., places=10)
        self.assertGreater(report["damageoverlap"], 0)
        self.assertAlmostEqual(report["damagenonlinear"], 0., places=7)

    def testinteractionablation(self):
        surface, _ = self.surface()
        self.assertNotEqual(surface.predict(.5, .5, "additive")["damage"], surface.predict(.5, .5)["damage"])
        self.assertAlmostEqual(surface.predict(.5, .5)["scores"]["score"], 1.75)
        actual = surface.image(.5, .5)
        good = surface.calibration(actual, {"score": 1.75})
        self.assertLess(good["rgb"], 1e-7)
        self.assertEqual(good["score"], 0)
        self.assertGreater(surface.calibration(actual + .03, {"score": 1.75})["rgb"], .029)

    def testverifiedselection(self):
        result = select(self.base, self.low, self.high, self.render, self.score, self.origin, self.mask)
        self.assertEqual(result.report["selected"], "low")
        self.assertLess(result.report["damagechange"], 0)
        self.assertGreaterEqual(result.report["gain"], -1e-7)
        self.assertTrue(result.report["constraintmet"])
        self.assertTrue(torch.equal(result.image, self.render(result.field)))
        self.assertLessEqual(result.report["evaluations"], 9)
        json.dumps(result.report, allow_nan=False)

    def testpositivegainfallback(self):
        result = select(self.base, self.low, self.high, self.render, self.score, self.origin, self.mask, Config(gain=.5))
        self.assertTrue(result.report["fallback"])
        self.assertFalse(result.report["constraintmet"])
        self.assertTrue(torch.equal(result.field, self.base))

    def testrejectsunseenviolation(self):
        def curved(field):
            a, b = float(field.mean()), float((field * self.high).mean())
            image = self.origin.clone()
            image[..., :2, :] += .05 * (a + b - a * b)
            image[..., 2:, :] += .1 * a - .2 * b + 4 * a * (a - .5) ** 2 * b * (1 - b)
            return image

        result = select(self.base, self.low, self.high, curved, self.score, self.origin, self.mask, Config(checks=1, budget=0))
        self.assertTrue(result.report["trusted"])
        self.assertEqual(len(result.report["verification"]), 1)
        self.assertFalse(result.report["verification"][0]["feasible"])
        self.assertTrue(result.report["fallback"])

    def testauditpairs(self):
        from audit import paired, summarize
        left = [{"key": key, "group": group, "seed": seed, "metrics": {"quality": 1.}, "report": {}} for key, group, seed in (("a", "image", 1), ("a", "image", 2), ("b", "image", 1), ("c", "other", 1))]
        right = [{**row, "metrics": {"quality": 3.}} for row in left]
        report = paired(left, right, repeats=10)
        self.assertEqual(report["metrics"]["metric:quality"]["difference"], 2)
        self.assertEqual(report["metrics"]["metric:quality"]["images"], 2)
        self.assertEqual(report["metrics"]["metric:quality"]["instances"], 3)
        self.assertEqual(summarize(left)["metrics"]["metric:quality"]["mean"], 1)
        with self.assertRaises(ValueError):
            paired(left, right[:-1])
        right[0]["report"] = {"source": "changed"}
        with self.assertRaises(ValueError):
            paired(left, right)

    def testinvalidsupport(self):
        for mask in (torch.zeros_like(self.mask), torch.ones_like(self.mask)):
            result = select(self.base, self.low, self.high, self.render, self.score, self.origin, mask)
            self.assertTrue(result.report["fallback"])
            self.assertEqual(result.report["evaluations"], 1)
            self.assertFalse(result.report["constraintmet"])

    def testnonlinearcalibration(self):
        def curved(field):
            image = self.render(field)
            a, b = float(field.mean()), float((field * self.high).mean())
            return image + .4 * a * (1 - a) * b * (1 - b)

        result = select(self.base, self.low, self.high, curved, self.score, self.origin, self.mask, Config(tolerance=.001))
        self.assertFalse(result.report["trusted"])
        self.assertEqual(result.report["verification"], [])
        self.assertTrue(result.report["constraintmet"])

    def testscalarcontrol(self):
        result = select(self.base, self.low, self.high, self.render, self.score, self.origin, self.mask, Config(decision="scalar"))
        self.assertTrue(result.report["fallback"])
        for probe in result.report["verification"]:
            self.assertEqual(probe["weights"][0], probe["weights"][1])

    def teststagediagnostics(self):
        count = [0]

        def before(field):
            count[0] += 1
            return self.render(field)

        result = select(self.base, self.low, self.high, self.render, self.score, self.origin, self.mask, transport=before)
        self.assertEqual(count[0], 4)
        self.assertEqual(result.report["stages"]["renders"], 4)
        self.assertAlmostEqual(result.report["stages"]["before"]["interactionenergy"], result.report["stages"]["after"]["interactionenergy"])

    def testdegeneratefields(self):
        zero = torch.zeros_like(self.base)
        result = select(zero, zero, zero, self.render, self.score, self.origin, self.mask)
        self.assertEqual(result.report["evaluations"], 1)
        self.assertTrue(result.report["fallback"])

    def testdensecontrol(self):
        result = select(self.base, self.low, self.high, self.render, self.score, self.origin, self.mask, Config(decision="dense", levels=3, checks=0))
        self.assertEqual(result.report["evaluations"], 9)
        self.assertEqual(len(result.report["verification"]), 4)

    def testvalidation(self):
        for options in ({"strength": 0}, {"levels": 4}, {"levels": 35}, {"checks": -1}, {"budget": float("nan")}, {"decision": "unknown"}, {"slack": -1}, {"gain": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                Config(**options).validate()
        with self.assertRaises(ValueError):
            select(self.base, self.high, self.low, self.render, self.score, self.origin, self.mask)
        with self.assertRaises(ValueError):
            coefficients([1, 2, 3])

    def testsemanticguard(self):
        reference = {"score": 1., "local": 2.}
        self.assertFalse(eligible({"score": 1.1, "local": 1.}, .01, reference, .02, Config()))
        self.assertFalse(eligible({"score": 1.1, "local": 2.}, .03, reference, .02, Config()))
        self.assertTrue(eligible({"score": 1.1, "local": 2.}, .01, reference, .02, Config()))

    def testarguments(self):
        args = common(argparse.ArgumentParser()).parse_args(["--clip", "./clip", "--method", "JOINT", "--lowtime", ".72", "--hightime", ".76"])
        self.assertEqual(args.method, "joint")
        self.assertEqual(jointconfig(args).interaction, "full")
        args.lowtime = 0
        with self.assertRaises(ValueError):
            jointconfig(args)

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu 2026
# Email: wlzchina05@gmail.com
