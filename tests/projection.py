# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import json
import unittest
import numpy as np
import torch
from engine.projection import Config, build, factorial, select, solve
from engine.response import Protection
from engine.wavelet import gates, split

class ProjectionTests(unittest.TestCase):
    def config(self, **changes):
        return Config(**{"lock": "semantic", **changes})

    def testsemanticcounterexample(self):
        matrix = np.array([[1., -1.]])
        residual = np.array([1.])
        semantics = np.array([[1., 1.]])
        weights, report = solve(matrix, residual, semantics, self.config(radius=1., ridge=.1))
        expected = np.array([-1., 1.]) / 2.1
        np.testing.assert_allclose(weights, expected, atol=1e-9)
        self.assertAlmostEqual((semantics @ weights).item(), 0., places=10)
        self.assertLess(report["predicted"], report["initial"])
        self.assertEqual(report["rank"], 1)
        self.assertEqual(report["nullity"], 1)
        self.assertLess(report["lockerror"], 1e-9)
        json.dumps(report, allow_nan=False)

    def testdependentlocks(self):
        matrix = np.array([[1., -1.]])
        residual = np.array([1.])
        constraints = np.array([[1., 1.], [2., 2.], [0., 0.]])
        weights, report = solve(matrix, residual, constraints, self.config(radius=1.))
        reference, _ = solve(matrix, residual, constraints[:1], self.config(radius=1.))
        np.testing.assert_allclose(weights, reference, atol=1e-9)
        self.assertEqual(report["rank"], 1)
        self.assertEqual(report["nullity"], 1)

    def testfulllockfallback(self):
        weights, report = solve(np.eye(2), np.array([1., 2.]), np.eye(2), self.config())
        np.testing.assert_array_equal(weights, np.zeros(2))
        self.assertEqual(report["nullity"], 0)
        self.assertAlmostEqual(report["initial"], report["predicted"])

    def testtrustboundary(self):
        weights, report = solve(np.eye(2), np.array([2., 0.]), np.empty((0, 2)), self.config(radius=.2, ridge=.1))
        np.testing.assert_allclose(weights, [-.2, 0.], atol=1e-8)
        self.assertAlmostEqual(report["norm"], .2, places=8)
        self.assertGreater(report["dual"], 0.)
        self.assertAlmostEqual(report["predicted"], 3.24, places=7)

    def testphysicalmetricinvariance(self):
        matrix = np.array([[1., .2], [-.3, 2.], [.5, -.5]])
        residual = np.array([1., -.5, .2])
        constraints = np.array([[1., -.7]])
        metric = np.array([[2., .3], [.3, 1.]])
        transform = np.array([[2., .4], [0., .5]])
        config = self.config(radius=.25, ridge=.05)
        weights, report = solve(matrix, residual, constraints, config, metric)
        changed, other = solve(matrix @ transform, residual, constraints @ transform, config, transform.T @ metric @ transform)
        np.testing.assert_allclose(weights, transform @ changed, atol=1e-8)
        self.assertAlmostEqual(report["predicted"], other["predicted"], places=8)
        self.assertAlmostEqual(report["norm"], other["norm"], places=8)
        self.assertLessEqual(float(weights @ metric @ weights), config.radius ** 2 + 1e-9)

    def testmetrictruststationarity(self):
        matrix = np.array([[2., .3], [.2, 1.]])
        residual = np.array([2., -1.])
        metric = np.array([[3., .4], [.4, 1.]])
        config = self.config(radius=.1, ridge=.02)
        weights, report = solve(matrix, residual, np.empty((0, 2)), config, metric)
        gradient = matrix.T @ (matrix @ weights + residual) + (config.ridge + report["dual"]) * metric @ weights
        np.testing.assert_allclose(gradient, np.zeros(2), atol=1e-7)
        self.assertAlmostEqual(float(weights @ metric @ weights), config.radius ** 2, places=8)

    def testzeroprotectresponse(self):
        weights, report = solve(np.zeros((3, 2)), np.ones(3), np.zeros((1, 2)))
        np.testing.assert_array_equal(weights, np.zeros(2))
        self.assertEqual(report["rank"], 0)
        self.assertAlmostEqual(report["initial"], 3.)
        self.assertAlmostEqual(report["predicted"], 3.)

    def testinvalidsystems(self):
        cases = [(np.ones((2, 2)), np.ones(3), np.zeros((1, 2)), None), (np.ones((2, 2)), np.ones(2), np.zeros((1, 3)), None), (np.array([[np.nan, 0.]]), np.ones(1), np.zeros((1, 2)), None), (np.eye(2), np.ones(2), np.zeros((1, 2)), np.array([[1., 2.], [0., 1.]])), (np.eye(2), np.ones(2), np.zeros((1, 2)), np.diag([1., -1.]))]
        for matrix, residual, constraints, metric in cases:
            with self.subTest(shape=matrix.shape, metric=metric), self.assertRaises(ValueError):
                solve(matrix, residual, constraints, metric=metric)

    def testconfiguration(self):
        for changes in ({"grid": 0}, {"step": 0.}, {"radius": -.1}, {"ridge": -.1}, {"backtrack": -1}, {"slack": -.1}, {"minreduce": -.1}, {"ranktol": -1.}, {"lock": "unknown"}, {"coupling": "unknown"}, {"step": float("nan")}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.config(**changes).validate()

    def testalignedorthogonalbasis(self):
        generator = torch.Generator().manual_seed(7)
        base = torch.randn(1, 2, 8, 8, generator=generator)
        vectors, metadata = build(base, grid=2)
        self.assertEqual(len(vectors), 8)
        self.assertIsInstance(metadata, dict)
        scale = float(base.square().mean())
        matrix = torch.stack([value.flatten() for value in vectors])
        gram = matrix @ matrix.T / base.numel() / scale
        self.assertTrue(torch.allclose(gram, torch.eye(len(vectors)), atol=2e-5, rtol=2e-5))
        windows = gates(base, 2)
        for value in vectors:
            low, high = split(value)
            self.assertLess(min(float(low.norm()), float(high.norm())), 1e-5 * float(value.norm()))
            self.assertTrue(any(torch.allclose(value * gate, value, atol=1e-7) for gate in windows))
        json.dumps(metadata, allow_nan=False)

    def testdonordeduplication(self):
        generator = torch.Generator().manual_seed(11)
        base = torch.randn(1, 2, 8, 8, generator=generator)
        donor = torch.randn(1, 2, 8, 8, generator=generator)
        vectors, _ = build(base, {.4: base, .5: base + donor, .6: base - donor}, grid=2)
        self.assertEqual(len(vectors), 16)
        matrix = torch.stack([value.flatten() for value in vectors])
        gram = matrix @ matrix.T / base.numel() / base.square().mean()
        self.assertTrue(torch.allclose(gram, torch.eye(len(vectors)), atol=3e-5, rtol=3e-5))

    def testdegeneratebasis(self):
        vectors, _ = build(torch.zeros(1, 2, 4, 4), grid=2)
        self.assertEqual(vectors, [])
        vectors, _ = build(torch.ones(1, 1, 4, 4), grid=2)
        self.assertEqual(len(vectors), 4)
        for vector in vectors:
            self.assertLess(float(split(vector)[1].abs().max()), 1e-7)

    def testinvalidbasis(self):
        with self.assertRaises(ValueError):
            build(torch.ones(2, 4, 4))
        with self.assertRaises(ValueError):
            build(torch.full((1, 1, 4, 4), float("nan")))
        with self.assertRaises(ValueError):
            build(torch.ones(1, 1, 4, 4), {.5: torch.ones(1, 1, 8, 8)})

    def fixture(self, curved=False):
        origin = torch.full((1, 3, 4, 4), .3)
        mask = torch.zeros(1, 1, 4, 4)
        mask[..., :2, :] = 1.
        base = torch.ones(1, 1, 4, 4)
        states = []

        def render(field):
            states.append(torch.is_grad_enabled())
            left, right = field[..., :2].mean(), field[..., 2:].mean()
            pixels = origin.clone()
            pixels[..., :2, :] += .1 - .1 * (left - right).square() if curved else .05 * (left + right)
            pixels[..., 2:, :] += .03 + .05 * (left - right)
            return pixels

        def score(image):
            value = float(image[..., :2, :].mean())
            return {"score": value, "global": value, "local": value, "target": value}

        return base, render, score, origin, mask, states

    def testdirectspatialrepair(self):
        base, render, score, origin, mask, states = self.fixture()
        baseline = render(base)
        states.clear()
        result = select(base, render, score, origin, mask, self.config(radius=.5, ridge=1e-6, slack=1e-6))
        self.assertTrue(states)
        self.assertFalse(any(states))
        protection = Protection(origin, mask)
        self.assertLess(protection.energy(result.image), protection.energy(baseline) * .1)
        self.assertGreaterEqual(score(result.image)["score"], score(baseline)["score"] - 1e-6)
        self.assertFalse(torch.equal(result.field, base))
        self.assertTrue(torch.allclose(result.image, render(result.field), atol=1e-7))
        self.assertFalse(result.field.requires_grad)
        json.dumps(result.report, allow_nan=False)

    def testexternalguardselection(self):
        base, render, score, origin, mask, _ = self.fixture()
        protection = Protection(origin, mask)
        admit = lambda pixels, scores: protection.energy(pixels) > .0001
        result = select(base, render, score, origin, mask, self.config(radius=.5, ridge=1e-6, slack=1e-6, calibration="fixed"), admit=admit)
        self.assertFalse(result.report["fallback"])
        self.assertTrue(admit(result.image, score(result.image)))
        self.assertTrue(any(not row["referencevalid"] for row in result.report["verification"]))
        self.assertTrue(any(row["accepted"] for row in result.report["verification"]))
        self.assertLess(protection.energy(result.image), protection.energy(render(base)))
        with self.assertRaises(ValueError):
            select(base, render, score, origin, mask, admit=1)

    def testnonlinearsemanticrejection(self):
        base, render, score, origin, mask, _ = self.fixture(curved=True)
        result = select(base, render, score, origin, mask, self.config(radius=.5, ridge=1e-6, slack=0., backtrack=3))
        self.assertTrue(torch.equal(result.field, base))
        self.assertTrue(torch.equal(result.image, render(base)))
        json.dumps(result.report, allow_nan=False)

    def testinvalidsupportfallback(self):
        base, render, score, origin, mask, states = self.fixture()
        for support in (torch.zeros_like(mask), torch.ones_like(mask)):
            states.clear()
            result = select(base, render, score, origin, support)
            self.assertTrue(torch.equal(result.field, base))
            self.assertEqual(len(states), 1)

    def testzerofieldfallback(self):
        base, render, score, origin, mask, states = self.fixture()
        zero = torch.zeros_like(base)
        result = select(zero, render, score, origin, mask)
        self.assertTrue(torch.equal(result.field, zero))
        self.assertEqual(len(states), 1)

    def testconefullrankimprovement(self):
        weights, report = solve(np.eye(2), np.array([-1., -2.]), np.eye(2), Config(radius=3., ridge=.1))
        np.testing.assert_allclose(weights, [1 / 1.1, 2 / 1.1], atol=1e-9)
        self.assertEqual(report["semanticrank"], 2)
        self.assertEqual(report["rank"], 0)
        self.assertGreater(report["modeldecrease"], 0.)
        self.assertLess(report["violation"], 1e-9)
        equality, _ = solve(np.eye(2), np.array([-1., -2.]), np.eye(2), self.config(radius=3., ridge=.1))
        np.testing.assert_array_equal(equality, np.zeros(2))

    def testconeboundarykkt(self):
        matrix, residual, constraints = np.eye(2), np.array([2., -3.]), np.eye(2)
        config = Config(radius=.5, ridge=.1)
        weights, report = solve(matrix, residual, constraints, config)
        np.testing.assert_allclose(weights, [0., .5], atol=1e-8)
        self.assertEqual(report["active"], [0])
        self.assertGreater(report["dual"], 0.)
        gradient = matrix.T @ (matrix @ weights + residual) + (config.ridge + report["dual"]) * weights
        self.assertGreater(gradient[0], 0.)
        self.assertAlmostEqual(gradient[1], 0., places=7)

    def testconeopposingconstraints(self):
        constraints = np.array([[1., 0.], [-1., 0.], [0., 1.]])
        weights, report = solve(np.eye(2), np.array([-2., -1.]), constraints, Config(radius=2., ridge=.1))
        np.testing.assert_allclose(weights, [0., 1 / 1.1], atol=1e-8)
        self.assertTrue(np.all(constraints @ weights >= -1e-9))
        self.assertLess(report["objective"], report["initial"])

    def testconemetricinvariance(self):
        matrix = np.array([[1., .2], [-.3, 2.], [.5, -.5]])
        residual, constraints = np.array([1., -.5, .2]), np.array([[1., -.7], [.1, 1.]])
        metric, transform = np.array([[2., .3], [.3, 1.]]), np.array([[2., .4], [0., .5]])
        config = Config(radius=.25, ridge=.05)
        weights, report = solve(matrix, residual, constraints, config, metric)
        changed, other = solve(matrix @ transform, residual, constraints @ transform, config, transform.T @ metric @ transform)
        np.testing.assert_allclose(weights, transform @ changed, atol=1e-8)
        self.assertAlmostEqual(report["objective"], other["objective"], places=8)
        self.assertTrue(np.all(constraints @ weights >= -1e-9))

    def testconegridreference(self):
        matrix, residual = np.array([[1., -.5], [.2, 1.]]), np.array([1., -.4])
        constraints, config = np.array([[1., 1.], [-1., 2.]]), Config(radius=.6, ridge=.05)
        weights, report = solve(matrix, residual, constraints, config)
        points = np.array([[x, y] for x in np.linspace(-.6, .6, 121) for y in np.linspace(-.6, .6, 121)])
        points = points[(np.linalg.norm(points, axis=1) <= config.radius + 1e-12) & np.all(points @ constraints.T >= -1e-12, axis=1)]
        losses = np.square(residual + points @ matrix.T).sum(axis=1) + config.ridge * np.square(points).sum(axis=1)
        self.assertLessEqual(report["objective"], float(losses.min()) + 1e-9)
        self.assertLessEqual(np.linalg.norm(weights), config.radius + 1e-9)

    def testconerowlimit(self):
        with self.assertRaises(ValueError):
            solve(np.eye(2), np.ones(2), np.ones((9, 2)))
        weights, report = solve(np.eye(2), np.ones(2), np.zeros((9, 2)))
        self.assertEqual(report["faces"], 1)
        self.assertTrue(np.isfinite(weights).all())

    def testchannelmatchedbasis(self):
        base = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(31))
        frequency, _ = build(base, grid=2)
        channel, report = build(base, grid=2, basis="channel")
        self.assertEqual(len(frequency), len(channel))
        self.assertEqual(report["kind"], "channel")
        matrix = torch.stack([value.flatten() for value in channel])
        gram = matrix @ matrix.T / base.numel() / base.square().mean()
        self.assertTrue(torch.allclose(gram, torch.eye(8), atol=2e-5))
        for value, atom in zip(channel, report["atoms"]):
            group = value[:, 2:] if atom["band"] == "first" else value[:, :2]
            self.assertEqual(float(group.abs().max()), 0.)
            self.assertTrue(torch.allclose(value * gates(base, 2)[atom["region"]], value))
            low, high = split(value)
            self.assertGreater(float(low.norm()), 0.)
            self.assertGreater(float(high.norm()), 0.)
        with self.assertRaises(ValueError):
            build(torch.ones(1, 1, 4, 4), basis="channel")

    def testconerealverification(self):
        base, render, score, origin, mask, _ = self.fixture()
        result = select(base, render, score, origin, mask, Config(radius=.5, ridge=1e-6, slack=1e-6))
        self.assertFalse(result.report["fallback"])
        self.assertEqual(result.report["solver"]["lock"], "cone")
        for attempt in result.report["verification"]:
            self.assertEqual(len(attempt["semanticprediction"]), 3)
            self.assertTrue(np.isfinite(attempt["semanticerror"]))
            if attempt["accepted"]:
                self.assertLess(attempt["actual"], result.report["baseline"]["damage"])
        json.dumps(result.report, allow_nan=False)

    def testjointfeedbackcompensation(self):
        base, original, score, origin, mask, _ = self.fixture()

        def render(field):
            left, right = field[..., :2].mean(), field[..., 2:].mean()
            image = original(field)
            image[..., :2, :] -= .1 * (left - right).square()
            return image

        fixed = select(base, render, score, origin, mask, Config(radius=.5, ridge=1e-6, calibration="fixed"))
        feedback = select(base, render, score, origin, mask, Config(radius=.5, ridge=1e-6))
        shrink = select(base, render, score, origin, mask, Config(radius=.5, ridge=1e-6, calibration="shrink"))
        self.assertTrue(fixed.report["fallback"])
        self.assertTrue(shrink.report["fallback"])
        self.assertFalse(feedback.report["fallback"])
        self.assertGreater(feedback.report["feedback"]["updates"], 0)
        self.assertLessEqual(feedback.report["evaluations"], fixed.report["evaluations"])
        self.assertLessEqual(feedback.report["evaluations"], shrink.report["evaluations"])
        self.assertTrue(all(row["fit"] is None for row in shrink.report["feedback"]["models"]))
        self.assertGreaterEqual(score(feedback.image)["score"], score(render(base))["score"] - 1e-10)
        self.assertLess(Protection(origin, mask).energy(feedback.image), fixed.report["baseline"]["damage"])
        self.assertGreater(feedback.report["selectedmodel"], 0)
        selected = [row for row in feedback.report["verification"] if row["model"] > 0]
        self.assertTrue(any(row["semanticimprovement"] > 0 for row in selected))
        self.assertTrue(torch.allclose(feedback.image, render(feedback.field), atol=1e-7))
        json.dumps(feedback.report, allow_nan=False)

    def testfixedlinearagreement(self):
        base, render, score, origin, mask, _ = self.fixture()
        common = {"radius": .5, "ridge": 1e-6, "slack": 1e-6}
        fixed = select(base, render, score, origin, mask, Config(**common, calibration="fixed"))
        feedback = select(base, render, score, origin, mask, Config(**common))
        self.assertEqual(feedback.report["feedback"]["updates"], 0)
        self.assertEqual(feedback.report["evaluations"], fixed.report["evaluations"])
        self.assertTrue(torch.equal(feedback.field, fixed.field))

    def testfeedbackzeroandinvalid(self):
        base, render, score, origin, mask, _ = self.fixture(curved=True)
        fixed = select(base, render, score, origin, mask, Config(updates=0, slack=0.))
        self.assertEqual(fixed.report["feedback"]["updates"], 0)
        for changes in ({"updates": -1}, {"updates": 9}, {"updates": True}, {"defect": -.1}, {"defect": float("nan")}, {"calibration": "other"}, {"factorial": 1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                Config(**changes).validate()

    def testfactorialintervention(self):
        origin = torch.full((1, 3, 4, 4), .3)
        mask = torch.zeros(1, 1, 4, 4)
        mask[..., :2, :] = 1.
        low = torch.ones(1, 1, 4, 4)
        high = torch.tensor([[[[1., -1., 1., -1.], [-1., 1., -1., 1.], [1., -1., 1., -1.], [-1., 1., -1., 1.]]]])

        def render(field, mixed=True):
            first, second = field.mean(), (field * high).mean()
            return origin + .03 * first + .02 * second + (.04 * first * second if mixed else 0.)

        base = torch.zeros_like(low)
        report = factorial(base, [low, high], [{"band": "low"}, {"band": "high"}], np.ones(2), render, Protection(origin, mask), render(base), render(low + high), lambda field: render(field, False))
        self.assertTrue(report["measured"])
        self.assertAlmostEqual(report["protected"], .04, places=6)
        self.assertAlmostEqual(report["rgb"], .04, places=6)
        self.assertAlmostEqual(report["transport"], 0., places=6)
        self.assertAlmostEqual(report["refinementchange"], .04, places=6)
        self.assertEqual(report["evaluations"], 2)
        self.assertEqual(report["decodes"], 6)
        self.assertAlmostEqual(report["overlap"], .0012, places=6)
        self.assertAlmostEqual(report["nonlinear"], .0056, places=6)
        self.assertLess(report["identityerror"], 1e-12)
        missing = factorial(base, [low], [{"band": "low"}], np.ones(1), render, Protection(origin, mask), render(base), render(low))
        self.assertFalse(missing["measured"])

    def testfactorialsemanticconflict(self):
        origin = torch.full((1, 3, 4, 4), .5)
        mask = torch.zeros(1, 1, 4, 4)
        mask[..., :2, :] = 1.
        first = torch.zeros(1, 1, 4, 4)
        first[..., :2] = 1.
        second = 1 - first
        base = torch.zeros_like(first)

        def render(field):
            left, right = field[..., :2].mean(), field[..., 2:].mean()
            image = origin.clone()
            image[..., 2:, :] += .1 - .03 * (left + right)
            image[..., :2, :] += .01 * (left + right) - .05 * left * right
            return image

        def score(image):
            value = float(image[..., :2, :].mean())
            return {"score": value, "global": value}

        report = factorial(base, [first, second], [{"band": "low"}, {"band": "high"}], np.ones(2), render, Protection(origin, mask), render(base), render(first + second), score=score)
        self.assertTrue(report["singlevalidjointinvalid"])
        self.assertTrue(all(row["accepted"] for row in report["interventions"][:2]))
        self.assertFalse(report["interventions"][2]["semanticvalid"])
        self.assertTrue(report["interventions"][2]["improves"])
        self.assertAlmostEqual(report["semanticinteraction"]["score"], -.05, places=6)
        json.dumps(report, allow_nan=False)

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
