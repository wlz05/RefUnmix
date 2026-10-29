# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import json
import unittest
import numpy as np
import torch
from engine.response import Config, Protection, region, select, solve
from engine.wavelet import dictionary, gates, sources, split

class WaveletTest(unittest.TestCase):
    def testreconstruction(self):
        for shape in ((1, 3, 8, 6), (1, 2, 7, 5), (1, 1, 1, 3)):
            value = torch.randn(shape, generator=torch.Generator().manual_seed(9))
            low, high = split(value)
            torch.testing.assert_close(low + high, value)
            torch.testing.assert_close(split(low)[0], low)
            self.assertLess(abs(float((low * high).sum())), 2e-5)

    def testlattice(self):
        value = torch.randn(1, 2, 7, 9)
        windows = gates(value, 2)
        torch.testing.assert_close(sum(windows), torch.ones(1, 1, 7, 9))
        low, high = split(value)
        for gate in windows:
            self.assertLess(float(split(low * gate)[1].abs().max()), 1e-7)
            self.assertLess(float(split(high * gate)[0].abs().max()), 2e-7)

    def testpixelgate(self):
        low = torch.ones(1, 1, 2, 2)
        pixel = torch.tensor([[[[1., 0.], [1., 0.]]]])
        self.assertGreater(float(split(low * pixel)[1].norm()), 0)

    def testalltimes(self):
        base = torch.zeros(1, 1, 4, 4)
        pool = {0.72: torch.ones_like(base), 0.76: torch.arange(16).float().reshape_as(base)}
        atoms = dictionary(base, pool[0.72], pool, 2)
        self.assertIn("0.76", {atom.time for atom in atoms})
        for atom in atoms:
            other = split(atom.value)[1 if atom.band == "low" else 0]
            self.assertLess(float(other.abs().max()), 1e-7)
        restricted = dictionary(base, pool[0.72], pool, 2, "high")
        self.assertTrue(all(atom.band == "high" for atom in restricted))

    def testinvalid(self):
        value = torch.zeros(1, 1, 2, 2)
        for time in (0, -0.2, float("nan"), 1.1, True):
            with self.assertRaises(ValueError):
                sources(value, {time: value})
        with self.assertRaises(ValueError):
            gates(value, 2)
        with self.assertRaises(ValueError):
            split(torch.zeros(2, 1, 2, 2))

class ResponseSolverTest(unittest.TestCase):
    def testjointcancellation(self):
        gradient = np.ones(2)
        full = np.array([[1., -1.], [-1., 1.]])
        config = Config(radius=1, active=2, sparse=0, ridge=0.001)
        weights = solve(gradient, full, np.zeros(2), 0, 0, ["low", "high"], config)
        np.testing.assert_allclose(weights, [0.5, 0.5], atol=1e-7)
        diagonal = solve(gradient, np.eye(2), np.zeros(2), 0, 0, ["low", "high"], config)
        np.testing.assert_allclose(diagonal, [0, 0])

    def testbaselinecross(self):
        config = Config(radius=1, sparse=0, ridge=0)
        weights = solve(np.array([1.]), np.array([[1.]]), np.array([-1.]), 1, 1, ["low"], config)
        self.assertAlmostEqual(float(weights[0]), 1)

    def testconstraints(self):
        generator = np.random.default_rng(6)
        for _ in range(10):
            response = generator.normal(size=(7, 4))
            original = generator.normal(size=7)
            matrix = response.T @ response
            linear = response.T @ original
            initial = float(original @ original)
            config = Config(radius=1.5, active=2, sparse=0)
            weights = solve(np.ones(4), matrix, linear, initial, initial + 0.1, ["a", "a", "b", "b"], config)
            self.assertGreaterEqual(weights.min(), 0)
            self.assertLessEqual(weights.sum(), config.radius + 1e-9)
            self.assertLessEqual((weights > 1e-10).sum(), 2)
            self.assertLessEqual(weights[:2].sum(), 1 + 1e-9)
            self.assertLessEqual(weights[2:].sum(), 1 + 1e-9)
            self.assertLessEqual(initial + 2 * linear @ weights + weights @ matrix @ weights, initial + 0.10000001)

    def testzerobenefit(self):
        weights = solve(np.array([-1., 0.]), np.eye(2), np.zeros(2), 0, 1, ["a", "b"], Config())
        np.testing.assert_array_equal(weights, [0, 0])

    def testinvalid(self):
        for config in (Config(step=0), Config(step=1, radius=0.5), Config(budget=-1), Config(active=True), Config(grid=0), Config(coupling="unknown")):
            with self.assertRaises(ValueError):
                config.validate()
        with self.assertRaises(ValueError):
            solve(np.ones(1), np.array([[-1.]]), np.zeros(1), 0, 1, ["a"], Config())

class ProtectionTest(unittest.TestCase):
    def testauditidentity(self):
        origin = torch.ones(1, 3, 8, 8) * 0.5
        mask = torch.rand(1, 1, 4, 4)
        protect = Protection(origin, mask)
        result = (origin + torch.randn_like(origin) * 0.1).clamp(0, 1)
        audit = protect.audit(result)
        self.assertEqual(set(audit), {"detail", "cross", "coarse", "mse", "coverage"})
        self.assertAlmostEqual(audit["mse"], audit["detail"] + audit["cross"] + audit["coarse"], places=7)

    def testsignedcross(self):
        origin = torch.ones(1, 3, 2, 2) * 0.5
        mask = torch.tensor([[[[0., 1.], [1., 1.]]]])
        residual = torch.tensor([[[[-0.1, 0.3], [0.3, 0.3]]]]).expand_as(origin)
        audit = Protection(origin, mask).audit(origin + residual)
        self.assertLess(audit["cross"], 0)
        self.assertAlmostEqual(audit["cross"], -0.12, places=7)
        self.assertAlmostEqual(audit["mse"], audit["detail"] + audit["cross"] + audit["coarse"], places=7)
        unweighted = Protection(origin, torch.zeros_like(mask)).audit(origin + residual)
        self.assertAlmostEqual(unweighted["cross"], 0, places=7)

    def testmissingprotection(self):
        origin = torch.zeros(1, 3, 2, 2)
        protect = Protection(origin, torch.ones(1, 1, 2, 2))
        self.assertIsNone(protect.energy(origin))
        self.assertIsNone(protect.audit(origin)["mse"])
        self.assertFalse(protect.valid)

    def testareanormalization(self):
        origin = torch.zeros(1, 3, 4, 4)
        full = Protection(origin, torch.zeros(1, 1, 4, 4))
        mask = torch.ones(1, 1, 4, 4)
        mask[..., :2, :] = 0
        half = Protection(origin, mask)
        self.assertAlmostEqual(full.energy(origin + 0.1), half.energy(origin + 0.1), places=7)

    def testfrozenmask(self):
        base = torch.zeros(1, 1, 4, 4)
        self.assertEqual(float(region(base).sum()), 0)
        with self.assertRaises(ValueError):
            region(base, torch.ones(1, 2, 4, 4))

class SelectionTest(unittest.TestCase):
    def setUp(self):
        self.base = torch.zeros(1, 1, 4, 4)
        self.origin = torch.ones(1, 3, 4, 4) * 0.5
        self.mask = torch.ones_like(self.base)

    def render(self, field):
        return (self.origin + 0.1 * field.expand(-1, 3, -1, -1)).clamp(0, 1)

    def score(self, image):
        return {"score": float(image.mean())}

    def testlocalotherstime(self):
        target = self.base.clone()
        target[..., :2, :2] = 1
        target[..., 2:, :] = -2
        candidate = select(self.base, self.base, {0.72: target, 0.76: -target}, self.render, self.score, self.origin, self.mask, Config(bands="low", sparse=0, ridge=0))
        self.assertGreater(candidate.report["localgain"], 0)
        self.assertTrue(any(weight > 0 and atom["time"] in {"0.72", "0.76"} for weight, atom in zip(candidate.report["weights"], candidate.report["atoms"])))
        self.assertGreaterEqual(candidate.report["gain"], 0)
        self.assertEqual(candidate.report["schema"], 3)
        self.assertEqual(candidate.report["auditkind"], "protected RGB Haar residual energy")
        self.assertEqual(set(candidate.report["audit"]), {"detail", "cross", "coarse", "mse", "coverage"})
        json.dumps(candidate.report, allow_nan=False)

    def testonlyhigh(self):
        value = torch.tensor([[[[1., -1., 1., -1.]]]]).expand(1, 1, 4, 4)

        def objective(image):
            return {"score": float((image[:, :1] * value).mean())}

        candidate = select(self.base, self.base, {0.72: value}, self.render, objective, self.origin, self.mask, Config(bands="high", local=False))
        self.assertGreater(candidate.report["gain"], 0)
        self.assertLess(float(split(candidate.field)[0].abs().max()), 1e-7)

    def testbudgetfallback(self):
        candidate = select(self.base, self.base, {0.72: torch.ones_like(self.base)}, self.render, self.score, self.origin, torch.zeros_like(self.base), Config(budget=0, sparse=0))
        torch.testing.assert_close(candidate.image, self.origin)
        self.assertEqual(candidate.report["selected"], "base")
        self.assertEqual(candidate.report["audit"]["mse"], 0)

    def testnonlinearverification(self):
        field = torch.ones_like(self.base)

        def nonlinear(value):
            return self.origin + 0.1 * value.expand(-1, 3, -1, -1).square()

        candidate = select(self.base, self.base, {0.72: field}, nonlinear, self.score, self.origin, torch.zeros_like(self.base), Config(globalstage=False, grid=1, budget=0.00001, sparse=0))
        self.assertLessEqual(candidate.report["audit"]["mse"], candidate.report["ceiling"] + 1e-10)
        self.assertGreaterEqual(candidate.report["score"], 0.5)

    def testzeroedit(self):
        candidate = select(self.base, self.base, {0.72: self.base}, self.render, self.score, self.origin, self.mask)
        self.assertEqual(candidate.report["atoms"], [])
        self.assertEqual(candidate.report["selected"], "base")
        self.assertEqual(candidate.report["evaluations"], 1)

    def testnonfinitejudge(self):
        with self.assertRaises(ValueError):
            select(self.base, self.base, {0.72: self.base}, self.render, lambda image: float("nan"), self.origin, self.mask)

    def testinvalidrgb(self):
        with self.assertRaises(ValueError):
            select(self.base, self.base, {0.72: self.base}, lambda field: self.origin * 3, self.score, self.origin, self.mask)

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
