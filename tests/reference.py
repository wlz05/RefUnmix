# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import unittest
import torch
from engine.reference import center, decomposition, safeguard

class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.origin = torch.full((1, 3, 4, 4), .5)
        self.mask = torch.zeros(1, 1, 4, 4)

    def testidentity(self):
        neutral = self.origin + .17
        self.assertTrue(torch.equal(center(self.origin, neutral, neutral), self.origin))

    def testaffine(self):
        bias = torch.full_like(self.origin, .2)
        response = torch.linspace(-.1, .1, self.origin.numel()).reshape_as(self.origin)
        neutral = self.origin + bias
        result = center(self.origin, neutral + response, neutral)
        self.assertTrue(torch.allclose(result, self.origin + response, atol=1e-7))

    def testunreachable(self):
        neutral = self.origin.clone()
        neutral[..., 2:, :] += .2
        response = torch.zeros_like(self.origin)
        response[..., :2, :] = .1
        mask = self.mask.clone()
        mask[..., :2, :] = 1
        for scale in (0., .5, 1.):
            report = decomposition(self.origin, neutral + scale * response, neutral, mask)
            self.assertAlmostEqual(report["rawdamage"], .04, places=6)
            self.assertEqual(report["centerdamage"], 0.)
            self.assertTrue(torch.allclose(center(self.origin, neutral + scale * response, neutral), self.origin + scale * response, atol=1e-7))

    def testscale(self):
        neutral = self.origin + .2
        response = torch.full_like(self.origin, .1)
        original = decomposition(self.origin, neutral + response, neutral, self.mask)
        tiny = decomposition(self.origin, neutral + .01 * response, neutral, self.mask)
        self.assertGreater(tiny["rawdamage"], .039)
        self.assertAlmostEqual(tiny["centerdamage"] / original["centerdamage"], .0001, places=7)

    def testclamp(self):
        neutral, edited = self.origin + .7, self.origin + .8
        correct = center(self.origin, edited, neutral)
        incorrect = center(self.origin, edited.clamp(0, 1), neutral.clamp(0, 1))
        self.assertTrue(torch.allclose(correct, self.origin + .1, atol=1e-7))
        self.assertTrue(torch.equal(incorrect, self.origin))

    def testmixed(self):
        def raw(left, right):
            return self.origin + .1 + .03 * left + .04 * right + .02 * left * right

        neutral = raw(0, 0)
        corners = [raw(a, b) for a, b in ((0, 0), (1, 0), (0, 1), (1, 1))]
        centered = [center(self.origin, value, neutral) for value in corners]
        mixed = lambda values: values[3] - values[1] - values[2] + values[0]
        self.assertTrue(torch.allclose(mixed(corners), mixed(centered), atol=1e-7))
        self.assertGreater(float(mixed(centered).abs().max()), .019)

    def testdecomposition(self):
        neutral, edited = self.origin + .2, self.origin + .05
        report = decomposition(self.origin, edited, neutral, self.mask)
        self.assertLess(report["protected"]["cross"], 0)
        self.assertLess(report["protected"]["identityerror"], 1e-12)
        self.assertLess(report["unclipped"]["identityerror"], 1e-12)
        self.assertGreater(report["centerdamage"], report["rawdamage"])

    def testsaturation(self):
        neutral, edited = self.origin + .7, self.origin + 1.5
        report = decomposition(self.origin, edited, neutral)
        self.assertEqual(report["saturation"], 1.)
        self.assertTrue(torch.equal(center(self.origin, edited, neutral), torch.ones_like(self.origin)))

    def testsemanticfallback(self):
        fallback, candidate = self.origin + .2, self.origin.clone()
        image, report = safeguard(candidate, fallback, self.origin, self.mask, {"score": .1}, {"score": .2})
        self.assertFalse(report["accepted"])
        self.assertTrue(report["improves"])
        self.assertTrue(torch.equal(image, fallback))

    def testlocalfallback(self):
        scores = {"score": .3, "local0": .4, "local1": .1}
        raw = {"score": .2, "local0": .2, "local1": .2}
        _, report = safeguard(self.origin, self.origin + .1, self.origin, self.mask, scores, raw)
        self.assertFalse(report["semanticvalid"])

    def testaccepted(self):
        candidate = self.origin + .03
        image, report = safeguard(candidate, self.origin + .1, self.origin, self.mask, {"score": .2}, {"score": .2})
        self.assertTrue(report["accepted"])
        self.assertTrue(torch.equal(image, candidate))

    def testdamagefallback(self):
        _, report = safeguard(self.origin + .2, self.origin + .1, self.origin, self.mask, {"score": .4}, {"score": .2})
        self.assertFalse(report["accepted"])
        self.assertTrue(report["semanticvalid"])

    def testemptyprotection(self):
        _, report = safeguard(self.origin, self.origin + .1, self.origin, torch.ones_like(self.mask), {"score": .2}, {"score": .2})
        self.assertFalse(report["accepted"])
        self.assertIsNone(report["damagechange"])
        decompositionreport = decomposition(self.origin, self.origin, self.origin, torch.ones_like(self.mask))
        self.assertEqual(decompositionreport["protected"], {})
        self.assertIsNone(decompositionreport["centerdamage"])

    def testtolerance(self):
        for value in (-1, True, float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                safeguard(self.origin, self.origin, self.origin, self.mask, {"score": .2}, {"score": .2}, slack=value)
        with self.assertRaises(ValueError):
            safeguard(self.origin, self.origin, self.origin, self.mask, {"score": .2}, {"score": .2, "local": .2})

    def testvalidation(self):
        for value in (self.origin.double() * float("nan"), self.origin[:, :1], self.origin[..., :2], self.origin.long()):
            with self.subTest(shape=value.shape), self.assertRaises(ValueError):
                center(self.origin, value, self.origin)
        with self.assertRaises(ValueError):
            center(self.origin + 1, self.origin, self.origin)

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
