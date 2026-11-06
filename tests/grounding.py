# -*- coding: utf-8 -*-
# RefUnmix: Reference-Grounded Response Unmixing for One-Step Image Editing

import argparse
import unittest
import torch
from engine.grounding import reference, support, supply, restrict
from run import common, validate
from engine.serial import finite
from session import editor
from PIL import Image

class GroundingTests(unittest.TestCase):
    def testconstant(self):
        for value in (0., .4):
            self.assertTrue(torch.equal(support(torch.full((1, 1, 4, 4), value), (8, 8)), torch.zeros(1, 1, 8, 8)))

    def testlocalidentity(self):
        origin = torch.rand(1, 3, 4, 4)
        decoded = torch.rand_like(origin)
        attention = torch.zeros(1, 1, 4, 4)
        attention[:, :, :, 2:] = 1
        result, weight = reference(origin, decoded, attention)
        self.assertTrue(torch.equal(result[:, :, :, :2], decoded[:, :, :, :2]))
        self.assertTrue(torch.allclose(result[:, :, :, 2:], origin[:, :, :, 2:]))
        same, _ = reference(origin, origin, attention)
        self.assertTrue(torch.equal(same, origin))
        self.assertTrue(torch.equal(weight, support(7 * attention, (4, 4))))

        decoded = 3 * decoded - 1
        result, _ = reference(origin, decoded, attention)
        self.assertTrue(torch.equal(result[:, :, :, :2], decoded[:, :, :, :2]))

    def testinvalid(self):
        origin = torch.rand(1, 3, 4, 4)
        with self.assertRaises(ValueError):
            reference(origin, origin, torch.full((1, 1, 4, 4), float("nan")))
        with self.assertRaises(ValueError):
            reference(origin + 2, origin, torch.ones(1, 1, 4, 4))
        with self.assertRaises(ValueError):
            support(torch.ones(1, 2, 4, 4), (4, 4))
        with self.assertRaises(ValueError):
            reference(origin, origin[:,:,:2], torch.zeros(1, 1, 4, 4))

    def testserialization(self):
        session = editor().prepare(Image.new("RGB", (16, 16)), "source", "target")
        session.attentionlog[749] = {"maps": 3, "tokens": [[1], [2]]}
        report = session.stats()
        finite(report)
        self.assertEqual(report["attention"]["749"]["maps"], 3)

    def testallocation(self):
        weight = torch.rand(1, 1, 8, 8)
        self.assertTrue(torch.equal(supply(weight, torch.zeros_like(weight)), weight))
        self.assertTrue(torch.equal(supply(weight, torch.ones_like(weight)), torch.ones_like(weight)))
        scope = torch.rand_like(weight)
        result = supply(weight, scope)
        self.assertTrue(torch.all(result >= weight) and torch.all(result <= 1))
        with self.assertRaises(ValueError):
            supply(weight, scope * float("nan"))
        with self.assertRaises(ValueError):
            supply(weight, scope[:, :, :4])

    def testcontraction(self):
        origin, pixels = torch.rand(1, 3, 8, 8), torch.rand(1, 3, 8, 8)
        scope = torch.rand(1, 1, 8, 8)
        result, report = restrict(origin, pixels, scope)
        source = (origin * 255).round()
        before = ((pixels * 255).round() - source).square().sum(1)
        after = ((result * 255).round() - source).square().sum(1)
        self.assertTrue(torch.all(after <= before))
        self.assertLessEqual(report["digitalerrorafter"], report["digitalerrorbefore"])
        zero, _ = restrict(origin, pixels, torch.zeros_like(scope))
        full, _ = restrict(origin, pixels, torch.ones_like(scope))
        self.assertTrue(torch.equal((zero * 255).round(), source))
        self.assertTrue(torch.equal((full * 255).round(), (pixels * 255).round()))
        with self.assertRaises(ValueError):
            restrict(origin, pixels, scope + 2)

    def testoptions(self):
        args = common(argparse.ArgumentParser()).parse_args(["--single", "--singlemode", "contrast", "--singleend", "responsibility", "--recenter", "0", "--no-singlebias", "--noise", "1", "--delta", "0", "--method", "raw", "--no-control", "--referencelocal"])
        validate(args)
        args.noise = 2
        with self.assertRaises(ValueError):
            validate(args)

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
