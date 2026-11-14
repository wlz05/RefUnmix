# -*- coding: utf-8 -*-
# RefUnmix: Reference-Grounded Response Unmixing for One-Step Image Editing

import unittest
import torch
from engine.attraction import attract

class Attraction(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(42)
        self.field, self.anchor, self.target = [torch.randn((1, 4, 8, 8), generator=generator) for _ in range(3)]

    def testidentity(self):
        result, report = attract(self.field, self.anchor, self.target, 0.)
        self.assertIs(result, self.field)
        self.assertEqual(report["extraqueries"], 0)

    def testformula(self):
        result, report = attract(self.field, self.anchor, self.target, .5)
        torch.testing.assert_close(result, self.field + .5 * (self.target - self.anchor), rtol=0, atol=0)
        self.assertEqual(report["weight"], .5)

    def teststationary(self):
        result, _ = attract(self.field, self.anchor, self.anchor, 1.)
        torch.testing.assert_close(result, self.field, rtol=0, atol=0)

    def testinvalid(self):
        for weight in (-.1, 1.1, float("nan"), float("inf"), True, "0.5"):
            with self.assertRaises(ValueError):
                attract(self.field, self.anchor, self.target, weight)
        with self.assertRaises(ValueError):
            attract(self.field, self.anchor, self.target[:, :, :-1])
        bad = self.target.clone()
        bad[0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            attract(self.field, self.anchor, bad)

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yeming Dai, Yaming Lv
# Email: wlzchina05@gmail.com
