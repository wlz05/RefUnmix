# -*- coding: utf-8 -*-
# RefUnmix: Reference-Grounded Response Unmixing for One-Step Image Editing

import unittest
import torch
from engine.grounding import restrict

class Context(unittest.TestCase):
    def testretained(self):
        source = torch.full((1, 3, 8, 8), 100 / 255)
        target = torch.full_like(source, 140 / 255)
        scope = torch.zeros_like(source[:, :1])
        pixels, report = restrict(source, target, scope, .25)
        torch.testing.assert_close(pixels, torch.full_like(source, 110 / 255), rtol=0, atol=0)
        self.assertEqual(report["retainedcontext"], .25)

    def testlimit(self):
        generator = torch.Generator().manual_seed(42)
        source, target = [torch.rand((1, 3, 8, 8), generator=generator) for _ in range(2)]
        scope = torch.rand((1, 1, 8, 8), generator=generator)
        pixels, _ = restrict(source, target, scope, 1.)
        torch.testing.assert_close((pixels * 255).round(), (target * 255).round(), rtol=0, atol=0)

    def testbudget(self):
        generator = torch.Generator().manual_seed(42)
        source, target = [torch.rand((1, 3, 17, 19), generator=generator) for _ in range(2)]
        scope = torch.rand((1, 1, 17, 19), generator=generator)
        before = ((target * 255).round() - (source * 255).round()).double().square().sum(1)
        for retained in (0., .15, .5, 1.):
            pixels, report = restrict(source, target, scope, retained)
            after = ((pixels * 255).round() - (source * 255).round()).double().square().sum(1)
            self.assertTrue(torch.all(after <= before))
            self.assertLessEqual(report["digitalerrorafter"], report["digitalerrorbefore"])

    def testinvalid(self):
        image = torch.zeros((1, 3, 8, 8))
        for value in (-.1, 1.1, float("nan"), float("inf"), True, "0.1"):
            with self.assertRaises(ValueError):
                restrict(image, image, image[:, :1], value)

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
