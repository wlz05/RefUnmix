# -*- coding: utf-8 -*-
# RefUnmix: Reference-Grounded Response Unmixing for One-Step Image Editing

import unittest
import torch
from engine import fixed as module

class FixedAdoption(unittest.TestCase):
    def test_unit_gain(self):
        torch.manual_seed(42)
        x = torch.rand(1, 3, 9, 11)
        d, a = torch.randn_like(x), torch.randn_like(x)
        q = torch.rand(1, 1, 9, 11)
        actual, report = module.resolve(x, d, a, q, 1.)
        expected = ((x.double() + d.double() + q.double() * a.double()).clamp(0, 1) * 255).round().float() / 255
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(report["effective"], 1.)

    def test_cancellation_allows_gain(self):
        x = torch.full((1, 3, 1, 1), .5)
        d, a = torch.full_like(x, .1), torch.full_like(x, -.3)
        result, report = module.resolve(x, d, a, torch.ones_like(x[:, :1]), 3.)
        self.assertAlmostEqual(report["effective"], 3.)
        self.assertTrue(torch.all((result - x).abs() < 1 / 255))

    def test_aligned_response(self):
        x = torch.full((1, 3, 1, 1), .5)
        d = torch.full_like(x, .1)
        _, report = module.resolve(x, d, d, torch.ones_like(x[:, :1]), 4.)
        self.assertEqual(report["effective"], 1.)

    def test_grid_oracle(self):
        torch.manual_seed(81)
        x = torch.full((1, 3, 1, 1), .5)
        for _ in range(40):
            d, a = torch.randn_like(x), torch.randn_like(x)
            q = torch.rand_like(x[:, :1])
            _, report = module.resolve(x, d, a, q, 4.)
            grid = torch.linspace(1, 4, 30001, dtype=torch.float64)
            base = (d.double() + q.double() * a.double()).square().sum()
            vectors = grid[:, None] * d.double().flatten() + q.double().item() * a.double().flatten()
            feasible = vectors.square().sum(1) <= base + 1e-12
            maximum = grid[feasible][-1].item()
            self.assertLessEqual(abs(report["effective"] - maximum), 1.1e-4)

    def test_digital_budget(self):
        torch.manual_seed(19)
        for _ in range(20):
            x = torch.rand(1, 3, 17, 19)
            d, a = torch.randn_like(x), torch.randn_like(x)
            q = torch.rand(1, 1, 17, 19)
            result, _ = module.resolve(x, d, a, q, 4.)
            baseline = ((x.double() + d.double() + q.double() * a.double()).clamp(0, 1) * 255).round()
            original = (x.double() * 255).round()
            before = (baseline - original).square().sum(1)
            after = ((result.double() * 255).round() - original).square().sum(1)
            self.assertTrue(torch.all(after <= before))

    def test_zero_conditional(self):
        x = torch.full((1, 3, 2, 2), .5)
        result, report = module.resolve(x, torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x[:, :1]), 4.)
        self.assertEqual(report["effective"], 1.)
        self.assertTrue(torch.isfinite(result).all())

    def test_invalid(self):
        x = torch.zeros(1, 3, 2, 2)
        for gain in [True, 0., 5., float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                module.resolve(x, x, x, x[:, :1], gain)
        with self.assertRaises(ValueError):
            module.resolve(x, x, x, torch.full_like(x[:, :1], 2.))

if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yeming Dai, Yaming Lv
# Email: wlzchina05@gmail.com
