# -*- coding: utf-8 -*-
# RefUnmix: Reference-Grounded Response Unmixing for One-Step Image Editing

import unittest
import numpy as np
from scipy.optimize import lsq_linear
from engine.coupling import bounded

class CouplingTests(unittest.TestCase):
    def testprojection(self):
        rng = np.random.default_rng(42)
        for count in (1, 2):
            for index in range(30):
                matrix = rng.normal(size=(30, count))
                target = rng.normal(size=30)
                gram, linear = matrix.T @ matrix, matrix.T @ target
                actual = bounded(gram, linear)
                reference = lsq_linear(matrix, target, bounds=(0, 1)).x
                self.assertLessEqual(np.linalg.norm(matrix @ actual - target), np.linalg.norm(matrix @ reference - target) + 1e-7)
                self.assertTrue(np.all((actual >= 0) & (actual <= 1)))

    def testdegenerate(self):
        self.assertTrue(np.array_equal(bounded(np.zeros((2, 2)), np.zeros(2)), np.zeros(2)))
        gram = np.ones((2, 2))
        actual = bounded(gram, np.array([.5, .5]))
        self.assertAlmostEqual(actual.sum(), .5)

    def testdamage(self):
        rng = np.random.default_rng(3407)
        for index in range(40):
            origin = rng.uniform(size=60)
            candidate = rng.uniform(size=60)
            common = rng.normal(size=(60, 2))
            weight = rng.uniform(size=60)
            residual = candidate - origin
            gram = common.T @ (weight[:, None] * common)
            linear = common.T @ (weight * residual)
            value = bounded(gram, linear)
            corrected = np.clip(candidate - common @ value, 0, 1)
            self.assertLessEqual(np.sum(weight * (corrected - origin) ** 2), np.sum(weight * residual ** 2) + 1e-10)

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
