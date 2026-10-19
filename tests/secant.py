# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import json
import unittest
import numpy as np
from engine.secant import complete, defect

class SecantTests(unittest.TestCase):
    def testexactchords(self):
        initial = np.array([[1., 2., 3.], [-1., .5, 2.]])
        directions = np.array([[1., 0.], [0., 1.], [1., 1.]])
        observed = np.array([[5., 7.], [2., -3.]])
        fitted, report = complete(initial, directions, observed, np.eye(3))
        np.testing.assert_allclose(fitted @ directions, observed, atol=1e-10)
        complement = np.array([-1., -1., 1.])
        np.testing.assert_allclose(fitted @ complement, initial @ complement, atol=1e-10)
        self.assertEqual(report["rank"], 2)
        self.assertLess(report["fiterror"], 1e-10)
        json.dumps(report, allow_nan=False)

    def testphysicalinvariance(self):
        initial = np.array([[1., 2.], [3., -1.]])
        directions, observed = np.array([[.3], [-.2]]), np.array([[1.], [2.]])
        metric, transform = np.array([[2., .3], [.3, 1.]]), np.array([[2., .4], [0., .5]])
        fitted, first = complete(initial, directions, observed, metric)
        changed, second = complete(initial @ transform, np.linalg.solve(transform, directions), observed, transform.T @ metric @ transform)
        np.testing.assert_allclose(changed, fitted @ transform, atol=1e-10)
        self.assertAlmostEqual(first["change"], second["change"], places=9)
        complement = np.array([.11, .54])
        self.assertAlmostEqual((directions.T @ metric @ complement).item(), 0., places=10)
        np.testing.assert_allclose(fitted @ complement, initial @ complement, atol=1e-10)

    def testlatestdependentchord(self):
        directions, observed = np.array([[1., .5, 0.], [0., 0., 0.]]), np.array([[1., 2., 0.]])
        fitted, report = complete(np.zeros((1, 2)), directions, observed, np.eye(2))
        np.testing.assert_allclose(fitted, [[4., 0.]], atol=1e-10)
        self.assertEqual(report["kept"], [1])
        self.assertEqual(report["pruned"], [0, 2])

    def testemptyobservations(self):
        initial = np.eye(2)
        fitted, report = complete(initial, np.empty((2, 0)), np.empty((2, 0)), np.eye(2))
        np.testing.assert_array_equal(fitted, initial)
        self.assertEqual(report["rank"], 0)
        self.assertEqual(defect(np.zeros(2), np.zeros(2))["relative"], 0.)
        self.assertEqual(defect(np.ones(2), -np.ones(2))["relative"], 2.)

    def testinvaliddata(self):
        cases = [(np.eye(2), np.ones((3, 1)), np.ones((2, 1)), np.eye(3), 1e-6), (np.eye(2), np.ones((2, 1)), np.ones((2, 1)), np.diag([1., -1.]), 1e-6), (np.eye(2), np.ones((2, 1)), np.full((2, 1), np.nan), np.eye(2), 1e-6), (np.eye(2), np.ones((2, 1)), np.ones((2, 1)), np.eye(2), 0.)]
        for arrays in cases:
            with self.assertRaises(ValueError):
                complete(*arrays)
        with self.assertRaises(ValueError):
            defect(np.ones(2), np.ones(3))

    def testneardependentchords(self):
        directions = np.array([[1., 1.], [0., 1e-8]])
        fitted, report = complete(np.zeros((1, 2)), directions, np.array([[1., 2.]]), np.eye(2))
        self.assertEqual(report["kept"], [1])
        np.testing.assert_allclose(fitted @ directions[:, 1], [2.], atol=1e-10)

    def testauditmechanism(self):
        from audit import values
        report = {"method": "projection", "feedback": {"updates": 1}, "verification": [{"accepted": False, "model": 0, "responseimprovement": 0.}, {"accepted": True, "model": 1, "responseimprovement": .2, "semanticimprovement": .01}], "factorial": {"singlevalidjointinvalid": True, "semanticinteraction": {"score": -.03}, "nonlinear": .1}}
        row = values({"report": report})
        self.assertEqual(row["projection:updates"], 1.)
        self.assertEqual(row["projection:responseimprovement"], .2)
        self.assertEqual(row["factorial:singlevalidjointinvalid"], 1.)
        self.assertEqual(row["factorial:semantic:score"], -.03)

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
