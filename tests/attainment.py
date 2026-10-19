# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import argparse
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from engine.attainment import capacity, choose, gain, maximize, propose
from engine.limit import Exhausted
from engine.poll import Config, select
from run import common, validate
from trial import variants

class AttainmentTests(unittest.TestCase):
    def row(self, key="raw", value=0., cost=.1, feasible=True, complete=True):
        return SimpleNamespace(key=key, taskgain=value, objective=cost, feasible=feasible, complete=complete)

    def model(self):
        return SimpleNamespace(anchor=np.array([1., 1., .5]), active=np.array([0, 2]), metric=np.eye(2), root=np.ones(2), residual=np.array([.2]), damage=.04, response=np.zeros((1, 2)), gram=np.zeros((2, 2)), cross=np.zeros(2), scores=np.zeros(2), semantic=np.array([[1., .5], [-1., .5]]), penalty=0., slope=np.zeros(2))

    def testbottleneck(self):
        self.assertEqual(gain([2., -.1], [0., 0.], [1., 1.]), -.1)
        self.assertEqual(gain([2., -.1, -100.], [0., 0., 0.], [2., .1, 1.], [0, 1]), -1.)

    def testgainvalidation(self):
        for values in (([1], [0, 0], [1]), ([np.nan], [0], [1]), ([1], [0], [0]), ([1], [0], [1], []), ([1], [0], [1], [True]), ([1], [0], [1], [0, 0]), ([1], [0], [1], [1])):
            with self.subTest(values=values), self.assertRaises(ValueError):
                gain(*values)

    def testpuresemanticgain(self):
        raw, improved = self.row(), self.row("edit", .2)
        selected, report = choose([raw, improved])
        self.assertIs(selected, improved)
        self.assertEqual(report["gainfloor"], .2)
        self.assertIs(choose([raw, improved], mode="protect")[0], raw)

    def testnomaskingfailure(self):
        raw, partial = self.row(), self.row("partial", -.1, 0.)
        self.assertIs(choose([raw, partial])[0], raw)

    def testboundedband(self):
        raw, high, near = self.row(), self.row("high", .2, .09), self.row("near", .19, .01)
        selected, report = choose([raw, high, near], .02)
        self.assertIs(selected, near)
        self.assertGreaterEqual(selected.taskgain, report["bestgain"] - .02)

    def testnoexcessivepriorveto(self):
        raw, edited = self.row(), self.row("edited", .2, .11)
        self.assertIs(choose([raw, edited])[0], edited)

    def testnobaselinedegradation(self):
        raw, worse = self.row(), self.row("worse", -.001, 0.)
        self.assertIs(choose([raw, worse], 1.)[0], raw)

    def testincompleteexclusion(self):
        raw, edited = self.row(), self.row("edited", 10., 0., complete=False)
        self.assertIs(choose([raw, edited])[0], raw)

    def testselectionvalidation(self):
        for loss in (-1, np.nan, True):
            with self.assertRaises(ValueError):
                choose([self.row()], loss)
        with self.assertRaises(ValueError):
            choose([])
        with self.assertRaises(ValueError):
            choose([self.row(cost=np.inf)])

    def testjointnotseparate(self):
        model = self.model()
        full = maximize(model, .5, [-1., -1.], .04, [0., 0.], [1., 1.], [0, 1])
        field = maximize(model, .5, [-1., -1.], .04, [0., 0.], [1., 1.], [0, 1], columns=np.array([0]))
        self.assertAlmostEqual(full["gain"], .25, places=5)
        self.assertAlmostEqual(field["gain"], 0., places=6)
        self.assertGreaterEqual(full["upper"], .25 - 1e-7)
        self.assertLess(full["upper"] - full["lower"], 1e-5)

    def testbalancedcapacity(self):
        report = capacity(self.model(), .5, [-1., -1.], .04, [0., 0.], [1., 1.], [0, 1])
        self.assertTrue(report["verified"])
        self.assertTrue(report["positiveidentified"])
        self.assertLessEqual(report["incrementlower"], .25)
        self.assertGreaterEqual(report["incrementupper"], .25)
        json.dumps(report, allow_nan=False)

    def testgainphase(self):
        result = propose(self.model(), .5, [-1., -1.], [1., 1.], .04, 1., .004, 1e-6, [0., 0.], [0, 1])
        self.assertEqual(result["phase"], "attainment")
        self.assertLess(result["predicted"], result["start"])

    def testallfloorsretained(self):
        model = self.model()
        model.semantic = np.vstack((model.semantic, [0., -1.]))
        model.scores = np.zeros(3)
        solved = maximize(model, .5, [-1., -1., 0.], .04, [0., 0., 0.], [1., 1., 1.], [0, 1])
        self.assertLessEqual(solved["gain"], 1e-7)

    def testfixedcolumns(self):
        solved = maximize(self.model(), .5, [-1., -1.], .04, [0., 0.], [1., 1.], [0, 1], columns=np.array([], dtype=int))
        self.assertTrue(solved["verified"])
        self.assertEqual((solved["lower"], solved["upper"]), (0., 0.))

    def testmissingnotzero(self):
        with patch("engine.attainment.solve", return_value={"point": None, "dualbound": None, "iterations": 1}):
            report = capacity(self.model(), .5, [-1., -1.], .04, [0., 0.], [1., 1.], [0, 1])
        self.assertFalse(report["verified"])
        self.assertIsNone(report["incrementlower"])

    def testinvalidbudgets(self):
        for changes in ({"radius": 0.}, {"radius": 1e300}, {"radius": 1e-300}, {"damage": -1.}, {"ridge": 0.}, {"iterations": True}):
            args = dict(radius=.5, lower=[-1., -1.], damage=.04, reference=[0., 0.], scales=[1., 1.], indices=[0, 1], **{})
            args.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                maximize(self.model(), **args)

    def testsharedbudget(self):
        def stop():
            raise Exhausted("shared limit")

        with self.assertRaises(Exhausted):
            maximize(self.model(), .5, [-1., -1.], .04, [0., 0.], [1., 1.], [0, 1], checkpoint=stop)

    def testconfig(self):
        self.assertEqual(validate(common(argparse.ArgumentParser()).parse_args(["--clip", "clip"])).selection, "attain")
        self.assertEqual(variants("attainment")["protected"]["selection"], "protect")
        for values in ({"selection": "unknown"}, {"gainloss": -1}, {"gainloss": True}, {"archive": 1}, {"goals": ("retain0",)}, {"goals": ([],)}, {"goals": ("target", "target")}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Config(**values).validate()

    def testzerodamagestillsearches(self):
        class Path:
            def __init__(self):
                self.session = SimpleNamespace(origin=torch.zeros(1, 3, 2, 2), limit=None)
                self.native = SimpleNamespace(continuous=True, metadata=lambda: {})
                self.seed = 42

            def raw(self):
                pixels = torch.zeros_like(self.session.origin)
                pixels[..., 1] = .2
                return pixels.clone(), pixels

            def initialize(self):
                return None

            def canonical(self, point):
                return tuple(point)

            def axes(self, requested):
                return (2,)

            def candidate(self, point):
                latent, pixels = self.raw()
                pixels[..., 1] = .3 + .1 * point[2]
                return pixels.clone(), pixels

            def metadata(self):
                return {}

        class Prior:
            def __call__(self, latent):
                return float(latent.mean())

            def metadata(self):
                return {}

        path = Path()
        weight = torch.zeros_like(path.session.origin[:, :1])
        weight[..., 0] = 1.
        result = select(path, lambda image: {"target": float(image.mean()), "retain0": 1.}, weight, Prior(), Config(candidates=3, axes="path", archive=True))
        self.assertFalse(result.report["fallback"])
        self.assertEqual(result.report["damage"], 0.)
        self.assertEqual(result.report["penalty"], 0.)
        self.assertGreater(result.report["attainment"]["selectedgain"], 0.)
        self.assertEqual(result.report["attainment"]["goals"], ["target"])
        self.assertEqual(result.report["commonpool"]["protect"]["selected"], "raw")
        self.assertNotEqual(result.report["commonpool"]["attain"]["selected"], "raw")
        self.assertEqual(len(path.session.artifacts), 4)
        json.dumps(result.report, allow_nan=False)

    def testgeometryvalidation(self):
        for field, value in (("metric", np.eye(3)), ("cross", np.zeros(3)), ("gram", -np.eye(2)), ("active", np.array([0., 2.])), ("anchor", np.array([1., 1., 1.5])), ("metric", np.zeros((2, 2)))):
            model = self.model()
            setattr(model, field, value)
            for columns in (None, np.array([], dtype=int)):
                with self.subTest(field=field, columns=columns), self.assertRaises(ValueError):
                    maximize(model, .5, [-1., -1.], .04, [0., 0.], [1., 1.], [0, 1], columns=columns)

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu 2026
# Email: wlzchina05@gmail.com
