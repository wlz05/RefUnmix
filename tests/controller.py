# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import argparse
from copy import deepcopy
import json
from pathlib import Path as File
import unittest
from unittest.mock import patch
from PIL import Image
import torch
from assess import coverage as metrics
from evaluate import coverage
from engine.curvature import fallback
from engine.limit import Budget
from engine.native import Native
from engine.path import Path
from engine.poll import Config, select
from engine.prior import Prior
from run import common, run, validate
from trial import frozen, settings, variants
from session import Clip, Processor, editor
from engine.judge import ClipJudge
from trust import Expensive, Fake

class ControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def args(self, *values):
        return common(argparse.ArgumentParser()).parse_args(["--clip", "clip", "--size", "16", *values])

    def session(self):
        return editor().prepare(Image.new("RGB", (16, 16), (96, 128, 160)), "source", "target long")

    def testdefaults(self):
        args = self.args()
        self.assertEqual((args.prior, args.poststeps, args.release, args.compensation), ("off", 1, True, False))
        self.assertEqual(validate(args), Config())

    def testmainidentity(self):
        suites = ("null", "core", "reference", "prior", "search", "guidance", "budget", "candidates", "mechanism", "cost", "release", "geometry", "solver", "attainment", "leverage", "frontier", "diagnosis", "goals")
        reference = variants("core")["full"]
        for suite in suites:
            self.assertEqual(variants(suite)["full"], reference)
            for values in variants(suite).values():
                args = self.args()
                for key, value in values.items():
                    setattr(args, key, value)
                validate(args)
        self.assertEqual(variants("fields")["pullbackfull"], reference)
        self.assertEqual(variants("alignment")["controlled"], reference)

    def testisolatedfactors(self):
        for suite, variant, expected in (("mechanism", "multistep", {"poststeps"}), ("release", "joint", {"release"}), ("leverage", "compensated", {"compensation"}), ("geometry", "latent", {"geometry"}), ("search", "poll", {"algorithm"}), ("candidates", "large", {"candidates"})):
            controls = variants(suite)
            before, after = controls["full"], controls[variant]
            changed = {key for key in before.keys() | after.keys() if before.get(key) != after.get(key)}
            self.assertEqual(changed, expected)

    def testrealbudgets(self):
        controls = variants("budget")
        for key, value in (("ten", 10), ("twelve", 12), ("fifteen", 15), ("twenty", 20), ("thirty", 30)):
            self.assertEqual((controls[key]["maxsamples"], controls[key]["maxcalls"]), (value, value))
            self.assertNotIn("candidates", controls[key])
        self.assertEqual(variants("prior")["hardprior"]["prior"], "ddpm")
        self.assertEqual(variants("prior", "flow")["boundedprior"]["prior"], "flow")

    def testseedidentity(self):
        first = self.args()
        first.seeds = [3407, 3047, 52, 42]
        second = deepcopy(first)
        second.seeds.reverse()
        self.assertEqual(settings(first), settings(second))
        second.seeds = [42]
        self.assertNotEqual(settings(first), settings(second))

    def testfrozenidentity(self):
        args = self.args()
        args.suite, args.recipe, args.manifest = "core", "recipe.json", "split.json"
        payload = {"schema": 1, "split": "val", "suite": "core", "settings": settings(args), "variants": variants("core"), "digests": {}, "manifestdigest": "manifest"}
        with patch("trial.locate", return_value=File("recipe.json")), patch("trial.read", return_value=payload), patch("trial.identities", return_value={}), patch("trial.digest", return_value="manifest"):
            self.assertEqual(frozen(args, []), variants("core"))
            args.seed = 52
            with self.assertRaises(ValueError):
                frozen(args, [])
            args.seed, args.suite = 42, "geometry"
            with self.assertRaises(ValueError):
                frozen(args, [])

    def testcostprediction(self):
        for mode in ("latent", "rgb", "mix", "random", "cfg"):
            for kind, steps in (("embedding", 1), ("prediction", 1), ("embedding", 2)):
                session = self.session()
                path = Path(Native(session, steps=steps, path=kind), session.field(.9, .15), mode)
                for point in ((0., 0., 0.), (1., 1., .5), (1., 1., 1.), (.5, 1.5, .25), (.5, 1.5, .25)):
                    expected = path.cost(point)
                    before = session.samples, session.calls
                    path.candidate(point)
                    self.assertEqual((session.samples - before[0], session.calls - before[1]), expected)

    def testfixedreferences(self):
        session = self.session()
        path = Path(Native(session), session.field(.9, .15))
        path.initialize((0.,))
        self.assertEqual(set(path.references), {0.})
        self.assertEqual(path.cost((1., 1., 0.)), (0, 0))

    def testconditionalreferences(self):
        session = self.session()
        native = Native(session, path="prediction")
        path = Path(native, torch.full_like(session.anchor, .05), "cfg")
        with patch.object(native, "endpoint", side_effect=lambda field, eta=0.: field * (1 + eta)):
            path.initialize()
            self.assertFalse(path.references)
            path.references = {0.: torch.zeros_like(path.base), 1.: torch.zeros_like(path.base)}
            self.assertIn(2, path.axes((0, 1, 2)))
            self.assertFalse(torch.equal(path.latent((1., 1., 0.)), path.latent((1., 1., 1.))))

    def testlowbudgetsearch(self):
        args = self.args("--maxsamples", "12", "--maxcalls", "12")
        image = Image.new("RGB", (16, 16), (96, 128, 160))
        _, report, session = run(editor(), ClipJudge(Clip(), Processor()), image, "source", "target long", args, Image.new("L", image.size, 128))
        initial = report["initialization"]
        self.assertTrue(initial["complete"])
        self.assertGreater(initial["skipped"], 0)
        self.assertLess(initial["evaluated"], 7)
        self.assertTrue(report["history"])
        self.assertTrue(any(row.get("budgetfallback") for row in report["history"]))
        self.assertLessEqual(session.samples, 12)
        self.assertLessEqual(session.calls, 12)
        self.assertFalse(report["rescue"]["enabled"])
        self.assertTrue(report["release"]["enabled"])
        json.dumps(report, allow_nan=False)

    def testatomicprior(self):
        session = self.session()
        path = Path(Native(session), session.field(.9, .15))
        path.initialize()
        session.limit = Budget(3)
        known = set(path.fields)
        result = select(path, lambda image: {"target": 0.}, torch.ones_like(session.origin[:, :1]), Prior(session, mode="rec"))
        self.assertEqual(session.limit.used, 2)
        self.assertEqual(set(path.fields), known)
        self.assertEqual(result.report["selected"], "raw")
        self.assertFalse(result.report["candidates"])

    def testcachedreferencebudget(self):
        session = self.session()
        path = Path(Native(session, path="prediction"), session.field(.9, .15))
        path.initialize()
        session.limit = Budget(1)
        result = select(path, lambda image: {"target": 0.}, torch.ones_like(session.origin[:, :1]), Prior(session, mode="off"))
        self.assertIn([1., 1., 1.], [row["point"] for row in result.report["candidates"]])
        self.assertEqual(session.limit.used, 1)

    def testlightreferences(self):
        path = Fake()
        report = select(path, lambda image: {"target": 1.}, torch.ones_like(path.session.origin[:, :1]), Expensive(), Config(candidates=3, axes="path")).report
        self.assertEqual({row["point"][2] for row in report["candidates"]}, {0., .5, 1.})
        self.assertFalse(report["history"])

    def testemptyprotection(self):
        class Linear(Fake):
            def candidate(self, point):
                image = torch.full_like(self.session.origin, .2 + .2 * point[2])
                return image.clone(), image

        path = Linear()
        result = select(path, lambda image: {"target": float(image.mean())}, torch.zeros_like(path.session.origin[:, :1]), Expensive(), Config())
        self.assertFalse(result.report["fallback"])
        self.assertFalse(result.report["protectedvalid"])
        self.assertEqual(result.report["damage"], 0.)
        self.assertGreater(result.report["attainment"]["selectedgain"], 0.)
        self.assertTrue(result.report["candidates"])

    def testrejectedrescue(self):
        path = Fake()
        with patch("engine.attainment.propose", return_value=None), patch("engine.curvature.fallback", wraps=fallback):
            report = select(path, lambda image: {"target": 1.}, torch.zeros_like(path.session.origin[:, :1]), Expensive(), Config(release=False)).report
        rejected = [row for row in report["history"] if row.get("kind") == "fallback" and row.get("accepted") is False]
        self.assertGreaterEqual(len(rejected), 2)
        self.assertLess(rejected[1]["radius"], rejected[0]["radius"])

    def testcoverage(self):
        rows = [{"report": {"method": "path", "initialization": {"complete": True}, "release": {"enabled": True, "queries": 0, "accepted": 0}, "rescue": {"enabled": False}, "attainment": {"bottlenecks": ["global"]}}}, {"report": {"method": "path", "release": {"enabled": True, "queries": 2, "accepted": 1}}}]
        result = coverage(rows)
        self.assertEqual(result["release"], {"enabled": 2, "queried": 1, "accepted": 1, "queries": 2})
        self.assertEqual(result["bottlenecks"], {"global": 1})
        self.assertEqual(result["initialized"], 1)

    def testmetriccoverage(self):
        rows = [{"key": "a", "status": {"psnr": "finite"}}, {"key": "a", "status": {"psnr": "finite"}}, {"key": "b", "status": {"psnr": "empty protected region"}}]
        result = metrics(rows)["psnr"]
        self.assertEqual(result["finite"], {"rows": 2, "sources": ["a"]})
        self.assertEqual(result["empty protected region"], {"rows": 1, "sources": ["b"]})

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
