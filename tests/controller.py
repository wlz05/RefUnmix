# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import argparse
import contextlib
from copy import deepcopy
import io
import json
import math
import os
from pathlib import Path as File
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from assess import coverage as metrics
from audit import values
from evaluate import coverage
from engine.curvature import fallback
from engine.field import Field, build, measure, solve
from engine.limit import Budget
from engine.native import Native
from engine.path import Path
from engine.poll import Config, select
from engine.prior import Prior
from engine.protocol import signature
from run import common, run, validate
from trial import design, frozen, settings, variants
from session import Clip, Processor, editor
from engine.judge import ClipJudge
from trust import Expensive, Fake
from field import Native as Linear, Session

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
        suites = ("temporal", "null", "core", "reference", "prior", "search", "guidance", "budget", "candidates", "mechanism", "cost", "release", "geometry", "solver", "attainment", "leverage", "frontier", "diagnosis", "goals")
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

class MeasurementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def args(self, *values):
        return common(argparse.ArgumentParser()).parse_args(["--size", "16", "--prior", "off", *values])

    def testzero(self):
        session = Session()
        constructed = build(session, None, "zero")
        self.assertTrue(torch.equal(constructed.value, torch.zeros_like(session.anchor)))
        self.assertEqual(session.samples, 0)
        self.assertEqual(constructed.report["actual"], "zero")
        self.assertEqual(constructed.report["probes"], 0)

    def testnullmatrix(self):
        args = self.args("--clip", "clip")
        args.suite = "null"
        controls = variants("null", base=args)
        plan = design(args, controls)
        self.assertEqual(set(controls), {"native", "centered", "condition", "reference", "raw", "full"})
        self.assertEqual(plan["variants"]["full"]["changes"], {})
        self.assertEqual(controls["centered"]["eta"], 0.)
        self.assertEqual(controls["condition"]["eta"], 1.)
        for key in ("native", "centered", "condition", "reference"):
            self.assertEqual(controls[key]["field"], "zero")
        for key in ("native", "centered", "condition"):
            current = deepcopy(args)
            for name, value in controls[key].items():
                setattr(current, name, value)
            validate(current)

    def testnulloutputs(self):
        image = Image.new("RGB", (16, 16), (96, 128, 160))
        source, target = "source", "target long"
        session = editor().prepare(image, source, target)
        native = Native(session, path="embedding")
        zero = torch.zeros_like(session.anchor)
        targetzero, sourcezero = native.endpoint(zero), native.endpoint(zero, 1.)
        expected = {"native": session.decode(targetzero).clamp(0, 1), "centered": session.decode(session.anchor).clamp(0, 1), "condition": session.decode(session.anchor + targetzero - sourcezero).clamp(0, 1)}
        controls = variants("null")
        actual = {}
        for key in expected:
            args = self.args()
            for name, value in controls[key].items():
                setattr(args, name, value)
            actual[key], report, _ = run(editor(), None, image, source, target, args)
            self.assertTrue(torch.allclose(actual[key], expected[key], atol=1e-6))
            self.assertEqual(report["fieldnorm"], 0.)
            self.assertEqual(report["field"]["actual"], "zero")
        self.assertFalse(torch.equal(actual["centered"], session.origin))
        self.assertFalse(torch.equal(actual["condition"], actual["centered"]))

    def testzerocache(self):
        session = editor().prepare(Image.new("RGB", (16, 16)), "source", "target long")
        path = Path(Native(session), torch.zeros_like(session.anchor))
        self.assertEqual(path.cost((1., 1., 0.)), (1, 1))
        before = session.samples
        latent, _ = path.candidate((1., 1., 0.))
        self.assertEqual(session.samples - before, 1)
        self.assertTrue(torch.equal(latent, session.anchor))
        self.assertEqual(path.cost((1., 1., 0.)), (0, 0))
        self.assertIs(path.fields[(1., 1.)], path.references[0.])

    def testzeroprepaid(self):
        session = editor().prepare(Image.new("RGB", (16, 16)), "source", "target long")
        native = Native(session)
        neutral = native.endpoint(torch.zeros_like(session.anchor))
        path = Path(native, torch.zeros_like(session.anchor), neutral=neutral)
        session.limit = Budget(1)
        latent, _ = path.candidate((1., 1., 0.))
        self.assertEqual(session.limit.used, 0)
        self.assertTrue(torch.equal(latent, session.anchor))

    def testcachedrawbudget(self):
        image = Image.new("RGB", (16, 16))
        args = self.args("--method", "raw", "--maxsamples", "3", "--maxcalls", "2")

        def built(session, native, *args):
            zero = torch.zeros_like(session.anchor)
            return Field(zero, {"actual": "pullback"}, native.endpoint(zero))

        with patch("engine.field.build", side_effect=built):
            _, report, session = run(editor(), None, image, "source", "target long", args)
        self.assertEqual(session.samples, 3)
        self.assertEqual(report["selected"], "raw")
        self.assertNotIn("budgetstopped", report)

    def testcachecosts(self):
        for scale in (0., 1.):
            for mode in ("latent", "rgb", "mix", "random", "cfg"):
                session = editor().prepare(Image.new("RGB", (16, 16)), "source", "target long", scale=scale)
                path = Path(Native(session), torch.zeros_like(session.anchor), mode)
                for point in ((1., 1., 0.), (.5, 1.5, .5), (1., 1., 1.)):
                    expected = path.cost(point)
                    before = session.samples, session.calls
                    path.candidate(point)
                    self.assertEqual((session.samples - before[0], session.calls - before[1]), expected)

    def testresponse(self):
        request = torch.tensor([1., 2., 3., 4.]).reshape(1, 1, 2, 2)
        result = measure(request * 2, request, request)
        self.assertAlmostEqual(result["relativefit"], 1.)
        self.assertAlmostEqual(result["responsecosine"], 1.)
        self.assertAlmostEqual(result["relativegap"], 1.)
        self.assertEqual(result["actualfit"], result["modelgap"])
        json.dumps(result, allow_nan=False)

    def testzeroresponse(self):
        zero = torch.zeros(1, 1, 2, 2)
        result = measure(zero, torch.ones_like(zero))
        self.assertEqual(result["responsestatus"], "zero response")
        self.assertIsNone(result["responsecosine"])
        self.assertEqual(result["relativefit"], 1.)

    def testzerorequest(self):
        zero = torch.zeros(1, 1, 2, 2)
        result = measure(torch.ones_like(zero), zero)
        self.assertEqual(result["responsestatus"], "zero request")
        self.assertIsNone(result["responsecosine"])
        self.assertIsNone(result["relativefit"])
        json.dumps(result, allow_nan=False)

    def testinvalid(self):
        value = torch.ones(1, 1, 2, 2)
        for displacement, request, prediction in ((value, value[..., :1], None), (value, value, value[..., :1]), (value * float("nan"), value, None)):
            with self.assertRaises(ValueError):
                measure(displacement, request, prediction)

    def testoverflow(self):
        actual = torch.full((1, 1, 1, 1), 1e100, dtype=torch.float64)
        request = torch.full_like(actual, 1e-300)
        result = measure(actual, request)
        self.assertIsNone(result["relativefit"])
        self.assertEqual(result["actualfit"], 1e200)
        json.dumps(result, allow_nan=False)

    def testdirection(self):
        session = Session()
        native = Linear(session)
        field = build(session, native)
        before = session.samples
        field.observe(native.endpoint(field.value))
        self.assertEqual(session.samples - before, 1)
        self.assertLess(field.report["modelgap"], 1e-10)
        self.assertGreater(field.report["alignment"]["angle"], 0.)
        self.assertGreater(field.report["alignment"]["orthogonalratio"], 0.)
        self.assertEqual(field.report["rank"], 2)
        self.assertGreater(field.report["responsecosine"], .99)

    def testscalar(self):
        session = Session()
        field = build(session, Linear(session), geometry="scalar")
        self.assertAlmostEqual(field.report["alignment"]["angle"], 0., places=5)
        self.assertAlmostEqual(field.report["alignment"]["orthogonalratio"], 0., places=10)

    def testrankone(self):
        session = Session()
        field = build(session, Linear(session), delta=0.)
        self.assertEqual(field.report["rank"], 1)
        self.assertIsNone(field.report["alignment"]["singularratio"])
        self.assertAlmostEqual(field.report["alignment"]["orthogonalratio"], 0.)

    def testexternalreference(self):
        value = torch.ones(1, 1, 2, 2)
        field = Field(value, {})
        field.observe(value * 3, value, value * 2)
        self.assertEqual(field.report["actualfit"], 0.)
        self.assertEqual(field.report["responsecosine"], 1.)
        self.assertNotIn("modelgap", field.report)

    def testaudit(self):
        report = {"field": {"relativefit": .5, "responsecosine": .9, "alignment": {"angle": 10., "cosine": None, "scope": "input geometry"}}}
        result = values({"report": report})
        self.assertEqual(result["field:relativefit"], .5)
        self.assertEqual(result["field:inputangle"], 10.)
        self.assertNotIn("field:inputcosine", result)

    def testprobe(self):
        import probe
        origin = File.cwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder)
                Image.new("RGB", (16, 16), (96, 128, 160)).save("input.png")
                arguments = ["probe.py", "--image", "input.png", "--source", "source", "--target", "target long", "--size", "16", "--withfield", "--output", "probes"]
                with patch("sys.argv", arguments), patch("probe.load", return_value=(editor(), None)), contextlib.redirect_stdout(io.StringIO()):
                    probe.main()
                report = json.loads(File("probes/summary.json").read_text())
                self.assertIn("responsecosine", report["field"])
                self.assertIn("relativefit", report["field"])
                self.assertIn("controlsignature", report)
                self.assertIn("reproducibility", report)
            finally:
                os.chdir(origin)

class BoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def args(self, *values):
        return common(argparse.ArgumentParser()).parse_args(["--clip", "clip", "--size", "16", *values])

    def path(self):
        session = editor().prepare(Image.new("RGB", (16, 16)), "source", "target long")
        return Path(Native(session), session.field(.9, .15))

    def testtinynorm(self):
        request = torch.full((1, 1, 1, 1), 1e-200, dtype=torch.float64)
        report = measure(torch.zeros_like(request), request)
        self.assertEqual(report["targetnorm"], 1e-200)
        self.assertEqual(report["relativefit"], 1.)
        self.assertIsNone(report["actualfit"])
        self.assertEqual(report["fitstatus"], "underflow")

    def testtinygap(self):
        request = torch.full((1, 1, 1, 1), 1e-200, dtype=torch.float64)
        report = measure(torch.zeros_like(request), request, request)
        self.assertIsNone(report["modelgap"])
        self.assertEqual(report["gapstatus"], "underflow")
        self.assertEqual(report["relativegap"], 1.)
        json.dumps(report, allow_nan=False)

    def testrelativeoverflow(self):
        request = torch.full((1, 1, 1, 1), 1e-300, dtype=torch.float64)
        report = measure(torch.full_like(request, 1e100), request)
        self.assertGreater(report["targetnorm"], 0)
        self.assertEqual(report["relativefitstatus"], "overflow")
        self.assertIsNone(report["relativefit"])

    def testmeanunderflow(self):
        request = torch.zeros(1, 1, 2, 1024, dtype=torch.float64)
        request[..., 0, 0] = 1e-161
        report = measure(torch.zeros_like(request), request)
        self.assertGreater(report["actualfit"], 0)
        self.assertIsNone(report["actualfitmean"])
        self.assertEqual(report["fitstatus"], "underflow")

    def testgapmeanunderflow(self):
        request = torch.zeros(1, 1, 2, 1024, dtype=torch.float64)
        request[..., 0, 0] = 1e-161
        report = measure(torch.zeros_like(request), request, request)
        self.assertGreater(report["modelgap"], 0)
        self.assertIsNone(report["modelgapmean"])
        self.assertEqual(report["gapstatus"], "underflow")

    def testlargesolve(self):
        coefficients, report = solve(np.eye(2) * 1e100, np.ones(2) * 1e100, np.ones(2) * .01, radius=.1)
        self.assertTrue(math.isfinite(report["multiplier"]))
        self.assertLessEqual(math.hypot(*(coefficients - .01)), report["bound"] * (1 + 1e-10))
        self.assertGreater(report["multiplier"], 0)
        json.dumps(report, allow_nan=False)

    def testsmallball(self):
        _, report = solve(np.eye(2), np.ones(2) * 1e-200, np.ones(2) * 1e-200, radius=.1)
        self.assertGreater(report["bound"], 0)

    def testrawprior(self):
        with self.assertRaisesRegex(ValueError, "Raw outputs"):
            validate(self.args("--method", "raw", "--prior", "rec"))

    def testrawheldout(self):
        with self.assertRaisesRegex(ValueError, "Raw outputs"):
            validate(self.args("--method", "raw", "--heldout"))

    def testrawcontrol(self):
        with self.assertRaisesRegex(ValueError, "Raw outputs"):
            validate(self.args("--method", "raw", "--control"))

    def testheldoutoff(self):
        with self.assertRaisesRegex(ValueError, "enabled prior"):
            validate(self.args("--heldout", "--prior", "off"))

    def testablationraw(self):
        args = self.args("--prior", "rec", "--heldout", "--control")
        for suite in ("alignment", "fields", "core", "diagnosis"):
            for values in variants(suite, base=args).values():
                if values["method"] == "raw":
                    self.assertEqual(values["prior"], "off")
                    self.assertFalse(values["heldout"])
                    self.assertFalse(values["control"])

    def testcostnonfinite(self):
        path = self.path()
        for point in ((1., float("nan"), 0.), (1., 1., float("inf")), (True, 1., 0.)):
            with self.assertRaises(ValueError):
                path.cost(point)

    def testcostarity(self):
        path = self.path()
        for point in (None, (1., 1.), (1., 1., 0., 0.), "abc"):
            with self.assertRaises(ValueError):
                path.cost(point)

    def testcanonicalnonfinite(self):
        with self.assertRaises(ValueError):
            self.path().canonical((1., 1., float("nan")))

    def testcandidatebounds(self):
        path = self.path()
        for point in ((-.01, 1., 0.), (1., 2.01, 0.), (1., 1., 1.01), None):
            with self.assertRaises(ValueError):
                path.candidate(point)

    def testnonfiniteproposal(self):
        from engine.trust import Model

        def unresolved(*args, **kwargs):
            model = Model(*args, **kwargs)
            model.rank = 0
            return model

        path = Fake()
        with patch("engine.trust.Model", side_effect=unresolved), patch("engine.trust.geometry", return_value=(1., float("nan"), 0.)):
            with self.assertRaises(ValueError):
                select(path, lambda image: {"target": 1.}, torch.ones_like(path.session.origin[:, :1]), Expensive(), Config())

    def testconfigcontent(self):
        origin = File.cwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder)
                File("control.json").write_text('{"slacks": {}}', encoding="utf-8")
                args = self.args("--constraints", "control.json")
                before = signature(args)
                File("control.json").write_text('{"slacks": {"target": 0.1}}', encoding="utf-8")
                after = signature(args)
                self.assertNotEqual(before["sha256"], after["sha256"])
                self.assertNotEqual(before["settings"]["configurationhashes"], after["settings"]["configurationhashes"])
            finally:
                os.chdir(origin)

    def testconfigmissing(self):
        with self.assertRaises(FileNotFoundError):
            signature(self.args("--constraints", "nonexistent.json"))

    def testmanifestmutation(self):
        args = self.args()
        args.suite, args.recipe, args.manifest = "core", "recipe.json", "split.json"
        payload = {"schema": 1, "split": "val", "suite": "core", "settings": settings(args), "variants": variants("core"), "manifestdigest": "before", "digests": {}}
        with patch("trial.locate"), patch("trial.read", return_value=payload), patch("trial.digest", return_value="after"):
            with self.assertRaisesRegex(ValueError, "split manifest"):
                frozen(args, [])

    def testmanifestmissing(self):
        args = self.args()
        args.suite, args.recipe = "core", "recipe.json"
        payload = {"schema": 1, "split": "val", "suite": "core", "settings": settings(args), "variants": variants("core"), "digests": {}}
        with patch("trial.locate"), patch("trial.read", return_value=payload):
            with self.assertRaisesRegex(ValueError, "split manifest"):
                frozen(args, [])

    def testprobemode(self):
        import probe
        arguments = ["probe.py", "--image", "input.png", "--source", "source", "--target", "target", "--method", "path"]
        with patch("sys.argv", arguments), patch("probe.load") as load:
            with self.assertRaisesRegex(ValueError, "Native probes"):
                probe.main()
        load.assert_not_called()

    def testprobediscrete(self):
        import probe
        origin = File.cwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder)
                Image.new("RGB", (16, 16)).save("input.png")
                arguments = ["probe.py", "--image", "input.png", "--source", "source", "--target", "target long", "--referencepath", "discrete", "--size", "16"]
                with patch("sys.argv", arguments), patch("probe.load", return_value=(editor(), None)):
                    with self.assertRaisesRegex(ValueError, "Discrete references"):
                        probe.main()
                self.assertFalse(File("probes").exists())
            finally:
                os.chdir(origin)

class RobustnessTests(unittest.TestCase):
    def args(self, *flags):
        return common(argparse.ArgumentParser()).parse_args(["--clip", "clip", "--size", "16", *flags])

    def recipe(self, suite="temporal"):
        args = self.args()
        args.suite, args.recipe, args.manifest = suite, "recipe.json", "split.json"
        controls = variants(suite, base=args)
        payload = {"schema": 1, "split": "val", "suite": suite, "settings": settings(args), "variants": controls, "digests": {}, "manifestdigest": "manifest"}
        return args, payload

    def recover(self, args, payload):
        with patch("trial.locate", return_value=File("recipe.json")), patch("trial.read", return_value=payload), patch("trial.identities", return_value={}), patch("trial.digest", return_value="manifest"):
            return frozen(args, [])

    def testtemporalrecipe(self):
        args, payload = self.recipe()
        self.assertEqual(self.recover(args, payload), variants("temporal", base=args))

    def testallrecipes(self):
        for suite in ("alignment", "temporal", "fields", "null", "core", "reference", "diagnosis", "leverage", "frontier", "attainment", "release", "geometry", "prior", "mechanism", "search", "guidance", "budget", "candidates", "cost", "solver", "goals"):
            with self.subTest(suite=suite):
                args, payload = self.recipe(suite)
                self.assertEqual(self.recover(args, payload), payload["variants"])

    def testrecipestructure(self):
        args, _ = self.recipe()
        for payload in (None, [], [1], "recipe", 1):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.recover(args, payload)

    def testrecipebindings(self):
        for key, value in (("digests", []), ("digests", {"image": None}), ("group", "group"), ("family", {"family": 1})):
            args, payload = self.recipe("core")
            payload[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.recover(args, payload)

    def testrecipetypes(self):
        args, payload = self.recipe("core")
        payload["variants"] = deepcopy(payload["variants"])
        payload["variants"]["full"]["guidance"] = True
        with self.assertRaises(ValueError):
            self.recover(args, payload)

    def testrecipesettings(self):
        args, payload = self.recipe("core")
        payload["settings"]["guidance"] = True
        with self.assertRaises(ValueError):
            self.recover(args, payload)

    def testfixedprior(self):
        with self.assertRaises(ValueError):
            validate(self.args("--method", "fixed", "--axes", "none", "--prior", "ddpm"))

    def testbudgettypes(self):
        for changes in ({"seconds": "1"}, {"seconds": []}, {"seconds": 10 ** 400}, {"started": float("nan")}, {"started": 10 ** 400}, {"used": -1}, {"forwards": True}, {"phase": None}, {"stages": []}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                Budget(**changes)

    def testnumberrange(self):
        from engine.session import number
        with self.assertRaises(ValueError):
            number(10 ** 400, "scale")

    def testbudgetpreload(self):
        from run import load
        for key, value in (("maxsamples", 0), ("maxcalls", 0), ("seconds", -1)):
            args = self.args()
            setattr(args, key, value)
            with self.subTest(key=key), patch("engine.session.Editor.load") as checkpoint:
                with self.assertRaises(ValueError):
                    load(args)
                checkpoint.assert_not_called()

    def testbudgetatomic(self):
        budget = Budget()
        budget.phase = []
        with self.assertRaises(ValueError):
            budget.consume(1)
        self.assertEqual((budget.used, budget.forwards), (0, 0))
        budget.phase, budget.stages = "front", {"front": {"samples": 0}}
        with self.assertRaises(ValueError):
            budget.consume(1)
        self.assertEqual((budget.used, budget.forwards), (0, 0))

    def testtemporalrange(self):
        from engine.field import temporal
        report = temporal([[1e-200, -1e-200]], [1.])
        self.assertTrue(report["nonnegative"])
        self.assertLess(report["coneerror"], 1e-20)
        np.testing.assert_allclose(report["nonnegativeweights"], [1e200, 0.])

    def testtemporalunderflow(self):
        from engine.field import temporal
        report = temporal([[1.], [0.]], [1., 1e-200], tolerance=1e-250)
        for key in ("span", "cone"):
            self.assertIsNone(report[key + "error"])
            self.assertEqual(report[key + "status"], "underflow")
        self.assertFalse(report["nonnegative"])

    def testtemporalcoefficients(self):
        from engine.field import temporal
        for matrix, value in (([[1e-300]], [1e300]), ([[1e300]], [1e-300]), ([[1e-308, 0.], [0., 1e-308]], [1., 1.])):
            with self.subTest(matrix=matrix, value=value), self.assertRaises(ValueError):
                temporal(matrix, value)

    def testrmsrange(self):
        class Precise(Linear):
            def endpoint(self, value, eta=0.):
                return super().endpoint(value.double(), eta)

        for value in (1e-200, 1e200):
            session = Session()
            native = Precise(session, matrix=value * torch.eye(4, dtype=torch.float64))
            native.offset = torch.full((4,), value, dtype=torch.float64)
            result = Path(native, torch.ones_like(session.anchor)).mechanism()
            self.assertAlmostEqual(result["latent"]["bias"] / value, 1.)

    def testnativeoverflow(self):
        session = editor().prepare(Image.new("RGB", (16, 16)), "source", "target long", cleanup=False, scale=1e30)
        native = Native(session)
        with self.assertRaises(ValueError):
            native.endpoint(torch.full_like(session.anchor, 1e30))
        self.assertEqual((native.endpoints, session.samples), (0, 0))

    def testnativestate(self):
        session = editor().prepare(Image.new("RGB", (16, 16)), "source", "target long")
        session.noises.fill_(3e38)
        native = Native(session, steps=2)
        with self.assertRaises(ValueError):
            native.endpoint(torch.full_like(session.anchor, 3e38))
        self.assertEqual((native.endpoints, session.samples), (0, 0))

    def testmechanismphase(self):
        for maximum in (1, 8):
            session = editor().prepare(Image.new("RGB", (16, 16)), "source", "target long")
            base = session.field(.9, .15)
            session.limit = Budget(maximum, phase="prior")
            path = Path(Native(session), base)
            if maximum == 1:
                from engine.limit import Exhausted
                with self.assertRaises(Exhausted):
                    path.mechanism()
            else:
                path.mechanism()
            self.assertEqual(session.limit.phase, "prior")

    def testjsonkeys(self):
        from engine.serial import write
        with tempfile.TemporaryDirectory() as folder:
            path = File(folder) / "reports" / "result.json"
            with self.assertRaises(ValueError):
                write(path, {"rows": [{1: "first", "1": "second"}]})
            self.assertFalse(path.parent.exists())

    def testjsoninput(self):
        from engine.serial import loads
        for value in (None, 1, b"{}", []):
            with self.subTest(value=value), self.assertRaises(ValueError):
                loads(value)

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
