# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import argparse
import json
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from engine.field import build, options, orthogonal, solve
from engine.limit import Budget
from engine.path import Path
from run import common, validate
from trial import variants

class Session:
    def __init__(self, maximum=64, scale=1., cleanup=True):
        self.anchor = torch.zeros(1, 1, 2, 2)
        self.origin = torch.zeros(1, 3, 2, 2)
        self.scale, self.cleanup = scale, cleanup
        self.limit = Budget(maximum)
        self.calls, self.samples = 0, 0
        self.cache = {}
        self.editor = SimpleNamespace(index=lambda value: round(value, 6))

    def difference(self, tick):
        if tick not in self.cache:
            self.limit.consume(2)
            self.calls += 1
            self.samples += 2
            value = [2., 0., 0., 0.] if tick > .8 else [0., 2., 0., 0.]
            self.cache[tick] = torch.tensor(value).reshape_as(self.anchor)
        return self.cache[tick]

    def field(self, time, delta):
        first = self.difference(self.editor.index(time))
        return first if delta == 0 else (delta * first + time * self.difference(self.editor.index(time - delta))) / (time + delta)

class Native:
    def __init__(self, session, matrix=None):
        self.session = session
        self.matrix = torch.diag(torch.tensor([2., .5, 1., 1.])) if matrix is None else matrix
        self.offset = torch.tensor([.2, -.3, .1, .2])
        self.inputs = []

    def cost(self, eta):
        return 1

    def queries(self):
        return 1

    def endpoint(self, value, eta=0.):
        self.session.limit.consume(1)
        self.session.calls += 1
        self.session.samples += 1
        self.inputs.append(value.detach().clone())
        return (self.offset + self.matrix @ (self.session.scale * value.reshape(-1))).reshape_as(value)

class FieldTests(unittest.TestCase):
    def testinverse(self):
        response = np.array([[2., .4], [0., .5], [0., 0.]])
        target, anchor = np.array([1., 1., 0.]), np.ones(2)
        result, report = solve(response, target, anchor, 1e-8, 10.)
        self.assertTrue(np.allclose(response @ result, target, atol=1e-6))
        self.assertLess(report["after"], report["before"])
        self.assertLess(report["objective"], report["before"])
        json.dumps(report, allow_nan=False)

    def testcoupled(self):
        response = np.array([[1., 1.], [0., 1.]])
        result, _ = solve(response, np.array([1., 0.]), np.ones(2), 1e-8, 10.)
        self.assertAlmostEqual(result[0], 1., places=5)
        self.assertAlmostEqual(result[1], 0., places=5)

    def testidentity(self):
        anchor = np.array([1., 2.])
        result, report = solve(np.eye(2), anchor, anchor)
        self.assertTrue(np.array_equal(result, anchor))
        self.assertEqual(report["objective"], 0.)

    def testzeroresponse(self):
        result, report = solve(np.zeros((3, 2)), np.ones(3), np.ones(2))
        self.assertTrue(np.array_equal(result, np.ones(2)))
        self.assertEqual(report["condition"], 1.)

    def testball(self):
        response, target, anchor = np.diag([.01, 3.]), np.ones(2), np.ones(2)
        result, report = solve(response, target, anchor, 1e-6, .1)
        self.assertLessEqual(np.linalg.norm(result - anchor), .1 * np.linalg.norm(anchor) + 1e-12)
        self.assertGreater(report["multiplier"], 0.)
        self.assertLessEqual(report["objective"], report["before"])

    def testrotation(self):
        rotation = np.array([[.6, -.8], [.8, .6]])
        response, target, anchor = np.array([[2., 1.], [0., .5]]), np.ones(2), np.ones(2)
        first, _ = solve(response, target, anchor, .1, .3)
        second, _ = solve(response @ rotation, target, rotation.T @ anchor, .1, .3)
        self.assertTrue(np.allclose(first, rotation @ second, atol=1e-10))

    def testzeroanchor(self):
        result, _ = solve(np.eye(2), np.ones(2), np.zeros(2))
        self.assertTrue(np.array_equal(result, np.zeros(2)))

    def testinvalid(self):
        for ridge, radius in ((0., 1.), (-1., 1.), (.1, -1.), (float("nan"), 1.)):
            with self.subTest(ridge=ridge, radius=radius), self.assertRaises(ValueError):
                solve(np.eye(2), np.ones(2), np.ones(2), ridge, radius)
        for response, target, anchor in ((np.eye(2), np.ones(3), np.ones(2)), (np.eye(2) * np.nan, np.ones(2), np.ones(2)), (np.empty((2, 0)), np.ones(2), np.empty(0))):
            with self.assertRaises(ValueError):
                solve(response, target, anchor)

    def testbasis(self):
        value = torch.arange(4.).reshape(1, 1, 2, 2)
        basis, _ = orthogonal([value, value * 2])
        self.assertEqual(basis.shape, (4, 1))
        self.assertTrue(np.allclose(basis.T @ basis, np.eye(1)))
        basis, _ = orthogonal([torch.zeros_like(value)])
        self.assertEqual(basis.shape[1], 0)

    def testbuild(self):
        session = Session()
        native = Native(session)
        result = build(session, native)
        value = result.value.reshape(-1)
        target = torch.tensor([1., 1., 0., 0.])
        before = torch.linalg.vector_norm(native.matrix @ target - target)
        after = torch.linalg.vector_norm(native.matrix @ value - target)
        self.assertLess(after, before)
        self.assertEqual(result.report["rank"], 2)
        self.assertEqual(result.report["probes"], 3)
        self.assertEqual(session.limit.report()["stages"]["field"]["samples"], 3)
        self.assertFalse(result.report["usesclip"])
        self.assertEqual(session.limit.phase, "front")
        self.assertTrue(torch.equal(result.neutral.reshape(-1), native.offset))
        previous = session.samples
        endpoint = (native.offset + native.matrix @ result.value.reshape(-1)).reshape_as(result.value)
        result.observe(endpoint)
        self.assertEqual(session.samples, previous)
        self.assertAlmostEqual(result.report["actualfit"], result.report["model"]["after"], places=5)
        self.assertLess(result.report["modelgap"], 1e-10)
        json.dumps(result.report, allow_nan=False)

    def testcache(self):
        session = Session()
        native = Native(session)
        result = build(session, native)
        path = Path(native, result.value, neutral=result.neutral)
        previous = session.samples
        self.assertTrue(torch.equal(path.zero(0.), result.neutral))
        self.assertEqual(session.samples, previous)

    def testnativegeometry(self):
        class Malformed(Native):
            def endpoint(self, value, eta=0.):
                return super().endpoint(value, eta).reshape(1, 1, 1, 4)

        session = Session()
        with self.assertRaisesRegex(ValueError, "request geometry"):
            build(session, Malformed(session))
        self.assertEqual(session.limit.phase, "front")

    def testbudget(self):
        session = Session(maximum=7)
        result = build(session, Native(session))
        self.assertEqual(result.report["actual"], "average")
        self.assertEqual(result.report["status"], "budget")
        self.assertEqual(session.samples, 4)
        self.assertIsNone(result.neutral)

    def testscale(self):
        for scale in (.5, 2.):
            session = Session(scale=scale)
            native = Native(session)
            result = build(session, native)
            expected = scale * torch.tensor([1., 1., 0., 0.])
            actual = native.matrix @ (scale * result.value.reshape(-1))
            raw = native.matrix @ expected
            self.assertLess(torch.linalg.vector_norm(actual - expected), torch.linalg.vector_norm(raw - expected))

    def testdegenerate(self):
        for values in ({"scale": 0.}, {"cleanup": False}):
            session = Session(**values)
            result = build(session, Native(session))
            self.assertEqual(result.report["status"], "degenerate")
            self.assertEqual(session.samples, 4)
        session = Session()
        result = build(session, Native(session), radius=0)
        self.assertEqual(result.report["status"], "degenerate")

    def testfamilies(self):
        for kind in ("instant", "chord", "average", "pullback"):
            session = Session()
            result = build(session, Native(session), kind)
            self.assertEqual(result.value.shape, session.anchor.shape)
            self.assertEqual(result.report["actual"], kind)
        session = Session()
        result = build(session, Native(session), delta=0)
        self.assertEqual(result.report["rank"], 1)
        self.assertEqual(result.report["times"], [.9])
        self.assertAlmostEqual(result.report["singular"][0], 2.)

    def testoptions(self):
        args = common(argparse.ArgumentParser()).parse_args(["--method", "raw", "--delta", "0"])
        validate(args)
        self.assertEqual(args.field, "pullback")
        for kind, ridge, probe, radius, tolerance in (("bad", .1, .1, 1., 1e-6), ("pullback", 0., .1, 1., 1e-6), ("pullback", .1, 0., 1., 1e-6), ("pullback", .1, .1, -1., 1e-6)):
            with self.assertRaises(ValueError):
                options(kind, ridge, probe, radius, tolerance)

    def testmatrix(self):
        matrix = variants("fields")
        self.assertEqual(len(matrix), 12)
        for field in ("pullback", "chord", "instant", "average"):
            self.assertEqual({matrix[field + suffix]["method"] for suffix in ("raw", "fixed", "full")}, {"raw", "fixed", "path"})
            for suffix in ("raw", "fixed", "full"):
                args = common(argparse.ArgumentParser()).parse_args(["--clip", "clip"])
                for key, value in matrix[field + suffix].items():
                    setattr(args, key, value)
                validate(args)

    def testshortbudget(self):
        from PIL import Image
        from run import run
        from session import editor
        args = common(argparse.ArgumentParser()).parse_args(["--method", "raw", "--size", "16", "--maxsamples", "1", "--prior", "off"])
        output, report, session = run(editor(), None, Image.new("RGB", (16, 16)), "source", "target", args)
        self.assertTrue(report["budgetstopped"])
        self.assertEqual(report["selected"], "source")
        self.assertTrue(torch.equal(output, session.origin))
        self.assertEqual(session.samples, 0)

    def testsharedregion(self):
        from PIL import Image
        from run import run
        from session import editor
        hashes = set()
        for kind in ("instant", "average", "chord", "pullback"):
            args = common(argparse.ArgumentParser()).parse_args(["--method", "raw", "--field", kind, "--size", "16", "--prior", "off"])
            _, report, _ = run(editor(), None, Image.new("RGB", (16, 16), (96, 128, 160)), "source", "target long", args)
            hashes.add(report["protectionsha256"])
            if kind == "pullback":
                self.assertIn("actualfit", report["field"])
        self.assertEqual(len(hashes), 1)

    def testfixedbudget(self):
        from PIL import Image
        from run import run
        from session import editor
        args = common(argparse.ArgumentParser()).parse_args(["--method", "fixed", "--field", "average", "--axes", "none", "--size", "16", "--prior", "off", "--maxsamples", "5"])
        output, report, session = run(editor(), None, Image.new("RGB", (16, 16)), "source", "target", args)
        self.assertEqual(output.shape, session.origin.shape)
        self.assertEqual(report["selected"], "raw")
        self.assertTrue(report["budgetstopped"])
        self.assertEqual(session.samples, 5)

    def testcostcoverage(self):
        from evaluate import costs
        rows = [{"key": "first", "report": {"stats": {"budget": {"stages": {"field": {"samples": 3, "calls": 3}}}}}}, {"key": "second", "report": {"stats": {"budget": {"stages": {}}}}}]
        result = costs(rows)["metrics"]["fieldsamples"]
        self.assertEqual(result["mean"], 1.5)
        self.assertEqual(result["sources"], 2)

    def testentrypoints(self):
        import contextlib
        import io
        import probe
        import run
        import study
        import trial
        from unittest.mock import patch
        for module in (probe, run, study, trial):
            with self.subTest(module=module.__name__), patch("sys.argv", [module.__name__, "--help"]), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as result:
                    module.main()
                self.assertEqual(result.exception.code, 0)

    def testunscoredcontrols(self):
        from PIL import Image
        from run import run
        from session import editor

        class Forbidden:
            def stats(self):
                return {"evaluations": 0}

            def prepare(self, *args, **kwargs):
                raise AssertionError("Uncontrolled runs must not prepare rewards.")

        for mode in ("raw", "fixed"):
            args = common(argparse.ArgumentParser()).parse_args(["--method", mode, "--field", "chord", "--axes", "none", "--size", "16", "--prior", "off"])
            _, report, _ = run(editor(), Forbidden(), Image.new("RGB", (16, 16)), "source", "target", args)
            self.assertFalse(report["controlactive"])
            self.assertIsNone(report["scores"])
            self.assertEqual(report["evaluations"], 0)
            self.assertEqual(report["fieldnorm"], report["adoptedfieldnorm"])

    def testminimalfixed(self):
        from PIL import Image
        from run import run
        from session import editor
        for eta in (0., .5, 1.):
            args = common(argparse.ArgumentParser()).parse_args(["--method", "fixed", "--field", "chord", "--axes", "none", "--eta", str(eta), "--size", "16", "--prior", "off"])
            _, report, session = run(editor(), None, Image.new("RGB", (16, 16)), "source", "target long", args)
            self.assertEqual(session.samples, 6)
            self.assertEqual(report["backend"]["endpoints"], 2)

    def testcanonicalqueries(self):
        from PIL import Image
        from engine.native import create
        from session import editor
        session = editor().prepare(Image.new("RGB", (16, 16)), "source", "target long")
        path = Path(create(session), session.field(.9, .15))
        previous = session.samples
        self.assertEqual(path.canonical((1., 1., 0.)), (1., 1., 0.))
        self.assertEqual(path.canonical((1., 1., 1.)), (1., 1., 1.))
        self.assertEqual(session.samples, previous)
        path.initialize()
        previous = session.samples, path.hits
        self.assertEqual(path.metadata(), path.metadata())
        self.assertEqual((session.samples, path.hits), previous)

    def testcorrection(self):
        response, target, anchor = np.array([[2., .4], [0., .5]]), np.ones(2), np.ones(2)
        coefficients, report = solve(response, target, anchor)
        correction = np.array(report["correction"])
        self.assertTrue(np.allclose(coefficients, anchor + correction))
        residual = target - response @ anchor
        expected = np.linalg.norm(response @ correction - residual) ** 2 + report["damping"] * (correction @ correction)
        self.assertAlmostEqual(expected, report["objective"])

    def testnormmatched(self):
        first, second = Session(), Session()
        coupled = build(first, Native(first))
        scalar = build(second, Native(second), geometry="scalar")
        self.assertAlmostEqual(float(coupled.value.norm()), float(scalar.value.norm()), places=6)
        self.assertEqual(first.samples, second.samples)
        self.assertEqual(first.calls, second.calls)
        direction = torch.tensor([1., 1., 0., 0.]).reshape_as(scalar.value)
        self.assertTrue(torch.allclose(scalar.value / scalar.value.norm(), direction / direction.norm()))
        self.assertGreater(scalar.report["model"]["appliedafter"], coupled.report["model"]["appliedafter"])

    def testalignmentmatrix(self):
        matrix = variants("alignment")
        self.assertEqual(len(matrix), 4)
        for changes in matrix.values():
            args = common(argparse.ArgumentParser()).parse_args(["--clip", "clip"])
            for key, value in changes.items():
                setattr(args, key, value)
            validate(args)

    def testnormoptions(self):
        with self.assertRaises(ValueError):
            options("chord", .05, .25, 1., 1e-6, "scalar")

    def testlazyreferences(self):
        from PIL import Image
        from engine.native import Native as Backend
        from session import editor
        for mode in ("latent", "mix", "random", "rgb", "cfg"):
            for steps in (1, 2):
                for eta in (0., .5, 1.):
                    with self.subTest(mode=mode, steps=steps, eta=eta):
                        first = editor().prepare(Image.new("RGB", (16, 16)), "source", "target long")
                        second = editor().prepare(Image.new("RGB", (16, 16)), "source", "target long")
                        lazy = Path(Backend(first, steps), first.field(.9, .15), mode)
                        eager = Path(Backend(second, steps), second.field(.9, .15), mode)
                        eager.initialize()
                        left, right = lazy.candidate((1., 1., eta)), eager.candidate((1., 1., eta))
                        self.assertTrue(torch.allclose(left[0], right[0], atol=1e-7))
                        self.assertTrue(torch.allclose(left[1], right[1], atol=1e-7))
                        self.assertLessEqual(first.samples, second.samples)

    def testmemorycoverage(self):
        from evaluate import costs
        rows = [{"key": "first", "vram": 14., "residentvram": 10., "incrementalvram": 4., "report": {}}, {"key": "second", "vram": 16., "residentvram": 10., "incrementalvram": 6., "report": {}}]
        result = costs(rows)["metrics"]
        self.assertEqual(result["vram"]["mean"], 15.)
        self.assertEqual(result["residentvram"]["mean"], 10.)
        self.assertEqual(result["incrementalvram"]["mean"], 5.)

    def testunresolvedrequest(self):
        class Unresolved(Session):
            def difference(self, tick):
                value = [1., 1e-8, 0., 0.] if tick > .8 else [-1., 1e-8, 0., 0.]
                return torch.tensor(value).reshape_as(self.anchor)

        for geometry in ("coupled", "scalar"):
            session = Unresolved()
            field = build(session, Native(session), geometry=geometry)
            self.assertFalse(torch.any(field.value != 0).item())
            self.assertEqual(field.report["status"], "degenerate")
            self.assertEqual(field.report["rank"], 1)
            self.assertEqual(field.report["probes"], 0)
            self.assertEqual(session.samples, 0)

    def testscorerlifetime(self):
        import gc
        from weakref import ref
        from engine.judge import ClipJudge
        from session import Clip, Processor
        judge = ClipJudge(Clip(), Processor())
        previous = gc.isenabled()
        gc.disable()
        try:
            scorer = judge.prepare(torch.zeros(1, 3, 16, 16), "source", "target")
            scorer(torch.ones(1, 3, 16, 16))
            self.assertEqual(scorer.evaluations, 1)
            reference = ref(scorer)
            del scorer
            self.assertIsNone(reference())
        finally:
            if previous:
                gc.enable()

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu 2026
# Email: wlzchina05@gmail.com
