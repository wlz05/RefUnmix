# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import argparse
import json
import math
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from audit import values
from engine.field import build, capacity, compatible, options, orthogonal, solve, temporal
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

    def testrepairspan(self):
        matrix = torch.eye(4)
        matrix[2, 0] = 1.
        session = Session()
        native = Native(session, matrix)
        result = build(session, native, repair=1)
        report = result.report["repair"]
        self.assertEqual(report["accepted"], 1)
        self.assertEqual(report["history"][0]["rank"], 3)
        self.assertLess(report["history"][0]["after"], report["history"][0]["before"])
        self.assertGreater(result.report["temporal"]["spanerror"], 0.)
        self.assertFalse(result.report["usesclip"])
        self.assertEqual(result.report["calls"], session.limit.stages["field"]["calls"])
        self.assertEqual(result.report["samples"], session.limit.stages["field"]["samples"])
        previous = session.samples
        path = Path(native, result.value, neutral=result.neutral, endpoint=result.endpoint)
        self.assertTrue(torch.equal(path.target(), result.endpoint))
        self.assertEqual(session.samples, previous)
        result.observe(result.endpoint)
        summary = values({"report": {"field": result.report}})
        self.assertEqual(summary["field:repairaccepted"], 1)
        self.assertEqual(summary["field:repairfinalerror"], report["finalerror"])
        self.assertNotIn("field:repairfinalerror", values({"report": {}}))
        json.dumps(result.report, allow_nan=False)

    def testrepairbudget(self):
        for maximum in range(8, 16):
            session = Session(maximum=maximum)
            result = build(session, Native(session), repair=2)
            self.assertLessEqual(session.samples, maximum)
            self.assertEqual(session.samples, session.limit.used)
            self.assertEqual(session.limit.phase, "front")
            if result.report.get("repair", {}).get("status") == "budget":
                self.assertIsNotNone(result.endpoint)
                self.assertTrue(torch.isfinite(result.value).all())

    def testrepairrejection(self):
        class Curved(Native):
            def endpoint(self, value, eta=0.):
                output = super().endpoint(value, eta)
                if len(self.inputs) >= 7:
                    output = output + 100.
                return output

        session = Session()
        native = Curved(session)
        result = build(session, native, repair=1)
        report = result.report["repair"]
        self.assertEqual(report["status"], "rejected")
        self.assertEqual(report["accepted"], 0)
        self.assertTrue(torch.equal(result.value, native.inputs[3]))
        self.assertTrue(torch.allclose(result.endpoint.reshape(-1), native.offset + native.matrix @ result.value.reshape(-1)))

    def testrepairbounds(self):
        for value in (True, -1, 3, 1.5):
            session = Session()
            with self.assertRaises(ValueError):
                build(session, Native(session), repair=value)
        for values in ({"kind": "chord"}, {"geometry": "scalar"}):
            session = Session()
            with self.assertRaises(ValueError):
                build(session, Native(session), repair=1, **values)
        move, report = solve(np.eye(2), np.ones(2), np.zeros(2), bound=.1)
        self.assertLessEqual(np.linalg.norm(move), .1 + 1e-12)
        self.assertGreater(report["before"], report["after"])
        with self.assertRaises(ValueError):
            solve(np.eye(2), np.ones(2), np.zeros(2), bound=float("nan"))

    def testrepairnonlinear(self):
        class Curved(Native):
            def endpoint(self, value, eta=0.):
                return super().endpoint(value, eta) + .5 * value.square()

        session = Session()
        result = build(session, Curved(session), repair=2)
        accepted = [row for row in result.report["repair"]["history"] if row["accepted"]]
        self.assertTrue(accepted)
        for row in accepted:
            self.assertLess(row["after"], row["before"])

    def testrepairrepeatability(self):
        state = torch.random.get_rng_state().clone()
        results = []
        for _ in range(2):
            session = Session()
            result = build(session, Native(session), repair=2)
            results.append((result.value, result.report["repair"]))
        self.assertTrue(torch.equal(results[0][0], results[1][0]))
        self.assertEqual(results[0][1], results[1][1])
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

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
        output, report, session = run(editor(), None, Image.new("RGB", (16, 16)), "source", "target long", args)
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

class CapacityTests(unittest.TestCase):
    def testanisotropy(self):
        row = capacity(np.diag([2., .5]), np.ones(2), np.ones(2))
        self.assertAlmostEqual(row["lineerror"], 9 / 34)
        self.assertAlmostEqual(row["spanerror"], 0.)
        self.assertAlmostEqual(row["directionroom"], 9 / 34)
        json.dumps(row, allow_nan=False)

    def testunreachable(self):
        row = capacity([[1.], [0.]], [1., 1.], [1.])
        self.assertAlmostEqual(row["lineerror"], .5)
        self.assertAlmostEqual(row["spanerror"], .5)
        self.assertAlmostEqual(row["directionroom"], 0.)

    def testidentity(self):
        row = capacity(np.eye(2), np.ones(2), np.ones(2))
        self.assertAlmostEqual(row["directionroom"], 0.)
        self.assertAlmostEqual(row["linecosine"], 1.)

    def testnegative(self):
        row = capacity([[-2.], [0.]], [1., 0.], [1.])
        self.assertAlmostEqual(row["lineerror"], 0.)
        self.assertAlmostEqual(row["directionroom"], 0.)
        self.assertAlmostEqual(row["linecosine"], -1.)

    def testzerorequest(self):
        row = capacity(np.eye(2), np.zeros(2), np.ones(2))
        self.assertEqual(row["status"], "zero request")
        self.assertIsNone(row["directionroom"])
        self.assertEqual(row["rank"], 2)

    def testzeroresponse(self):
        row = capacity(np.zeros((2, 2)), np.ones(2), np.ones(2))
        self.assertEqual(row["status"], "zero response")
        self.assertEqual(row["rank"], 0)
        self.assertAlmostEqual(row["lineerror"], 1.)
        self.assertAlmostEqual(row["spanerror"], 1.)
        self.assertEqual(row["directionroom"], 0.)
        self.assertIsNone(row["linecosine"])

    def testzeroanchor(self):
        row = capacity(np.eye(2), [1., 0.], [0., 0.])
        self.assertEqual(row["status"], "zero baseline")
        self.assertEqual(row["directionroom"], 1.)

    def testnullanchor(self):
        row = capacity([[1., -1.], [0., 0.]], [1., 0.], [1., 1.])
        self.assertEqual(row["status"], "zero baseline")
        self.assertEqual(row["directionroom"], 1.)

    def testtruncation(self):
        row = capacity(np.diag([1., 1e-8]), [0., 1.], [0., 1.])
        self.assertEqual(row["rank"], 1)
        self.assertEqual(row["status"], "zero baseline")
        self.assertEqual(row["spanerror"], 1.)
        self.assertEqual(row["directionroom"], 0.)

    def testunits(self):
        reference = capacity(np.diag([2., .5]), np.ones(2), np.ones(2))
        for scale in (1e-250, 1e250):
            row = capacity(scale * np.diag([2., .5]), scale * np.ones(2), scale * np.ones(2))
            for key in ("lineerror", "spanerror", "directionroom", "linecosine"):
                self.assertAlmostEqual(reference[key], row[key])

    def testcovariance(self):
        rotation = np.array([[.6, -.8], [.8, .6]])
        matrix, target, anchor = np.array([[2., .4], [0., .5], [0., 0.]]), np.array([1., 1., 1.]), np.ones(2)
        first = capacity(matrix, target, anchor)
        second = capacity(matrix @ rotation, target, rotation.T @ anchor)
        for key in ("lineerror", "spanerror", "directionroom", "linecosine"):
            self.assertAlmostEqual(first[key], second[key])

    def testinvalid(self):
        for matrix, target, anchor in (([], [], []), (np.eye(2), [1.], [1., 1.]), ([[float("inf")]], [1.], [1.]), ([[1.]], [float("nan")], [1.])):
            with self.assertRaises(ValueError):
                capacity(matrix, target, anchor)
        for tolerance in (0., 1., -1., True, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                capacity(np.eye(2), np.ones(2), np.ones(2), tolerance)

    def testpaidprobes(self):
        first, second = Session(), Session()
        coupled = build(first, Native(first))
        scalar = build(second, Native(second), geometry="scalar")
        self.assertEqual(coupled.report["capacity"], scalar.report["capacity"])
        self.assertGreater(coupled.report["capacity"]["directionroom"], .25)
        self.assertEqual((first.samples, first.calls), (7, 5))
        self.assertEqual((first.samples, first.calls), (second.samples, second.calls))
        measured = values({"report": {"field": coupled.report}})
        self.assertEqual(measured["field:responsedirectionroom"], coupled.report["capacity"]["directionroom"])

class TransportTests(unittest.TestCase):
    def testblend(self):
        for weight in (0., .25, .5, .75, 1.):
            session = Session()
            result = build(session, Native(session), "blend", weight=weight)
            expected = torch.tensor([2 * (1 - weight), 2 * weight, 0., 0.]).reshape_as(session.anchor)
            self.assertTrue(torch.equal(result.value, expected))
            self.assertEqual((session.samples, result.report["probes"]), (4, 0))

    def testmean(self):
        session = Session()
        mean = build(session, Native(session), "average")
        blend = build(session, Native(session), "blend", weight=.5)
        self.assertTrue(torch.equal(mean.value, blend.value))

    def testect(self):
        from PIL import Image
        from session import editor
        session = editor().prepare(Image.new("RGB", (16, 16)), "source", "target long")
        result = build(session, None, "ect", start=.75, delta=0)
        variance = float(session.editor.alphas[session.editor.index(.75)])
        sigma = math.sqrt(1 - variance)
        expected = math.atan2(sigma, math.sqrt(variance)) / sigma - .7
        self.assertAlmostEqual(result.report["gain"], expected)
        contrast = session.difference(session.editor.index(.75))
        self.assertTrue(torch.allclose(result.value, (expected * contrast.double()).float()))
        self.assertEqual((session.samples, session.calls), (2, 1))
        self.assertEqual(result.report["times"], [.75])

    def testectlimit(self):
        from PIL import Image
        from session import editor
        session = editor().prepare(Image.new("RGB", (16, 16)), "source", "target long")
        session.editor.alphas[session.editor.index(.75)] = 1.
        result = build(session, None, "ect", start=.75, delta=0)
        self.assertAlmostEqual(result.report["gain"], .3)
        self.assertFalse(torch.any(result.value != 0).item())

    def testectparameterization(self):
        from PIL import Image
        from session import editor
        for kind in ("sample", "v_prediction"):
            session = editor(kind).prepare(Image.new("RGB", (16, 16)), "source", "target long")
            with self.assertRaisesRegex(ValueError, "epsilon"):
                build(session, None, "ect", delta=0)
            self.assertEqual(session.samples, 0)
        with self.assertRaisesRegex(ValueError, "VP"):
            compatible(SimpleNamespace(metadata={"family": "flux"}, kind="epsilon"), "ect")

    def testectvalidation(self):
        for flags in (("--field", "ect"), ("--field", "ect", "--delta", "0", "--backbone", "flux"), ("--localizationtime", "nan"), ("--localizationtime", "0"), ("--localizationtime", ".9", "--localization", "attention")):
            args = common(argparse.ArgumentParser()).parse_args(["--clip", "clip", *flags])
            with self.assertRaises(ValueError):
                validate(args)

    def testectdirect(self):
        from PIL import Image
        from run import run
        from session import editor
        args = common(argparse.ArgumentParser()).parse_args(["--method", "raw", "--field", "ect", "--start", ".75", "--delta", "0", "--no-cleanup"])
        _, report, _ = run(editor(), None, Image.new("RGB", (16, 16)), "source", "target long", args)
        self.assertEqual((report["field"]["actual"], report["stats"]["samples"], report["stats"]["calls"]), ("ect", 2, 1))

    def testlocalization(self):
        from PIL import Image
        from run import run
        from session import editor
        parser = common(argparse.ArgumentParser())
        mean = parser.parse_args(["--method", "raw", "--field", "average"])
        ect = parser.parse_args(["--method", "raw", "--field", "ect", "--start", ".75", "--delta", "0", "--localizationtime", ".9"])
        image = Image.new("RGB", (16, 16))
        _, first, _ = run(editor(), None, image, "source", "target long", mean)
        _, second, session = run(editor(), None, image, "source", "target long", ect)
        self.assertEqual(first["protectionsha256"], second["protectionsha256"])
        self.assertEqual(second["localization"]["time"], .9)
        self.assertTrue({session.editor.index(.9), session.editor.index(.75)} <= session.cache.keys())
        self.assertEqual(second["stats"]["samples"], 5)

    def testtemporalweights(self):
        report = temporal(np.eye(2), [1., 3.])
        np.testing.assert_allclose(report["coefficients"], [1., 3.])
        self.assertTrue(report["nonnegative"])
        self.assertEqual(report["strength"], 4.)
        self.assertEqual(report["weight"], .75)
        self.assertAlmostEqual(report["spanerror"], 0.)

    def testtemporalsigned(self):
        report = temporal(np.eye(2), [1., -1.])
        self.assertFalse(report["nonnegative"])
        self.assertAlmostEqual(report["spanerror"], 0.)
        self.assertAlmostEqual(report["coneerror"], .5)

    def testtemporaldependent(self):
        report = temporal([[1., -1.], [0., 0.]], [1., 0.])
        self.assertTrue(report["nonnegative"])
        np.testing.assert_allclose(report["nonnegativeweights"], [1., 0.])
        self.assertEqual(report["rank"], 1)

    def testtemporalincomplete(self):
        report = temporal([[1.], [0.]], [1., 1.])
        self.assertAlmostEqual(report["spanerror"], .5)
        self.assertFalse(report["nonnegative"])
        self.assertIsNone(report["weight"])

    def testtemporalzero(self):
        report = temporal(np.eye(2), np.zeros(2))
        self.assertEqual(report["status"], "zero field")
        self.assertIsNone(report["nonnegative"])
        self.assertIsNone(report["spanerror"])
        self.assertIsNone(report["coneerror"])

    def testtemporalunits(self):
        for scale in (1e-250, 1e250):
            report = temporal(scale * np.eye(2), scale * np.array([1., 3.]))
            np.testing.assert_allclose(report["coefficients"], [1., 3.])
            self.assertEqual(report["weight"], .75)

    def testtemporalinvalid(self):
        for matrix, value in (([], []), (np.eye(3), np.ones(3)), (np.eye(2), [1.]), ([[float("nan")]], [1.])):
            with self.assertRaises(ValueError):
                temporal(matrix, value)
        for tolerance in (0., 1., True, float("nan")):
            with self.assertRaises(ValueError):
                temporal(np.eye(2), np.ones(2), tolerance)

    def testtemporalsuite(self):
        from trial import design
        args = common(argparse.ArgumentParser()).parse_args(["--clip", "clip"])
        args.suite = "temporal"
        controls = variants("temporal", base=args)
        plan = design(args, controls)
        self.assertEqual(len(controls), 15)
        self.assertFalse(plan["variants"]["full"]["changes"])
        self.assertFalse(controls["ectdirect"]["cleanup"])
        self.assertIsNone(controls["ectdirect"]["localizationtime"])
        self.assertEqual(controls["ectfull"]["localizationtime"], args.start)
        self.assertEqual((controls["ectfull"]["method"], controls["ectraw"]["method"]), ("path", "raw"))

    def testtemporalreport(self):
        session = Session()
        report = build(session, Native(session)).report
        self.assertLess(report["temporal"]["spanerror"], 1e-12)
        self.assertTrue(report["temporal"]["nonnegative"])
        self.assertIn("field:temporalweight", values({"report": {"field": report}}))
        json.dumps(report, allow_nan=False)

class TangentTests(unittest.TestCase):
    def testmetricallocation(self):
        from engine.geometry import tangent
        anchor = np.array([1., 0., 0.])
        metric = np.diag([2., 3., 4.])
        value, report = tangent(np.eye(3), np.array([[0., 0., 1.]]), np.ones(3), anchor, metric, 1., .01, .3)
        self.assertAlmostEqual(float(anchor @ metric @ value), 0., places=12)
        self.assertLessEqual(float(value @ metric @ value), .09 + 1e-10)
        self.assertGreater(value[1], value[2])
        self.assertEqual(report["rank"], 2)
        json.dumps(report, allow_nan=False)

    def testtangentcovariance(self):
        from engine.geometry import tangent
        edit = np.array([[1., .3, 0.], [0., 1., 2.]])
        protect = np.array([[.2, 1., .3]])
        anchor, metric = np.array([1., .2, -.1]), np.diag([1., 2., 3.])
        transform = np.array([[1., .3, 0.], [0., 2., .2], [.1, 0., 1.]])
        left, _ = tangent(edit, protect, np.ones(2), anchor, metric)
        right, _ = tangent(edit @ transform, protect @ transform, np.ones(2), np.linalg.solve(transform, anchor), transform.T @ metric @ transform)
        np.testing.assert_allclose(left, transform @ right, atol=1e-9)

    def testprotectioninterval(self):
        from engine.geometry import adoption
        self.assertAlmostEqual(adoption([-.1], [1.])[0], .2)
        self.assertEqual(adoption([.1], [1.])[0], 0.)
        self.assertEqual(adoption([0.], [1.])[0], 0.)
        self.assertEqual(adoption([0.], [0.])[0], 1.)
        self.assertEqual(adoption([], [])[0], 1.)
        with self.assertRaises(ValueError):
            adoption([0.], [np.nan])

    def testtangentdegenerate(self):
        from engine.geometry import tangent
        for anchor in ([0.], [1.]):
            value, report = tangent(np.ones((1, 1)), np.zeros((0, 1)), [1.], anchor, np.eye(1))
            np.testing.assert_array_equal(value, [0.])
        for metric in (np.zeros((2, 2)), np.array([[1., 1.], [0., 1.]])):
            with self.assertRaises(ValueError):
                tangent(np.eye(2), np.eye(2), [1., 1.], [1., 0.], metric)

    def fixture(self, maximum=64, harmful=False):
        from engine.field import Field
        session = Session(maximum=maximum)
        session.origin = torch.full((1, 3, 2, 2), .5)
        session.decodes = 0
        session.difference = lambda time: torch.tensor([1., 0., 0., 0.]).reshape_as(session.anchor)
        native = Native(session, torch.eye(4))
        native.offset = torch.zeros(4)
        def decode(latent):
            session.decodes += 1
            result = session.origin.clone()
            amount = float(latent.flatten()[1])
            result[..., 0, 0] += .1 * (1 + amount if harmful and len(native.inputs) >= 3 else 1 - amount)
            result[..., 0, 1] += .1 * amount
            return result
        session.decode = decode
        value = torch.tensor([1., 0., 0., 0.]).reshape_as(session.anchor)
        request = torch.tensor([1., 1., 0., 0.]).reshape_as(session.anchor)
        field = Field(value.clone(), {"times": [.9, .75]}, torch.zeros_like(value), request)
        mask = torch.zeros(1, 1, 2, 2)
        mask[..., 0, 1] = 1
        def score(image):
            value = float(image[..., 0, 1].mean())
            return {"target": value, "local": value, "localviews": 1}
        return session, native, field, mask, score

    def testactualredirect(self):
        from engine.field import redirect
        session, native, field, mask, score = self.fixture()
        baseline = field.value.clone()
        redirect(field, native, score, mask)
        report = field.report["tangent"]
        self.assertTrue(report["accepted"])
        self.assertGreater(float(field.value.flatten()[1]), 0.)
        self.assertLessEqual(report["damage"], report["baselinedamage"] + 1e-12)
        self.assertLess(abs(float((baseline * (field.value - baseline)).sum())), 1e-6)
        self.assertFalse(report["fieldmasking"])
        self.assertEqual(session.samples, session.limit.used)
        self.assertEqual(session.limit.phase, "front")
        self.assertTrue(torch.equal(field.baseline, baseline))
        json.dumps(report, allow_nan=False)

    def testtemporalcapacity(self):
        from engine.field import redirect
        session, native, field, mask, score = self.fixture()
        redirect(field, native, score, mask, family="temporal")
        self.assertFalse(field.report["tangent"]["accepted"])
        self.assertEqual(field.report["tangent"]["rank"], 1)
        self.assertEqual(session.samples, 1)

    def testactualfallback(self):
        from engine.field import redirect
        for maximum, harmful, match in ((2, False, False), (64, True, False), (64, True, True)):
            with self.subTest(maximum=maximum, harmful=harmful, match=match):
                session, native, field, mask, score = self.fixture(maximum, harmful)
                baseline = field.value.clone()
                redirect(field, native, score, mask, match=match)
                self.assertFalse(field.report["tangent"]["accepted"])
                self.assertTrue(torch.equal(field.value, baseline))
                self.assertTrue(torch.equal(field.endpoint, baseline))
                self.assertLessEqual(session.samples, maximum)
                self.assertEqual(session.limit.phase, "front")

    def testtangentflags(self):
        from run import validate
        for flags in (["--fieldrepair", "1"], ["--field", "chord"], ["--fieldgeometry", "scalar"], ["--no-cleanup"], ["--method", "fixed"], ["--tangentchecks", "0"]):
            args = common(argparse.ArgumentParser()).parse_args(["--clip", "clip", "--tangent", *flags])
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                validate(args)
        args = common(argparse.ArgumentParser()).parse_args(["--method", "raw", "--tangent"])
        with self.assertRaises(ValueError):
            validate(args)

    def testtangentemptyprotection(self):
        from engine.field import redirect
        session, native, field, mask, score = self.fixture()
        redirect(field, native, score, torch.ones_like(mask))
        report = field.report["tangent"]
        self.assertTrue(report["accepted"])
        self.assertFalse(report["protectedvalid"])
        self.assertEqual(report["damage"], 0.)

    def testtangentzeroanchor(self):
        from engine.field import redirect
        session, native, field, mask, score = self.fixture()
        field.value.zero_()
        redirect(field, native, score, mask)
        report = field.report["tangent"]
        self.assertFalse(report["accepted"])
        self.assertEqual(report["gain"], 0.)
        self.assertEqual(session.samples, 0)

    def testtangentconflictingheads(self):
        from engine.field import redirect
        session, native, field, mask, score = self.fixture()
        def conflicting(image):
            value = float(image[..., 0, 1].mean())
            return {"target": value, "local": -value, "localviews": 1}
        redirect(field, native, conflicting, mask)
        self.assertFalse(field.report["tangent"]["accepted"])
        self.assertEqual(field.report["tangent"]["gain"], 0.)

    def testtangentretention(self):
        from engine.field import redirect
        session, native, field, mask, score = self.fixture()
        def retaining(image):
            value = score(image)
            value["retain0"] = 1 - value["target"]
            return value
        redirect(field, native, retaining, mask, slacks={"retain0": 0.})
        self.assertFalse(field.report["tangent"]["accepted"])
        self.assertEqual(field.report["tangent"]["gain"], 0.)

    def testresidualallocation(self):
        from engine.geometry import tangent
        edit, protect = np.array([[0., 1., 1.]]), np.array([[0., 1., -1.]])
        first, _ = tangent(edit, protect, [1.], [1., 0., 0.], np.eye(3))
        second, _ = tangent(edit, protect, [1.], [1., 0., 0.], np.eye(3), residual=[1.])
        self.assertAlmostEqual(float((protect @ first).item()), 0.)
        self.assertLess(float((protect @ second).item()), 0.)
        self.assertGreater(float((edit @ second).item()), 0.)

    def testnormintervals(self):
        from engine.geometry import matching
        generator = np.random.default_rng(42)
        for index in range(40):
            displacement, effect = generator.normal(size=(2, 4))
            if index == 0:
                effect = -3 * displacement
            intervals = matching(displacement, effect, 1., .01)
            for alpha in np.linspace(0, 1, 501):
                actual = abs(np.linalg.norm(displacement + alpha * effect) / np.linalg.norm(displacement) - 1) <= .01 + 1e-13
                predicted = any(low - 1e-14 <= alpha <= high + 1e-14 for low, high in intervals)
                self.assertEqual(actual, predicted)
            for scale in (1e-200, 1e200):
                np.testing.assert_allclose(matching(scale * displacement, scale * effect), intervals, atol=1e-12)
        self.assertEqual(matching([0.], [1.]), [])
        self.assertEqual(matching([1.], [0.]), [(0., 1.)])
        self.assertEqual(len(matching([1.], [-3.])), 2)

    def testtolerantnorm(self):
        from engine.field import redirect
        session, native, field, mask, score = self.fixture()
        redirect(field, native, score, mask, match=True)
        report = field.report["tangent"]
        self.assertTrue(report["accepted"])
        self.assertLessEqual(abs(report["endpointnormratio"] - 1), .01)
        self.assertLessEqual(report["damage"], report["baselinedamage"])
        self.assertEqual(report["probes"], report["rank"] - 1)

    def testacceptedbudget(self):
        from engine.field import redirect
        session, native, field, mask, score = self.fixture(3)
        redirect(field, native, score, mask)
        report = field.report["tangent"]
        self.assertTrue(report["accepted"])
        self.assertTrue(report["budgetstopped"])
        self.assertEqual(report["checks"], 1)
        self.assertEqual(report["samples"], 3)
        self.assertEqual(report["calls"], session.limit.forwards)
        self.assertEqual(report["decodes"], report["evaluations"])
        self.assertEqual(report["endpoints"], 1 + report["probes"] + report["checks"])
        self.assertAlmostEqual(report["inputnormratio"], float(field.value.double().norm()))
        self.assertTrue(torch.equal(native.endpoint.__self__.inputs[-1], field.value))
        self.assertEqual(score(field.pixels), field.scores)
        self.assertEqual(session.limit.phase, "front")

    def testcachedbaseline(self):
        from engine.field import redirect
        session, native, field, mask, score = self.fixture()
        field.endpoint = native.endpoint(field.value)
        before = session.samples
        redirect(field, native, score, mask)
        report = field.report["tangent"]
        self.assertEqual(report["samples"], session.samples - before)
        self.assertEqual(report["endpoints"], report["probes"] + report["checks"])
        self.assertEqual(report["evaluations"], report["endpoints"] + 1)

    def testprotectedgeometry(self):
        from engine.field import redirect
        session, native, field, mask, score = self.fixture()
        session.decode = lambda latent: torch.ones(1, 3, 1, 1)
        with self.assertRaisesRegex(ValueError, "source geometry"):
            redirect(field, native, score, mask)
        self.assertEqual(session.limit.phase, "front")

    def testfixedamplitude(self):
        from engine.field import rescale
        for factor in (0., .5, .75, 1., 1.25, 1.5, 2.):
            session = Session()
            field = build(session, Native(session))
            initial = field.value.clone()
            before = session.samples, session.calls
            rescale(field, factor)
            torch.testing.assert_close(field.value, initial * factor)
            self.assertEqual((session.samples, session.calls), before)
            self.assertEqual(field.report["radial"]["factor"], factor)
            if factor == 0:
                torch.testing.assert_close(field.endpoint, field.neutral)
            elif factor != 1:
                self.assertIsNone(field.endpoint)

    def testfeedbackresidual(self):
        from engine.field import redirect
        session, native, field, mask, score = self.fixture()
        original = session.decode
        def nonlinear(latent):
            pixels = original(latent)
            amount = float(latent.flatten()[1])
            pixels[..., 0, 1] -= .07 * amount * amount
            return pixels
        session.decode = nonlinear
        redirect(field, native, score, mask, feedback=True)
        report = field.report["tangent"]
        self.assertTrue(report["feedback"])
        self.assertTrue(report["updates"])
        self.assertGreater(report["updates"][0]["error"]["semantic"], 0)
        self.assertTrue(all(row["queries"] == 0 for row in report["updates"]))
        self.assertLessEqual(report["checks"], 3)
        self.assertEqual(report["samples"], 1 + report["probes"] + report["checks"])
        self.assertLessEqual(report["damage"], report["baselinedamage"])
        self.assertEqual(session.limit.used, session.samples)
        self.assertEqual(score(field.pixels), field.scores)
        json.dumps(report, allow_nan=False)

    def testfeedbackbudget(self):
        from engine.field import redirect
        session, native, field, mask, score = self.fixture(3)
        redirect(field, native, score, mask, feedback=True)
        report = field.report["tangent"]
        self.assertTrue(report["accepted"])
        self.assertTrue(report["budgetstopped"])
        self.assertEqual(report["checks"], 1)
        self.assertEqual(report["calls"], 3)
        self.assertEqual(session.limit.phase, "front")

    def testattentionoperator(self):
        from engine.geometry import tangent
        value, report = tangent([[0., 2., .1]], [[0., .5, 1.]], [1.], [1., 0., 0.], np.eye(3), residual=[.2])
        attention = report["attention"]
        self.assertTrue(all(0 <= gate <= 1 for gate in attention["gates"]))
        self.assertAlmostEqual(sum(attention["energy"]), 1.)
        self.assertAlmostEqual(value[0], 0.)

    def testzerorequestnorm(self):
        from engine.field import Field
        field = Field(torch.zeros(1, 1, 2, 2), {}, torch.zeros(1, 1, 2, 2))
        field.observe(field.neutral)
        self.assertEqual(field.report["responsenorm"], 0)
        self.assertEqual(field.report["responsestatus"], "zero response")

    def testfeedbackfailure(self):
        from engine.field import redirect
        from unittest.mock import patch
        session, native, field, mask, score = self.fixture()
        with patch("engine.theory.fit", side_effect=np.linalg.LinAlgError("Singular response.")):
            redirect(field, native, score, mask, feedback=True)
        report = field.report["tangent"]
        self.assertTrue(report["accepted"])
        self.assertEqual(report["checks"], 1)
        self.assertIn("feedbackfailure", report)
        self.assertEqual(score(field.pixels), field.scores)
        self.assertEqual(session.limit.phase, "front")
        self.assertLessEqual(report["damage"], report["baselinedamage"])

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
