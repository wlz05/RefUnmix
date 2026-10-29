# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import argparse
import hashlib
import json
import os
from pathlib import Path as File
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
from engine.native import Native
from engine.adapter import Adapter
from engine.limit import Budget, Exhausted
from engine.path import Path
from engine.poll import Config, select
from engine.prior import Prior
from engine.trust import Model, geometry, propose
from engine.checkpoint import configuration, fingerprint, signature
from run import common, jsonfile, run, validate
from engine.judge import ClipJudge
from trial import settings, variants
from data import destination, localdescription
from blind import build, collect
from report import assess
from session import Clip, Processor, editor

class Fake:
    def __init__(self):
        self.session = SimpleNamespace(origin=torch.zeros(1, 3, 4, 4), anchor=torch.zeros(1, 3, 4, 4), limit=None)
        self.native = SimpleNamespace(continuous=True, metadata=lambda: {})
        self.seed = 42

    def raw(self):
        image = torch.full_like(self.session.origin, .3)
        return image.clone(), image

    def initialize(self):
        return None

    def canonical(self, point):
        return tuple(round(float(value), 12) for value in point)

    def axes(self, active):
        return active

    def candidate(self, point):
        low, high, eta = point
        value = .05 + .1 * (low - .5) ** 2 + .02 * (high - .7) ** 2 + .01 * (eta - .8) ** 2
        pixels = torch.full_like(self.session.origin, value)
        return pixels.clone(), pixels

    def metadata(self):
        return {}

class Expensive:
    mode = "ddpm"

    def __init__(self):
        self.evaluations = 0

    def __call__(self, latent):
        self.evaluations += 1
        return 0. if float(latent.mean()) > .29 else 1e100

    def metadata(self):
        return {"evaluations": self.evaluations}

class TrustTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def session(self):
        return editor().prepare(Image.new("RGB", (16, 16), (96, 128, 160)), "source", "target long")

    def search(self, **values):
        path = Fake()
        return select(path, lambda image: {"target": 1., "attribute0": .5}, torch.ones_like(path.session.origin[:, :1]), Expensive(), Config(**values))

    def model(self, rank=True, penalty=True):
        anchor = np.array([1., 1., .5])
        points = np.array([anchor, anchor + [.5, 0, 0], anchor + [-.5, 0, 0], anchor + [0, .5, 0], anchor + [0, -.5, 0], anchor + [0, 0, .5], anchor + [0, 0, -.5]])
        if not rank:
            points = points[:3]
        residuals = [.2 + np.array([[1., 0., .2], [0., 1., -.1]]) @ (point - anchor) for point in points]
        scores = [np.array([1.]) + np.array([[0., .1, .3]]) @ (point - anchor) for point in points]
        penalties = [.04 + .03 * (point[0] - 1) for point in points] if penalty else [None] * len(points)
        return Model(anchor, points, residuals, scores, penalties, (0, 1, 2), regularization=1e-10), points

    def testboundedbias(self):
        result = self.search()
        report = result.report
        self.assertFalse(report["fallback"])
        self.assertLess(report["penalty"], report["maximumpriorbias"])
        self.assertLessEqual(report["objective"], report["rawdamage"])
        self.assertLessEqual(report["damage"], report["rawdamage"])
        self.assertGreater(report["rawdamage"] - report["damage"], report["maximumpriorbias"])
        self.assertTrue(all(row["prior"] is not None for row in report["candidates"] if row["feasible"]))
        json.dumps(report, allow_nan=False)

    def testhardcomparison(self):
        result = self.search(priorcontrol="hard")
        self.assertTrue(result.report["fallback"])
        self.assertTrue(all("prior" in row["reasons"] for row in result.report["candidates"]))

    def testunboundedcomparison(self):
        result = self.search(priorcontrol="soft", algorithm="poll")
        self.assertTrue(result.report["fallback"])

    def testinitialization(self):
        report = self.search(candidates=7).report
        points = np.array([row["point"] for row in report["candidates"]])
        self.assertEqual(points.shape, (7, 3))
        self.assertEqual(np.linalg.matrix_rank(points - [1, 1, .5]), 3)
        self.assertEqual(report["evaluations"], 8)

    def testsmallbudget(self):
        for count in (1, 2, 3, 5):
            result = self.search(candidates=count)
            self.assertEqual(len(result.report["candidates"]), count)

    def testalgorithmbaselines(self):
        for method in ("trust", "poll", "random"):
            report = self.search(algorithm=method).report
            self.assertLessEqual(len(report["candidates"]), 12)
            self.assertLessEqual(report["damage"], report["rawdamage"])
            self.assertTrue(any(row["kind"] == method for row in report["history"]))

    def testindividualslacks(self):
        result = self.search(slacks={"attribute0": .1}, scales={"attribute0": .2})
        self.assertEqual(result.report["config"]["slacks"], {"attribute0": .1})
        with self.assertRaises(ValueError):
            self.search(slacks={"unknown": .1})

    def testinvalidcontrols(self):
        for values in ({"priorweight": 0}, {"priorweight": 1}, {"secant": 0}, {"condition": float("nan")}, {"algorithm": "unknown"}, {"priorcontrol": "unknown"}, {"eta": True}, {"slacks": {"target": -1}}, {"scales": {"target": 0}}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Config(**values).validate()

    def testfitreuse(self):
        model, points = self.model()
        self.assertEqual(model.rank, 3)
        self.assertEqual(len(model.indices), 7)
        self.assertLess(np.linalg.norm(model.response - [[1, 0, .2], [0, 1, -.1]]), 1e-8)
        self.assertLess(np.linalg.norm(model.semantic - [[0, .1, .3]]), 1e-8)
        self.assertAlmostEqual(model.slope[0], .03, places=8)

    def testproposalpieces(self):
        model, _ = self.model()
        result = propose(model, .5, np.array([.98]), np.array([1.]), .2, 1., .1, .001)
        self.assertIsNotNone(result)
        move = result["move"]
        predicted, semantic, penalty = model.prediction(move, .1)
        self.assertLessEqual(float(move @ model.metric @ move), .25 + 1e-8)
        self.assertGreaterEqual(float(semantic[0]), .98 - 1e-8)
        self.assertLessEqual(predicted, .2 + 1e-8)
        self.assertTrue(0 <= penalty <= .1)
        self.assertLess(result["predicted"], result["start"])

    def testrecoverynoprior(self):
        model, _ = self.model(penalty=False)
        result = propose(model, .5, np.array([1.1]), np.array([1.]), .01, 1., .1, .001, recovery=True)
        self.assertIsNotNone(result)
        self.assertLessEqual(result["predicted"], result["start"] + 1e-8)
        self.assertFalse(model.metadata()["priorobserved"])
        with self.assertRaises(ValueError):
            propose(model, .5, np.array([1.]), np.array([1.]), .2, 1., .1, .001)

    def testgeometry(self):
        model, points = self.model(rank=False)
        point = geometry(model, points, .5)
        self.assertEqual(model.rank, 1)
        self.assertIsNotNone(point)
        move = np.array(point) - model.anchor
        self.assertLessEqual(np.linalg.norm(move * model.root), .5 + 1e-12)
        self.assertTrue(all(0 <= value <= limit for value, limit in zip(point, (2, 2, 1))))

    def testembeddingnative(self):
        session = self.session()

        class Nonlinear(torch.nn.Module):
            def forward(self, sample, timestep, encoder_hidden_states, return_dict=False):
                value = encoder_hidden_states.float().mean(dim=(1, 2)).square().reshape(-1, 1, 1, 1)
                return (sample * .1 + value,)

        session.editor.unet = Nonlinear()
        native = Native(session, path="embedding")
        zero = torch.zeros_like(session.anchor)
        source, target, middle = native.endpoint(zero, 1), native.endpoint(zero), native.endpoint(zero, .5)
        self.assertFalse(native.affine)
        self.assertFalse(torch.allclose(middle, (source + target) / 2, atol=1e-6))
        self.assertEqual(session.samples, 3)
        self.assertEqual(Native(session, path="auto").path, "embedding")
        self.assertEqual(Native(session, cfg=True, path="auto").path, "prediction")

    def testnonlinearclosedpath(self):
        session = self.session()
        a, b = float(session.source.mean()), float(session.target.mean())

        class Loop(torch.nn.Module):
            def forward(self, sample, timestep, encoder_hidden_states, return_dict=False):
                value = encoder_hidden_states.float().mean(dim=(1, 2)).reshape(-1, 1, 1, 1)
                return (sample * .1 + (value - a) * (value - b),)

        session.editor.unet = Loop()
        path = Path(Native(session, path="embedding"), torch.ones_like(session.anchor))
        path.initialize()
        self.assertTrue(torch.equal(path.zero(0), path.zero(1)))
        self.assertFalse(torch.equal(path.zero(.5), path.zero(0)))
        self.assertIn(2, path.axes((0, 1, 2)))

    def testdiscretecapability(self):
        session = self.session()
        native = Native(session, path="discrete")
        with self.assertRaises(ValueError):
            native.endpoint(torch.zeros_like(session.anchor), .5)
        path = Path(native, session.field(.9, .15))
        result = select(path, lambda image: {"target": 1.}, torch.ones_like(session.origin[:, :1]), Prior(session, mode="off"))
        self.assertTrue(all(row["point"][2] in {0, 1} for row in result.report["candidates"]))

    def testzerobands(self):
        session = self.session()
        path = Path(Native(session), torch.zeros_like(session.anchor))
        path.initialize()
        self.assertEqual(path.canonical((0, 2, 0)), (1., 1., 0.))
        self.assertNotIn(0, path.axes((0, 1, 2)))
        self.assertNotIn(1, path.axes((0, 1, 2)))

    def testprobabilityoverflow(self):
        session = self.session()
        prior = Prior(session, probabilities=[1e308] * session.editor.maxtime)
        self.assertAlmostEqual(sum(prior.probabilities), 1.)

    def testzerovariance(self):
        session = self.session()
        session.editor.alphas[0] = 1.
        with self.assertRaises(ValueError):
            Prior(session)
        self.assertEqual(Prior(session, variance="fixed_large").variance, "fixed_large")

    def testcheckpointidentity(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = File(directory) / "first", File(directory) / "second"
            first.mkdir()
            second.mkdir()
            (first / "weight.bin").write_bytes(b"same")
            (second / "renamed.safetensors").write_bytes(b"same")
            self.assertEqual(fingerprint(first), fingerprint(second))
            (second / "renamed.safetensors").write_bytes(b"changed")
            self.assertNotEqual(fingerprint(first), fingerprint(second))

    def testfrozentrials(self):
        args = common(argparse.ArgumentParser()).parse_args(["--clip", "clip"])
        self.assertEqual(validate(args).algorithm, "trust")
        self.assertEqual(settings(args)["configuration"], {"constraints": None, "probabilities": None, "radiusprofile": None})
        for suite in ("core", "reference", "prior", "search", "guidance", "budget", "mechanism"):
            self.assertIn("full", variants(suite))

    def testparameteridentity(self):
        first, second = torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)
        second.load_state_dict(first.state_dict())
        self.assertEqual(signature(first), signature(second))
        with torch.no_grad():
            second.weight.add_(1)
        self.assertNotEqual(signature(first), signature(second))

    def testconfigurationhash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = File(directory)
            self.assertIsNone(configuration(path))
            (path / "config.json").write_text("{\"value\": 1}", encoding="utf-8")
            first = configuration(path)
            (path / "config.json").write_text("{\"value\": 2}", encoding="utf-8")
            self.assertNotEqual(first, configuration(path))

    def testhumanratings(self):
        key = {"schema": 1, "assignments": [{"id": "first", "key": "image", "seed": 1, "group": "family", "a": "right"}, {"id": "second", "key": "image", "seed": 2, "group": "family", "a": "left"}]}
        rows = [{"id": "first", "rater": "rater", "completion": "A", "preservation": "tie", "naturalness": "unsure"}, {"id": "second", "rater": "rater", "completion": "B", "preservation": "B", "naturalness": "unsure"}]
        with patch("blind.jsonfile", side_effect=[key, rows]):
            result = collect("key", "responses", repeats=10)
        self.assertEqual(result["metrics"]["completion"]["rightpreference"], 1.)
        self.assertEqual(result["metrics"]["preservation"]["rightpreference"], .75)
        self.assertIsNone(result["metrics"]["naturalness"]["rightpreference"])
        self.assertIsNone(result["metrics"]["completion"]["interval"])

    def testhumaninvalidratings(self):
        key = {"schema": 1, "assignments": [{"id": "first", "key": "image", "seed": 1, "group": "family", "a": "right"}]}
        row = {"id": "first", "rater": "rater", "completion": "A", "preservation": "B", "naturalness": "tie"}
        for rows in ([row, row], [{**row, "id": "unknown"}], [{**row, "completion": "best"}]):
            with patch("blind.jsonfile", side_effect=[key, rows]), self.assertRaises(ValueError):
                collect("key", "responses", repeats=10)

    def testsourceoutputguard(self):
        with self.assertRaises(ValueError):
            destination("engine/result.json")
        with self.assertRaises(ValueError):
            localdescription(7)
        self.assertEqual(localdescription(" target "), "target")

    def testnativeadapter(self):
        session = self.session()
        session.limit = Budget(4)

        class Generator(torch.nn.Module):
            def forward(self, sample, embedding):
                return sample + embedding.float().mean().square()

        noise = torch.randn_like(session.anchor)
        native = Adapter(session, Generator(), lambda field: noise + 2 * field, fields=True)
        zero = torch.zeros_like(session.anchor)
        first, second = native.endpoint(zero), native.endpoint(zero, 1)
        self.assertFalse(torch.equal(first, second))
        self.assertEqual(session.limit.used, 2)
        self.assertEqual(session.calls, 2)
        self.assertTrue(torch.allclose(native.endpoint(torch.ones_like(zero)) - first, torch.full_like(zero, 2), atol=1e-6))
        self.assertFalse(native.affine)

    def testadapterdiagnosticonly(self):
        session = self.session()

        class Generator(torch.nn.Module):
            def forward(self, sample, embedding):
                return sample

        native = Adapter(session, Generator(), lambda field: torch.zeros_like(session.anchor))
        with self.assertRaises(ValueError):
            native.endpoint(torch.ones_like(session.anchor))
        native = Adapter(session, Generator(), lambda field: field, fields=True, continuous=False)
        with self.assertRaises(ValueError):
            native.endpoint(torch.zeros_like(session.anchor), .5)

    def testadapterbudget(self):
        session = self.session()
        session.limit = Budget(1)

        class Generator(torch.nn.Module):
            def forward(self, sample, embedding):
                return sample

        native = Adapter(session, Generator(), lambda field: field, fields=True)
        native.endpoint(torch.zeros_like(session.anchor))
        with self.assertRaises(Exhausted):
            native.endpoint(torch.zeros_like(session.anchor), 1)
        self.assertEqual(session.samples, 1)

    def testmatchedadapterinputs(self):
        session = self.session()
        calls = []

        class Generator(torch.nn.Module):
            def forward(self, sample, embedding):
                return sample

        def inputs(field):
            calls.append(True)
            return torch.randn_like(field) + field

        native = Adapter(session, Generator(), inputs, fields=True)
        field = torch.ones_like(session.anchor)
        first, second = native.endpoint(field), native.endpoint(field, 1)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(len(calls), 2)
        self.assertEqual(native.metadata()["mappedinputs"], 2)

    def testzerofieldscale(self):
        session = self.session()
        session.scale = 0.
        path = Path(Native(session), torch.ones_like(session.anchor))
        path.initialize()
        self.assertNotIn(0, path.axes((0, 1, 2)))
        self.assertNotIn(1, path.axes((0, 1, 2)))
        self.assertEqual(path.canonical((0, 2, 0)), (1., 1., 0.))

    def testnonaffinecostbound(self):
        for options in (["--referencepath", "embedding"], ["--cfg", "--referencepath", "prediction", "--poststeps", "5", "--guidance", "2", "--maxsamples", "512"]):
            args = common(argparse.ArgumentParser()).parse_args(["--clip", "clip", "--size", "16", *options])
            image = Image.new("RGB", (16, 16), (96, 128, 160))
            _, report, session = run(editor(), ClipJudge(Clip(), Processor()), image, "source", "target long", args, Image.new("L", image.size, 128))
            branches = 3 if args.guidance != 1 else 2
            bound = 4 + args.poststeps * branches * (3 + 2 * args.candidates) + args.priorprobes * (args.candidates + 1)
            self.assertLessEqual(session.samples, min(args.maxsamples, bound))
            self.assertLessEqual(session.decodes, args.candidates + 1)
            self.assertFalse(report["backend"]["affine"])

    def testadapterpriorboundary(self):
        args = common(argparse.ArgumentParser()).parse_args(["--clip", "clip", "--adapter", "native.py", "--prior", "ddpm"])
        with self.assertRaises(ValueError):
            validate(args)
        args.prior = "off"
        validate(args)
        args.cfg = True
        with self.assertRaises(ValueError):
            validate(args)

    def testblindexportandtasklabels(self):
        original = os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                image = Image.new("RGB", (4, 4), (96, 128, 160))
                image.save("source.png")
                pixels = (np.asarray(image, dtype=np.float32) / 255).transpose(2, 0, 1)[None].copy()
                provenance = {"source": "source", "target": "target", "imagesha256": hashlib.sha256(pixels.tobytes()).hexdigest(), "imageshape": [1, 3, 4, 4], "masksha256": None, "constraintmet": True}
                for method, color in (("left", "blue"), ("right", "green")):
                    File(method).mkdir()
                    Image.new("RGB", (4, 4), color).save(File(method) / "image.png")
                    row = {"key": "sample", "seed": 42, "group": "family", "sourceimage": "source.png", "instruction": "change color", "image": "image.png", "resultsha256": hashlib.sha256((File(method) / "image.png").read_bytes()).hexdigest(), "metrics": {}, "report": provenance}
                    File(method, "summary.json").write_text(json.dumps({"schema": 1, "split": "test", "oracle": False, "rows": [row]}), encoding="utf-8")
                public, private = build("left/summary.json", "right/summary.json", "blind", seed=7)
                self.assertEqual(len(public), 1)
                self.assertNotIn("a", public[0])
                self.assertTrue(File("blind", public[0]["panel"]).is_file())
                labels = [{"key": "sample", "seed": 42, "imagesha256": hashlib.sha256(File("right/image.png").read_bytes()).hexdigest(), "success": False, "preserved": True, "artifact": False, "naturalness": 4}]
                File("labels.json").write_text(json.dumps(labels), encoding="utf-8")
                reported = assess("right/summary.json", "labels.json", repeats=10)
                self.assertEqual(reported["failurecounts"]["proxypassfailure"], 1)
                self.assertEqual(reported["metrics"]["success"]["mean"], 0.)
                self.assertEqual(reported["metrics"]["naturalness"]["mean"], 4.)
                Image.new("RGB", (4, 4), "red").save("right/image.png")
                with self.assertRaises(ValueError):
                    assess("right/summary.json", "labels.json", repeats=10)
            finally:
                os.chdir(original)

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
