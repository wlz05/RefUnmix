# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import argparse
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from engine.limit import Budget, Exhausted
from engine.native import Native
from engine.path import Path
from engine.poll import Config, feedback, protection, select
from engine.prior import Prior
from engine.theory import fit, solve, support
from engine.judge import ClipJudge
from run import common, run, validate
from session import Clip, Processor, editor

class PathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def prepare(self, **values):
        image = Image.new("RGB", (16, 16), (96, 128, 160))
        return editor().prepare(image, "source", "target long", **values)

    def path(self, mode="latent", steps=1, **values):
        session = self.prepare(**values)
        base = session.field(.9, .15)
        native = Native(session, steps)
        path = Path(native, base, mode)
        path.initialize()
        return path

    def testdefault(self):
        args = common(argparse.ArgumentParser()).parse_args(["--clip", "clip"])
        self.assertEqual(args.method, "path")
        self.assertEqual(validate(args), Config())
        self.assertEqual((args.priorprobes, args.poststeps, args.maxsamples), (2, 1, 64))

    def testinvalidconfiguration(self):
        for values in ({"candidates": 0}, {"step": 0}, {"minimum": .75}, {"eta": 2}, {"semanticscale": 0}, {"priorslack": -1}, {"axes": "invalid"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Config(**values).validate()
        for options in (["--size", "8"], ["--guidance", "2"], ["--reference", "pixel"], ["--donors", ".75"]):
            with self.subTest(options=options), self.assertRaises(ValueError):
                validate(common(argparse.ArgumentParser()).parse_args(["--clip", "clip", *options]))

    def testbudget(self):
        budget = Budget(3, calls=2)
        budget.consume(2)
        budget.phase = "prior"
        budget.consume(1)
        self.assertEqual(budget.report()["stages"]["prior"]["samples"], 1)
        with self.assertRaises(Exhausted):
            budget.consume(1)
        with self.assertRaises(ValueError):
            budget.consume(-1)
        self.assertEqual(budget.used, 3)

    def testwall(self):
        budget = Budget(seconds=1.)
        with patch("engine.limit.perf_counter", return_value=budget.started + 2):
            with self.assertRaises(Exhausted):
                budget.require()

    def testfrontcap(self):
        session = self.prepare(noise=2)
        session.limit = Budget(3)
        with self.assertRaises(Exhausted):
            session.field(.9, .15)
        self.assertEqual(session.samples, 0)

    def testnativeendpoints(self):
        session = self.prepare()
        native = Native(session)
        zero = torch.zeros_like(session.anchor)
        target, source = native.endpoint(zero), native.endpoint(zero, 1.)
        middle = native.endpoint(zero, .5)
        self.assertTrue(torch.allclose(middle, (target + source) / 2, atol=1e-7))
        self.assertEqual(session.samples, 4)

    def testcfg(self):
        session = self.prepare()
        zero = torch.zeros_like(session.anchor)
        native = Native(session, guidance=2., cfg=True)
        mixed = native.endpoint(zero, .5)
        self.assertEqual(session.samples, 3)
        tick = session.editor.index(session.end)
        target = session.editor.predict(session.anchor, session.noises[:1], tick, session.target)
        source = session.editor.predict(session.anchor, session.noises[:1], tick, session.source)
        unconditional = session.editor.predict(session.anchor, session.noises[:1], tick, session.editor.prompt(""))
        self.assertTrue(torch.allclose(mixed, target + source - unconditional, atol=1e-7))
        with self.assertRaises(ValueError):
            Native(session, guidance=2.)

    def testnativebudget(self):
        session = self.prepare()
        session.limit = Budget(1)
        with self.assertRaises(Exhausted):
            Native(session).endpoint(torch.zeros_like(session.anchor), .5)
        self.assertEqual(session.calls, 0)

    def testdecoderonly(self):
        path = self.path(cleanup=False)
        self.assertTrue(torch.equal(path.zero(0.), path.zero(1.)))
        self.assertTrue(torch.equal(path.latent((1., 1., 0.)), path.latent((1., 1., 1.))))
        self.assertEqual(path.native.cost(.5), 0)

    def testzeroisreconstruction(self):
        path = self.path()
        latent, pixels = path.candidate((0., 0., 0.))
        self.assertTrue(torch.equal(latent, path.session.anchor))
        self.assertGreater(float((pixels - path.session.origin).abs().max()), .001)

    def testcaching(self):
        path = self.path()
        before = path.session.samples
        for eta in (0., .25, .5, .75, 1.):
            path.candidate((1., 1., eta))
        self.assertEqual(path.session.samples, before)
        self.assertEqual(path.session.decodes, 5)
        self.assertEqual(path.native.endpoints, 3)

    def testmixtureequivalence(self):
        path = self.path()
        mix = Path(path.native, path.base, "mix")
        mix.fields, mix.references = path.fields, path.references
        first = path.candidate((.5, 1.5, .5))[1]
        second = mix.candidate((.5, 1.5, .5))[1]
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(mix.metadata()["affineequivalence"])

    def testrandomnorm(self):
        path = self.path("random")
        difference = path.zero(0.) - path.zero(1.)
        self.assertAlmostEqual(float(path.direction.double().norm()), float(difference.double().norm()), places=6)

    def testmechanism(self):
        path = self.path()
        report = path.mechanism(decoded=True)
        self.assertLess(report["identityerror"], 1e-6)
        self.assertGreater(report["latent"]["condition"], 0)
        self.assertTrue(report["decoded"])

    def testmultistepnative(self):
        session = self.prepare()

        class Nonlinear(torch.nn.Module):
            def forward(self, sample, timestep, encoder_hidden_states, return_dict=False):
                condition = encoder_hidden_states.mean(dim=(1, 2))[:, None, None, None]
                return (sample * condition * .05 + sample.square() * .1 + condition * .01,)

        session.editor.unet = Nonlinear()
        path = Path(Native(session, 5), session.field(.9, .15))
        path.initialize()
        first, last = path.zero(0.), path.zero(1.)
        before = session.samples
        middle = path.zero(.5)
        self.assertEqual(session.samples - before, 10)
        self.assertFalse(torch.allclose(middle, (first + last) / 2, atol=1e-6))
        path.zero(.5)
        self.assertEqual(session.samples - before, 10)
        self.assertFalse(path.native.affine)

    def testmultistepbound(self):
        session = self.prepare()
        session.limit = Budget(4)
        with self.assertRaises(Exhausted):
            Native(session, 5).endpoint(torch.zeros_like(session.anchor))
        self.assertEqual(session.samples, 0)

    def testpriorconversion(self):
        session = self.prepare()
        prior = Prior(session, probes=3, seed=7, mode="rec")
        with patch.object(session, "predict", side_effect=lambda latent, noise, tick, target: latent + .2 * noise):
            value = prior(session.anchor)
        expected = np.mean([float((.2 * noise.double()).square().mean()) * float(session.editor.alphas[tick] / (1 - session.editor.alphas[tick])) for tick, noise in zip(prior.ticks, prior.noises)])
        self.assertAlmostEqual(value, expected, places=7)

    def testpriorweights(self):
        session = self.prepare()
        prior = Prior(session, probes=3, seed=7)
        for tick, weight in zip(prior.ticks, prior.weights):
            cumulative = float(session.editor.alphas[tick])
            previous = float(session.editor.alphas[tick - 1])
            step, beta = cumulative / previous, 1 - cumulative / previous
            variance = beta * (1 - previous) / (1 - cumulative)
            expected = beta ** 2 / (2 * variance * step * (1 - cumulative)) * session.editor.maxtime
            self.assertAlmostEqual(weight, expected, places=9)

    def testpriorshared(self):
        session = self.prepare()
        initial = torch.random.get_rng_state().clone()
        first, second = Prior(session, seed=7), Prior(session, seed=7)
        self.assertTrue(torch.equal(initial, torch.random.get_rng_state()))
        self.assertEqual(first.ticks, second.ticks)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first.noises, second.noises)))
        self.assertEqual(first(session.anchor), second(session.anchor))

    def testpriorbudget(self):
        session = self.prepare()
        prior = Prior(session, probes=2)
        session.limit = Budget(1)
        with self.assertRaises(Exhausted):
            prior(session.anchor)
        self.assertEqual(session.calls, 0)

    def testinvalidprior(self):
        session = self.prepare()
        with self.assertRaises(ValueError):
            Prior(session, probabilities=[1, 0])
        with self.assertRaises(ValueError):
            Prior(session, variance="learned")
        session.editor.alphas[1] = 1
        with self.assertRaises(ValueError):
            Prior(session)

    def testfeedback(self):
        result = feedback({"score": 1., "target": 2., "local0": 3., "attribute0": 4., "retain0": 5., "localviews": 1})
        self.assertEqual(set(result), {"target", "local0", "attribute0", "retain0"})
        for value in ({}, {"score": float("nan")}, {"target": True}):
            with self.assertRaises(ValueError):
                feedback(value)

    def testprotectiondenominator(self):
        origin = torch.zeros(1, 3, 4, 4)
        damage, mass = protection(origin, torch.ones(1, 1, 4, 4))
        self.assertEqual(mass, 16)
        self.assertEqual(damage(torch.ones_like(origin)), 3.)

    def testnoprotection(self):
        path = self.path()
        prior = Prior(path.session)
        result = select(path, lambda image: {"score": float(image.mean())}, torch.zeros_like(path.session.origin[:, :1]), prior)
        self.assertFalse(result.report["protectedvalid"])
        self.assertEqual(result.report["damage"], 0.)
        self.assertGreater(prior.evaluations, 0)

    def testrawfallback(self):
        path = self.path()
        rawimage = path.raw()[1]
        prior = Prior(path.session, mode="off")
        result = select(path, lambda image: {"score": -float((image - rawimage).square().sum())}, torch.ones_like(rawimage[:, :1]), prior, Config(slack=0.))
        self.assertTrue(result.report["fallback"])
        self.assertTrue(torch.equal(result.image, rawimage))
        self.assertLessEqual(len(result.report["candidates"]), 12)
        self.assertTrue(all(not row["feasible"] for row in result.report["candidates"]))

    def testinitbudgetfallback(self):
        session = self.prepare()
        session.limit = Budget(5)
        base = session.field(.9, .15)
        path = Path(Native(session), base)
        result = select(path, lambda image: {"score": 0.}, torch.ones_like(session.origin[:, :1]), Prior(session))
        self.assertEqual(result.report["selected"], "raw")
        self.assertEqual(session.limit.used, 5)

    def testmainpipeline(self):
        args = common(argparse.ArgumentParser()).parse_args(["--clip", "clip", "--size", "16", "--candidates", "12", "--prior", "rec", "--heldout", "--control"])
        image = Image.new("RGB", (16, 16), (96, 128, 160))
        mask = Image.new("L", (16, 16), 128)
        output, report, session = run(editor(), ClipJudge(Clip(), Processor()), image, "source", "target long", args, mask)
        self.assertEqual(report["method"], "path")
        self.assertEqual(output.shape, (1, 3, 16, 16))
        self.assertLessEqual(session.samples, args.maxsamples)
        self.assertEqual(session.samples, report["stats"]["budget"]["samples"])
        self.assertIn("heldout", report)
        self.assertIn("mechanism", report)
        self.assertLessEqual(report["damage"], report["rawdamage"])
        json.dumps(report, allow_nan=False)

    def testmaincostbound(self):
        args = common(argparse.ArgumentParser()).parse_args(["--clip", "clip", "--size", "16", "--cfg", "--referencepath", "prediction"])
        image = Image.new("RGB", (16, 16), (96, 128, 160))
        _, report, session = run(editor(), ClipJudge(Clip(), Processor()), image, "source", "target long", args, Image.new("L", image.size, 128))
        self.assertLessEqual(session.samples, 4 + 6 + args.candidates + args.priorprobes * (args.candidates + 1))
        self.assertLessEqual(session.decodes, args.candidates + 1)

    def testnoop(self):
        args = common(argparse.ArgumentParser()).parse_args(["--clip", "clip", "--size", "16"])
        image = Image.new("RGB", (16, 16), (96, 128, 160))
        output, report, session = run(editor(), ClipJudge(Clip(), Processor()), image, "source", "SOURCE", args)
        self.assertTrue(torch.equal(output, session.origin))
        self.assertTrue(report["noop"])
        self.assertEqual(session.samples, 0)

    def testfullresolutionmask(self):
        pixels = np.zeros((16, 16), dtype=np.uint8)
        pixels[:3] = 255
        session = self.prepare(region=Image.fromarray(pixels))
        self.assertEqual(session.area.shape, session.origin[:, :1].shape)
        self.assertEqual(int(session.area.sum()), 3 * 16)

    def testsupportcertificate(self):
        report = support([[1]], [0], [[1]], [[1]], [.2], [1], [1], 1., .01)
        self.assertTrue(report["infeasible"])
        self.assertAlmostEqual(report["margin"], .1)

    def testregularizedsecant(self):
        rng = np.random.default_rng(20261009)
        initial, directions, observed = rng.normal(size=(4, 3)), rng.normal(size=(3, 5)), rng.normal(size=(4, 5))
        fitted = fit(initial, directions, observed, .2)
        residual = fitted - initial + 5 * (fitted @ directions - observed) @ directions.T
        self.assertLess(float(np.linalg.norm(residual)), 1e-12)

    def testqcqpscope(self):
        result = solve([[1]], [.2], [[1]], [[1]], [-.3], .5, .05)
        self.assertTrue(result["feasible"])
        self.assertAlmostEqual(result["point"][0], -.2 / 1.001, places=6)
        impossible = solve([[1]], [0], [[1]], [[1]], [.2], 1., .01)
        self.assertFalse(impossible["feasible"])

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
