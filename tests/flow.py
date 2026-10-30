# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import argparse
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
from engine.flow import Consistency, Editor, Native, identifiers, pack, stability, unpack
from engine.access import describe
from engine.judge import ClipJudge
from engine.limit import Budget, Exhausted
from engine.native import create
from engine.path import Path
from engine.prior import Prior
from run import common, run, validate
from session import Clip, Processor, Vae

class Velocity(torch.nn.Module):
    config = SimpleNamespace(guidance_embeds=True)

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.1))
        self.inputs = []

    def forward(self, hidden_states, timestep, encoder_hidden_states, pooled_projections, return_dict=False, **options):
        self.inputs.append({"state": hidden_states.detach().clone(), "time": timestep.detach().clone(), "pool": pooled_projections.detach().clone(), **options})
        dims = (hidden_states.shape[0],) + (1,) * (hidden_states.ndim - 1)
        condition = encoder_hidden_states.float().mean(dim=(1, 2)).square() + pooled_projections.float().mean(dim=1)
        return (hidden_states * self.weight + condition.reshape(dims) * .1,)

class Pipeline:
    def __init__(self, family):
        self.family, self.transformer, self.vae = family, Velocity(), Vae()
        self.vae.config = SimpleNamespace(scaling_factor=.5, shift_factor=.3)
        self.scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))
        self.tokenizer = SimpleNamespace()
        self.text_encoder = torch.nn.Linear(1, 1)

    def encode_prompt(self, prompt, **options):
        value = len(prompt) / 20
        embedding = torch.full((1, 2, 3), value)
        pool = torch.tensor([[value, value * .5]])
        return (embedding, pool, torch.zeros(2, 3)) if self.family == "flux" else (embedding, None, pool, None)

def editor(family="flux", consistency=True):
    return Editor(Pipeline(family), family=family, size=16, consistency=consistency)

class FlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def session(self, family="flux", **options):
        return editor(family).prepare(Image.new("RGB", (16, 16), (96, 128, 160)), "source", "target long", **options)

    def testpacking(self):
        value = torch.arange(2 * 3 * 4 * 6).reshape(2, 3, 4, 6)
        self.assertEqual(pack(value).shape, (2, 6, 12))
        self.assertTrue(torch.equal(unpack(pack(value), value.shape), value))
        with self.assertRaises(ValueError):
            pack(torch.zeros(1, 3, 3, 2))
        with self.assertRaises(ValueError):
            unpack(torch.zeros(1, 2, 12), (1, 3, 4, 4))

    def testpositions(self):
        value = identifiers(4, 6, "cpu", torch.float32)
        self.assertEqual(value.shape, (6, 3))
        self.assertTrue(torch.equal(value[-1], torch.tensor([0., 1., 2.])))

    def testendpointbound(self):
        self.assertAlmostEqual(stability(.25, 0., 1.), .5)
        self.assertGreaterEqual(stability(.25, 2., 1.), .5)
        self.assertAlmostEqual(stability(.25, 1e-12, 1.), .5, places=10)
        self.assertEqual(stability(.25, 2., 0.), 0.)
        with self.assertRaises(ValueError):
            stability(-1., 0., 1.)
        with self.assertRaises(ValueError):
            stability(.25, 1000., 1.)
        self.assertEqual(stability(0., 1000., 1.), 0.)

    def testconditionregistrationatomicity(self):
        current = editor()
        embedding, pool = torch.ones(1, 2, 3), torch.ones(1, 2)
        with self.assertRaises(ValueError):
            current.register(embedding, pool, torch.zeros(1, 3))
        self.assertEqual(current.pools, {})
        self.assertEqual(current.textids, {})
        with self.assertRaises(ValueError):
            current.register(embedding, pool)

    def testvelocitytickbounds(self):
        session = self.session()
        for tick in (-1, 1001, True):
            with self.subTest(tick=tick), self.assertRaises(ValueError):
                session.editor.velocity(session.anchor, tick, session.target)

    def testdegeneratenoisestate(self):
        for time in (0., 1.):
            with self.assertRaises(ValueError):
                Native(self.session(end=time))

    def testcleansign(self):
        for family in ("flux", "sd3"):
            session = self.session(family)
            tick = session.editor.index(.3)
            with patch.object(session.editor, "velocity", return_value=torch.full_like(session.anchor, .2)):
                state = .7 * session.anchor + .3 * session.noises[:1]
                clean = session.predict(session.anchor, session.noises[:1], tick, session.target)
                self.assertTrue(torch.allclose(clean, state - .3 * .2, atol=1e-7))
            self.assertEqual(session.samples, 1)

    def testvaeshift(self):
        session = self.session()
        expected = (session.editor.vae.encode(session.origin * 2 - 1).latent_dist.mode() - .3) * .5
        self.assertTrue(torch.equal(expected, session.anchor))
        decoded = session.editor.vae.decode(session.anchor / .5 + .3).sample
        self.assertTrue(torch.equal(session.decode(session.anchor), (decoded + 1) / 2))

    def testmodelclock(self):
        for family in ("flux", "sd3"):
            session = self.session(family)
            session.predict(session.anchor, session.noises[:1], session.editor.index(.3), session.target)
            value = session.editor.unet.inputs[-1]
            self.assertAlmostEqual(float(value["time"][0]), .3 if family == "flux" else 300., places=5)
            self.assertTrue(torch.equal(value["pool"], session.editor.pools[session.editor.key(session.target)]))
            self.assertEqual(value["state"].ndim, 3 if family == "flux" else 4)

    def testmatchedfield(self):
        session = self.session(noise=2)
        value = session.field(.9, .15)
        self.assertEqual(value.shape, session.anchor.shape)
        self.assertEqual(session.samples, 8)
        for item in session.editor.unet.inputs:
            self.assertTrue(torch.equal(item["state"][:2], item["state"][2:]))

    def testfrozen(self):
        current = editor()
        self.assertTrue(all(not parameter.requires_grad for model in (current.unet, current.vae, current.pipeline.text_encoder) for parameter in model.parameters()))

    def testembeddingpoolpath(self):
        session = self.session()
        native = Native(session, path="embedding")
        zero = torch.zeros_like(session.anchor)
        first, second, middle = native.endpoint(zero), native.endpoint(zero, 1), native.endpoint(zero, .5)
        self.assertFalse(torch.allclose(middle, (first + second) / 2, atol=1e-7))
        pools = session.editor.pools
        expected = .5 * (pools[session.editor.key(session.source)] + pools[session.editor.key(session.target)])
        self.assertTrue(torch.equal(session.editor.unet.inputs[-1]["pool"], expected))

    def testaffinecache(self):
        session = self.session()
        path = Path(Native(session, path="prediction"), session.field(.9, .15))
        path.initialize()
        before = session.samples
        for eta in (.25, .5, .75):
            path.candidate((1., 1., eta))
        self.assertEqual(session.samples, before)
        self.assertTrue(path.native.affine)

    def testtwostepstate(self):
        session = self.session()
        native = Native(session, steps=2)
        zero = torch.zeros_like(session.anchor)
        output = native.endpoint(zero)
        state = .7 * session.anchor + .3 * session.noises[:1]
        for sigma in (.3, .15):
            state = state - .15 * session.editor.velocity(state, sigma, session.target)
        self.assertTrue(torch.allclose(output, state, atol=1e-7))
        self.assertEqual(session.samples, 2)
        self.assertFalse(native.affine)

    def testnativebudget(self):
        session = self.session()
        session.limit = Budget(1)
        with self.assertRaises(Exhausted):
            Native(session, steps=2).endpoint(torch.zeros_like(session.anchor))
        self.assertEqual(session.samples, 0)

    def testrawoverflow(self):
        for family in ("flux", "sd3"):
            session = self.session(family, cleanup=False, scale=1e30)
            native = Native(session)
            with self.subTest(family=family), self.assertRaises(ValueError):
                native.endpoint(torch.full_like(session.anchor, 1e30))
            self.assertEqual((native.endpoints, session.samples), (0, 0))

    def testintegrationoverflow(self):
        session = self.session()
        session.noises.fill_(3e38)
        native = Native(session, steps=2)
        with patch.object(session.editor, "velocity", return_value=torch.full_like(session.anchor, -3e38)):
            with self.assertRaises(ValueError):
                native.endpoint(torch.full_like(session.anchor, 3e38))
        self.assertEqual(native.endpoints, 0)
        self.assertEqual(session.samples, 1)

    def testpriorformula(self):
        session = self.session()
        prior = Prior(session, probes=3, mode="flow")
        with patch.object(session, "velocity", side_effect=lambda state, tick, embedding: torch.zeros_like(state)):
            result = prior(session.anchor)
        expected = np.mean([weight * float((noise.double() - session.anchor.double()).square().mean()) for noise, weight in zip(prior.noises, prior.weights)])
        self.assertAlmostEqual(result, expected, places=10)
        self.assertEqual(prior.metadata()["velocitydirection"], "data-to-noise")

    def testcleanresidualequivalence(self):
        session = self.session()
        latent = session.anchor
        sigma = .42
        noise = torch.randn_like(latent)
        state = (1 - sigma) * latent + sigma * noise
        velocity = session.editor.velocity(state, sigma, session.target)
        clean = state - sigma * velocity
        expected = -(velocity - noise + latent) * sigma
        self.assertTrue(torch.allclose(clean - latent, expected, atol=1e-7))

    def testpriorrngandphases(self):
        session = self.session()
        original = torch.random.get_rng_state().clone()
        first, second = Consistency(session, seed=7), Consistency(session, seed=7)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), original))
        self.assertEqual(first.ticks, second.ticks)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first.noises, second.noises)))
        session.limit = Budget(4)
        session.limit.phase = "backend"
        self.assertEqual(first(session.anchor), second(session.anchor))
        self.assertEqual(session.limit.phase, "backend")
        self.assertEqual(session.limit.stages["prior"]["samples"], 4)

    def testpriorcapability(self):
        session = editor(consistency=False).prepare(Image.new("RGB", (16, 16)), "source", "target")
        with self.assertRaises(ValueError):
            Prior(session, mode="flow")
        with self.assertRaises(ValueError):
            Prior(session, mode="ddpm")
        self.assertIsNone(Prior(session, mode="off")(session.anchor))
        access = describe(Native(session), Prior(session, mode="off"))
        self.assertTrue(access["fullcontrol"])
        self.assertFalse(access["timequeriesused"])

    def testpriorbudgetatomic(self):
        session = self.session()
        session.limit = Budget(1)
        with self.assertRaises(Exhausted):
            Consistency(session, probes=2)(session.anchor)
        self.assertEqual(session.samples, 0)

    def testpriorinvalidprobabilities(self):
        session = self.session()
        for probabilities in ([1.], [0.] * 999, [float("nan")] * 999):
            with self.assertRaises(ValueError):
                Consistency(session, probabilities=probabilities)
        prior = Consistency(session, probabilities=[1e308] * 999)
        self.assertAlmostEqual(sum(prior.probabilities), 1.)

    def testdiscreteandcfg(self):
        session = self.session()
        native = Native(session, cfg=True, path="discrete")
        self.assertFalse(native.continuous)
        with self.assertRaises(ValueError):
            native.endpoint(torch.zeros_like(session.anchor), .5)
        self.assertEqual(create(session).metadata()["kind"], "flow")
        with self.assertRaises(ValueError):
            Native(session, guidance=2.)

    def testcliarchitectureguard(self):
        for options in (["--backbone", "flux", "--prior", "ddpm"], ["--prior", "flow"], ["--backbone", "sd3", "--prior", "flow"], ["--backbone", "flux", "--prior", "off", "--localization", "attention"]):
            with self.assertRaises(ValueError):
                validate(common(argparse.ArgumentParser()).parse_args(["--clip", "clip", *options]))
        validate(common(argparse.ArgumentParser()).parse_args(["--clip", "clip", "--backbone", "flux", "--prior", "flow", "--flowconsistency"]))
        validate(common(argparse.ArgumentParser()).parse_args(["--clip", "clip", "--backbone", "flux"]))

    def testfullpipeline(self):
        for family in ("flux", "sd3"):
            args = common(argparse.ArgumentParser()).parse_args(["--clip", "clip", "--size", "16", "--backbone", family, "--prior", "flow", "--flowconsistency", "--poststeps", "2", "--maxsamples", "128", "--heldout", "--control"])
            image = Image.new("RGB", (16, 16), (96, 128, 160))
            pixels, report, session = run(editor(family), ClipJudge(Clip(), Processor()), image, "source", "target long", args, Image.new("L", image.size, 128))
            self.assertEqual(pixels.shape, session.origin.shape)
            self.assertEqual(report["backend"]["kind"], "flow")
            self.assertEqual(report["prior"]["kind"], "flow")
            self.assertTrue(report["access"]["timequeriesused"])
            self.assertIn("heldout", report)
            self.assertLessEqual(session.samples, 128)
            self.assertLessEqual(report["damage"], report["rawdamage"])
            self.assertLess(report["mechanism"]["identityerror"], 1e-6)
            json.dumps(report, allow_nan=False)

    def testrepairedflow(self):
        for family in ("flux", "sd3"):
            args = common(argparse.ArgumentParser()).parse_args(["--clip", "clip", "--size", "16", "--backbone", family, "--fieldrepair", "1", "--semantic", "region"])
            image = Image.new("RGB", (16, 16), (96, 128, 160))
            pixels, report, session = run(editor(family), ClipJudge(Clip(), Processor()), image, "source", "target long", args)
            self.assertEqual(pixels.shape, session.origin.shape)
            self.assertEqual(report["semanticviews"]["layout"], "region")
            self.assertIn("repair", report["field"])
            self.assertLessEqual(session.samples, 64)
            self.assertEqual(session.samples, session.limit.used)
            self.assertLessEqual(report["damage"], report["rawdamage"])
            json.dumps(report, allow_nan=False)

    def testtangentflow(self):
        for family in ("flux", "sd3"):
            args = common(argparse.ArgumentParser()).parse_args(["--clip", "clip", "--size", "16", "--backbone", family, "--tangent", "--method", "raw"])
            image = Image.new("RGB", (16, 16), (96, 128, 160))
            pixels, report, session = run(editor(family), ClipJudge(Clip(), Processor()), image, "source", "target long", args)
            self.assertEqual(pixels.shape, session.origin.shape)
            self.assertIn("tangent", report["field"])
            self.assertLessEqual(session.samples, 64)
            self.assertEqual(session.samples, session.limit.used)
            json.dumps(report, allow_nan=False)

    def testintegratedflow(self):
        for family in ("flux", "sd3"):
            args = common(argparse.ArgumentParser()).parse_args(["--clip", "clip", "--size", "16", "--backbone", family, "--method", "field", "--semantic", "region"])
            image = Image.new("RGB", (16, 16), (96, 128, 160))
            pixels, report, session = run(editor(family), ClipJudge(Clip(), Processor()), image, "source", "target long", args)
            self.assertEqual(pixels.shape, session.origin.shape)
            self.assertIn("tangent", report["field"])
            self.assertLessEqual(session.samples, 64)
            self.assertEqual(session.samples, session.limit.used)
            json.dumps(report, allow_nan=False)

    def testfeedbackflow(self):
        for family in ("flux", "sd3"):
            args = common(argparse.ArgumentParser()).parse_args(["--clip", "clip", "--size", "16", "--backbone", family, "--method", "field", "--semantic", "region", "--tangentfeedback"])
            image = Image.new("RGB", (16, 16), (96, 128, 160))
            pixels, report, session = run(editor(family), ClipJudge(Clip(), Processor()), image, "source", "target long", args)
            self.assertEqual(pixels.shape, session.origin.shape)
            self.assertIn("tangent", report["field"])
            self.assertLessEqual(session.samples, 64)
            self.assertEqual(session.samples, session.limit.used)
            json.dumps(report, allow_nan=False)

    def testconditionalflow(self):
        for family in ("flux", "sd3"):
            args = common(argparse.ArgumentParser()).parse_args(["--clip", "clip", "--size", "16", "--backbone", family, "--method", "field", "--semantic", "region", "--tangentfeedback", "--tangentbasis", "shared", "--tangentpolicy", "conditional"])
            image = Image.new("RGB", (16, 16), (96, 128, 160))
            pixels, report, session = run(editor(family), ClipJudge(Clip(), Processor()), image, "source", "target long", args)
            self.assertEqual(pixels.shape, session.origin.shape)
            self.assertIn("tangent", report["field"])
            self.assertLessEqual(session.samples, 64)
            self.assertEqual(session.samples, session.limit.used)
            json.dumps(report, allow_nan=False)

    def testrgbshiftpath(self):
        session = self.session()
        path = Path(Native(session), session.field(.9, .15), mode="rgb")
        path.initialize()
        latent, pixels = path.candidate((1., 1., .5))
        self.assertTrue(torch.equal(latent, session.editor.encode(pixels)))

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
