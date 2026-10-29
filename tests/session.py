# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import unittest
from types import SimpleNamespace
import torch
import torch.nn.functional as functional
from PIL import Image
from engine.session import Editor, orient, relative, toimage
from engine.judge import ClipJudge
from engine.response import Config, region, select

class Unet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, sample, timestep, encoder_hidden_states, return_dict=False):
        self.inputs.append(sample.clone())
        condition = encoder_hidden_states.mean(dim=(1, 2)).reshape(-1, 1, 1, 1)
        return (sample * 0.1 + condition * 0.02 + timestep.reshape(-1, 1, 1, 1) * 0.001,)

class Vae(torch.nn.Module):
    config = SimpleNamespace(scaling_factor=0.5)

    def encode(self, image):
        value = functional.avg_pool2d(image, 8) + 0.04
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: value))

    def decode(self, latent):
        return SimpleNamespace(sample=functional.interpolate(latent, scale_factor=8, mode="nearest"))

class Text(torch.nn.Module):
    config = SimpleNamespace(max_position_embeddings=77)

    def forward(self, input_ids):
        return (input_ids.float().unsqueeze(-1),)

class Tokenizer:
    model_max_length = 77

    def __call__(self, values, **kwargs):
        return SimpleNamespace(input_ids=torch.tensor([[len(value)] for value in values]))

def editor(kind="epsilon"):
    config = SimpleNamespace(num_train_timesteps=101, prediction_type=kind)
    scheduler = SimpleNamespace(config=config, alphas_cumprod=torch.linspace(0.99, 0.01, 101))
    return Editor(Unet(), scheduler, Vae(), Tokenizer(), Text(), size=16)

class Processor:
    image_processor = SimpleNamespace(size={"shortest_edge": 4}, crop_size={"height": 4, "width": 4}, image_mean=[0.5] * 3, image_std=[0.5] * 3)

    def __call__(self, text, **kwargs):
        return {"input_ids": torch.tensor([[1, 0, 0, 1] if item == "source" else [0, 1, 0, 1] for item in text])}

class Vision(torch.nn.Module):
    def forward(self, pixel_values):
        values = pixel_values.mean(dim=(-2, -1))
        return SimpleNamespace(pooler_output=torch.cat((values, values.new_ones((values.shape[0], 1))), dim=1))

class ClipText(torch.nn.Module):
    def forward(self, input_ids):
        return SimpleNamespace(pooler_output=input_ids.float())

class Clip(torch.nn.Module):
    config = SimpleNamespace(vision_config=SimpleNamespace(image_size=4))

    def __init__(self):
        super().__init__()
        self.vision_model, self.text_model = Vision(), ClipText()
        self.visual_projection, self.text_projection = torch.nn.Identity(), torch.nn.Identity()

class SessionTests(unittest.TestCase):
    def setUp(self):
        self.editor = editor()
        self.image = Image.new("RGB", (24, 16), (96, 128, 160))

    def prepare(self, **kwargs):
        return self.editor.prepare(self.image, "source", "target long", **kwargs)

    def testoriginal(self):
        session = self.prepare(cleanup=False)
        self.assertEqual(session.origin.shape, (1, 3, 16, 16))
        self.assertAlmostEqual(float(session.origin[0, 0, 0, 0]), 96 / 255, places=6)
        reconstruction = session.render(torch.zeros_like(session.anchor))
        self.assertGreater(float((reconstruction - session.origin).abs().max()), 0.01)

    def testcache(self):
        session = self.prepare(noise=3)
        first = session.field(0.75)
        again = session.field(0.7501)
        self.assertTrue(torch.equal(first, again))
        self.assertEqual(session.stats()["calls"], 1)
        self.assertEqual(session.stats()["samples"], 6)
        self.assertTrue(torch.equal(self.editor.unet.inputs[0][:3], self.editor.unet.inputs[0][3:]))
        again.zero_()
        self.assertTrue(torch.equal(first, session.field(0.75)))

    def testchord(self):
        session = self.prepare()
        current, previous = session.field(0.90), session.field(0.75)
        expected = (0.15 * current + 0.90 * previous) / 1.05
        self.assertTrue(torch.allclose(session.field(0.90, 0.15), expected))
        self.assertEqual(session.stats()["calls"], 2)

    def testcleanupcost(self):
        session = self.prepare(noise=2)
        field = session.field(0.75)
        first, second = session.render(field), session.render(field)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(session.stats()["calls"], 3)
        self.assertEqual(session.stats()["samples"], 6)
        self.assertEqual(session.stats()["decodes"], 2)

    def testcleanupoverride(self):
        session = self.prepare(cleanup=True)
        field = session.field(.75)
        before = session.stats()["calls"]
        raw = session.render(field, cleanup=False)
        self.assertEqual(session.stats()["calls"], before)
        refined = session.render(field)
        self.assertEqual(session.stats()["calls"], before + 1)
        self.assertTrue(session.cleanup)
        self.assertFalse(torch.equal(raw, refined))
        self.assertTrue(torch.equal(raw, session.render(field, cleanup=False)))
        with self.assertRaises(ValueError):
            session.render(field, cleanup=1)

    def testfractionalprobe(self):
        session = self.prepare(cleanup=False)
        zero = torch.zeros_like(session.anchor)
        first, second = session.render(zero), session.render(zero + 1e-5)
        self.assertGreater(float((second - first).abs().max()), 0)
        self.assertLess(float((second - first).abs().max()), 1 / 255)
        self.assertEqual(toimage(first).size, (16, 16))

    def testreferenceidentity(self):
        for cleanup in (False, True):
            with self.subTest(cleanup=cleanup):
                session = self.prepare(cleanup=cleanup)
                zero = torch.zeros_like(session.anchor)
                self.assertTrue(torch.equal(session.render(zero, reference="PIXEL"), session.origin))
                self.assertFalse(torch.equal(session.render(zero), session.origin))
                self.assertFalse(torch.equal(session.render(zero, reference="latent"), session.origin))

    def testreferencescale(self):
        session = self.prepare(scale=0.)
        field = torch.ones_like(session.anchor)
        self.assertTrue(torch.equal(session.render(field, reference="pixel"), session.origin))
        self.assertFalse(torch.equal(session.render(field), session.origin))

    def testreferencecache(self):
        session = self.prepare()
        field = torch.ones_like(session.anchor) * .01
        first = session.render(field, reference="pixel")
        second = session.render(field, reference="pixel")
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(session.calls, 3)
        self.assertEqual(session.decodes, 3)
        self.assertEqual(len(session.references), 1)
        self.assertTrue(torch.equal(self.editor.unet.inputs[0], self.editor.unet.inputs[2]))
        self.assertFalse(torch.equal(self.editor.unet.inputs[0], self.editor.unet.inputs[1]))
        session.render(field, cleanup=False, reference="pixel")
        self.assertEqual(session.calls, 3)
        self.assertEqual(session.decodes, 5)
        session.end = .4
        session.render(field, reference="pixel")
        self.assertEqual(session.calls, 5)
        self.assertEqual(session.decodes, 7)
        self.assertEqual(len(session.references), 3)

    def testreferenceresponse(self):
        session = self.prepare(cleanup=False)
        field = torch.ones_like(session.anchor) * .01
        self.assertTrue(torch.allclose(session.render(field, reference="pixel"), session.origin + .01, atol=1e-7))
        self.assertFalse(torch.equal(session.render(field, reference="latent"), session.origin + .01))
        with self.assertRaises(ValueError):
            session.render(field, reference="unknown")
        with self.assertRaises(ValueError):
            session.render(field, cleanup=1, reference="pixel")

    def testreferenceunclipped(self):
        session = self.prepare(cleanup=False)
        original = self.editor.vae.decode
        self.editor.vae.decode = lambda latent: SimpleNamespace(sample=original(latent).sample + 3.)
        field = torch.ones_like(session.anchor) * .01
        self.assertTrue(torch.allclose(session.render(field, reference="pixel"), session.origin + .01, atol=2e-7))
        self.assertTrue(torch.equal(session.render(field), torch.ones_like(session.origin)))

    def testdiagnostic(self):
        session = self.prepare()
        report = session.diagnostic(torch.ones_like(session.anchor) * .01)
        self.assertEqual(report["zeropixelerror"], 0.)
        self.assertGreater(report["reconstruction"], 0.)
        self.assertGreater(report["refinementdrift"], 0.)
        self.assertLess(report["protected"]["identityerror"], 1e-10)
        self.assertEqual(report["cost"], {"calls": 2, "samples": 2, "decodes": 3})
        again = session.diagnostic(torch.zeros_like(session.anchor))
        self.assertEqual(again["cost"], {"calls": 1, "samples": 1, "decodes": 1})
        self.assertEqual(again["centerdamage"], 0.)

    def testmaskpolarity(self):
        mask = Image.new("L", self.image.size, 64)
        editing = self.prepare(region=mask)
        protected = self.prepare(region=mask, polarity="PROTECT")
        self.assertEqual(editing.mask.shape, (1, 1, 2, 2))
        self.assertTrue(torch.allclose(editing.mask + protected.mask, torch.ones_like(editing.mask)))
        field = torch.ones_like(editing.anchor) * 0.01
        unmasked = self.prepare()
        self.assertTrue(torch.equal(editing.render(field), unmasked.render(field)))

    def testpredictiontypes(self):
        sample, estimate = torch.ones(1, 3, 2, 2), torch.full((1, 3, 2, 2), 0.5)
        alpha, sigma = torch.tensor(0.8), torch.tensor(0.6)
        for kind, expected in (("epsilon", 0.875), ("sample", 0.5), ("v_prediction", 0.5)):
            result = editor(kind).recover(sample, estimate, alpha, sigma)
            self.assertTrue(torch.allclose(result, torch.full_like(result, expected)))

    def testvalidation(self):
        for kwargs in ({"noise": 0}, {"noise": True}, {"scale": -1}, {"end": 1.1}, {"cleanup": 1}, {"polarity": "protect"}, {"seed": 2 ** 65}, {"region": Image.new("L", (8, 8))}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.prepare(**kwargs)
        session = self.prepare()
        for time, delta in ((-0.1, 0), (1.1, 0), (0.75, 0.75), (0.75, -0.1), (float("nan"), 0)):
            with self.subTest(time=time, delta=delta), self.assertRaises(ValueError):
                session.field(time, delta)
        with self.assertRaises(ValueError):
            session.render(torch.zeros(1, 3, 4, 4))

    def testpaths(self):
        self.assertEqual(str(relative("./models")), "models")
        # Reject absolute and escaping paths.
        for path in ("D:/models", "../models", "/models", "\\\\host\\models", ""):
            with self.subTest(path=path), self.assertRaises(ValueError):
                relative(path)

    def testorientation(self):
        image = Image.new("RGB", (24, 16))
        image.getexif()[274] = 6
        mask = Image.new("L", image.size)
        mask.putpixel((0, 0), 255)
        aligned, regionmask = orient(image, mask)
        self.assertEqual(aligned.size, (16, 24))
        self.assertEqual(regionmask.getpixel((15, 0)), 255)
        self.assertIsNone(aligned.getexif().get(274))

    def testattentionlifecycle(self):
        session = self.prepare()
        session.localization("source", "target long")
        session.field(.75)
        self.assertIsNone(session.attention())
        self.assertIn(75, session.attentionlog)
        with self.assertRaises(ValueError):
            session.localization("source", "target")

class JudgeTests(unittest.TestCase):
    def setUp(self):
        self.judge = ClipJudge(Clip(), Processor())
        self.origin = torch.full((1, 3, 16, 16), 0.5)

    def testfloatresponse(self):
        objective = self.judge.prepare(self.origin, "source", "target", torch.ones(1, 1, 2, 2))
        first = objective(self.origin)
        changed = self.origin.clone()
        changed[:, 1] += 1e-4
        second = objective(changed)
        self.assertEqual(first["score"], 0)
        self.assertGreater(second["score"], 0)
        self.assertEqual(objective.evaluations, 2)
        self.assertEqual(self.judge.stats()["imagecalls"], 3)
        self.assertEqual(first["localviews"], 4)

    def testfrozencrop(self):
        mask = torch.zeros(1, 1, 16, 16)
        mask[..., 3:9, 4:8] = 1
        objective = self.judge.prepare(self.origin, "source", "target", mask)
        self.assertEqual(objective.boxes, ((4, 3, 8, 9),))
        objective(self.origin + 0.01)
        self.assertEqual(objective.boxes, ((4, 3, 8, 9),))

    def testemptymask(self):
        objective = self.judge.prepare(self.origin, "source", "target", torch.zeros(1, 1, 2, 2))
        result = objective(self.origin)
        self.assertEqual(result["localviews"], 0)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in result.values()))

    def testindividualregions(self):
        objective = self.judge.prepare(self.origin, "source", "target", torch.ones(1, 1, 2, 2))
        changed = self.origin.clone()
        changed[:, 1, :8, :8] += .01
        changed[:, 1, 8:, :8] -= .01
        values = objective(changed)
        self.assertGreater(values["local0"], 0)
        self.assertLess(values["local1"], 0)
        self.assertEqual(len([key for key in values if key.startswith("local") and key[5:].isdigit()]), 4)

    def testgeometry(self):
        pixels = self.judge.pixels(torch.full((1, 3, 3, 9), 0.5))
        self.assertEqual(pixels.shape, (1, 3, 4, 4))
        self.assertLess(float(pixels.abs().max()), 1e-6)
        with self.assertRaises(ValueError):
            self.judge.pixels(torch.full((1, 3, 4, 4), 1.1))
        with self.assertRaises(ValueError):
            self.judge.prepare(self.origin, "source", "target", torch.full((1, 1, 2, 2), -1.0))

    def testregiongoal(self):
        from engine.poll import feedback
        from engine.attainment import gain
        patch = self.judge.prepare(self.origin, "source", "target", layout="patches")
        region = self.judge.prepare(self.origin, "source", "target", layout="region")
        changed = self.origin.clone()
        changed[:, 1, :8, :8] += .01
        left, right = patch(changed), region(changed)
        patchgoals = [left["local" + str(index)] for index in range(4)]
        self.assertEqual(gain(patchgoals, [0.] * 4, [1.] * 4), 0.)
        self.assertGreater(right["local0"], 0.)
        self.assertEqual(right["localviews"], 1)
        self.assertNotIn("view0", feedback(right))
        self.assertEqual(region(self.origin)["local0"], 0.)

    def testregionsupport(self):
        mask = torch.zeros(1, 1, 16, 16)
        mask[..., 2:4, 1:15] = 1
        objective = self.judge.prepare(self.origin, "source", "target", mask, layout="region")
        self.assertEqual(objective.boxes, ((0, 0, 16, 16), (1, 2, 15, 4)))
        changed = self.origin.clone()
        changed[:, 1, 2:4, 1:3] += .1
        self.assertGreater(objective(changed)["local0"], 0.)
        self.assertEqual(objective.layout, "region")
        empty = self.judge.prepare(self.origin, "source", "target", torch.zeros_like(mask), layout="region")
        self.assertEqual(empty(changed)["localviews"], 0)
        self.assertNotIn("local0", empty(changed))
        with self.assertRaises(ValueError):
            self.judge.prepare(self.origin, "source", "target", layout="unknown")

class IntegrationTests(unittest.TestCase):
    def testreferencecontrol(self):
        import argparse
        from run import common, run
        model = editor()
        judge = ClipJudge(Clip(), Processor())
        image = Image.new("RGB", (24, 16), (96, 128, 160))
        mask = Image.new("L", image.size, 128)
        args = common(argparse.ArgumentParser()).parse_args(["--method", "projection", "--field", "chord", "--clip", "./clip", "--size", "16", "--grid", "1", "--control"])
        output, report, session = run(model, judge, image, "source", "target long", args, mask)
        self.assertIn("zeroscores", report["control"])
        self.assertEqual(report["control"]["zeropixelerror"], 0.)
        if not report["referenceguard"]["accepted"]:
            self.assertEqual(report["selected"], "raw")
            self.assertEqual(report["weights"], [])
            self.assertTrue(torch.equal(output, session.render(session.field(.9, .15))))
        args.reference = "raw"
        _, rawreport, _ = run(editor(), judge, image, "source", "target long", args, mask)
        self.assertNotIn("referenceguard", rawreport)
        self.assertEqual(rawreport["renderer"], "raw")

    def testjointpipeline(self):
        import argparse
        from run import common, run
        model = editor()
        judge = ClipJudge(Clip(), Processor())
        image = Image.new("RGB", (24, 16), (96, 128, 160))
        mask = Image.new("L", image.size, 128)
        args = common(argparse.ArgumentParser()).parse_args(["--method", "joint", "--field", "chord", "--clip", "./clip", "--size", "16", "--stages", "--lowtime", ".72", "--hightime", ".76"])
        output, report, session = run(model, judge, image, "source", "target long", args, mask)
        self.assertEqual(report["method"], "joint")
        self.assertEqual(report["bandtimes"]["low"], .72)
        self.assertEqual(report["bandtimes"]["high"], .76)
        self.assertEqual(report["partition"]["mode"], "none")
        self.assertEqual(session.decodes, report["evaluations"] + 4)
        self.assertEqual(session.calls, len(session.cache) + report["evaluations"])
        self.assertEqual(output.shape, session.origin.shape)
        self.assertTrue(torch.isfinite(output).all())

    def testprojectionpipeline(self):
        import argparse
        from run import common, run
        model = editor()
        judge = ClipJudge(Clip(), Processor())
        image = Image.new("RGB", (24, 16), (96, 128, 160))
        mask = Image.new("L", image.size, 128)
        args = common(argparse.ArgumentParser()).parse_args(["--method", "projection", "--field", "chord", "--clip", "./clip", "--size", "16", "--grid", "1", "--donors", ".72", ".76"])
        output, report, session = run(model, judge, image, "source", "target long", args, mask)
        self.assertEqual(report["method"], "projection")
        self.assertEqual(report["donors"], [.72, .76])
        self.assertEqual(report["partition"]["mode"], "aligned")
        self.assertEqual(report["renderer"], "pixel")
        self.assertIn("referenceguard", report)
        self.assertEqual(session.decodes, report["evaluations"] + 1)
        self.assertEqual(session.calls, len(session.cache) + report["evaluations"] + 1)
        self.assertEqual(output.shape, session.origin.shape)
        self.assertTrue(torch.isfinite(output).all())
        self.assertIn("local0", report["scores"])

    def testresponsepipeline(self):
        model = editor()
        image = Image.new("RGB", (24, 16), (96, 128, 160))
        mask = Image.new("L", image.size, 128)
        session = model.prepare(image, "source", "target long", noise=2, region=mask)
        base = session.field(0.90, 0.15)
        pool = {time: session.field(time) for time in (0.45, 0.75)}
        before = session.stats()
        support = region(base, session.mask)
        judge = ClipJudge(Clip(), Processor())
        objective = judge.prepare(session.origin, "source", "target", support)
        result = select(session.anchor, base, pool, session.render, objective, session.origin, support, Config(grid=1, active=2, step=0.2, radius=0.5, budget=0.0005))
        after = session.stats()
        self.assertEqual(result.image.shape, session.origin.shape)
        self.assertTrue(torch.isfinite(result.image).all())
        self.assertEqual(after["decodes"] - before["decodes"], result.report["evaluations"])
        self.assertEqual(after["calls"] - before["calls"], result.report["evaluations"])
        self.assertEqual(after["samples"] - before["samples"], result.report["evaluations"])
        self.assertEqual(objective.evaluations, result.report["evaluations"])
        self.assertGreaterEqual(result.report["gain"], -1e-10)
        self.assertLessEqual(result.report["audit"]["mse"], result.report["ceiling"] + 1e-10)
        self.assertGreater(result.report["evaluations"], 1)
        self.assertEqual(result.report["times"], [0.45, 0.75])

    def testbaselinefallback(self):
        model = editor()
        session = model.prepare(Image.new("RGB", (16, 16), (96, 128, 160)), "source", "target long", cleanup=False)
        base = session.field(0.90, 0.15)
        pool = {0.45: session.field(0.45)}
        baseline = session.render(base)

        def objective(image):
            return {"score": -float((image - baseline).square().sum())}

        result = select(session.anchor, base, pool, session.render, objective, session.origin, torch.zeros(1, 1, 2, 2), Config(grid=1, budget=0))
        self.assertEqual(result.report["selected"], "base")
        self.assertTrue(result.report["fallback"])
        self.assertTrue(torch.equal(result.image, baseline))
        self.assertTrue(torch.equal(result.field, base))
        self.assertEqual(result.report["gain"], 0)
        self.assertLessEqual(result.report["audit"]["mse"], result.report["ceiling"] + 1e-10)

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
