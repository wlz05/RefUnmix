# -*- coding: utf-8 -*-
# RefUnmix: Separating Prompt Response from Shared Transport in One-Step Image Editing

import argparse
from pathlib import Path
import unittest
from unittest.mock import patch
from PIL import Image
import torch
import conduct
from engine.linear import Editor as Linear
from engine.native import create, load
from evaluate import datasetargs
from run import common, run, validate
from session import editor

def linear():
    base = editor()
    return Linear(base.unet, base.scheduler, base.vae, base.tokenizer, base.text, size=16)

class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def settings(self, backbone="vp", profile="naive"):
        options = conduct.parser().parse_args(["--backbone", backbone, "--profile", profile, "--batch", "1"])
        command = conduct.generation(options, 42, Path("outputs") / profile / "42")
        args = datasetargs(common(argparse.ArgumentParser())).parse_args(command[3:])
        args.size, args.device = 16, "cpu"
        validate(args)
        return args

    def testpairedbackbonepresets(self):
        for name in ("sdturbo", "swiftbrush", "instaflow"):
            baseline, main = self.settings(name), self.settings(name, "main")
            self.assertEqual(baseline.backbone, main.backbone)
            self.assertTrue(baseline.naive)
            self.assertFalse(main.naive)
            self.assertEqual((baseline.start, baseline.end, baseline.scale, baseline.noise), (.90, .30, 1., 1))
            self.assertEqual((main.conditiongain, main.singlegain, main.innovationgain), (1.25, 1.5, 2.))

    def testnativevpdispatch(self):
        for name in ("vp", "sdturbo", "swiftbrush"):
            current = editor()
            with patch("engine.session.Editor.load", return_value=current) as loader:
                self.assertIs(load("models", name, "cpu", "fp32", 16), current)
            loader.assert_called_once_with("models", "cpu", "fp32", 16)
            if name == "vp":
                self.assertNotIn("architecture", current.metadata)
            else:
                self.assertEqual(current.metadata["architecture"], name)
                self.assertEqual(current.metadata["native"]["prediction"], "epsilon")
        with patch("engine.session.Editor.load", return_value=editor("vprediction")):
            with self.assertRaises(ValueError):
                load("models", "swiftbrush")
        with self.assertRaises(ValueError):
            load("models", "unknown")

    def testnativeflowdispatch(self):
        current = linear()
        with patch("engine.linear.Editor.load", return_value=current) as loader:
            self.assertIs(load("models/insta", "instaflow", "cpu", "fp32", 16), current)
        loader.assert_called_once_with("models/insta", "cpu", "fp32", 16)
        self.assertEqual(current.metadata["native"]["clock"], "1000 * sigma")

    def testlinearbridgeclockandsign(self):
        current = linear()
        for batch in (1, 2, 3):
            anchor = torch.full((batch, 3, 2, 2), .4)
            noise = torch.full_like(anchor, .8)
            embedding = torch.ones(batch, 2, 1)
            for sigma in (0., .001, .30, .75, 1.):
                with self.subTest(batch=batch, sigma=sigma), patch.object(current.unet, "forward", return_value=(torch.full_like(anchor, .2),)) as model:
                    result = current.predict(anchor, noise, current.index(sigma), embedding)
                    expected = (1 - sigma) * anchor + sigma * noise + sigma * .2
                    torch.testing.assert_close(result, expected)
                    query = model.call_args.kwargs
                    torch.testing.assert_close(query["sample"], (1 - sigma) * anchor + sigma * noise)
                    self.assertTrue(torch.equal(query["timestep"], torch.full((batch,), sigma * 1000)))

    def testofficialonestepparity(self):
        current = linear()
        noise = torch.randn(2, 3, 2, 2, generator=torch.Generator().manual_seed(17))
        embedding = torch.ones(2, 1, 1)
        officialvelocity = noise * .1 + .02 + 1.
        native = current.predict(torch.full_like(noise, 9.), noise, current.index(1.), embedding)
        torch.testing.assert_close(native, noise + officialvelocity)
        torch.testing.assert_close(current.state(noise, 1000, embedding), native)

    def testlinearvalidation(self):
        current = linear()
        anchor, embedding = torch.ones(1, 3, 2, 2), torch.ones(1, 2, 1)
        for tick in (-1, 1001, True, .3, float("nan")):
            with self.subTest(tick=tick), self.assertRaises(ValueError):
                current.predict(anchor, anchor, tick, embedding)
        for noise, condition in ((anchor[:, :, :1], embedding), (anchor * float("nan"), embedding), (anchor, embedding.expand(2, -1, -1)), (anchor, embedding * float("inf"))):
            with self.assertRaises(ValueError):
                current.predict(anchor, noise, 300, condition)
        with self.assertRaises(ValueError):
            current.predict(anchor, anchor, 300, embedding, keys=.5)
        with self.assertRaises(ValueError):
            current.recover(anchor, anchor, 1., .3)
        with patch.object(current.unet, "forward", return_value=(anchor * float("nan"),)), self.assertRaises(RuntimeError):
            current.state(anchor, 300, embedding)
        with patch.object(current.unet, "forward", return_value=(None,)), self.assertRaises(RuntimeError):
            current.state(anchor, 300, embedding)

    def testnaivecompositionandaccounting(self):
        image = Image.new("RGB", (16, 16), (96, 128, 160))
        for name in ("sdturbo", "swiftbrush", "instaflow"):
            with self.subTest(backbone=name):
                current = linear() if name == "instaflow" else editor()
                args = self.settings(name)
                reference = current.prepare(image, "source", "target long", seed=42)
                noise, anchor = reference.noises[:1], reference.anchor
                tick = current.index(.9)
                source = current.predict(anchor, noise, tick, reference.source)
                target = current.predict(anchor, noise, tick, reference.target)
                endpoint = current.predict(anchor + target - source, noise, current.index(.3), reference.target)
                expected = ((current.decode(endpoint) + 1) / 2).clamp(0, 1)
                pixels, report, session = run(current, None, image, "source", "target long", args)
                torch.testing.assert_close(pixels, expected)
                self.assertEqual((session.samples, session.calls, session.decodes, session.encodes), (3, 2, 1, 1))
                self.assertEqual(report["method"], "naive")
                self.assertFalse(report["controller"]["enabled"])
                self.assertEqual(report["candidates"], [])
                self.assertEqual([query["samples"] for query in session.queries], [2, 1])

    def testmainkeepsmatchedendpoints(self):
        image = Image.new("RGB", (16, 16), (96, 128, 160))
        for name in ("sdturbo", "swiftbrush", "instaflow"):
            args = self.settings(name, "main")
            current = linear() if name == "instaflow" else editor()
            pixels, report, session = run(current, None, image, "source", "target long", args)
            self.assertEqual((session.samples, session.calls, session.decodes), (5, 2, 3))
            self.assertEqual(report["renderer"], "responsibility")
            self.assertEqual(report["sourcecalibration"], [])
            self.assertTrue(torch.isfinite(pixels).all())

    def testunsupportedcontrols(self):
        for values in ({"single": True}, {"delta": .15}, {"cleanup": False}, {"cfg": True}, {"oracle": True}, {"adapter": "external.py"}):
            args = self.settings()
            for key, value in values.items():
                setattr(args, key, value)
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate(args)
        for values in ({"singlebias": True}, {"singlemode": "recenter"}, {"batch": 3}):
            args = self.settings("instaflow", "main")
            for key, value in values.items():
                setattr(args, key, value)
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate(args)
        image = Image.new("RGB", (16, 16))
        for profile in ("naive", "main"):
            with self.assertRaises(ValueError):
                run(editor(), None, image, "source", "target long", self.settings(profile=profile), Image.new("L", image.size, 255))
        session = linear().prepare(Image.new("RGB", (16, 16)), "source", "target")
        for options in ({"steps": 2}, {"cfg": True}):
            with self.assertRaises(ValueError):
                create(session, **options)

    def testinstaflowprofilebounds(self):
        for options in (["--backbone", "instaflow"], ["--backbone", "instaflow", "--batch", "1", "--profile", "calibration"]):
            args = conduct.parser().parse_args(options)
            with self.assertRaises(ValueError):
                conduct.generation(args, 42, Path("outputs/main/42"))

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
