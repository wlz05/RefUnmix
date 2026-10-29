# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import argparse
import json
import unittest
import torch
from engine.locality import Config, locate, normalize, partition, shortlist
from engine.response import Config as Response, region, select
from engine.wavelet import Atom, dictionary, split
from run import common, run, validate

class LocalizationTests(unittest.TestCase):
    def setUp(self):
        self.base = torch.zeros(1, 2, 16, 16)
        self.field = self.base.clone()
        self.field[..., 2:4, 2:4] = .1
        self.field[..., 12:14, 12:14] = .2
        self.pool = {.6: self.field, .8: self.field * 2}

    def testsmallnormalization(self):
        value = torch.zeros(1, 1, 32, 32)
        value[..., 3, 3] = .1
        result = normalize(value)
        self.assertEqual(float(result.sum()), 1)
        self.assertEqual(float(normalize(value * 1e-10).sum()), 0)
        self.assertGreater(float(region(value).sum()), 0)

    def testmultiplesupports(self):
        mask, report = locate(self.base, self.pool, config=Config(mode="activity", threshold=.2, dilation=0))
        self.assertEqual(report["retained"], 2)
        self.assertGreater(float(mask[..., 2:4, 2:4].sum()), 0)
        self.assertGreater(float(mask[..., 12:14, 12:14].sum()), 0)
        self.assertFalse(report["uncertain"])

    def testtinysupport(self):
        field = self.base.clone()
        field[..., 0, 0] = .1
        mask, report = locate(self.base, {.6: field}, config=Config(mode="activity", minimum=8, small=1, dilation=0))
        self.assertEqual(report["small"], 1)
        self.assertGreater(float(mask.sum()), 0)

    def testattentionfallback(self):
        mask, report = locate(self.base, self.pool, config=Config(mode="attention"))
        self.assertTrue(report["uncertain"])
        self.assertEqual(float(mask.sum()), 0)
        mask, report = locate(self.base, self.pool, config=Config(mode="hybrid"))
        self.assertTrue(report["fallback"])
        self.assertEqual(report["source"], "activity")
        _, report = locate(self.base, self.pool, attention=torch.ones_like(self.base[:, :1]), config=Config(mode="hybrid"))
        self.assertTrue(report["fallback"])

    def testsourcealiases(self):
        atoms = dictionary(self.base, self.base, {.6: self.field, .8: self.field.clone()})
        self.assertTrue(any("0.8" in atom.aliases for atom in atoms))

    def testexplicitpriority(self):
        supplied = torch.rand(1, 1, 16, 16)
        mask, report = locate(self.base, self.pool, supplied, config=Config(mode="attention"))
        torch.testing.assert_close(mask, supplied)
        self.assertEqual(report["source"], "explicit")

    def testattentionnovelregion(self):
        attention = torch.zeros(1, 1, 16, 16)
        attention[..., 8:10, 8:10] = 1
        mask, report = locate(self.base, self.pool, attention=attention, config=Config(mode="attention", dilation=0))
        self.assertGreater(float(mask[..., 8:10, 8:10].sum()), 0)
        self.assertEqual(report["source"], "attention")

    def testinvalidconfig(self):
        for config in (Config(floor=0), Config(blend=2), Config(threshold=0), Config(regions=True), Config(small=-1), Config(mode="invalid")):
            with self.assertRaises(ValueError):
                config.validate()

    def testadaptivepartition(self):
        mask = torch.zeros_like(self.base[:, :1])
        mask[..., :8, :8] = 1
        windows, report = partition(self.base, self.pool, mask, config=Config(partition="adaptive", regions=7, penalty=0))
        self.assertLessEqual(len(windows), 7)
        self.assertGreater(len(windows), 1)
        torch.testing.assert_close(sum(windows), torch.ones_like(mask))
        for window in windows:
            self.assertEqual(float(split(window)[1].abs().max()), 0)
        atoms = dictionary(self.base, self.base, self.pool, windows=windows)
        self.assertTrue(atoms)
        self.assertEqual(report["count"], len(windows))

    def testuniformpartition(self):
        windows, report = partition(self.base, {.6: torch.ones_like(self.base)}, torch.ones_like(self.base[:, :1]), config=Config(partition="adaptive", regions=7))
        self.assertEqual(len(windows), 1)

    def testoddpartition(self):
        base = torch.zeros(1, 1, 7, 9)
        field = base.clone()
        field[..., :3, :5] = 1
        windows, _ = partition(base, {.6: field}, field, config=Config(partition="adaptive", regions=4, penalty=0))
        torch.testing.assert_close(sum(windows), torch.ones_like(base))
        self.assertTrue(dictionary(base, base, {.6: field}, windows=windows))

    def testgatesrejectoverlap(self):
        gate = torch.ones_like(self.base[:, :1])
        with self.assertRaises(ValueError):
            dictionary(self.base, self.base, self.pool, windows=[gate, gate])
        gate[..., 0, 0] = 0
        with self.assertRaises(ValueError):
            dictionary(self.base, self.base, self.pool, windows=[gate])

    def testshortlist(self):
        atoms = dictionary(self.base, self.base, self.pool)
        kept, report = shortlist(atoms, torch.ones_like(self.base[:, :1]), 3)
        self.assertEqual(len(kept), 3)
        self.assertFalse(report["complete"])
        self.assertEqual(len(report["dropped"]), len(atoms) - 3)
        self.assertEqual({atom.time for atom in kept}, {"0.6", "0.8"})
        json.dumps(report, allow_nan=False)

    def testopposites(self):
        vector = torch.ones(1, 1, 4, 4)
        atoms = [Atom("low", 0, "base", vector), Atom("low", 0, "base", vector * 1.01), Atom("low", 0, "base", -vector)]
        kept, _ = shortlist(atoms, vector, 2)
        self.assertTrue(any(float(atom.value.mean()) < 0 for atom in kept))

    def testnovelhighfrequency(self):
        base = torch.zeros(1, 1, 4, 4)
        high = torch.tensor([[[[1., -1.], [-1., 1.]]]]).repeat(1, 1, 2, 2)
        atoms = dictionary(base, base, {.72: high}, bands="high")
        self.assertTrue(atoms)
        self.assertTrue(all(atom.band == "high" for atom in atoms))

    def testresponseextensions(self):
        base = torch.zeros(1, 1, 4, 4)
        origin = torch.full((1, 3, 4, 4), .5)
        field = torch.arange(16).float().reshape_as(base) * .001
        mask = torch.full_like(base, .5)
        result = select(base, base, {.6: field, .8: field * 2}, lambda value: (origin + value + value.square()).clamp(0, 1), lambda image: float(image.mean()), origin, mask, Response(probes=3, maps=2, trust="curvature", globalstage=False))
        self.assertLessEqual(len(result.report["atoms"]), 3)
        self.assertLessEqual(result.report["trust"]["radius"], .75)
        self.assertIn("influence", result.report["responses"]["curvature"][0])
        self.assertLessEqual(result.report["audit"]["mse"], .0005 + 1e-8)
        json.dumps(result.report, allow_nan=False)

    def testclivalidation(self):
        args = common(argparse.ArgumentParser()).parse_args(["--method", "thread", "--field", "chord", "--clip", "./clip", "--partition", "ADAPTIVE", "--localization", "HYBRID", "--probes", "12", "--trust", "CURVATURE"])
        self.assertEqual(validate(args).probes, 12)
        args.threshold = float("nan")
        with self.assertRaises(ValueError):
            validate(args)

    def testpipelinehybrid(self):
        from PIL import Image
        from session import editor, Clip, Processor
        from engine.judge import ClipJudge
        args = common(argparse.ArgumentParser()).parse_args(["--method", "thread", "--field", "chord", "--clip", "./clip", "--localization", "hybrid", "--partition", "adaptive", "--probes", "2", "--maps", "2", "--trust", "curvature", "--grid", "1", "--size", "16"])
        output, report, state = run(editor(), ClipJudge(Clip(), Processor()), Image.new("RGB", (24, 16), (96, 128, 160)), "source", "target long", args)
        self.assertEqual(output.shape, (1, 3, 16, 16))
        self.assertEqual(report["partition"]["mode"], "adaptive")
        self.assertTrue(report["localization"]["fallback"])
        self.assertTrue(state.attentionlog)
        self.assertLessEqual(len(report["atoms"]), 2)
        json.dumps(report, allow_nan=False)

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
