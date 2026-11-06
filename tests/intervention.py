# -*- coding: utf-8 -*-
# RefUnmix: Separating Prompt Response from Shared Transport in One-Step Image Editing

import argparse
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
import torch.nn.functional as functional
import conduct
from evaluate import datasetargs
from run import common, validate
from engine.single import edit
from engine.intervention import edit as intervention

class Collector:
    def __init__(self, unet, indices, shape, **options):
        self.indices = indices

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def result(self):
        return torch.arange(16).float().reshape(1, 1, 4, 4) / 15

    def report(self):
        return {"captured": 1}

class Session:
    def __init__(self):
        generator = torch.Generator().manual_seed(381)
        self.anchor = torch.rand(1, 4, 4, 4, generator=generator) - .5
        self.noises = torch.randn(1, 4, 4, 4, generator=generator)
        self.origin = torch.rand(1, 3, 16, 16, generator=generator) * .5 + .25
        self.source = torch.zeros(1, 8, 4)
        self.target = torch.ones(1, 8, 4)
        self.editor = SimpleNamespace(index=lambda value: round(value * 1000), tokenizer=None, tokens=8, unet=None)
        self.end = .3
        self.cache, self.artifacts, self.attentions, self.attentionlog = {}, {}, {}, {}
        self.samples, self.calls, self.decodes = 0, 0, 0
        self.queries = []

    def predict(self, anchors, noises, tick, embeddings, *options):
        self.queries.append(anchors.clone())
        self.samples += len(anchors)
        self.calls += 1
        condition = embeddings.mean((1, 2)).reshape(-1, 1, 1, 1)
        return anchors + .03 * noises + .07 * condition + .00001 * tick

    def decode(self, latent):
        self.decodes += len(latent)
        return functional.interpolate(latent[:, :3], (16, 16), mode="bilinear", align_corners=False) + .5

    def attention(self):
        return next(reversed(self.attentions.values()))

    def stats(self):
        return {"samples": self.samples, "calls": self.calls, "decodes": self.decodes, "encodes": 1}

class Interventions(unittest.TestCase):
    def setUp(self):
        options = conduct.parser().parse_args(["--seeds", "42", "--batch", "1"])
        command = conduct.generation(options, 42, Path("outputs/controltest"))
        command[command.index("--singlegain") + 1] = "1"
        command[command.index("--innovationgain") + 1] = "3"
        command.append("--referencelocal")
        self.args = datasetargs(common(argparse.ArgumentParser())).parse_args(command[3:])
        self.metadata = {"source": "a green cube", "target": "a red cube"}
        self.collector = patch("engine.attention.Collector", Collector)
        self.tokens = patch("engine.attention.tokens", return_value=((1,), (1,)))
        self.collector.start()
        self.tokens.start()
        self.addCleanup(self.collector.stop)
        self.addCleanup(self.tokens.stop)

    def render(self, mode):
        args = copy.deepcopy(self.args)
        args.ablation = mode
        validate(args)
        return edit(Session(), args, dict(self.metadata))

    def test_full_renderer_parity(self):
        pixels, report, original = self.render("full")
        session = Session()
        session.samples, session.calls = 2, 1
        session.collector = Collector(None, ((1,), (1,)), (4, 4))
        actual, details, _ = intervention(session, self.args, original.queries[1][1:2], dict(self.metadata), [.75], report["field"]["calibration"])
        self.assertTrue(torch.equal(actual, pixels))
        self.assertEqual(details["stats"], report["stats"])

    def test_actual_query_and_decode_counts(self):
        for mode in ["assignment", "reference", "gating", "target", "grounding", "attraction"]:
            with self.subTest(mode=mode):
                pixels, report, session = self.render(mode)
                expected = (3, 2, 1) if mode == "target" else (5, 2, 2) if mode == "reference" else (5, 2, 3)
                self.assertEqual((session.samples, session.calls, session.decodes), expected)
                self.assertTrue(torch.isfinite(pixels).all())
                self.assertTrue(torch.all((pixels >= 0) & (pixels <= 1)))
                self.assertEqual(report["sourcecalibration"], [])

    def test_fixed_assignment_does_not_resolve_adoption(self):
        with patch("engine.intervention.resolve", side_effect=AssertionError("Coupled solve entered")):
            _, report, _ = self.render("assignment")
        self.assertEqual(report["field"]["response"]["requested"], 3.)
        self.assertIn("fixed", report["field"]["response"]["assignment"])

    def test_attraction_is_removed_alone(self):
        _, full, _ = self.render("full")
        _, removed, _ = self.render("attraction")
        self.assertEqual(full["field"]["calibration"]["targetattraction"]["weight"], .15)
        self.assertEqual(removed["field"]["calibration"]["targetattraction"]["weight"], 0.)
        self.assertTrue(removed["grounding"]["enabled"])
        self.assertIn("innovation", removed["field"]["response"])

    def test_grounding_retains_field_and_innovation(self):
        _, report, _ = self.render("grounding")
        self.assertFalse(report["grounding"]["enabled"])
        self.assertNotIn("support", report["field"]["response"])
        self.assertIn("innovation", report["field"]["response"])
        self.assertEqual(report["field"]["calibration"]["targetattraction"]["weight"], .15)

    def test_new_controls_require_grounding_path(self):
        self.args.referencelocal = False
        for mode in ["grounding", "attraction"]:
            self.args.ablation = mode
            with self.assertRaises(ValueError):
                validate(self.args)

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
