# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import argparse
import contextlib
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import tempfile
from threading import Barrier
import unittest
from unittest.mock import patch
from PIL import Image
import torch
from engine.batch import Batch, options
from engine.judge import ClipJudge
from engine.protocol import signature
from evaluate import datasetargs, execute
from run import common, run, validate
from session import Clip, Processor, editor

class BatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def args(self, *flags):
        return common(argparse.ArgumentParser()).parse_args(["--size", "16", "--clip", "clip", "--maxsamples", "20", "--maxcalls", "20", *flags])

    def testconfiguration(self):
        self.assertEqual(datasetargs(argparse.ArgumentParser()).parse_args([]).batch, 3)
        for value in (0, 5, True, 3., None):
            args = self.args()
            args.batch = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate(args)
        for flags in (("--deterministic",), ("--localization", "attention"), ("--adapter", "custom.py"), ("--method", "joint", "--field", "chord")):
            args = self.args(*flags)
            args.batch = 3
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                validate(args)

    def testidentity(self):
        args = self.args()
        before = signature(args)
        args.batch = 3
        self.assertEqual(before, signature(args))

    def testsharing(self):
        current = editor()
        gate = Barrier(3)
        with Batch((current, None, None), 3, .1) as batching:
            def work(index):
                backend, _, _ = batching.replicas()
                self.assertIs(backend.unet, current.unet)
                self.assertIs(backend.scheduler, current.scheduler)
                gate.wait(5)
                return backend.decode(torch.full((1, 3, 2, 2), float(index)))

            outputs = batching.map(work, [1, 2, 3])
            row = batching.report()["operations"]["decode"]
        self.assertEqual(row, {"requests": 3, "batches": 1, "samples": 3})
        for index, output in enumerate(outputs, 1):
            self.assertTrue(torch.equal(output, torch.full((1, 3, 16, 16), 2. * index)))

    def testmismatchedticks(self):
        current = editor()
        gate = Barrier(3)
        with Batch((current, None, None), 3, .02) as batching:
            def work(tick):
                gate.wait(5)
                return batching.call(current, "predict", torch.ones(1, 3, 2, 2), torch.zeros(1, 3, 2, 2), tick, torch.ones(1, 1, 1))

            outputs = batching.map(work, [10, 20, 30])
            self.assertEqual(batching.report()["denoisercalls"], 3)
        self.assertFalse(torch.equal(outputs[0], outputs[1]))

    def testmixedsizes(self):
        current = editor()
        gate = Barrier(3)
        with Batch((current, None, None), 3, .1) as batching:
            def work(count):
                gate.wait(5)
                return batching.call(current, "decode", torch.full((count, 3, 2, 2), float(count)))

            outputs = batching.map(work, [1, 2, 4])
        self.assertEqual([value.shape[0] for value in outputs], [1, 2, 4])
        self.assertEqual(batching.report()["operations"]["decode"]["samples"], 7)

    def testisolatedrequests(self):
        image = Image.new("RGB", (16, 16), (70, 90, 120))
        args = self.args()
        serial = []
        for seed in (42, 52, 3047):
            args.seed = seed
            serial.append(run(editor(), ClipJudge(Clip(), Processor()), image, "source", "target long", args))
        current, judge = editor(), ClipJudge(Clip(), Processor())
        args.batch = 3
        with Batch((current, judge, None), 3, .01) as batching:
            def work(seed):
                backend, scorer, _ = batching.replicas()
                local = deepcopy(args)
                local.seed = seed
                return run(backend, scorer, image, "source", "target long", local)

            parallel = batching.map(work, [42, 52, 3047])
        for first, second in zip(serial, parallel):
            torch.testing.assert_close(first[0], second[0])
            for key in ("samples", "calls", "decodes", "encodes", "textcalls"):
                self.assertEqual(first[1]["stats"][key], second[1]["stats"][key])
            self.assertEqual(first[1]["judgestats"], second[1]["judgestats"])
        self.assertLess(batching.report()["denoisercalls"], sum(item[1]["stats"]["calls"] for item in parallel))

    def testseedisolation(self):
        current = editor()
        image = Image.new("RGB", (16, 16))
        expected = {seed: current.prepare(image, "source", "target", seed).noises for seed in (42, 52, 3047, 3407)}
        with Batch((current, None, None), 4) as batching:
            def work(seed):
                backend, _, _ = batching.replicas()
                return backend.prepare(image, "source", "target", seed).noises

            actual = batching.map(work, list(expected))
        for value, target in zip(actual, expected.values()):
            self.assertTrue(torch.equal(value, target))

    def testfailedmodel(self):
        current = editor()
        gate = Barrier(3)
        with patch.object(current, "decode", side_effect=ValueError("broken decoder")):
            with self.assertRaisesRegex(ValueError, "broken decoder"):
                with Batch((current, None, None), 3, .1) as batching:
                    def work(index):
                        gate.wait(5)
                        return batching.call(current, "decode", torch.zeros(1, 3, 2, 2))

                    batching.map(work, [0, 1, 2])
        self.assertFalse(batching.thread.is_alive())

    def testinterruption(self):
        with self.assertRaises(KeyboardInterrupt):
            with Batch((editor(), None, None), 3) as batching:
                def work(index):
                    if index == 1:
                        raise KeyboardInterrupt()
                    return batching.call(batching.resources[0], "decode", torch.zeros(1, 3, 2, 2))

                batching.map(work, [0, 1, 2])
        self.assertFalse(batching.thread.is_alive())

    def testflow(self):
        from flow import editor as floweditor
        image = Image.new("RGB", (16, 16), (70, 90, 120))
        args = self.args("--backbone", "flux")
        args.batch = 3
        with Batch((floweditor(), ClipJudge(Clip(), Processor()), None), 3) as batching:
            def work(seed):
                backend, judge, _ = batching.replicas()
                local = deepcopy(args)
                local.seed = seed
                return run(backend, judge, image, "source", "target long", local)

            results = batching.map(work, [42, 52, 3047])
        for result in results:
            self.assertTrue(torch.isfinite(result[0]).all())
            self.assertLessEqual(result[1]["stats"]["samples"], args.maxsamples)

    def testloadmemory(self):
        from run import load
        with patch("engine.session.Editor.load", side_effect=torch.cuda.OutOfMemoryError("synthetic OOM")):
            with self.assertRaisesRegex(RuntimeError, "resident weight memory"):
                load(self.args())

    def testautomaticdevice(self):
        from run import load
        from types import SimpleNamespace
        current = SimpleNamespace(device=torch.device("cuda:0"))
        with patch("engine.session.Editor.load", return_value=current), patch("engine.judge.ClipJudge.load") as scorer:
            load(self.args())
        self.assertEqual(scorer.call_args.args[1], current.device)

    def testoom(self):
        current = editor()
        with patch.object(current, "decode", side_effect=torch.cuda.OutOfMemoryError("synthetic OOM")):
            with self.assertRaisesRegex(RuntimeError, "--batch 1"):
                with Batch((current, None, None), 3) as batching:
                    batching.map(lambda value: batching.call(current, "decode", torch.zeros(1, 3, 2, 2)), [0, 1, 2])
        self.assertFalse(batching.thread.is_alive())

    def testlifecycle(self):
        batching = Batch((editor(), None, None), 3)
        with self.assertRaises(RuntimeError):
            batching.call(batching.resources[0], "decode", torch.zeros(1, 3, 2, 2))
        with batching:
            with self.assertRaises(ValueError):
                batching.map(lambda value: value, list(range(4)))
        with self.assertRaises(RuntimeError):
            batching.__enter__()

    def testprior(self):
        args = self.args("--prior", "rec", "--priorprobes", "1", "--maxsamples", "40", "--maxcalls", "40")
        args.batch = 3
        image = Image.new("RGB", (16, 16), (70, 90, 120))
        with Batch((editor(), ClipJudge(Clip(), Processor()), None), 3) as batching:
            def work(seed):
                current = deepcopy(args)
                current.seed = seed
                backend, judge, _ = batching.replicas()
                return run(backend, judge, image, "source", "target long", current)

            results = batching.map(work, [42, 52, 3047])
        for _, report, session in results:
            self.assertLessEqual(report["stats"]["samples"], 40)
            self.assertGreater(report["stats"]["budget"]["stages"]["prior"]["samples"], 0)

    def testjournal(self):
        origin = Path.cwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder)
                Path("data/images").mkdir(parents=True)
                Image.new("RGB", (16, 16)).save("data/images/source.png")
                Path("data/map.json").write_text(json.dumps({"sample": {"image": "source.png", "source": "source", "target": "target long"}}), encoding="utf-8")
                args = datasetargs(common(argparse.ArgumentParser())).parse_args(["--dataset", "data", "--mapping", "map.json", "--images", "images", "--split", "all", "--method", "raw", "--size", "16", "--batch", "1", "--seeds", "42", "52", "--output", "outputs"])
                from evaluate import request

                def fail(current, item, seed, *values, **flags):
                    if seed == 52:
                        raise RuntimeError("synthetic request failure")
                    return request(current, item, seed, *values, **flags)

                with patch("evaluate.request", side_effect=fail), contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "synthetic request failure"):
                    execute(args, (editor(), None, None, False))
                failure = json.loads(Path("outputs/failure.json").read_text())
                self.assertEqual(len(failure["completed"]), 1)
                self.assertFalse(Path("outputs/summary.json").exists())
                self.assertTrue((Path("outputs") / failure["completed"][0]).is_file())
            finally:
                os.chdir(origin)

    def testdataset(self):
        from flow import editor as floweditor
        origin = Path.cwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder)
                Path("data/images").mkdir(parents=True)
                Image.new("RGB", (16, 16), (70, 90, 120)).save("data/images/source.png")
                Path("data/map.json").write_text(json.dumps({"sample": {"image": "source.png", "source": "source", "target": "target long"}}), encoding="utf-8")
                args = datasetargs(common(argparse.ArgumentParser())).parse_args(["--dataset", "data", "--mapping", "map.json", "--images", "images", "--split", "all", "--seeds", "42", "52", "3047", "3407", "--size", "16", "--clip", "clip", "--maxsamples", "15", "--maxcalls", "15", "--output", "outputs"])
                for family, build in (("vp", editor), ("flux", floweditor)):
                    args.backbone, args.output = family, "outputs" + family
                    current = build()
                    resources = (current, ClipJudge(Clip(), Processor()), ClipJudge(Clip(), Processor()), True)
                    with self.subTest(family=family), contextlib.redirect_stdout(io.StringIO()):
                        payload = execute(args, resources)
                    self.assertEqual([row["seed"] for row in payload["rows"]], [42, 52, 3047, 3407])
                    self.assertEqual(payload["execution"]["width"], 3)
                    self.assertGreater(payload["execution"]["throughput"], 0)
                    for row in payload["rows"]:
                        self.assertTrue(row["concurrent"])
                        self.assertIsNone(row["vram"])
                        self.assertGreater(row["report"]["reportstats"]["evaluations"], 0)
                        self.assertTrue((Path(args.output) / row["image"]).is_file())
                    self.assertTrue((Path(args.output) / "summary.json").is_file())
                    if family == "flux":
                        self.assertEqual((current.pools, current.textids), ({}, {}))
            finally:
                os.chdir(origin)

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
