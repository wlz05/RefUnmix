# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import argparse
import contextlib
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import random
import tempfile
from time import monotonic, sleep
from threading import Barrier
import unittest
from unittest.mock import patch
from weakref import ref
import numpy as np
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

    def testidlereferences(self):
        current = editor()
        with Batch((current, None, None), 4) as batching:
            def work(index):
                value = torch.ones(1, 3, 2, 2)
                output = batching.call(current, "decode", value)
                return ref(value), ref(output)

            references = batching.map(work, list(range(4)))
            deadline = monotonic() + 1
            while any(reference() is not None for pair in references for reference in pair) and monotonic() < deadline:
                sleep(.005)
            self.assertTrue(all(reference() is None for pair in references for reference in pair))

    def testflowcounts(self):
        from flow import editor as floweditor
        current = floweditor()
        image = Image.new("RGB", (16, 16))
        session = current.prepare(image, "source", "target long")
        count = 3
        with Batch((current, None, None), 3) as batching:
            def work(index):
                return batching.call(current, "velocity", session.anchor.expand(count, -1, -1, -1), .3, session.target.expand(count, -1, -1))

            batching.map(work, list(range(3)))
            row = batching.report()["operations"]["velocity"]
        self.assertEqual(row["samples"], 9)
        self.assertEqual(row["batches"], 3)

    def testerrorreferences(self):
        current = editor()
        captured = []
        with Batch((current, None, None), 3) as batching:
            def fail(value):
                captured.append(ref(value))
                raise ValueError("synthetic decoder failure")

            with patch.object(current, "decode", new=fail):
                with self.assertRaises(ValueError):
                    batching.map(lambda index: batching.call(current, "decode", torch.ones(1, 3, 2, 2)), list(range(3)))
                with self.assertRaises(ValueError), patch.object(batching.executor, "submit") as submit:
                    batching.map(lambda value: value, [0])
                submit.assert_not_called()
        self.assertTrue(all(reference() is None for reference in captured))
        self.assertIsNone(batching.failure.__traceback__)
        self.assertEqual(batching.queue.unfinished_tasks, 0)

    def testmanygroups(self):
        for width in (3, 4):
            current = editor()
            references = []
            with self.subTest(width=width), Batch((current, None, None), width, .001) as batching:
                def work(index):
                    value = torch.full((1, 3, 2, 2), float(index))
                    output = batching.call(current, "decode", value)
                    return ref(value), ref(output)

                for group in range(30):
                    references.extend(batching.map(work, list(range(width))))
                    self.assertTrue(all(reference() is None for pair in references for reference in pair))
                self.assertEqual(batching.report()["operations"]["decode"]["samples"], width * 30)
            self.assertEqual(batching.queue.unfinished_tasks, 0)

    def testrandombranches(self):
        args = self.args("--algorithm", "random", "--representation", "random", "--prior", "rec", "--priorprobes", "1", "--maxsamples", "64", "--maxcalls", "64", "--candidates", "8")
        image = Image.new("RGB", (16, 16), (70, 90, 120))
        seeds = (42, 52, 3047, 3407)
        expected = {}
        for seed in seeds:
            args.seed = seed
            expected[seed] = run(editor(), ClipJudge(Clip(), Processor()), image, "source", "target long", args)
        for width in (3, 4):
            current, judge = editor(), ClipJudge(Clip(), Processor())
            pythonstate, numpystate, torchstate = random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()
            args.batch = width
            with self.subTest(width=width), Batch((current, judge, None), width, .002) as batching:
                def work(seed):
                    local = deepcopy(args)
                    local.seed = seed
                    backend, scorer, _ = batching.replicas()
                    return seed, run(backend, scorer, image, "source", "target long", local)

                actual = []
                order = (3407, 42, 3047, 52)
                for offset in range(0, 4, width):
                    actual.extend(batching.map(work, order[offset:offset + width]))
            self.assertEqual(random.getstate(), pythonstate)
            self.assertEqual(np.random.get_state()[0], numpystate[0])
            np.testing.assert_array_equal(np.random.get_state()[1], numpystate[1])
            self.assertEqual(np.random.get_state()[2:], numpystate[2:])
            self.assertTrue(torch.equal(torch.random.get_rng_state(), torchstate))
            for seed, result in actual:
                torch.testing.assert_close(result[0], expected[seed][0])
                for key in ("calls", "samples", "decodes", "textcalls"):
                    self.assertEqual(result[1]["stats"][key], expected[seed][1]["stats"][key])

    def testnegativeentropy(self):
        args = self.args("--seed", "-1", "--algorithm", "random", "--representation", "random")
        image = Image.new("RGB", (16, 16), (70, 90, 120))
        first = run(editor(), ClipJudge(Clip(), Processor()), image, "source", "target long", args)
        args.seed = 2 ** 64 - 1
        second = run(editor(), ClipJudge(Clip(), Processor()), image, "source", "target long", args)
        torch.testing.assert_close(first[0], second[0])

    def testmixedbudgets(self):
        image = Image.new("RGB", (16, 16), (70, 90, 120))
        caps, seeds = (10, 12, 15, 40), (42, 52, 3047, 3407)
        for width in (3, 4):
            with self.subTest(width=width), Batch((editor(), ClipJudge(Clip(), Processor()), None), width) as batching:
                def work(index):
                    args = self.args("--cfg", "--guidance", "2", "--referencepath", "prediction", "--poststeps", "2")
                    args.batch, args.seed, args.maxsamples, args.maxcalls, args.noise = width, seeds[index], caps[index], caps[index], 1 + index % 2
                    backend, scorer, _ = batching.replicas()
                    return run(backend, scorer, image, "source", "target long", args)[1]

                reports = []
                for offset in range(0, 4, width):
                    reports.extend(batching.map(work, list(range(offset, min(offset + width, 4)))))
                totals = batching.report()
            for cap, report in zip(caps, reports):
                stats = report["stats"]
                self.assertLessEqual(stats["samples"], cap)
                self.assertLessEqual(stats["calls"], cap)
                self.assertEqual(sum(row["samples"] for row in stats["budget"]["stages"].values()), stats["samples"])
                self.assertEqual(sum(row["calls"] for row in stats["budget"]["stages"].values()), stats["calls"])
            self.assertEqual(totals["denoisersamples"], sum(report["stats"]["samples"] for report in reports))
            self.assertEqual(totals["denoiserrequests"], sum(report["stats"]["calls"] for report in reports))

    def testsharedprobabilities(self):
        from engine.prior import Prior
        from flow import editor as floweditor
        image = Image.new("RGB", (16, 16))
        seeds = (42, 52, 3047, 3407)
        for family, build in (("vp", editor), ("flow", floweditor)):
            current = build()
            sessions = [current.prepare(image, "source", "target long", seed) for seed in seeds]
            count = current.training - 1 if family == "flow" else current.maxtime
            probability = torch.arange(1, count + 1, dtype=torch.float64, requires_grad=True)
            original = probability.detach().clone()
            mode = "flow" if family == "flow" else "rec"
            expected = [Prior(session, 3, seed, mode, probability) for session, seed in zip(sessions, seeds)]
            for width in (3, 4):
                with self.subTest(family=family, width=width), Batch((current, None, None), width) as batching:
                    def work(index):
                        return Prior(sessions[index], 3, seeds[index], mode, probability)

                    actual = []
                    for offset in range(0, 4, width):
                        actual.extend(batching.map(work, list(range(offset, min(offset + width, 4)))))
                self.assertTrue(torch.equal(probability.detach(), original))
                for first, second in zip(expected, actual):
                    self.assertEqual(first.ticks, second.ticks)
                    self.assertEqual(first.weights, second.weights)
                    self.assertTrue(all(torch.equal(one, two) for one, two in zip(first.noises, second.noises)))

    def testkeyfailure(self):
        with self.assertRaisesRegex(ValueError, "invalid queue key"):
            with Batch((editor(), None, None), 3) as batching:
                with patch.object(batching, "key", side_effect=ValueError("invalid queue key")):
                    batching.map(lambda index: batching.call(batching.resources[0], "decode", torch.ones(1, 3, 2, 2)), list(range(3)))
        self.assertFalse(batching.thread.is_alive())
        self.assertEqual(batching.queue.unfinished_tasks, 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable.")
    def testcudaretention(self):
        from types import SimpleNamespace
        device = torch.device("cuda:0")
        current = SimpleNamespace(metadata={"family": "vp"}, decode=lambda value: value * 2)
        warmup = torch.ones(1, 3, 512, 512, device=device)
        self.assertEqual(float((warmup * 2).mean()), 2.)
        del warmup
        torch.cuda.synchronize(device)
        baseline = torch.cuda.memory_allocated(device)
        for width in (3, 4):
            readings = []
            with self.subTest(width=width), Batch((current, None, None), width, .002) as batching:
                def work(index):
                    value = torch.ones(1, 3, 512, 512, device=device)
                    output = batching.call(current, "decode", value)
                    return float(output.mean())

                for group in range(20):
                    self.assertEqual(batching.map(work, list(range(width))), [2.] * width)
                    torch.cuda.synchronize(device)
                    readings.append(torch.cuda.memory_allocated(device))
            self.assertLessEqual(max(readings[5:]) - min(readings[5:]), 2 ** 20)
            self.assertLessEqual(max(readings[5:]), baseline + 2 ** 20)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable.")
    def testcudarng(self):
        base = editor()
        device = torch.device("cuda:0")
        current = type(base)(base.unet, base.scheduler, base.vae, base.tokenizer, base.text, device=device, size=16)
        judge = ClipJudge(Clip(), Processor(), device)
        image = Image.new("RGB", (16, 16), (70, 90, 120))
        args = self.args("--algorithm", "random", "--representation", "random", "--prior", "rec", "--priorprobes", "1", "--maxsamples", "64", "--maxcalls", "64")
        state = torch.cuda.get_rng_state(device).clone()
        for width in (3, 4):
            args.batch = width
            with Batch((current, judge, None), width) as batching:
                def work(seed):
                    local = deepcopy(args)
                    local.seed = seed
                    backend, scorer, _ = batching.replicas()
                    return run(backend, scorer, image, "source", "target long", local)[1]["stats"]["samples"]

                values = (42, 52, 3047, 3407)
                for offset in range(0, 4, width):
                    self.assertTrue(all(value <= 64 for value in batching.map(work, values[offset:offset + width])))
            self.assertTrue(torch.equal(torch.cuda.get_rng_state(device), state))

    def testisolatedrequests(self):
        image = Image.new("RGB", (16, 16), (70, 90, 120))
        args = self.args()
        serial = []
        seeds = (42, 52, 3047, 3407)
        for seed in seeds:
            args.seed = seed
            serial.append(run(editor(), ClipJudge(Clip(), Processor()), image, "source", "target long", args))
        for width in (3, 4):
            current, judge = editor(), ClipJudge(Clip(), Processor())
            args.batch = width
            with Batch((current, judge, None), width, .01) as batching:
                def work(seed):
                    backend, scorer, _ = batching.replicas()
                    local = deepcopy(args)
                    local.seed = seed
                    return run(backend, scorer, image, "source", "target long", local)

                parallel = []
                for offset in range(0, len(seeds), width):
                    parallel.extend(batching.map(work, seeds[offset:offset + width]))
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
                    self.assertEqual(set(payload["byseed"]), {"42", "52", "3047", "3407"})
                    for seed in (42, 52, 3047, 3407):
                        row = next(row for row in payload["rows"] if row["seed"] == seed)
                        self.assertEqual(payload["byseed"][str(seed)]["mse"]["mean"], row["metrics"]["mse"])
                    self.assertAlmostEqual(payload["summary"]["mse"]["mean"], sum(payload["byseed"][str(seed)]["mse"]["mean"] for seed in (42, 52, 3047, 3407)) / 4)
                    self.assertEqual(payload["execution"]["width"], 3)
                    self.assertGreater(payload["execution"]["throughput"], 0)
                    self.assertTrue(payload["execution"]["accounting"]["checked"])
                    self.assertTrue(payload["execution"]["accounting"]["matched"])
                    self.assertEqual(payload["execution"]["denoisersamples"], sum(row["report"]["stats"]["samples"] for row in payload["rows"]))
                    self.assertEqual(payload["execution"]["denoiserrequests"], sum(row["report"]["stats"]["calls"] for row in payload["rows"]))
                    for row in payload["rows"]:
                        self.assertTrue(row["concurrent"])
                        self.assertIsNone(row["vram"])
                        self.assertGreater(row["report"]["reportstats"]["evaluations"], 0)
                        self.assertTrue((Path(args.output) / row["image"]).is_file())
                    self.assertTrue((Path(args.output) / "summary.json").is_file())
                    if family == "flux":
                        self.assertEqual((current.pools, current.textids), ({}, {}))
                original = Batch.report

                def tampered(self):
                    result = original(self)
                    result["denoisersamples"] += 1
                    return result

                args.output = "outputsbroken"
                with patch.object(Batch, "report", new=tampered), contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "counts do not match"):
                    execute(args, (floweditor(), ClipJudge(Clip(), Processor()), None, False))
                self.assertFalse(Path("outputsbroken/summary.json").exists())
                self.assertTrue(Path("outputsbroken/failure.json").is_file())
            finally:
                os.chdir(origin)

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
