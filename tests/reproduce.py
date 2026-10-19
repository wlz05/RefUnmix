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
import types
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from doctor import check, dependencies, main, parser, weights
from engine.reproduce import digest, seed, snapshot, worker
from evaluate import datasetargs, execute
from run import common

class ReproduceTests(unittest.TestCase):
    def setUp(self):
        self.previous = Path.cwd()
        self.temporary = tempfile.TemporaryDirectory()
        os.chdir(self.temporary.name)
        Path("pie/images").mkdir(parents=True)
        Image.new("RGB", (16, 16), "red").save("pie/images/source.png")
        mapping = {"a": {"image": "source.png", "source": "Red.", "target": "Blue.", "attributes": ["Blue."], "localtext": "A blue object."}}
        Path("pie/map.json").write_text(json.dumps(mapping), encoding="utf-8")

    def tearDown(self):
        os.chdir(self.previous)
        self.temporary.cleanup()

    def args(self, *values):
        return parser().parse_args(["--dataset", "pie", "--mapping", "map.json", "--images", "images", *values])

    def testseed(self):
        seed(17)
        first = random.random(), float(np.random.rand()), float(torch.rand(1))
        seed(17)
        self.assertEqual(first, (random.random(), float(np.random.rand()), float(torch.rand(1))))
        for value in (True, 2 ** 64, 1.5):
            with self.assertRaises(ValueError):
                seed(value)

    def testworker(self):
        with patch("torch.initial_seed", return_value=2 ** 32 + 7):
            worker(0)
            first = random.random(), float(np.random.rand())
            worker(0)
            self.assertEqual(first, (random.random(), float(np.random.rand())))

    def testdeterministicinvalid(self):
        with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": "invalid"}), self.assertRaises(ValueError):
            seed(17, True)

    def testdeterministiclate(self):
        with patch.dict(os.environ), patch("torch.cuda.is_initialized", return_value=True):
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            with self.assertRaisesRegex(ValueError, "restart"):
                seed(17, True)

    def testdeterministic(self):
        controls = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(), torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic, torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        try:
            with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}):
                seed(17, True)
                self.assertTrue(snapshot()["torch"]["deterministic"])
                self.assertFalse(snapshot()["torch"]["benchmark"])
        finally:
            torch.use_deterministic_algorithms(controls[0], warn_only=controls[1])
            torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = controls[2:4]
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = controls[4:]

    def testsnapshot(self):
        result = snapshot()
        self.assertEqual(len(result["codesha256"]), 64)
        self.assertIn("torch", result["packages"])
        self.assertEqual(result["format"]["alpha"], "white RGB matte")
        json.dumps(result, allow_nan=False)
        self.assertEqual(len(digest("pie/map.json")), 64)

    def testdoctor(self):
        result = check(self.args())
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["dataset"]["inspected"], 1)
        self.assertTrue(result["dataset"]["complete"])

    def testdoctorbroken(self):
        Path("pie/images/source.png").write_bytes(b"invalid")
        result = check(self.args())
        self.assertFalse(result["ok"])
        self.assertTrue(any("Record a" in value for value in result["errors"]))

    def testdoctorselection(self):
        for arguments in (("--limit", "0"), ("--precision", "fp16"), ("--device", "cuda:999"), ("--split", "test")):
            with self.subTest(arguments=arguments):
                self.assertFalse(check(self.args(*arguments))["ok"])

    def testdoctormissing(self):
        with patch("doctor.metadata.version", side_effect=Exception("missing")):
            result = check(self.args())
        self.assertFalse(result["ok"])
        self.assertNotIn("dataset", result)

    def testdoctorweights(self):
        Path("models").mkdir()
        with self.assertRaises(ValueError):
            weights("models", "vp")
        for name in ("unet", "scheduler", "vae", "tokenizer", "text_encoder"):
            Path("models", name).mkdir()
        with self.assertRaises(ValueError):
            weights("models", "vp")
        for name in ("unet", "vae", "text_encoder"):
            Path("models", name, "weights.safetensors").write_bytes(b"fixture")
        self.assertEqual(weights("models", "vp")["parts"]["text"], "models/text_encoder")

    def testdoctorcli(self):
        with patch("sys.argv", ["doctor.py", "--dataset", "pie", "--mapping", "map.json", "--images", "images", "--report", "check.json"]), contextlib.redirect_stdout(io.StringIO()):
            main()
        self.assertTrue(json.loads(Path("check.json").read_text())["ok"])

    def testimmutableevaluation(self):
        args = datasetargs(common(argparse.ArgumentParser())).parse_args(["--dataset", "pie", "--mapping", "map.json", "--images", "images", "--split", "all", "--method", "base", "--size", "16", "--seed", "5", "--seeds", "9", "7", "--output", "evaluation"])
        original = deepcopy(vars(args))
        image = torch.ones(1, 3, 16, 16)

        def inference(editor, judge, source, origin, target, current, region):
            report = {"settings": deepcopy(vars(current))}
            return image, report, types.SimpleNamespace(origin=image, artifacts={})

        with patch("evaluate.run", side_effect=inference), contextlib.redirect_stdout(io.StringIO()):
            result = execute(args, (types.SimpleNamespace(device="cpu"), None, None, False))
        self.assertEqual(vars(args), original)
        self.assertEqual(result["settings"], original)
        self.assertEqual(result["seeds"], [7, 9])
        self.assertEqual(result["rows"][0]["report"]["settings"]["attributes"], ["Blue."])
        self.assertEqual(result["rows"][1]["report"]["settings"]["seed"], 9)
        self.assertEqual(result["inputs"]["mapping"]["sha256"], digest("pie/map.json"))
        self.assertEqual(result["rows"][0]["inputsha256"], digest("pie/images/source.png"))

    def testarchivereportevaluation(self):
        args = datasetargs(common(argparse.ArgumentParser())).parse_args(["--dataset", "pie", "--mapping", "map.json", "--images", "images", "--split", "all", "--method", "base", "--size", "16", "--output", "evaluation"])
        image = torch.ones(1, 3, 16, 16)

        class Judge:
            def __init__(self):
                self.count = 0

            def stats(self):
                return {"evaluations": self.count}

            def prepare(self, *values):
                def score(pixels):
                    self.count += 1
                    return {"whole": float(pixels.mean())}

                return score

        judge = Judge()
        session = types.SimpleNamespace(origin=image, artifacts={"raw": image, "candidate0": image * .5})
        with patch("evaluate.run", return_value=(image, {}, session)), contextlib.redirect_stdout(io.StringIO()):
            result = execute(args, (types.SimpleNamespace(device="cpu"), None, judge, False))
        report = result["rows"][0]["report"]
        self.assertNotIn("mechanism", report)
        self.assertEqual(report["artifactevaluation"]["candidate0"]["whole"], .5)
        self.assertEqual(report["reportstats"]["evaluations"], 3)
        self.assertEqual(report["artifactsha256"]["candidate0"], digest(report["artifacts"]["candidate0"]))

    def testrequestlifetime(self):
        from weakref import ref
        args = datasetargs(common(argparse.ArgumentParser())).parse_args(["--dataset", "pie", "--mapping", "map.json", "--images", "images", "--split", "all", "--method", "raw", "--size", "16", "--seeds", "1", "2", "--output", "evaluation"])
        previous = []

        class Context:
            def __init__(self, image):
                self.origin, self.artifacts = image, {}

        def inference(*values):
            self.assertTrue(all(reference() is None for reference in previous))
            image = torch.ones(1, 3, 16, 16)
            session = Context(image)
            previous[:] = [ref(image), ref(session)]
            return image, {}, session

        with patch("evaluate.run", side_effect=inference), contextlib.redirect_stdout(io.StringIO()):
            result = execute(args, (types.SimpleNamespace(device="cpu"), None, None, False))
        self.assertEqual(len(result["rows"]), 2)
        self.assertTrue(all(reference() is None for reference in previous))

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu 2026
# Email: wlzchina05@gmail.com
