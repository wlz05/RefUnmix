# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from __future__ import annotations
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from dataclasses import replace
from PIL import Image
from benchmark import seam, table
from data import Record, clean, records
from diagnose import averageprecision, calibrate, measure, nested, spearman, supportscore
from engine.model import EditConfig, Editor
from engine.frequency import FrequencyRouter
from engine.feedback import FeedbackConfig, FeedbackRouter, inspect
from dataset import EditDataset, annotation, loader
from split import create, select
from engine.router import RouterConfig, ThreadRouter

class ConfigTest(unittest.TestCase):
    def testcase(self) -> None:
        config = EditConfig(method="THREAD")
        config.validate()
        self.assertEqual(config.method, "thread")
        self.assertEqual(config.mode, "surprisal")
        self.assertFalse(config.edge)
        with self.assertRaises(ValueError):
            EditConfig(noise=True).validate()
        with self.assertRaises(ValueError):
            EditConfig(scale=float("nan")).validate()
        with self.assertRaises(ValueError):
            EditConfig(scale=-1.0).validate()
        with self.assertRaises(ValueError):
            EditConfig(method="frequency", couple=float("nan")).validate()

    def testrouter(self) -> None:
        for fields in ({"gain": 0.9}, {"floor": 1.1}, {"path": 0.0, "grid": 0.0}, {"pivot": -1.0}, {"signal": 0.0}, {"rough": float("inf")}, {"band": 3, "reach": 1}, {"route": False, "scalar": True}):
            with self.assertRaises((ValueError, TypeError)):
                RouterConfig(**fields).validate()

    def testschedule(self) -> None:
        editor = Editor.__new__(Editor)
        editor.alphas = torch.tensor([1e-8])
        editor.device = torch.device("cpu")
        tensor = torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16)
        alpha, sigma = editor.schedule(tensor, torch.tensor([0]))
        self.assertTrue(torch.isfinite(alpha).all())
        self.assertTrue(torch.isfinite(sigma).all())
        self.assertGreater(alpha.item(), 0.0)
        self.assertLess(alpha.item(), torch.finfo(torch.bfloat16).eps)

    def testrecover(self) -> None:
        editor = Editor.__new__(Editor)
        sample = torch.tensor([2.0])
        estimate = torch.tensor([0.5])
        alpha = torch.tensor([0.8])
        sigma = torch.tensor([0.6])
        expected = {"epsilon": torch.tensor([2.125]), "sample": estimate, "vprediction": torch.tensor([1.3])}
        for kind, value in expected.items():
            editor.kind = kind
            self.assertTrue(torch.allclose(editor.recover(sample, estimate, alpha, sigma), value))

class DataTest(unittest.TestCase):
    def testcase(self) -> None:
        origin = os.getcwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder)
                root = Path("PIE")
                (root / "Images").mkdir(parents=True)
                Image.new("RGB", (2, 2)).save(root / "Images" / "Photo.PNG")
                content = {"Sample": {"IMAGE_PATH": "PHOTO.PNG", "ORIGINAL_PROMPT": "A Cat", "EDITING_PROMPT": "A Dog"}}
                (root / "Mapping.JSON").write_text(json.dumps(content), encoding="utf-8")
                item = records("PIE", "MAPPING.JSON", "IMAGES")[0]
                self.assertEqual(item.key, "sample")
                self.assertEqual(item.relative.as_posix(), "photo.png")
                self.assertEqual(item.source, "A Cat")
            finally:
                os.chdir(origin)

    def testpath(self) -> None:
        for value in ("../pie", str(Path.cwd().anchor)):
            with self.assertRaises(ValueError):
                clean(value, "dataset")

class FrequencyTest(unittest.TestCase):
    def testprojection(self) -> None:
        router = FrequencyRouter()
        for shape in ((1, 4, 8, 8), (2, 3, 7, 5), (1, 1, 1, 1)):
            value = torch.randn(shape)
            low = router.low(value)
            high = router.high(value)
            self.assertTrue(torch.allclose(low + high, value))
            self.assertTrue(torch.allclose(router.low(high), torch.zeros_like(high), atol=1e-6))

    def teststripes(self) -> None:
        router = FrequencyRouter()
        anchor = torch.zeros(1, 4, 8, 8)
        stripes = torch.arange(8).remainder(2).float().mul(2).sub(1).reshape(1, 1, 1, 8).expand_as(anchor)
        output = router(anchor, stripes, stripes, anchor, anchor, 0.9, 0.75)
        self.assertGreater(float(output.maps["highweight"].mean()), 0.5)
        self.assertLess(float(output.maps["lowweight"].mean()), 0.1)
        self.assertTrue(torch.allclose(output.field, router.high(output.field), atol=1e-6))
        perfect = router.audit(anchor, output.target, output.target, output.mask)
        self.assertTrue(all(torch.count_nonzero(perfect[key]) == 0 for key in ("proxyhh", "proxyhl", "proxyll")))
        original = router.audit(anchor, anchor, output.target, output.mask)
        self.assertGreater(float(original["proxyhh"].mean()), 0.5)

    def testtime(self) -> None:
        router = FrequencyRouter()
        anchor = torch.zeros(1, 4, 8, 8)
        detail = torch.randn_like(anchor)
        current = detail * 0.5
        previous = detail
        output = router(anchor, current, previous, torch.ones_like(anchor), anchor, 0.9, 0.75)
        self.assertAlmostEqual(float(output.maps["hightime"].mean()), 0.75, places=6)
        self.assertTrue(torch.isfinite(output.field).all())

    def testzero(self) -> None:
        router = FrequencyRouter()
        anchor = torch.randn(1, 4, 7, 5)
        zero = torch.zeros_like(anchor)
        output = router(anchor, zero, zero, anchor, anchor, 0.9, 0.75)
        self.assertEqual(torch.count_nonzero(output.field).item(), 0)
        self.assertTrue(torch.isfinite(output.mask).all())

class RouterTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.target = torch.randn(2, 4, 8, 8)
        self.reference = torch.randn(2, 4, 8, 8)
        self.current = self.target - self.reference
        self.previous = self.current + 0.1 * torch.randn(2, 4, 8, 8)
        self.base = (0.15 * self.current + 0.9 * self.previous) / 1.05
        self.latent = torch.randn(2, 4, 8, 8)

    def route(self, **fields):
        return ThreadRouter(RouterConfig(**fields))(self.base, self.current, self.previous, self.target, self.reference, self.latent)

    def testmaps(self) -> None:
        result = self.route()
        for key in ("evidence", "surprisal", "probechange", "pathchange", "gridroughness", "trust", "gate", "relay", "anchor"):
            self.assertTrue(torch.isfinite(result.maps[key]).all())
        self.assertTrue(torch.all((result.maps["pathchange"] >= 0.0) & (result.maps["pathchange"] <= 2.0)))
        self.assertTrue(torch.all((result.maps["trust"] >= 0.0) & (result.maps["trust"] <= 1.0)))
        self.assertEqual(torch.count_nonzero(result.maps["relay"]).item(), 0)

    def testgate(self) -> None:
        config = RouterConfig(gain=1.5, floor=0.2, feather=0)
        result = ThreadRouter(config)(self.base, self.current, self.previous, self.target, self.reference, self.latent)
        expected = result.maps["evidence"] * (config.floor + (config.gain - config.floor) * result.maps["trust"])
        self.assertTrue(torch.allclose(result.maps["gate"], expected))
        direct = ThreadRouter(config).clip(expected * self.base, result.maps["cap"])
        self.assertTrue(torch.allclose(result.field, direct, atol=1e-6))

    def testbound(self) -> None:
        result = self.route(edge=True, gain=1.5)
        magnitude = result.field.square().sum(dim=1, keepdim=True).sqrt()
        self.assertTrue(torch.all(magnitude <= result.maps["cap"] + 1e-5))
        self.assertTrue(torch.all((result.mask >= 0.0) & (result.mask <= 1.0)))

    def testmodes(self) -> None:
        values = []
        for mode in ("surprisal", "relative", "invariant", "magnitude"):
            evidence = self.route(mode=mode).maps["evidence"]
            self.assertTrue(torch.all((evidence >= 0.0) & (evidence <= 1.0)))
            values.append(evidence)
        self.assertFalse(torch.allclose(values[0], values[3]))

    def testscale(self) -> None:
        router = ThreadRouter(RouterConfig(path=1.0, grid=1.0, rough=0.5))
        first = router.probemap(self.base, self.current, self.previous)
        second = router.probemap(self.base * 1e-8, self.current * 1e-8, self.previous * 1e-8)
        for maps in (first, second):
            for value in maps:
                self.assertTrue(torch.isfinite(value).all())
        self.assertTrue(torch.all(second[2] <= first[2] + 1e-6))

    def testscalar(self) -> None:
        diagonal = self.route()
        result = self.route(scalar=True)
        trust = result.maps["trust"]
        self.assertTrue(torch.allclose(trust, trust[:, :, :1, :1].expand_as(trust)))
        self.assertTrue(torch.allclose(diagonal.maps["gate"].mean(dim=(2, 3)), result.maps["gate"].mean(dim=(2, 3)), atol=1e-5))

    def testablate(self) -> None:
        result = self.route(route=False)
        self.assertTrue(torch.allclose(result.maps["trust"], torch.ones_like(result.maps["trust"])))
        self.assertTrue(torch.allclose(result.maps["gate"], result.maps["evidence"]))

    def testrelay(self) -> None:
        support = torch.zeros(1, 1, 16, 16)
        support[:, :, 7:9, 7:9] = 1.0
        router = ThreadRouter(RouterConfig(edge=True, band=3, reach=4))
        band, reached = router.bandmap(support)
        self.assertGreater(torch.count_nonzero(band).item(), 0)
        self.assertEqual(torch.count_nonzero((band > 0.0) & (reached <= 0.0)).item(), 0)
        cap = torch.full((1, 1, 1, 1), 1e6)
        relay = router.relayfield(torch.randn(1, 4, 16, 16), torch.randn(1, 4, 16, 16), support, torch.randn(1, 4, 16, 16), cap)
        self.assertTrue(torch.isfinite(relay).all())

    def testzero(self) -> None:
        zeros = torch.zeros_like(self.base)
        result = ThreadRouter(RouterConfig())(zeros, zeros, zeros, zeros, zeros, zeros)
        self.assertEqual(torch.count_nonzero(result.field).item(), 0)
        self.assertEqual(torch.count_nonzero(result.maps["pathchange"]).item(), 0)
        self.assertEqual(torch.count_nonzero(result.maps["gridroughness"]).item(), 0)

    def testsparse(self) -> None:
        target = torch.zeros(1, 4, 8, 8)
        target[:, :, 0, 0] = 1.0
        zeros = torch.zeros_like(target)
        result = ThreadRouter(RouterConfig(feather=0))(target, target, target, target, zeros, zeros)
        self.assertGreater(float(result.maps["evidence"][0, 0, 0, 0]), 0.99)
        self.assertGreater(float(result.field.abs().sum()), 0.0)

    def testinvalid(self) -> None:
        value = self.base.clone()
        value[0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            ThreadRouter(RouterConfig())(value, self.current, self.previous, self.target, self.reference, self.latent)
        with self.assertRaises(ValueError):
            ThreadRouter(RouterConfig())(self.base, self.current, self.previous, self.target, self.reference, self.target[:, :2])

class FeedbackTest(unittest.TestCase):
    def maps(self, editable: bool = True):
        low = torch.ones(1, 3, 16, 16) * 0.2
        high = torch.zeros_like(low)
        maps = {"lowfield": low, "highfield": high, "earlylow": low, "latelow": low, "earlyhigh": high, "latehigh": high, "mask": torch.full((1, 1, 16, 16), float(editable)), "base": low * 0.1}
        maps.update({key: torch.tensor(value) for key, value in {"lowweight": 0.5, "highweight": 0.5, "lowtime": 0.9, "hightime": 0.75, "earlytime": 0.9, "latetime": 0.75, "explicit": True}.items()})
        return maps

    def render(self, field):
        pixels = (field[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype(np.uint8)
        return field, Image.fromarray(pixels)

    def judge(self, image):
        value = float(np.asarray(image).mean() / 255)
        return {"global": value, "local": value, "direction": 0.0}

    def testcorrection(self) -> None:
        source = Image.new("RGB", (16, 16))
        latent, image, field, report = FeedbackRouter(FeedbackConfig()).run(self.render, torch.zeros(1, 3, 16, 16), source, self.maps(), self.judge)
        self.assertTrue(report["accepted"])
        self.assertEqual(report["selected"], "correction")
        self.assertGreater(report["weights"][0], 0.5)
        self.assertGreater(float(np.asarray(image).mean()), 25.5)
        scores = [row["score"] for row in report["history"] if row["accepted"]]
        self.assertEqual(scores, sorted(set(scores)))
        self.assertEqual(len(report["interaction"]), 1)

    def testbudget(self) -> None:
        source = Image.new("RGB", (16, 16))
        _, image, field, report = FeedbackRouter(FeedbackConfig(budget=0)).run(self.render, torch.zeros(1, 3, 16, 16), source, self.maps(False), self.judge)
        self.assertFalse(report["accepted"])
        self.assertEqual(report["selected"], "source")
        self.assertEqual(int(field.count_nonzero()), 0)
        self.assertTrue(np.array_equal(np.asarray(source), np.asarray(image)))
        self.assertTrue(all(not row["feasible"] for row in report["history"]))

    def testbackground(self) -> None:
        source = Image.new("RGB", (64, 64), (100, 100, 100))
        mask = torch.ones(1, 1, 64, 64)
        mask[..., 16:48, 16:48] = 0
        changed = np.asarray(source).copy()
        changed[mask[0, 0].numpy() > 0] = 200
        audit = inspect(source, Image.fromarray(changed), mask)
        self.assertEqual(audit["damage"], 0.0)
        self.assertEqual(audit["ll"], 0.0)
        self.assertGreater(audit["auditarea"], 0.0)
        changed[24:40, 24:40] = 0
        damaged = inspect(source, Image.fromarray(changed), mask)
        self.assertGreater(damaged["damage"], audit["damage"])
        self.assertGreater(damaged["ll"], audit["ll"])

    def testnonfinite(self) -> None:
        judge = lambda image: {"global": float("nan"), "local": 1.0, "direction": 0.0}
        _, _, _, report = FeedbackRouter(FeedbackConfig(rounds=0)).run(self.render, torch.zeros(1, 3, 16, 16), Image.new("RGB", (16, 16)), self.maps(), judge)
        self.assertFalse(report["accepted"])
        json.dumps(report, allow_nan=False)

class DatasetTest(unittest.TestCase):
    def testmask(self) -> None:
        item = Record("a", Path("a.png"), Path("a.png"), "a", "b", "change", mask=(0, 2, 14, 2), masksize=(4, 4))
        decoded = np.asarray(annotation(item))
        self.assertEqual(int((decoded > 0).sum()), 4)
        self.assertTrue(np.all(decoded.flatten()[[0, 1, 14, 15]] == 255))
        empty = replace(item, mask=())
        self.assertEqual(int(np.asarray(annotation(empty)).sum()), 0)
        self.assertGreater(int(np.asarray(annotation(empty, border=True)).sum()), 0)
        with self.assertRaises(ValueError):
            annotation(replace(item, mask=(1,)))
        with self.assertRaises(ValueError):
            annotation(replace(item, mask=(14, 9)))

    def testsplit(self) -> None:
        origin = os.getcwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder)
                items = []
                for index in range(9):
                    path = Path(str(index) + ".png")
                    Image.new("RGB", (8, 8), (index * 20, 0, 0)).save(path)
                    items.append(Record(str(index), path, path, "a", "b", "change"))
                items.append(replace(items[0], key="repeat"))
                payload = create(items)
                membership = {key: subset for subset, keys in payload["subsets"].items() for key in keys}
                self.assertEqual(membership["0"], membership["repeat"])
                self.assertEqual(payload, create(list(reversed(items))))
                Path("split.json").write_text(json.dumps(payload), encoding="utf-8")
                chosen = select(items, "split.json", "TEST")
                self.assertTrue(chosen)
                old = membership["repeat"]
                new = "test" if old != "test" else "train"
                payload["subsets"][old].remove("repeat")
                payload["subsets"][new].append("repeat")
                Path("split.json").write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(ValueError):
                    select(items, "split.json", "test")
                with self.assertRaises(ValueError):
                    EditDataset(items, paired=True)
                sample = EditDataset(items, size=8)[0]
                self.assertFalse(sample["haspair"])
                self.assertNotIn("edited", sample)
                paired = EditDataset([replace(items[0], edited=items[1].image)], size=8, paired=True)[0]
                self.assertTrue(paired["haspair"])
                self.assertFalse(torch.equal(paired["image"], paired["edited"]))
                self.assertEqual(tuple(next(iter(loader(EditDataset(items, size=8), batch=2)))["image"].shape), (2, 3, 8, 8))
            finally:
                os.chdir(origin)

class MetricTest(unittest.TestCase):
    def testspearman(self) -> None:
        left = torch.tensor([1.0, 2.0, 3.0, 4.0])
        right = torch.tensor([10.0, 20.0, 30.0, 40.0])
        self.assertAlmostEqual(spearman(left, right), 1.0, places=6)
        self.assertAlmostEqual(spearman(left, torch.flip(right, (0,))), -1.0, places=6)
        self.assertAlmostEqual(spearman(left, torch.ones_like(right)), 0.0, places=6)

    def testaverage(self) -> None:
        scores = torch.tensor([0.9, 0.8, 0.2, 0.1])
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
        self.assertAlmostEqual(averageprecision(scores, labels), 1.0, places=6)
        self.assertAlmostEqual(averageprecision(torch.flip(scores, (0,)), labels), 0.4166667, places=6)
        self.assertTrue(math.isnan(averageprecision(scores, torch.ones_like(labels))))

    def testseam(self) -> None:
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, :, 4:] = 1.0
        flat = Image.fromarray(torch.full((64, 64, 3), 128, dtype=torch.uint8).numpy())
        self.assertAlmostEqual(seam(flat, mask), 0.0, places=6)

    def testsupport(self) -> None:
        scores = torch.tensor([[[[0.9, 0.8], [0.2, 0.1]]]])
        target = torch.tensor([[[[True, True], [False, False]]]])
        ap, iou = supportscore(scores, target)
        self.assertAlmostEqual(ap, 1.0, places=6)
        self.assertAlmostEqual(iou, 1.0, places=6)

    def testtable(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "values.csv"
            table(path, "value\n", ["one"], False)
            table(path, "value\n", ["two"], False)
            self.assertEqual(path.read_text(encoding="utf-8"), "value\none\ntwo\n")

    def testnested(self) -> None:
        samples = []
        for seed in range(5):
            generator = torch.Generator().manual_seed(seed)
            features = torch.randn(32, 5, generator=generator)
            target = 2.0 * features[:, 3] + 0.1 * features[:, 0]
            samples.append((features, target, target > target.median()))
        result = nested(samples)
        self.assertGreater(result["pathgain"]["r2"]["mean"], 0.5)

class StubScheduler:
    class Config:
        num_train_timesteps = 1000
        prediction_type = "epsilon"

    def __init__(self) -> None:
        self.config = self.Config()
        self.alphas_cumprod = torch.linspace(0.99, 0.01, 1000).double()

class StubUNet(nn.Module):
    def forward(self, sample, timestep, encoder_hidden_states, return_dict=False):
        prompt = encoder_hidden_states.sum(dim=1, keepdim=True).transpose(1, 2)
        prompt = prompt[:, :sample.shape[1]].reshape(sample.shape[0], sample.shape[1], 1, 1)
        scale = timestep.float().reshape(-1, 1, 1, 1) / 1000.0
        return ((sample * 0.5 + prompt) * scale,)

class StubDistribution:
    def __init__(self, value) -> None:
        self.value = value

    def mode(self):
        return self.value

class StubVAE(nn.Module):
    class Config:
        scaling_factor = 0.18215

    def __init__(self) -> None:
        super().__init__()
        self.config = self.Config()

    def encode(self, tensor):
        pooled = torch.nn.functional.avg_pool2d(tensor, 8)
        stacked = torch.stack([pooled[:, index % 3] for index in range(4)], dim=1)

        class Encoded:
            latent_dist = StubDistribution(stacked)

        return Encoded()

    def decode(self, tensor):
        upsampled = torch.nn.functional.interpolate(tensor, scale_factor=8, mode="nearest")

        class Decoded:
            sample = upsampled[:, :3].clamp(-1, 1)

        return Decoded()

class StubTokenizer:
    model_max_length = 77

    def __call__(self, values, **fields):
        tokens = torch.zeros(len(values), 77, dtype=torch.long)
        for row, value in enumerate(values):
            for column, character in enumerate(value[:77]):
                tokens[row, column] = ord(character)

        class Tokens:
            input_ids = tokens

        return Tokens()

class StubText(nn.Module):
    class Config:
        max_position_embeddings = 77

    def __init__(self) -> None:
        super().__init__()
        self.config = self.Config()

    def forward(self, input_ids):
        hidden = torch.zeros(input_ids.shape[0], 77, 8)
        hidden[:, :, 0] = input_ids.float() / 10.0
        hidden[:, :, 1] = 1.0
        return (hidden,)

class EditTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        self.editor = Editor(StubUNet(), StubScheduler(), StubVAE(), StubTokenizer(), StubText(), "cpu", torch.float32, 64)
        self.image = Image.fromarray((torch.rand(64, 64, 3).numpy() * 255).astype("uint8"))

    def testprobe(self) -> None:
        anchor = self.editor.image(self.image)
        probe = self.editor.probe(anchor, self.editor.prompt("a cat"), self.editor.prompt("a dog"), [torch.randn_like(anchor)], 0.90, 0.15)
        self.assertEqual(tuple(probe.fields.shape), (1, 1, 4, 8, 8))
        self.assertTrue(torch.allclose(probe.current, probe.target - probe.reference))

    def testedit(self) -> None:
        for method in ("base", "thread", "frequency"):
            result = self.editor.edit(self.image, "a cat", "a dog", EditConfig(method=method, reach=4, band=2), seed=3)
            self.assertEqual(result.images[0].size, (64, 64))
            self.assertTrue(torch.isfinite(result.latent).all())
            self.assertIn("field", result.maps)
            if method == "frequency":
                self.assertTrue(all(key in result.maps for key in ("proxyhh", "proxyhl", "proxyll", "lowtime", "hightime", "lowweight", "highweight")))

    def testfeedback(self) -> None:
        class Judge:
            def prepare(self, source, original, target, mask, localtext):
                self.mask = mask
                return lambda image: {"global": 0.0, "local": 0.0, "direction": 0.0}

        judge = Judge()
        region = Image.new("L", self.image.size, 255)
        result = self.editor.edit(self.image, "a cat", "a dog", EditConfig(method="feedback"), region=region, polarity="PROTECT", judge=judge, feedback=FeedbackConfig(rounds=0))
        self.assertEqual(float(judge.mask.sum()), 0.0)
        self.assertEqual(result.report["selected"], "source")
        self.assertTrue(np.array_equal(np.asarray(result.images[0]), np.asarray(self.image)))
        self.assertEqual(result.report["samples"], 4 + result.report["evaluations"])
        self.assertEqual(result.report["calls"], 1 + result.report["evaluations"])
        with self.assertRaises(ValueError):
            self.editor.edit(self.image, "a cat", "a dog", EditConfig(method="feedback"))
        with self.assertRaises(ValueError):
            self.editor.edit(self.image, "a cat", "a dog", EditConfig(method="feedback"), judge=judge, polarity="protect")
        unchanged = self.editor.edit(self.image, "A Cat", "a cat", EditConfig(method="feedback"), judge=judge)
        self.assertEqual(unchanged.report["calls"], 0)
        self.assertEqual(unchanged.report["reason"], "identical prompts")

    def testreference(self) -> None:
        anchor = self.editor.image(self.image)
        source = self.editor.prompt("a cat")
        target = self.editor.prompt("a dog")
        noise = torch.randn_like(anchor)
        config = EditConfig(cleanup=False)
        result = self.editor.reference(anchor, source, target, noise, config)
        self.assertEqual(tuple(result["error"].shape), (1, 1, 8, 8))
        self.assertTrue(all(torch.isfinite(value).all() for value in result.values()))
        direction = self.editor.direction(anchor, source, target, noise, config)
        self.assertEqual(tuple(direction.shape), (1, 1, 8, 8))
        self.assertTrue(torch.isfinite(direction).all())

    def testdiagnose(self) -> None:
        maps, reference, routed, scan = measure(self.editor, self.image, "a cat", "a dog", EditConfig(cleanup=False), 4, [1e-4, 1e-3], [0.005, 0.01])
        self.assertEqual(tuple(maps["direction"].shape), (1, 1, 8, 8))
        self.assertEqual(tuple(reference["error"].shape), (1, 1, 8, 8))
        self.assertEqual(tuple(routed["error"].shape), (1, 1, 8, 8))
        self.assertEqual(len(scan["epsilon"]), 2)
        self.assertIn("editrelative", maps)

    def testcalibrate(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "image.png"
            self.image.save(path)
            item = Record("image", path, Path("image.png"), "a cat", "a dog", "change")
            config, values = calibrate(self.editor, [item], EditConfig(cleanup=False), 4)
            config.validate()
            self.assertGreater(values["signal"], 0.0)
            self.assertGreater(values["rough"], 0.0)
            self.assertGreater(values["pivot"], 0.0)

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu 2026
# Email: wlzchina05@gmail.com
