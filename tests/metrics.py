# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import contextlib
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from engine.metrics import Protocol, connection, pixels, relocate, storage
from assess import grouped, main, parser, population, prediction, verified
from data import Record

class Calculator:
    def __init__(self):
        self.calls = []

    def calculate_mse(self, origin, edited, first, second):
        a = np.asarray(origin, dtype=np.float32) / 255
        b = np.asarray(edited, dtype=np.float32) / 255
        return float(np.mean((a * first - b * second) ** 2))

    def calculate_psnr(self, *args):
        return -10 * np.log10(self.calculate_mse(*args))

    def calculate_lpips(self, *args):
        self.calls.append(("lpips", args[2].copy()))
        return .1

    def calculate_ssim(self, *args):
        return .9

    def calculate_structure_distance(self, *args):
        self.calls.append(("structure", len(args)))
        return .02

    def calculate_clip_similarity(self, image, text, mask=None):
        self.calls.append(("clip", mask))
        return 25. if mask is None else 22.

class MetricTests(unittest.TestCase):
    def setUp(self):
        self.origin = Image.new("RGB", (4, 4), 0)
        self.edited = Image.new("RGB", (4, 4), (255, 255, 255))
        values = np.zeros((4, 4), dtype=np.uint8)
        values[:2] = 255
        self.mask = Image.fromarray(values)

    @contextlib.contextmanager
    def workspace(self):
        origin = Path.cwd()
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ), patch.object(torch.hub, "_hub_dir", None):
            previous = str(Path(directory) / "previous")
            os.environ.update({key: previous for key in ("HF_HOME", "HF_HUB_CACHE", "TORCH_HOME", "TRANSFORMERS_CACHE")})
            modules = {name: SimpleNamespace(HF_HOME=previous, HF_HUB_CACHE=previous, TRANSFORMERS_CACHE=previous, default_cache_path=previous) for name in ("huggingface_hub.constants", "transformers.utils.hub", "transformers.file_utils")}
            with patch.dict(sys.modules, modules):
                os.chdir(directory)
                try:
                    yield modules
                finally:
                    os.chdir(origin)

    def teststorage(self):
        with self.workspace() as modules:
            result = storage("cache/metrics")
            self.assertEqual(result, {"root": "cache/metrics", "huggingface": "cache/metrics/huggingface/hub", "torch": "cache/metrics/torch/hub"})
            self.assertEqual(torch.hub.get_dir(), result["torch"])
            self.assertTrue(Path(result["root"]).is_dir())
            self.assertNotIn("TRANSFORMERS_CACHE", os.environ)
            for module in modules.values():
                self.assertEqual(module.HF_HUB_CACHE, result["huggingface"])
                self.assertEqual(module.TRANSFORMERS_CACHE, result["huggingface"])
            for key in ("HF_HOME", "HF_HUB_CACHE", "HF_ASSETS_CACHE", "HF_XET_CACHE", "TORCH_HOME", "XDG_CACHE_HOME"):
                self.assertFalse(Path(os.environ[key]).is_absolute())
            self.assertEqual(storage("cache/metrics"), result)

    def teststorageinvalid(self):
        with self.workspace():
            before = dict(os.environ)
            for path in (str(Path.cwd() / "cache"), "../cache", ".", ""):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    storage(path)
            self.assertEqual(dict(os.environ), before)
            self.assertFalse(Path("cache").exists())

    def teststoragefile(self):
        with self.workspace():
            Path("cache").write_text("occupied", encoding="utf-8")
            before = dict(os.environ)
            with self.assertRaises(FileExistsError):
                storage("cache")
            self.assertEqual(dict(os.environ), before)

    def testcachedbackend(self):
        with self.workspace():
            methods = ("psnr", "mse", "lpips", "ssim", "structure_distance", "clip_similarity")
            code = "import os\nclass MetricsCalculator:\n def __init__(self, device):\n  self.cache = os.environ['HF_HUB_CACHE']\n"
            code += "".join(" def calculate_" + name + "(self, *args): return 1.0\n" for name in methods)
            Path("protocol.py").write_text(code, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "network"):
                Protocol.load("protocol.py", cache="cache/metrics")
            self.assertFalse(Path("cache").exists())
            result = Protocol.load("protocol.py", network=True, cache="cache/metrics")
            self.assertEqual(result.calculator.cache, "cache/metrics/huggingface/hub")
            self.assertEqual(result.metadata["cache"]["root"], "cache/metrics")

    def testpixelcache(self):
        with self.workspace(), patch("engine.metrics.storage") as cache:
            result = Protocol.load(cache="cache/metrics")
            self.assertFalse(result.metadata["complete"])
            cache.assert_not_called()
            self.assertFalse(Path("cache").exists())

    def testofflineconnection(self):
        original, extended, lookup = socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo
        with self.assertRaisesRegex(RuntimeError, "Offline metric assets"):
            with connection(True):
                socket.getaddrinfo("example.invalid", 443)
        self.assertIs(socket.socket.connect, original)
        self.assertIs(socket.socket.connect_ex, extended)
        self.assertIs(socket.getaddrinfo, lookup)
        with self.assertRaises(ValueError):
            Protocol.load(network=True, offline=True)

    def testabsolutesnapshot(self):
        with self.workspace():
            hub = Path("cache/metrics/huggingface/hub")
            repository = hub / "models--example--probe"
            blob = repository / "blobs" / ("b" * 64)
            blob.parent.mkdir(parents=True)
            blob.write_bytes(b"portable weights")
            link = repository / "snapshots" / ("1" * 40) / "config.json"
            link.parent.mkdir(parents=True)
            link.write_bytes(b"stale link")
            target = str(Path.cwd() / "previous" / blob.name)
            for previous in (target, "Z:\\previous\\blobs\\" + blob.name):
                with patch.object(Path, "is_symlink", lambda path: path == link), patch("engine.metrics.os.readlink", return_value=previous):
                    self.assertEqual(relocate(hub), 1)
                self.assertEqual(link.read_bytes(), blob.read_bytes())
                self.assertEqual(relocate(hub), 0)
            self.assertFalse(list(link.parent.glob("*.tmp")))

    def testmissingblob(self):
        with self.workspace():
            hub = Path("cache/metrics/huggingface/hub")
            link = hub / "models--example--probe" / "snapshots" / "revision" / "config.json"
            link.parent.mkdir(parents=True)
            link.write_bytes(b"stale link")
            with patch.object(Path, "is_symlink", lambda path: path == link), patch("engine.metrics.os.readlink", return_value="Z:\\previous\\blobs\\missing"):
                with self.assertRaisesRegex(FileNotFoundError, "matching local blob"):
                    storage("cache/metrics", True)
            self.assertEqual(link.read_bytes(), b"stale link")

    def testofflinecachetransfer(self):
        with self.workspace():
            home = Path.cwd()
            Path("origin").mkdir()
            os.chdir("origin")
            storage("cache/metrics")
            hub = Path(torch.hub.get_dir())
            repository = hub / "example_probe_main"
            repository.mkdir(parents=True)
            (hub / "checkpoints").mkdir()
            torch.save({"value": torch.tensor(7)}, hub / "checkpoints" / "probe.pth")
            (repository / "hubconf.py").write_text("import torch\ndef model():\n return torch.hub.load_state_dict_from_url('https://example.invalid/probe.pth', file_name='probe.pth', map_location='cpu', weights_only=True)\n", encoding="utf-8")
            Path("cache/metrics/huggingface/hub").mkdir(parents=True)
            Path("cache/metrics/huggingface/hub/probe.json").write_text("{}", encoding="utf-8")
            code = "import os\nfrom pathlib import Path\nimport torch\nclass MetricsCalculator:\n def __init__(self, device):\n  self.value = torch.hub.load('example/probe:main', 'model', trust_repo=True)['value']\n  assert (Path(os.environ['HF_HUB_CACHE']) / 'probe.json').is_file()\n"
            code += "".join(" def calculate_" + name + "(self, *args): return 1.0\n" for name in ("psnr", "mse", "lpips", "ssim", "structure_distance", "clip_similarity"))
            Path("protocol.py").write_text(code, encoding="utf-8")
            os.chdir(home)
            Path("origin").rename("server")
            os.chdir("server")
            result = Protocol.load("protocol.py", offline=True, cache="cache/metrics")
            self.assertEqual(int(result.calculator.value), 7)
            self.assertFalse(result.metadata["networkauthorized"])
            self.assertTrue(result.metadata["offline"])
            self.assertEqual(result.evaluate(self.origin, self.edited, None, "target")["raw"]["whole"], 1.)
            Path("cache/metrics/torch/hub/checkpoints/probe.pth").unlink()
            original = socket.getaddrinfo
            with self.assertRaisesRegex(RuntimeError, "Offline metric assets"):
                Protocol.load("protocol.py", offline=True, cache="cache/metrics")
            self.assertIs(socket.getaddrinfo, original)

    def testdenominator(self):
        raw, status = pixels(self.origin, self.edited, self.mask)
        self.assertEqual(raw["mse"], .5)
        self.assertAlmostEqual(raw["psnr"], 3.0102999566)
        self.assertEqual(status["psnr"], "finite")

    def testemptyandidentity(self):
        empty, status = pixels(self.origin, self.origin, Image.new("L", (4, 4), 255))
        self.assertIsNone(empty["mse"])
        self.assertEqual(status["psnr"], "empty protected region")
        identity, status = pixels(self.origin, self.origin, self.mask)
        self.assertEqual(identity["mse"], 0)
        self.assertIsNone(identity["psnr"])
        self.assertEqual(status["psnr"], "positive infinity")
        json.dumps({"raw": identity, "status": status}, allow_nan=False)

    def testmissingmask(self):
        raw, status = pixels(self.origin, self.edited, None)
        self.assertIsNone(raw["mse"])
        self.assertEqual(status["psnr"], "missing mask")

    def testbinary(self):
        with self.assertRaises(ValueError):
            pixels(self.origin, self.edited, Image.new("L", (4, 4), 128))
        with self.assertRaises(ValueError):
            pixels(self.origin, Image.new("RGB", (5, 4)), self.mask)

    def testprotocolrouting(self):
        calculator = Calculator()
        report = Protocol(calculator).evaluate(self.origin, self.edited, self.mask, "target")
        self.assertEqual(report["table"]["mse"], 500.)
        self.assertEqual(report["table"]["lpips"], 100.)
        self.assertEqual(report["table"]["ssim"], 90.)
        self.assertEqual(report["table"]["structure"], 20.)
        self.assertEqual(report["table"]["whole"], 25.)
        self.assertEqual(report["table"]["edited"], 22.)
        self.assertIn(("structure", 2), calculator.calls)
        masks = [mask for name, mask in calculator.calls if name == "clip" and mask is not None]
        self.assertEqual(masks[0].shape, (4, 4, 3))
        self.assertEqual(float(masks[0][:2].mean()), 1.)

    def testpixelonly(self):
        result = Protocol.load().evaluate(self.origin, self.edited, self.mask, "target")
        self.assertFalse(result["backend"]["complete"])
        self.assertIsNone(result["raw"]["lpips"])

    def testrejectbackendmismatch(self):
        calculator = Calculator()
        with patch.object(calculator, "calculate_mse", return_value=1.):
            with self.assertRaises(RuntimeError):
                Protocol(calculator).evaluate(self.origin, self.edited, self.mask, "target")

    def testrejectpsnrmismatch(self):
        calculator = Calculator()
        with patch.object(calculator, "calculate_psnr", return_value=30.):
            with self.assertRaisesRegex(RuntimeError, "PSNR"):
                Protocol(calculator).evaluate(self.origin, self.edited, self.mask, "target")

    def testtinyerrorprotocol(self):
        origin = Image.new("RGB", (512, 512))
        values = np.zeros((512, 512, 3), dtype=np.uint8)
        values[0, 0, 0] = 1
        edited = Image.fromarray(values)
        mask = Image.new("L", origin.size, 0)
        calculator = Calculator()
        self.assertGreater(Protocol(calculator).evaluate(origin, edited, mask, "target")["raw"]["psnr"], 100.)
        for result, image in ((0., edited), (1e-13, origin)):
            with self.subTest(mse=result), patch.object(calculator, "calculate_mse", return_value=result):
                with self.assertRaisesRegex(RuntimeError, "masking"):
                    Protocol(calculator).evaluate(origin, image, mask, "target")

    def testexplicitbackend(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser().parse_args(["--input", "summary.json"])
            with self.assertRaises(SystemExit):
                parser().parse_args(["--input", "summary.json", "--pixels", "--protocol", "protocol.py"])
        self.assertTrue(Protocol(Calculator()).metadata["complete"])
        self.assertFalse(Protocol().metadata["complete"])
        for options in ({"network": True}, {"offline": True}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "Pixel diagnostics"):
                Protocol.load(**options)

    def grid(self, count=700, seeds=None):
        seeds = [42, 52, 3047, 3407] if seeds is None else seeds
        items = {str(index): SimpleNamespace(category=str(index % 10)) for index in range(count)}
        rows = [{"key": key, "seed": seed, "image": key + str(seed) + ".png"} for key in items for seed in seeds]
        return {"rows": rows, "seeds": seeds}, items, seeds

    def testbenchmarkgrid(self):
        payload, items, seeds = self.grid()
        result = population(payload, items, seeds, "benchmark")
        self.assertEqual((result["requests"], result["rows"]), (700, 2800))
        self.assertTrue(result["benchmarkcomplete"])
        self.assertEqual(set(result["categories"].values()), {70})

    def testmissingrequestgrid(self):
        payload, items, seeds = self.grid()
        for removed in (1, 4):
            with self.subTest(removed=removed), self.assertRaisesRegex(ValueError, "incomplete"):
                population({**payload, "rows": payload["rows"][:-removed]}, items, seeds, "benchmark")

    def testbenchmarkpopulation(self):
        payload, items, seeds = self.grid(699)
        with self.assertRaisesRegex(ValueError, "700"):
            population(payload, items, seeds, "benchmark")
        payload, items, seeds = self.grid()
        items["0"].category = "unknown"
        with self.assertRaisesRegex(ValueError, "categories"):
            population(payload, items, seeds, "benchmark")

    def testsubsetgrid(self):
        payload, items, seeds = self.grid(2, [42])
        result = population({**payload, "rows": payload["rows"][:1]}, items, seeds, "subset")
        self.assertEqual((result["requests"], result["datasetrequests"]), (1, 2))
        self.assertFalse(result["benchmarkcomplete"])

    def testinvalidgridseed(self):
        payload, items, seeds = self.grid(2, [42, 52])
        for value in (True, 42., "42", None, 7):
            changed = deepcopy(payload)
            changed["rows"][0]["seed"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "integer seed"):
                population(changed, items, seeds, "subset")
        for invalid in ([], [42, 42], [True], [2 ** 64], [-2 ** 63 - 1]):
            with self.subTest(seeds=invalid), self.assertRaisesRegex(ValueError, "unique supported"):
                population(payload, items, invalid, "subset")

    def testinvalidgridrows(self):
        payload, items, seeds = self.grid(2, [42])
        for rows in (None, [], [None], [{"configuration": {}}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                population({"rows": rows}, items, seeds, "subset")
        with self.assertRaisesRegex(ValueError, "unique"):
            population({"rows": payload["rows"] * 2}, items, seeds, "subset")
        changed = deepcopy(payload)
        changed["rows"][1]["image"] = changed["rows"][0]["image"]
        with self.assertRaisesRegex(ValueError, "separate prediction"):
            population(changed, items, seeds, "subset")

    def testdeclaredseeds(self):
        payload, items, seeds = self.grid(2, [42])
        for recorded in ([52], [42, 42], [True], "42"):
            with self.subTest(recorded=recorded), self.assertRaisesRegex(ValueError, "inference summary"):
                population({**payload, "seeds": recorded}, items, seeds, "subset")

    def testpredictionhash(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            root = Path(directory).resolve().relative_to(Path.cwd())
            path = root / "result.png"
            Image.new("RGB", (512, 512)).save(path)
            row = {"image": "result.png", "resultsha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            self.assertEqual(prediction(row, root / "summary.json"), path)
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "inference hash"):
                prediction(row, root / "summary.json")

    def testverifiedinputs(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            root = Path(directory).resolve().relative_to(Path.cwd())
            image = root / "source.png"
            Image.new("RGB", (512, 512)).save(image)
            Image.new("RGB", (512, 512), "white").save(root / "result.png")
            item = Record("a", image, Path("source.png"), "source", "target", "target", mask=(0, 1))
            row = {"image": "result.png", "report": {"source": "source", "target": "target"}}
            self.assertEqual(verified(row, root / "summary.json", item)[0].size, (512, 512))
            for key in ("source", "target", "imagesha256"):
                changed = deepcopy(row)
                changed["report"][key] = "changed"
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "no longer match"):
                    verified(changed, root / "summary.json", item)
            with self.assertRaisesRegex(ValueError, "inference hash"):
                verified({**row, "inputsha256": "changed"}, root / "summary.json", item)

    def testassessmentpreflight(self):
        for failure in ("mapping", "source", "prediction", "shape", "mask", "network"):
            with self.subTest(failure=failure), self.workspace(), tempfile.TemporaryDirectory(dir=".") as directory:
                root = Path(directory).resolve().relative_to(Path.cwd())
                (root / "images").mkdir()
                original, edited = root / "images/source.png", root / "result.png"
                Image.new("RGB", (512, 512)).save(original)
                Image.new("RGB", (512, 512), "white").save(edited)
                record = {"image": "source.png", "source": "source", "target": "target"}
                if failure != "mask":
                    record["mask"] = [0, 1]
                mapping = root / "mapping.json"
                mapping.write_text(json.dumps({"a": record}), encoding="utf-8")
                row = {"key": "a", "seed": 42, "image": "result.png", "inputsha256": hashlib.sha256(original.read_bytes()).hexdigest(), "resultsha256": hashlib.sha256(edited.read_bytes()).hexdigest()}
                mapped = "changed" if failure == "mapping" else hashlib.sha256(mapping.read_bytes()).hexdigest()
                if failure in {"source", "prediction"}:
                    row["inputsha256" if failure == "source" else "resultsha256"] = "changed"
                if failure == "shape":
                    Image.new("RGB", (16, 16)).save(edited)
                    row.pop("resultsha256")
                payload = {"rows": [row], "inputs": {"mapping": {"sha256": mapped}}}
                (root / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
                args = ["assess.py", "--input", str(root / "summary.json"), "--dataset", str(root), "--mapping", "mapping.json", "--images", "images", "--pixels", "--scope", "subset", "--seeds", "42", "--output", str(root / "assessment.json")]
                if failure == "network":
                    args.append("--network")
                expected = {"mapping": "annotation mapping", "source": "inference hash", "prediction": "inference hash", "shape": "512", "mask": "annotation mask", "network": "Pixel diagnostics"}[failure]
                with patch("sys.argv", args), patch("assess.Protocol.load") as load, self.assertRaisesRegex(ValueError, expected):
                    main()
                load.assert_not_called()
                self.assertFalse((root / "assessment.json").exists())

    def testimagegroupedmean(self):
        rows = [{"key": "a", "raw": {"mse": 1.}}, {"key": "a", "raw": {"mse": 3.}}, {"key": "b", "raw": {"mse": 10.}}]
        self.assertEqual(grouped(rows, "raw")["mse"], {"mean": 6., "count": 2})

    def testseedfirstmean(self):
        rows = [{"key": key, "seed": seed, "raw": {"mse": value}} for seed, values in ((42, (0., 2.)), (52, (10., None)), (3047, (20., None)), (3407, (30., None))) for key, value in zip(("a", "b"), values)]
        self.assertEqual(grouped(rows, "raw")["mse"], {"mean": 15.25, "count": 2})
        self.assertEqual([grouped([row for row in rows if row["seed"] == seed], "raw")["mse"]["mean"] for seed in (42, 52, 3047, 3407)], [1., 10., 20., 30.])

    def testundefinedseedmean(self):
        rows = [{"key": "a", "seed": 42, "raw": {"mse": 2.}}, {"key": "a", "seed": 52, "raw": {"mse": None}}]
        self.assertIsNone(grouped(rows, "raw")["mse"]["mean"])

    def testinfiniteaggregation(self):
        rows = [{"key": "a", "raw": {"psnr": None}, "status": {"psnr": "positive infinity"}}, {"key": "a", "raw": {"psnr": 30.}}, {"key": "b", "raw": {"psnr": 40.}}]
        result = grouped(rows, "raw")["psnr"]
        self.assertIsNone(result["mean"])
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["infinite"], 1)

    def testassessmentcli(self):
        with self.workspace(), tempfile.TemporaryDirectory(dir=".") as directory:
            root = Path(directory).resolve().relative_to(Path.cwd())
            (root / "images").mkdir()
            Image.new("RGB", (512, 512)).save(root / "images" / "source.png")
            Image.new("RGB", (512, 512), (32, 32, 32)).save(root / "result.png")
            mapping = {"A": {"image": "source.png", "source": "source", "target": "target", "mask": [0, 512]}}
            (root / "mapping.json").write_text(json.dumps(mapping), encoding="utf-8")
            (root / "summary.json").write_text(json.dumps({"rows": [{"key": "a", "seed": 1, "image": "result.png"}]}), encoding="utf-8")
            args = ["assess.py", "--input", str(root / "summary.json"), "--dataset", str(root), "--mapping", "mapping.json", "--images", "images", "--pixels", "--scope", "subset", "--seeds", "1", "--output", str(root / "assessment.json")]
            with patch("sys.argv", args), patch("builtins.print"):
                main()
            result = json.loads((root / "assessment.json").read_text(encoding="utf-8"))
            self.assertEqual(result["raw"]["mse"]["count"], 1)
            self.assertFalse(result["backend"]["complete"])
            self.assertFalse(result["population"]["benchmarkcomplete"])
            self.assertEqual(result["protocol"], "PIE pixel diagnostics")
            self.assertEqual(result["byseed"]["1"]["raw"], result["raw"])
            self.assertIsNone(result["rows"][0]["raw"]["lpips"])

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
