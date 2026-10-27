# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import argparse
import contextlib
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from data import Record, records
from dataset import EditDataset, annotation, training
from engine.corpus import asset, captions, convert, entries, image, normalize, partition, profile, rows
from evaluate import datasetargs, items, main as evaluatemain, metrics, paired
from prepare import main, parser
from split import create, main as splitmain, official, select

class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.origin = Path.cwd()
        self.temporary = tempfile.TemporaryDirectory()
        os.chdir(self.temporary.name)
        Path("raw").mkdir()
        Image.new("RGB", (16, 16), (10, 20, 30)).save("raw/source.png")
        Image.new("RGB", (16, 16), (40, 50, 60)).save("raw/target.png")
        Image.new("L", (16, 16), 255).save("raw/mask.png")

    def tearDown(self):
        os.chdir(self.origin)
        self.temporary.cleanup()

    def dump(self, value, path="raw/annotations.json"):
        Path(path).write_text(json.dumps(value), encoding="utf-8")
        return path

    def emu(self, **values):
        return {"image": "source.png", "input_caption": "A blue shirt.", "output_caption": "A striped shirt.", "instruction": "Add stripes.", "task": "Texture", "split": "dev", "idx": 0, **values}

    def testprofiles(self):
        for value, expected in (("EMU_EDIT", "emu"), ("PIE-BENCH", "pie"), ("MagicBrush", "magicbrush"), ("ImgEdit-Bench", "imgedit"), ("GEditBench", "gedit")):
            self.assertEqual(profile(value), expected)
        with self.assertRaises(ValueError):
            profile("geditv2")

    def testpartitions(self):
        for value, expected in (("DEV", "val"), ("validation", "val"), ("TRAINING", "train"), ("testing", "test"), (None, "")):
            self.assertEqual(partition(value), expected)
        with self.assertRaises(ValueError):
            partition("holdout")

    def testemu(self):
        result = normalize("EMU", "row", self.emu())
        self.assertEqual((result["key"], result["partition"], result["category"]), ("val-0", "val", "texture"))
        self.assertEqual(result["target"], "A striped shirt.")
        self.assertIsNone(result["edited"])

    def testemukeys(self):
        first = normalize("emu", "row", self.emu(hash="EDIT_SOURCE"))
        second = normalize("emu", "row", self.emu(split="test", hash="OTHER_SOURCE"))
        self.assertNotEqual(first["key"], second["key"])
        self.assertEqual(first["group"], second["group"])
        self.assertNotEqual(normalize("emu", "row", self.emu())["key"], normalize("emu", "row", self.emu(split="test"))["key"])

    def testemutrain(self):
        for value in ("train", ""):
            with self.subTest(partition=value), self.assertRaises(ValueError):
                normalize("emu", "row", self.emu(split=value))

    def teststoragetags(self):
        for tag in ("val", "validation", "dev"):
            result = normalize("emu", "row", self.emu(storage_split=tag))
            self.assertEqual(result["partition"], "val")
        for tag in ("train", "test"):
            with self.subTest(tag=tag), self.assertRaisesRegex(ValueError, "disagree"):
                normalize("emu", "row", self.emu(storage_split=tag))
        result = normalize("emu", "row", self.emu(split="", storage_split="validation"))
        self.assertEqual(result["partition"], "val")

    def testmagic(self):
        row = {"img_id": "SESSION", "turn_index": 2, "source_img": "source.png", "target_img": "target.png", "instruction": "Add stripes.", "storage_split": "dev"}
        override = {"session-2": {"source": "A shirt.", "target": "A striped shirt."}}
        result = normalize("magicbrush", "row", row, override)
        self.assertEqual((result["key"], result["group"], result["partition"], result["captionorigin"]), ("session-2", "session", "val", "sidecar"))
        self.assertEqual(result["edited"], "target.png")

    def testmagicturn(self):
        for value in (0, -1, True, "1", 1.5):
            row = {"img_id": "a", "turn_index": value}
            with self.subTest(turn=value), self.assertRaises(ValueError):
                normalize("magicbrush", "row", row)

    def testmagicpartition(self):
        with self.assertRaises(ValueError):
            normalize("magicbrush", "row", {"source": "A.", "target": "B."})

    def testimgedit(self):
        row = {"id": "source.png", "prompt": "Change the background.", "edit_type": "BACKGROUND"}
        with self.assertRaisesRegex(ValueError, "captions"):
            normalize("imgedit", "1082", row)
        result = normalize("imgedit", "1082", row, {"1082": {"source": "A person outdoors.", "target": "A person indoors."}})
        self.assertEqual((result["image"], result["category"], result["partition"]), ("source.png", "background", "test"))
        self.assertEqual(result["instruction"], row["prompt"])

    def testgedit(self):
        row = {"key": "A", "input_image": "source.png", "instruction": "Change the sky.", "task_type": "Background", "instruction_language": "English", "storage_split": "train"}
        result = normalize("gedit", "row", row, {"a": {"source": "A cloudy sky.", "target": "A blue sky."}})
        self.assertEqual((result["key"], result["partition"], result["category"]), ("a", "test", "background"))
        self.assertIsNone(normalize("gedit", "row", {**row, "instruction_language": "cn"}))

    def testbenchmarktraining(self):
        for kind in ("imgedit", "gedit"):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                normalize(kind, "a", {"source": "A.", "target": "B."}, split="train")

    def testpartitionoverride(self):
        with self.assertRaisesRegex(ValueError, "override"):
            normalize("emu", "a", self.emu(split="test"), split="val")
        self.assertEqual(normalize("emu", "a", self.emu(), split="val")["partition"], "val")
        with self.assertRaises(ValueError):
            normalize("gedit", "a", {"source": "A.", "target": "B.", "official_split": "train"})

    def testmagicsession(self):
        for value in (None, "", True):
            row = {"img_id": value, "turn_index": 1}
            with self.subTest(session=value), self.assertRaises(ValueError):
                normalize("magicbrush", "a", row)

    def testfieldcase(self):
        result = normalize("EMU", "a", {"IMAGE": "source.png", "INPUT_CAPTION": "Blue.", "OUTPUT_CAPTION": "Striped.", "SPLIT": "DEV"})
        self.assertEqual(result["source"], "Blue.")
        with self.assertRaises(ValueError):
            normalize("emu", "a", {**self.emu(), "INPUT_CAPTION": "Conflict."})

    def testcaptiontypes(self):
        for value in (None, "", 3, ["A shirt."]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize("emu", "a", self.emu(input_caption=value))

    def testmultisource(self):
        row = {"source": "A.", "target": "B.", "input_images": ["source.png", "target.png"]}
        with self.assertRaises(ValueError):
            normalize("imgedit", "a", row)
        row["input_images"] = ["source.png"]
        self.assertEqual(normalize("imgedit", "a", row)["image"], "source.png")

    def testmultiturn(self):
        with self.assertRaises(ValueError):
            list(entries([{"data": [{"prompt": "Edit."}]}]))
        with self.assertRaises(ValueError):
            normalize("imgedit", "a", {"source": "A.", "target": "B.", "input_images": ["source.png"], "DATA": [{"prompt": "Edit."}]})

    def testentries(self):
        self.assertEqual(list(entries({"records": [{"key": "a"}]})), [("0", {"key": "a"})])
        for value in (1, {"a": "text"}, [None]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                list(entries(value))

    def testsidecar(self):
        path = self.dump({"A": {"SOURCE": "Blue.", "TARGET": "Striped."}}, "raw/captions.json")
        self.assertEqual(captions(path)["a"]["target"], "Striped.")
        self.dump([{"key": "A"}, {"key": "a"}], path)
        with self.assertRaises(ValueError):
            captions(path)

    def testjsonl(self):
        Path("raw/annotations.jsonl").write_text("\n" + json.dumps(self.emu()) + "\n", encoding="utf-8")
        self.assertEqual(len(list(rows("RAW/ANNOTATIONS.JSONL"))), 1)
        Path("raw/annotations.jsonl").write_text("null\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            list(rows("raw/annotations.jsonl"))

    def testparquet(self):
        Path("raw/annotations.parquet").touch()
        batch = types.SimpleNamespace(to_pylist=lambda: [self.emu()])
        reader = types.SimpleNamespace(iter_batches=lambda batch_size: [batch])
        module = types.ModuleType("pyarrow.parquet")
        module.ParquetFile = lambda path: reader
        with patch.dict("sys.modules", {"pyarrow.parquet": module}):
            self.assertEqual(list(rows("raw/annotations.parquet"))[0][1]["idx"], 0)

    def testarrow(self):
        Path("raw/arrow").mkdir()
        module = types.ModuleType("datasets")

        class DatasetDict(dict):
            pass

        module.DatasetDict = DatasetDict
        module.load_from_disk = lambda path: DatasetDict(validation=[self.emu()], test=[self.emu(split="test")])
        with patch.dict("sys.modules", {"datasets": module}):
            result = list(rows("raw/arrow"))
        self.assertEqual([value[1]["storage_split"] for value in result], ["validation", "test"])

    def testformats(self):
        Path("raw/annotations.txt").touch()
        with self.assertRaises(ValueError):
            list(rows("raw/annotations.txt"))
        with self.assertRaises(FileNotFoundError):
            list(rows("raw/missing.json"))

    def testembedded(self):
        buffer = io.BytesIO()
        Image.new("RGB", (16, 16), "red").save(buffer, format="PNG")
        embedded = {"bytes": buffer.getvalue(), "path": None}
        self.assertEqual(image(embedded, Path("raw")).getpixel((0, 0)), (255, 0, 0))
        result, size = asset(embedded, Path("raw"), Path("stage"))
        self.assertEqual(size, (16, 16))
        self.assertEqual(result, asset(image(embedded, Path("raw")), Path("raw"), Path("stage"))[0])

    def testimagepaths(self):
        # Reject absolute and escaping path fixtures.
        for value in ("D:/source.png", "../source.png", "/source.png", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                image(value, Path("raw"))
        self.assertEqual(image("SOURCE.PNG", Path("raw")).size, (16, 16))

    def testconvert(self):
        path = self.dump({"a": self.emu()})
        result = convert("emu", path, "raw", "prepared")
        self.assertEqual(result["images"], "./images")
        selected = records("PREPARED", "MAPPING.JSON", "IMAGES")
        self.assertEqual((len(selected), selected[0].benchmark, selected[0].partition), (1, "emu", "val"))
        self.assertIsNone(annotation(selected[0]))
        self.assertFalse(Path("prepared/masks").exists())
        self.assertEqual({path.name for path in Path(".").iterdir()}, {"raw", "prepared"})

    def testmagicpair(self):
        path = self.dump({"a": {"img_id": "session", "turn_index": 1, "source_img": "source.png", "target_img": "target.png", "mask_img": "mask.png", "instruction": "Add stripes."}})
        sidecar = self.dump({"session-1": {"source": "Blue.", "target": "Striped."}}, "raw/captions.json")
        convert("magicbrush", path, "raw", "prepared", sidecar, "train")
        selected = records("prepared", "mapping.json", "images")
        record = selected[0]
        self.assertEqual(record.group, "session")
        self.assertEqual(annotation(record).size, (16, 16))
        self.assertEqual(annotation(record).getextrema(), (255, 255))
        sample = EditDataset(selected, 16, paired=True)[0]
        self.assertTrue(sample["hasmask"] and sample["haspair"])
        self.assertEqual(sample["edited"].shape, (3, 16, 16))
        target = torch.tensor([40, 50, 60], dtype=torch.float32)[None, :, None, None].expand(1, 3, 16, 16) / 255
        self.assertEqual(paired(target, record), {"pairmse": 0., "pairl1": 0.})

    def testpie(self):
        path = self.dump({"a": {"image_path": "source.png", "original_prompt": "Blue.", "editing_prompt": "Striped.", "mask": [0, 16], "masksize": [16, 16]}})
        convert("pie", path, "raw", "prepared")
        record = records("prepared", "mapping.json", "images")[0]
        self.assertEqual(int(np.asarray(annotation(record)).sum()), 16 * 255)
        self.assertEqual(record.benchmark, "pie")

    def testinvalidrle(self):
        for mask, shape in (([1], [16, 16]), ([True, 2], [16, 16]), ([257, 1], [16, 16]), ([0, 1], [0, 16])):
            row = {"image": "source.png", "source": "Blue.", "target": "Striped.", "mask": mask, "masksize": shape}
            with self.subTest(mask=mask, shape=shape), self.assertRaises(ValueError):
                convert("pie", self.dump({"a": row}), "raw", "prepared")
            self.assertFalse(Path("prepared").exists())

    def testmaskgeometry(self):
        Image.new("L", (8, 8), 255).save("raw/mask.png")
        path = self.dump({"a": {**self.emu(), "maskpath": "mask.png"}})
        with self.assertRaises(ValueError):
            convert("emu", path, "raw", "prepared")
        self.assertFalse(Path("prepared").exists())

    def testorientation(self):
        pixels = Image.new("RGB", (16, 16))
        metadata = pixels.getexif()
        metadata[274] = 6
        pixels.save("raw/oriented.jpg", exif=metadata)
        path = self.dump({"a": self.emu(image="oriented.jpg", maskpath="mask.png")})
        convert("emu", path, "raw", "prepared")
        record = records("prepared", "mapping.json", "images")[0]
        with Image.open(record.image) as handle:
            self.assertEqual(handle.getexif().get(274, 1), 1)
        self.assertEqual(annotation(record).size, (16, 16))

    def testduplicates(self):
        path = self.dump([self.emu(key="A"), self.emu(key="a")])
        with self.assertRaises(ValueError):
            convert("emu", path, "raw", "prepared")
        self.assertFalse(Path("prepared").exists())

    def testprotectedoutput(self):
        path = self.dump({"a": self.emu()})
        for value in ("raw/prepared", "raw", "../prepared"):
            with self.subTest(output=value), self.assertRaises(ValueError):
                convert("emu", path, "raw", value)
        Path("prepared").mkdir()
        with self.assertRaises(FileExistsError):
            convert("emu", path, "raw", "prepared")

    def testsourceoutput(self):
        os.chdir(self.origin)
        try:
            with self.assertRaises(ValueError):
                convert("emu", "missing.json", "tests", "engine/prepared")
        finally:
            os.chdir(self.temporary.name)

    def testtransaction(self):
        path = self.dump({"a": self.emu(input_caption=None)})
        with self.assertRaises(ValueError):
            convert("emu", path, "raw", "prepared")
        self.assertEqual({path.name for path in Path(".").iterdir()}, {"raw"})

    def testnonenglish(self):
        path = self.dump({"a": {"key": "a", "instruction_language": "cn"}})
        with self.assertRaisesRegex(ValueError, "No English"):
            convert("gedit", path, "raw", "prepared")

    def testofficialemptytrain(self):
        path = self.dump({"a": self.emu()})
        convert("emu", path, "raw", "prepared")
        selected = records("prepared", "mapping.json", "images")
        payload = official(selected)
        self.assertEqual(payload["subsets"]["train"], [])
        Path("split.json").write_text(json.dumps(payload), encoding="utf-8")
        self.assertEqual(select(selected, "split.json", "VAL"), selected)
        with self.assertRaises(ValueError):
            training("prepared", "split.json", "train", "mapping.json", "images")

    def testleakage(self):
        first = Record("a", Path("raw/source.png"), Path("source.png"), "A.", "B.", "Edit.", partition="val")
        for second in (replace(first, key="b", partition="test"), replace(first, key="b", image=Path("raw/target.png"), group="same", partition="test")):
            with self.subTest(second=second.key), self.assertRaises(ValueError):
                official([replace(first, group="same"), second])

    def testpixelduplicates(self):
        from PIL.PngImagePlugin import PngInfo
        metadata = PngInfo()
        metadata.add_text("note", "reencoded source")
        with Image.open("raw/source.png") as image:
            image.save("raw/copy.png", pnginfo=metadata)
        self.assertNotEqual(Path("raw/source.png").read_bytes(), Path("raw/copy.png").read_bytes())
        first = Record("a", Path("raw/source.png"), Path("source.png"), "A.", "B.", "Edit.", partition="val")
        second = replace(first, key="b", image=Path("raw/copy.png"), partition="test")
        with self.assertRaisesRegex(ValueError, "leakage"):
            official([first, second])

    def testpairedleakage(self):
        first = Record("a", Path("raw/source.png"), Path("source.png"), "A.", "B.", "Edit.", edited=Path("raw/target.png"), partition="val")
        second = replace(first, key="b", image=Path("raw/target.png"), edited=None, partition="test")
        with self.assertRaisesRegex(ValueError, "leakage"):
            official([first, second])

    def testallmanifest(self):
        item = Record("a", Path("raw/source.png"), Path("source.png"), "A.", "B.", "Edit.", partition="val")
        self.dump(official([item]), "split.json")
        with self.assertRaisesRegex(ValueError, "manifest"):
            select([item], "split.json", "all")

    def testpairedmutation(self):
        item = Record("a", Path("raw/source.png"), Path("source.png"), "A.", "B.", "Edit.", edited=Path("raw/target.png"), partition="val")
        self.dump(official([item]), "split.json")
        Image.new("RGB", (16, 16), (100, 110, 120)).save("raw/target.png")
        with self.assertRaises(ValueError):
            select([item], "split.json", "val")

    def testpixelbindings(self):
        item = Record("a", Path("raw/source.png"), Path("source.png"), "A.", "B.", "Edit.", partition="val")
        payload = official([item])
        payload.pop("pixels")
        self.dump(payload, "split.json")
        with self.assertRaisesRegex(ValueError, "regenerate"):
            select([item], "split.json", "val")

    def testtrialinputmutation(self):
        from engine.protocol import signature
        from engine.reproduce import digest
        from trial import main as trialmain
        Path("data/images").mkdir(parents=True)
        mapping = {}
        for index in range(4):
            Image.new("RGB", (16, 16), (index, 30, 40)).save(f"data/images/{index}.png")
            mapping[str(index)] = {"image": f"{index}.png", "source": "source", "target": "target long"}
        self.dump(mapping, "data/map.json")
        self.dump(create(records("data", "map.json", "images")), "split.json")
        arguments = ["trial.py", "--dataset", "data", "--mapping", "map.json", "--images", "images", "--manifest", "split.json", "--split", "val", "--suite", "core", "--size", "16", "--clip", "clip", "--batch", "1", "--output", "trial"]

        def execute(current, resources):
            result = {"controlsignature": signature(current), "inputs": {"mapping": {"sha256": digest("data/map.json")}, "manifest": {"sha256": digest("split.json")}}, "summary": {}, "cost": {}, "controller": {}, "rows": []}
            with Path("data/map.json").open("a", encoding="utf-8") as handle:
                handle.write("\n")
            return result

        with patch("sys.argv", arguments), patch("trial.load", return_value=(types.SimpleNamespace(metadata={"weights": "fixture"}), None)), patch("trial.execute", side_effect=execute) as runner:
            with self.assertRaisesRegex(ValueError, "inputs changed"):
                trialmain()
        self.assertEqual(runner.call_count, 1)
        self.assertFalse(Path("trial/recipe.json").exists())

    def testbenchmarktrials(self):
        from engine.protocol import signature
        from engine.reproduce import digest
        from split import identities
        from trial import main as trialmain
        Path("data/images").mkdir(parents=True)
        mapping = {}
        for index in range(3):
            Image.new("RGB", (16, 16), (index, 30, 40)).save(f"data/images/{index}.png")
            mapping[str(index)] = {"image": f"{index}.png", "source": "source", "target": "target long"}
        self.dump(mapping, "data/map.json")
        arguments = ["trial.py", "--dataset", "data", "--mapping", "map.json", "--images", "images", "--split", "all", "--suite", "core", "--size", "16", "--clip", "clip", "--batch", "1", "--output", "trial"]
        observed = []

        def execute(current, resources):
            chosen = items(current)
            observed.append([item.key for item in chosen])
            rows = [{"key": key, "inputsha256": value} for key, value in identities(chosen).items()]
            return {"controlsignature": signature(current), "inputs": {"mapping": {"sha256": digest("data/map.json")}}, "summary": {}, "cost": {}, "controller": {}, "rows": rows}

        with patch("sys.argv", arguments), patch("trial.load", return_value=(types.SimpleNamespace(metadata={"weights": "fixture"}), None)), patch("trial.execute", side_effect=execute):
            trialmain()
        payload = json.loads(Path("trial/summary.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["split"], "all")
        self.assertEqual(len(observed), 6)
        self.assertTrue(all(keys == ["0", "1", "2"] for keys in observed))
        self.assertTrue(all(arm["rows"] == 3 for arm in payload["variants"].values()))
        self.assertFalse(Path("trial/recipe.json").exists())

    def testofficialtag(self):
        item = Record("a", Path("raw/source.png"), Path("source.png"), "A.", "B.", "Edit.")
        with self.assertRaises(ValueError):
            official([item])

    def testofficialreplay(self):
        from engine.judge import ClipJudge
        from session import Clip, Processor, editor
        from trial import main as trialmain
        self.dump([self.emu(input_caption="source", output_caption="target long"), self.emu(image="target.png", input_caption="source", output_caption="target long", split="test", idx=1)])
        with patch("sys.argv", ["prepare.py", "--kind", "emu", "--input", "raw/annotations.json", "--dataset", "raw", "--output", "prepared"]), contextlib.redirect_stdout(io.StringIO()):
            main()
        with patch("sys.argv", ["split.py", "--dataset", "prepared", "--mapping", "mapping.json", "--images", "images", "--protocol", "official", "--output", "split.json"]), contextlib.redirect_stdout(io.StringIO()):
            splitmain()
        subsets = json.loads(Path("split.json").read_text(encoding="utf-8"))["subsets"]
        self.assertEqual(subsets, {"train": [], "val": ["val-0"], "test": ["test-1"]})
        for width in (3, 4):
            backend, judge = editor(), ClipJudge(Clip(), Processor())
            backend.metadata["weights"], judge.metadata["weights"] = "backbone", "control"
            shared = ["trial.py", "--dataset", "prepared", "--mapping", "mapping.json", "--images", "images", "--manifest", "split.json", "--suite", "core", "--size", "16", "--clip", "clip", "--batch", str(width), "--seed", "42", "--seeds", "42", "52", "3047", "3407", "--maxsamples", "64", "--maxcalls", "64"]
            validation, testing = f"val{width}", f"test{width}"
            for subset, output in (("val", validation), ("test", testing)):
                arguments = [*shared, "--split", subset, "--output", output]
                if subset == "test":
                    arguments.extend(["--recipe", validation + "/recipe.json"])
                home = Path.cwd()
                try:
                    if subset == "test":
                        server = home / f"server{width}"
                        server.mkdir()
                        shutil.copytree("prepared", server / "prepared")
                        shutil.copytree(validation, server / validation)
                        shutil.copyfile("split.json", server / "split.json")
                        shutil.copytree(Path(__file__).resolve().parent.parent, server / "RefUnmix", ignore=shutil.ignore_patterns(".git", "__pycache__"))
                        for source in (server / "RefUnmix").rglob("*.py"):
                            content = source.read_bytes().replace(b"\r\n", b"\n")
                            source.write_bytes(content.replace(b"\n", b"\r\n") if width == 3 else content)
                        os.chdir(server)
                        script = "import sys\nfrom pathlib import Path\nsys.path[:0] = [str(Path('RefUnmix').resolve()), str(Path('RefUnmix/tests').resolve())]\nfrom unittest.mock import patch\nfrom session import Clip, Processor, editor\nfrom engine.judge import ClipJudge\nfrom trial import main\nbackend, judge = editor(), ClipJudge(Clip(), Processor())\nbackend.metadata['weights'], judge.metadata['weights'] = 'backbone', 'control'\nwith patch('trial.load', return_value=(backend, judge)):\n main()\n"
                        completed = subprocess.run([sys.executable, "-B", "-c", script, *arguments[1:]], capture_output=True, text=True, timeout=90)
                        self.assertEqual(completed.returncode, 0, completed.stderr[-2500:])
                    else:
                        with patch("sys.argv", arguments), patch("trial.load", return_value=(backend, judge)), contextlib.redirect_stdout(io.StringIO()):
                            trialmain()
                    suite = json.loads(Path(output, "summary.json").read_text(encoding="utf-8"))
                    self.assertEqual(len(suite["variants"]), 6)
                    for label, variant in suite["variants"].items():
                        self.assertNotIn("\\", variant["path"])
                        self.assertTrue(Path(variant["path"]).is_file())
                        payload = json.loads(Path(output, label, "summary.json").read_text(encoding="utf-8"))
                        self.assertEqual({row["key"] for row in payload["rows"]}, set(subsets[subset]))
                        self.assertEqual({row["seed"] for row in payload["rows"]}, {42, 52, 3047, 3407})
                        self.assertEqual(len(payload["rows"]), 4)
                        self.assertTrue(payload["execution"]["accounting"]["matched"])
                        self.assertTrue(all(row["report"]["stats"]["samples"] <= 64 for row in payload["rows"]))
                    if subset == "test":
                        self.assertFalse(Path(output, "recipe.json").exists())
                finally:
                    os.chdir(home)
            arguments = [*shared, "--batch", "4" if width == 3 else "3", "--split", "test", "--recipe", validation + "/recipe.json", "--output", "mismatched"]
            with patch("sys.argv", arguments), patch("trial.load") as load:
                with self.assertRaisesRegex(ValueError, "unchanged validation"):
                    trialmain()
            load.assert_not_called()

    def testreshuffle(self):
        item = Record("a", Path("raw/source.png"), Path("source.png"), "A.", "B.", "Edit.", partition="test")
        with self.assertRaisesRegex(ValueError, "reshuffled"):
            create([item])

    def testmanifesttags(self):
        item = Record("a", Path("raw/source.png"), Path("source.png"), "A.", "B.", "Edit.", partition="val")
        payload = official([item])
        payload["subsets"] = {"train": ["a"], "val": [], "test": []}
        Path("split.json").write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "reassigns"):
            select([item], "split.json", "train")
        with self.assertRaisesRegex(ValueError, "require"):
            select([replace(item, partition="")], "split.json", "train")

    def testcategories(self):
        path = self.dump({"a": self.emu(task="BACKGROUND"), "b": self.emu(idx=1, task="local")})
        convert("emu", path, "raw", "prepared")
        selected = records("prepared", "mapping.json", "images")
        Path("split.json").write_text(json.dumps(official(selected)), encoding="utf-8")
        args = datasetargs(argparse.ArgumentParser()).parse_args(["--dataset", "prepared", "--mapping", "mapping.json", "--images", "images", "--manifest", "split.json", "--split", "VAL", "--categories", "BACKGROUND"])
        self.assertEqual([item.category for item in items(args)], ["background"])
        args.categories = ["missing"]
        with self.assertRaises(ValueError):
            items(args)

    def testmissingmask(self):
        result = metrics(torch.ones(1, 3, 16, 16), torch.zeros(1, 3, 16, 16))
        self.assertIsNone(result["protectedmse"])
        self.assertIsNone(result["editedmse"])
        item = Record("a", Path("raw/source.png"), Path("source.png"), "A.", "B.", "Edit.")
        self.assertEqual(paired(torch.zeros(1, 3, 16, 16), item), {})

    def testtemplate(self):
        self.assertEqual(parser().parse_args(["--kind", "EMU_EDIT", "--template"]).kind, "emu")
        output = io.StringIO()
        with patch("sys.argv", ["prepare.py", "--kind", "imgedit", "--template"]), contextlib.redirect_stdout(output):
            main()
        self.assertIn("caption_sidecar", json.loads(output.getvalue()))

    def testcustomtrain(self):
        selected = []
        for index in range(4):
            path = Path("raw") / f"{index}.png"
            Image.new("RGB", (8, 8), (index, 20, 30)).save(path)
            selected.append(Record(str(index), path, Path(path.name), "A.", "B.", "Edit.", partition="train"))
        payload = create(selected)
        Path("split.json").write_text(json.dumps(payload), encoding="utf-8")
        self.assertEqual(len(select(selected, "split.json", "val")), 1)

    def testassessmentprofile(self):
        from assess import main as assessmain
        convert("emu", self.dump({"a": self.emu()}), "raw", "prepared")
        self.dump({"rows": [{"key": "val-0", "seed": 1, "image": "result.png"}]}, "raw/summary.json")
        args = ["assess.py", "--input", "raw/summary.json", "--dataset", "prepared", "--mapping", "mapping.json", "--images", "images"]
        with patch("sys.argv", args), patch("assess.Protocol.load") as backend:
            with self.assertRaisesRegex(ValueError, "PIE protocol"):
                assessmain()
            backend.assert_not_called()

    def testcliflow(self):
        native = {"a": {"img_id": "session", "turn_index": 1, "source_img": "source.png", "target_img": "target.png", "mask_img": "mask.png", "instruction": "Add stripes.", "category": "Texture", "split": "dev"}}
        self.dump(native)
        self.dump({"session-1": {"source": "Blue.", "target": "Striped."}}, "raw/captions.json")
        arguments = ["prepare.py", "--kind", "MAGICBRUSH", "--input", "raw/annotations.json", "--dataset", "raw", "--output", "prepared", "--captions", "raw/captions.json", "--partition", "val"]
        with patch("sys.argv", arguments), contextlib.redirect_stdout(io.StringIO()):
            main()
        shared = ["--dataset", "prepared", "--mapping", "mapping.json", "--images", "images"]
        with patch("sys.argv", ["split.py", *shared, "--protocol", "OFFICIAL", "--output", "split.json"]), contextlib.redirect_stdout(io.StringIO()):
            splitmain()
        origin = torch.tensor([10, 20, 30], dtype=torch.float32)[None, :, None, None].expand(1, 3, 16, 16) / 255
        edited = torch.tensor([40, 50, 60], dtype=torch.float32)[None, :, None, None].expand_as(origin) / 255
        result = (edited, {"source": "Blue.", "target": "Striped."}, types.SimpleNamespace(origin=origin))
        arguments = ["evaluate.py", *shared, "--manifest", "split.json", "--split", "VAL", "--categories", "TEXTURE", "--method", "base", "--batch", "1", "--size", "16", "--seeds", "17", "42", "--output", "evaluation"]
        with patch("sys.argv", arguments), patch("evaluate.load", return_value=(types.SimpleNamespace(device="cpu"), None)), patch("evaluate.run", return_value=result) as inference, contextlib.redirect_stdout(io.StringIO()):
            evaluatemain()
        payload = json.loads(Path("evaluation/summary.json").read_text())
        self.assertEqual(len(payload["rows"]), 2)
        self.assertEqual(payload["summary"]["pairmse"], {"mean": 0., "count": 1})
        self.assertEqual(payload["categories"]["texture"]["pairl1"]["mean"], 0.)
        self.assertTrue(all(row["benchmark"] == "magicbrush" and row["partition"] == "val" and row["captionorigin"] == "sidecar" for row in payload["rows"]))
        self.assertTrue(all(call.args[-1] is None for call in inference.call_args_list))
        self.assertEqual(len(list(Path("evaluation").glob("*.png"))), 2)

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
