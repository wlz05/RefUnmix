# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from dataclasses import replace
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from data import Record, clean, destination, locate, records
from dataset import EditDataset, annotation, loader
from engine.corpus import convert, rows
from engine.format import fit, mask, orient, read, rgb, runs
from engine.serial import loads, read as readjson, write
from evaluate import regions
from run import readimage

class FormatTests(unittest.TestCase):
    def setUp(self):
        self.previous = Path.cwd()
        self.temporary = tempfile.TemporaryDirectory()
        os.chdir(self.temporary.name)
        Path("raw/images").mkdir(parents=True)
        Image.new("RGB", (16, 8), "red").save("raw/images/source.png")

    def tearDown(self):
        os.chdir(self.previous)
        self.temporary.cleanup()

    def record(self, **values):
        return Record("a", Path("raw/images/source.png"), Path("source.png"), "source", "target", "edit", **values)

    def mapping(self, value):
        Path("raw/map.json").write_text(json.dumps(value), encoding="utf-8")

    def oriented(self):
        image = Image.new("RGB", (16, 8), "black")
        image.paste((255, 0, 0), (0, 0, 8, 8))
        image.getexif()[274] = 6
        image.save("raw/images/source.png", exif=image.getexif())
        region = Image.new("L", image.size)
        region.paste(255, (0, 0, 8, 8))
        region.save("raw/mask.png")
        return image, region

    def testorientation(self):
        image, region = self.oriented()
        loaded = readimage("RAW/IMAGES/SOURCE.PNG")
        self.assertEqual(loaded.getexif().get(274), 6)
        output, aligned = orient(loaded, region)
        self.assertEqual(output.size, (8, 16))
        self.assertEqual(output.getexif().get(274, 1), 1)
        self.assertTrue(np.array_equal(np.asarray(output)[..., 0], np.asarray(aligned)))
        self.assertTrue(np.array_equal(np.asarray(output), np.asarray(orient(output)[0])))

    def testtrainingalignment(self):
        self.oriented()
        item = self.record(maskpath=Path("raw/mask.png"), masksize=(8, 16))
        sample = EditDataset([item], 8)[0]
        expected = (sample["image"][0] + 1) / 2
        self.assertTrue(torch.allclose(expected, sample["mask"][0], atol=.13))
        original, fitted = regions(item, read(item.image), 8)
        self.assertTrue(np.array_equal(np.asarray(fitted), sample["mask"][0].numpy() * 255))

    def testalpha(self):
        image = Image.new("RGBA", (2, 1), (255, 0, 0, 0))
        image.putpixel((1, 0), (255, 0, 0, 128))
        self.assertEqual(rgb(image).getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(rgb(image).getpixel((1, 0)), (255, 127, 127))

    def testpalette(self):
        image = Image.new("P", (1, 1))
        image.putpalette([255, 0, 0] + [0] * 765)
        image.info["transparency"] = 0
        self.assertEqual(rgb(image).getpixel((0, 0)), (255, 255, 255))

    def testmodes(self):
        for mode in ("I", "F", "I;16"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                rgb(Image.new(mode, (2, 2)))
        for mode in ("1", "L", "LA", "RGB", "CMYK", "YCbCr"):
            self.assertEqual(rgb(Image.new(mode, (2, 2))).mode, "RGB")

    def testanimated(self):
        Image.new("RGB", (2, 2), "red").save("animated.gif", save_all=True, append_images=[Image.new("RGB", (2, 2), "blue")], duration=100, loop=0)
        with self.assertRaises(ValueError):
            read("animated.gif")

    def testbinary(self):
        image = Image.fromarray(np.array([[0, 1]], dtype=np.uint8))
        self.assertEqual(np.asarray(mask(image)).flatten().tolist(), [0, 255])
        soft = Image.fromarray(np.array([[0, 127, 255]], dtype=np.uint8))
        self.assertTrue(np.array_equal(np.asarray(mask(soft)), np.asarray(soft)))

    def testmaskambiguity(self):
        for image in (Image.new("RGB", (2, 2), "red"), Image.new("RGBA", (2, 2), (0, 0, 0, 0))):
            with self.assertRaises(ValueError):
                mask(image)
        region = Image.new("L", (2, 2))
        region.getexif()[274] = 6
        with self.assertRaises(ValueError):
            mask(region)

    def testinvalidorientation(self):
        image = Image.new("RGB", (2, 2))
        image.getexif()[274] = 9
        with self.assertRaises(ValueError):
            rgb(image)
        with self.assertRaises(ValueError):
            orient(Image.new("RGB", (2, 2)), Image.new("L", (3, 2)))

    def testfit(self):
        for size in ((0, 2), (True, 2), (2.5, 2), (2,), None):
            with self.subTest(size=size), self.assertRaises(ValueError):
                fit(Image.new("RGB", (2, 2)), size)

    def testrle(self):
        self.assertEqual(runs((0, 2, 16, 0), (4, 4)), 16)
        for encoded in ((15, 2), (17, 0), (True, 1), (0, 1.5), (1,)):
            with self.subTest(encoded=encoded), self.assertRaises(ValueError):
                runs(encoded, (4, 4))
        with self.assertRaises(ValueError):
            annotation(self.record(mask=(15, 2), masksize=(4, 4)))
        with self.assertRaises(ValueError):
            annotation(self.record(mask=(), masksize=(4, 4)), 0)

    def testgeometry(self):
        Image.new("L", (4, 4)).save("raw/mask.png")
        item = self.record(maskpath=Path("raw/mask.png"), masksize=(4, 4))
        with self.assertRaises(ValueError):
            EditDataset([item], 8)[0]
        with self.assertRaises(ValueError):
            regions(item, read(item.image), 8)

    def testloadervalidation(self):
        dataset = EditDataset([self.record()], 8)
        for options in ({"seed": True}, {"seed": 2 ** 64}, {"shuffle": 1}, {"workers": -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                loader(dataset, **options)
        self.assertTrue(torch.equal(next(iter(loader(dataset)))["image"], next(iter(loader(dataset)))["image"]))

    def teststrictjson(self):
        for value in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}', '{"x":[-1e999]}'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                loads(value)
        self.assertEqual(loads('\ufeff{"x":1}'), {"x": 1})

    def testpublication(self):
        write("result.json", {"x": 1})
        self.assertEqual(readjson("result.json"), {"x": 1})
        with self.assertRaises(FileExistsError):
            write("result.json", {"x": 2})
        with self.assertRaises(ValueError):
            write("invalid.json", {"x": float("nan")})
        self.assertFalse(Path("invalid.json").exists())
        self.assertFalse(list(Path(".").glob("*.tmp")))

    def testfailedpublication(self):
        with patch("engine.serial.os.link", side_effect=OSError("unsupported")), self.assertRaises(OSError):
            write("result.json", {"x": 1})
        self.assertFalse(list(Path(".").glob("*.tmp")))

    def testinvalidrecords(self):
        for value in ({"a": None}, {"a": {"image": "source.png", "source": "A."}}, {"a": {"image": "source.png", "source": ["A."], "target": "B."}}, {"": {"image": "source.png", "source": "A.", "target": "B."}}):
            self.mapping(value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                records("raw", "map.json", "images")

    def testnumericcategory(self):
        self.mapping({"a": {"image": "source.png", "source": "A.", "target": "B.", "editing_type_id": 2}})
        self.assertEqual(records("raw", "map.json", "images")[0].category, "2")

    def testpaths(self):
        for value in (None, True, 42):
            with self.subTest(value=value), self.assertRaises(ValueError):
                clean(value, "input")
        Path("Mixed").mkdir()
        self.assertEqual(destination("mixed/output.png"), Path("Mixed/output.png"))
        with patch.object(Path, "iterdir", return_value=iter([Path("raw/images/A.png"), Path("raw/images/a.png")])):
            with self.assertRaises(ValueError):
                locate(Path("raw/images"), Path("a.png"))

    def testcsv(self):
        Path("raw/table.csv").write_text('\ufeffimage,source,target,mask,masksize\nsource.png,"A, blue shirt.",Striped.,"[0, 2]","[8, 16]"\n', encoding="utf-8")
        values = list(rows("raw/table.csv"))
        self.assertEqual(values[0][1]["mask"], [0, 2])
        self.assertEqual(values[0][1]["source"], "A, blue shirt.")
        convert("pie", "raw/table.csv", "raw/images", "prepared")
        self.assertEqual(records("prepared", "mapping.json", "images")[0].mask, (0, 2))

    def testcsvinvalid(self):
        for content in ("source,SOURCE\na,b\n", "source,target\na,b,c\n", "source,target\na\n"):
            Path("raw/table.csv").write_text(content, encoding="utf-8")
            with self.subTest(content=content), self.assertRaises(ValueError):
                list(rows("raw/table.csv"))

    def testconverterorientation(self):
        self.oriented()
        self.mapping({"a": {"image": "images/source.png", "source": "Red.", "target": "Blue.", "maskpath": "mask.png"}})
        convert("pie", "raw/map.json", "raw", "prepared")
        item = records("prepared", "mapping.json", "images")[0]
        self.assertEqual(read(item.image).size, (8, 16))
        self.assertTrue(np.array_equal(np.asarray(read(item.image))[..., 0], np.asarray(annotation(item))))

    def testconverterrleorientation(self):
        self.oriented()
        encoded = [value for row in range(8) for value in (row * 16, 8)]
        self.mapping({"a": {"image": "images/source.png", "source": "Red.", "target": "Blue.", "mask": encoded, "masksize": [8, 16]}})
        convert("pie", "raw/map.json", "raw", "prepared")
        item = records("prepared", "mapping.json", "images")[0]
        self.assertIsNone(item.mask)
        self.assertTrue(np.array_equal(np.asarray(read(item.image))[..., 0], np.asarray(annotation(item))))

    def testmaskcoordinates(self):
        for metadata in ({"maskbase": 1}, {"maskorder": "columnmajor"}, {"maskbase": False}):
            self.mapping({"a": {"image": "source.png", "source": "A.", "target": "B.", "mask": [0, 1], **metadata}})
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                records("raw", "map.json", "images")
            with self.assertRaises(ValueError):
                convert("pie", "raw/map.json", "raw/images", "prepared")

    def testliteralbrackets(self):
        self.mapping({"a": {"image": "source.png", "source": "A [label].", "target": "B [label].", "benchmark": "emu"}})
        self.assertEqual(records("raw", "map.json", "images")[0].source, "A [label].")

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu 2026
# Email: wlzchina05@gmail.com
