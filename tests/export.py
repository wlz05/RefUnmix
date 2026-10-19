# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from copy import deepcopy
import hashlib
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from PIL import Image
from engine.attainment import choose
from engine.serial import read, write
from replay import export
from report import template

class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.previous = Path.cwd()
        self.temporary = tempfile.TemporaryDirectory()
        os.chdir(self.temporary.name)
        Path("images").mkdir()
        Image.new("RGB", (16, 16), "red").save("images/raw.png")
        Image.new("RGB", (16, 16), "blue").save("images/edited.png")
        raw = {"key": "raw", "point": None, "scores": {"target": 0.}, "damage": .1, "prior": 0., "penalty": 0., "objective": .1, "taskgain": 0., "complete": True, "feasible": True}
        edited = {**raw, "key": "0", "point": [1., 1., .5], "scores": {"target": .2}, "taskgain": .2}
        artifacts = {"raw": "images/raw.png", "candidate0": "images/edited.png"}
        pool = [SimpleNamespace(**row) for row in (raw, edited)]
        common = {mode: {"selected": selected.key, **detail} for mode in ("attain", "protect") for selected, detail in [choose(pool, 0., mode)]}
        report = {"selected": "raw", "point": None, "gain": 0., "scores": raw["scores"], "rawscores": raw["scores"], "rawdamage": .1, "raw": raw, "candidates": [edited], "commonpool": common, "attainment": {"goals": ["target"]}, "candidatearchive": {"enabled": True}, "artifacts": artifacts, "artifactsha256": {key: hashlib.sha256(Path(path).read_bytes()).hexdigest() for key, path in artifacts.items()}, "config": {"selection": "attain", "gainloss": 0.}}
        self.payload = {"schema": 1, "split": "test", "oracle": False, "settings": {"archive": True}, "inputs": {"mapping": {"path": "map.json", "sha256": "frozen"}}, "summary": {"stale": 1.}, "cost": {"seconds": 1.}, "rows": [{"key": "a", "seed": 7, "image": "result.png", "metrics": {"stale": 1.}, "report": report}]}

    def tearDown(self):
        os.chdir(self.previous)
        self.temporary.cleanup()

    def runexport(self, payload=None, output="output"):
        write("summary.json", self.payload if payload is None else payload)
        return export("summary.json", output)

    def testfrozenchoices(self):
        result = self.runexport()
        attained, protected = read(result["attained"]), read(result["protected"])
        row = attained["rows"][0]
        self.assertEqual(row["report"]["selected"], "0")
        self.assertEqual(row["report"]["point"], [1., 1., .5])
        self.assertEqual(row["report"]["gain"], .2)
        self.assertEqual(row["report"]["replay"]["generatorqueries"], 0)
        self.assertEqual(row["metrics"], {})
        self.assertNotIn("summary", attained)
        self.assertNotIn("cost", attained)
        self.assertEqual(attained["inputs"], self.payload["inputs"])
        self.assertEqual(protected["rows"][0]["report"]["selected"], "raw")
        self.assertEqual(protected["rows"][0]["report"]["config"]["selection"], "protect")
        self.assertEqual(len(template(result["attained"])), 1)
        self.assertEqual(len(template(result["protected"])), 1)

    def testhashmismatch(self):
        Path("images/edited.png").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hashes"):
            self.runexport()
        self.assertFalse(Path("output").exists())

    def testchangedchoice(self):
        self.payload["rows"][0]["report"]["commonpool"]["attain"]["selected"] = "raw"
        with self.assertRaisesRegex(ValueError, "disagree"):
            self.runexport()

    def testmissingarchive(self):
        self.payload["rows"][0]["report"]["candidatearchive"]["enabled"] = False
        with self.assertRaisesRegex(ValueError, "archived"):
            self.runexport()

    def testduplicateidentities(self):
        self.payload["rows"].append(deepcopy(self.payload["rows"][0]))
        with self.assertRaisesRegex(ValueError, "identities"):
            self.runexport()

    def testduplicatefilenames(self):
        duplicate = deepcopy(self.payload["rows"][0])
        duplicate["key"] = "b"
        self.payload["rows"].append(duplicate)
        with self.assertRaisesRegex(ValueError, "unique flat"):
            self.runexport()

    def testnooverwrite(self):
        Path("output").mkdir()
        with self.assertRaises(FileExistsError):
            self.runexport()
        self.assertFalse(any(Path("output").iterdir()))

    def testrelativepaths(self):
        with self.assertRaises(ValueError):
            self.runexport(output="../outside")
        self.assertFalse(Path("output").exists())

    def testinfeasiblechoice(self):
        self.payload["rows"][0]["report"]["candidates"][0]["feasible"] = False
        with self.assertRaises(ValueError):
            self.runexport()

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu 2026
# Email: wlzchina05@gmail.com
