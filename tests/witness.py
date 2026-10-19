# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
import torch
from PIL import Image
from evidence import compare, CONTROLS, PROVENANCE
from report import assess, template
from run import common, run, validate
from trial import variants
from engine.native import create
from engine.path import Path as ReferencePath
from session import editor

class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def args(self, **changes):
        args = common(argparse.ArgumentParser()).parse_args(["--method", "fixed", "--field", "chord", "--axes", "none", "--prior", "off", "--size", "16"])
        for key, value in changes.items():
            setattr(args, key, value)
        return args

    def testfixedunfiltered(self):
        image = Image.new("RGB", (16, 16), (96, 128, 160))
        for eta in (0., .5, 1.):
            with self.subTest(eta=eta):
                args = self.args(eta=eta)
                output, report, session = run(editor(), None, image, "source", "target long", args)
                expected = ReferencePath(create(session, args.poststeps, args.guidance, args.cfg, args.referencepath), session.field(args.start, args.delta))
                expected.initialize()
                _, pixels = expected.candidate((1., 1., eta))
                self.assertTrue(torch.equal(output, pixels))
                self.assertEqual(report["selected"], "fixed")
                self.assertFalse(report["fallback"])
                self.assertEqual(report["selection"], "unfiltered intervention")
                self.assertNotIn("constraintmet", report)
                self.assertEqual(len(report["fieldsha256"]), 64)
                self.assertEqual(len(report["protectionsha256"]), 64)

    def testfixedinvalid(self):
        for changes in ({"prior": "ddpm"}, {"axes": "all"}, {"heldout": True}, {"eta": -1.}, {"eta": 1.1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate(self.args(**changes))

    def testdiagnosisdistinct(self):
        diagnosis = variants("diagnosis")
        for name, eta in (("target", 0.), ("middle", .5), ("source", 1.)):
            self.assertEqual(diagnosis[name]["method"], "fixed")
            self.assertEqual(diagnosis[name]["eta"], eta)
            self.assertEqual(diagnosis[name]["prior"], "off")
        self.assertNotIn("method", variants("core")["target"])
        self.assertEqual(diagnosis["random"]["representation"], "random")

    def write(self, path, payload):
        Path(path).write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")

    def fixture(self):
        for name, eta, success in (("target", 0., False), ("source", 1., True)):
            Path(name).mkdir()
            rows = []
            for key in ("first", "second"):
                image = Path(name, key + ".png")
                Image.new("RGB", (4, 4), "green" if success else "blue").save(image)
                settings = {key: "shared" for key in CONTROLS}
                report = {key: "shared" for key in PROVENANCE}
                report.update({"settings": settings, "method": "fixed", "renderer": "latent", "selection": "unfiltered intervention", "fallback": False, "requested": [1., 1., eta], "point": [1., 1., eta], "modelcheckpoint": {"weights": {"network": "hash"}}, "masksha256": None})
                rows.append({"key": key, "seed": 42, "group": key, "image": image.name, "resultsha256": hashlib.sha256(image.read_bytes()).hexdigest(), "metrics": {"psnr": 20.}, "report": report})
            summary = {"schema": 1, "split": "test", "oracle": False, "settings": {"frozenrecipehash": "hash"}, "rows": rows}
            path = name + "/summary.json"
            self.write(path, summary)
            labels = template(path)
            for label in labels:
                label.update({"success": success, "preserved": True, "artifact": False, "naturalness": 4.})
            self.write(name + "/labels.json", labels)
            self.write(name + "/task.json", assess(path, name + "/labels.json", repeats=10))

    def directory(self, action):
        original = os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                self.fixture()
                action()
            finally:
                os.chdir(original)

    def contrast(self, tasks=True):
        return compare("target/summary.json", "source/summary.json", "target/task.json" if tasks else None, "source/task.json" if tasks else None, repeats=10)

    def testpairedevidence(self):
        def action():
            result = self.contrast()
            self.assertEqual(result["status"], "paired evidence available")
            self.assertEqual(result["missing"], [])
            self.assertEqual(result["interventions"]["active"], 2)
            self.assertEqual(result["tasks"]["metrics"]["metric:jointsuccess"]["difference"], 1.)
            self.assertIn("no automatic", result["scope"])
            self.assertFalse(result["externalvalidity"]["externalauditprovided"])
            self.assertIn("another editor", result["externalvalidity"]["scope"])

        self.directory(action)

    def testmissinglabels(self):
        def action():
            result = self.contrast(False)
            self.assertEqual(result["status"], "incomplete evidence")
            self.assertIn("postselection independent task annotations", result["missing"])

        self.directory(action)

    def testmatchedcontrols(self):
        def action():
            payload = json.loads(Path("source/summary.json").read_text())
            payload["rows"][0]["report"]["fieldsha256"] = "different"
            self.write("source/summary.json", payload)
            with self.assertRaisesRegex(ValueError, "fieldsha256"):
                self.contrast(False)

        self.directory(action)

    def testfieldcontrast(self):
        def action():
            payload = json.loads(Path("source/summary.json").read_text())
            for row in payload["rows"]:
                row["report"]["fieldsha256"] = "otherfield"
                row["report"]["settings"]["field"] = "pullback"
            self.write("source/summary.json", payload)
            result = compare("target/summary.json", "source/summary.json", contrast="field", repeats=10)
            self.assertEqual(result["contrast"], "field")
            payload["rows"][0]["report"]["settings"]["maxsamples"] = 100
            self.write("source/summary.json", payload)
            with self.assertRaisesRegex(ValueError, "budget or controller"):
                compare("target/summary.json", "source/summary.json", contrast="field", repeats=10)

        self.directory(action)

    def testfilteredrejected(self):
        def action():
            payload = json.loads(Path("source/summary.json").read_text())
            payload["rows"][0]["report"]["method"] = "path"
            self.write("source/summary.json", payload)
            with self.assertRaisesRegex(ValueError, "unfiltered"):
                self.contrast(False)

        self.directory(action)

    def testchangedoutputs(self):
        def action():
            Image.new("RGB", (4, 4), "red").save("source/first.png")
            with self.assertRaisesRegex(ValueError, "bytes"):
                self.contrast()

        self.directory(action)

    def testchangedsummary(self):
        def action():
            payload = json.loads(Path("source/summary.json").read_text())
            payload["split"] = "val"
            self.write("source/summary.json", payload)
            with self.assertRaises(ValueError):
                compare("source/summary.json", "source/summary.json", "source/task.json", "source/task.json", contrast="outcome", repeats=10)

        self.directory(action)

    def testunratedtemplate(self):
        def action():
            labels = template("target/summary.json")
            self.assertIsNone(labels[0]["success"])
            self.write("unrated.json", labels)
            with self.assertRaisesRegex(ValueError, "boolean"):
                assess("target/summary.json", "unrated.json", repeats=10)

        self.directory(action)

    def testinconsistentjointlabels(self):
        def action():
            task = json.loads(Path("source/task.json").read_text())
            task["observations"][0]["metrics"]["jointsuccess"] = 0.
            self.write("source/task.json", task)
            with self.assertRaisesRegex(ValueError, "Joint success"):
                self.contrast()

        self.directory(action)

    def testjointsuccesscomponents(self):
        def action():
            labels = template("source/summary.json")
            for label in labels:
                label.update({"success": True, "preserved": False, "artifact": True, "naturalness": 2})
            self.write("labels.json", labels)
            task = assess("source/summary.json", "labels.json", repeats=10)
            self.assertEqual(task["metrics"]["success"]["mean"], 1.)
            self.assertEqual(task["metrics"]["jointsuccess"]["mean"], 0.)

        self.directory(action)

    def testincompleteprovenance(self):
        def action():
            for name in ("target", "source"):
                path = name + "/summary.json"
                payload = json.loads(Path(path).read_text())
                payload["settings"] = {}
                for row in payload["rows"]:
                    row["report"]["modelcheckpoint"] = None
                self.write(path, payload)
            result = self.contrast(False)
            self.assertIn("unverified model weight hashes", result["missing"])
            self.assertIn("shared frozen validation recipe hash", result["missing"])

        self.directory(action)

    def testinactivereference(self):
        image = Image.new("RGB", (16, 16), (96, 128, 160))
        args = self.args(cleanup=False, eta=1.)
        _, report, _ = run(editor(), None, image, "source", "target long", args)
        self.assertEqual(report["requested"], [1., 1., 1.])
        self.assertEqual(report["point"], [1., 1., 0.])

    def testnullcontrast(self):
        def action():
            path = "source/summary.json"
            payload = json.loads(Path(path).read_text())
            for row in payload["rows"]:
                row["report"]["point"] = [1., 1., 0.]
            self.write(path, payload)
            result = self.contrast(False)
            self.assertEqual(result["interventions"]["samepoint"], 2)
            self.assertEqual(result["interventions"]["active"], 0)

        self.directory(action)

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu 2026
# Email: wlzchina05@gmail.com
