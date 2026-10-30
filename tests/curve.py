# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from engine.curve import RADIAL, blend, compare, conditional, crossings, load
from engine.reproduce import digest

class CurveTests(unittest.TestCase):
    def fixture(self):
        arms, inference = {}, {}
        for name in [*RADIAL, "full", "matched", "feedback"]:
            factor = RADIAL.get(name, 1.)
            edited, mse = 20 + factor, 1 + factor ** 2
            if name not in RADIAL:
                edited, mse = {"full": (21.2, 1.8), "matched": (21.1, 1.5), "feedback": (21.3, 1.7)}[name]
            arms[name], inference[name] = [], []
            for seed in (42, 52, 3047, 3407):
                for key in ("a", "b"):
                    report = {"source": "source", "target": "target", "imagesha256": key * 64, "masksha256": None, "protectionsha256": "d" * 64, "anchorfieldsha256": "b" * 64, "fieldnorm": 2 * factor, "field": {"responsenorm": 2 * factor}}
                    row = {"key": key, "seed": seed, "group": key, "report": report, "metrics": {"edited": edited, "mse": mse, "psnr": 30 - mse, "lpips": 10 * mse}}
                    arms[name].append(row)
                    inference[name].append({key: value for key, value in row.items() if key != "metrics"})
        return arms, inference

    def testequalcurves(self):
        arms, inference = self.fixture()
        result = compare(arms, inference=inference, repeats=10)
        self.assertEqual(result["equaledit"]["status"], "matched")
        self.assertAlmostEqual(result["equaledit"]["point"]["metrics"]["edited"]["mean"], 21.2)
        self.assertGreater(result["equaledit"]["point"]["metrics"]["mse"]["mean"], 1.8)
        self.assertAlmostEqual(result["equalbudget"]["point"]["metrics"]["mse"]["mean"], 1.8)
        self.assertLess(result["equalbudget"]["point"]["metrics"]["edited"]["mean"], 21.2)
        self.assertEqual(result["normmatched"]["coverage"]["matched"], 8)
        self.assertEqual(result["target"]["seeds"], [42, 52, 3047, 3407])
        json.dumps(result, allow_nan=False)

    def testnonmonotone(self):
        points = [{"name": name, "metrics": {"edited": {"mean": value}}} for name, value in (("a", 1), ("b", 3), ("c", 2), ("d", 4))]
        self.assertEqual(len(crossings(points, 2.5, "edited")), 3)
        self.assertEqual(crossings(points, 5, "edited"), [])

    def testmissingseed(self):
        arms, _ = self.fixture()
        arms["full"] = [row for row in arms["full"] if row["seed"] != 52]
        with self.assertRaisesRegex(ValueError, "same requests"):
            compare(arms, repeats=10)

    def testundefinedseed(self):
        arms, _ = self.fixture()
        for rows in arms.values():
            for row in rows:
                if row["seed"] == 52:
                    row["metrics"]["edited"] = None
        result = compare(arms, repeats=10)
        self.assertIsNone(result["target"]["metrics"]["edited"]["mean"])
        self.assertEqual(result["equaledit"]["status"], "outside measured range or undefined")

    def testinfinity(self):
        arms, _ = self.fixture()
        for rows in arms.values():
            for row in rows:
                row["metrics"]["psnr"] = None
                row["status"] = {"psnr": "positive infinity"}
        result = compare(arms, repeats=10)
        self.assertEqual(result["equaledit"]["point"]["metrics"]["psnr"]["status"], "positive infinity")
        self.assertIsNone(result["normmatched"]["paired"]["metrics"]["metric:psnr"]["difference"])
        json.dumps(result, allow_nan=False)

    def testcoverage(self):
        arms, _ = self.fixture()
        arms["full"][0]["metrics"]["lpips"] = None
        with self.assertRaisesRegex(ValueError, "coverage"):
            compare(arms, repeats=10)

    def testnormviolation(self):
        arms, inference = self.fixture()
        inference["matched"][0]["report"]["field"]["responsenorm"] = 2.1
        with self.assertRaisesRegex(ValueError, "norm tolerance"):
            compare(arms, inference=inference, repeats=10)

    def testzeronorm(self):
        arms, inference = self.fixture()
        for name in ("anchor", "matched"):
            for row in inference[name]:
                row["report"]["field"]["responsenorm"] = 0
        result = compare(arms, inference=inference, repeats=10)
        self.assertEqual(result["normmatched"]["coverage"]["zeroanchor"], 8)
        self.assertEqual(result["normmatched"]["point"]["metrics"]["mse"]["count"], 2)

    def testweights(self):
        arms, _ = self.fixture()
        for weights in ({"anchor": -.1, "weak": 1.1}, {"anchor": .5}, {"unknown": 1}, {"anchor": True}):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                blend(arms, weights)
        self.assertEqual(blend(arms, {"anchor": 1., "weak": 0.})[0]["metrics"], arms["anchor"][0]["metrics"])

    def testconditionalcomparisons(self):
        arms, inference = self.fixture()
        mapping = {"full": "full", "anchor": "anchor", "expanded": "full", "plain": "full", "residual": "full", "feedback": "feedback", "matched": "matched"}
        result = conditional({key: arms[value] for key, value in mapping.items()}, {key: inference[value] for key, value in mapping.items()}, repeats=10)
        self.assertEqual(result["comparisons"]["rule"]["reference"], "expanded")
        self.assertEqual(result["comparisons"]["rule"]["comparison"], "full")
        self.assertEqual(result["normmatched"]["coverage"]["matched"], 8)
        self.assertEqual(result["arms"]["full"]["seeds"], [42, 52, 3047, 3407])
        json.dumps(result, allow_nan=False)

    def testloading(self):
        arms, inference = self.fixture()
        original = Path.cwd()
        with tempfile.TemporaryDirectory(prefix=".curve", dir=Path(__file__).resolve().parents[2]) as temporary:
            try:
                os.chdir(temporary)
                Path("runs").mkdir()
                Path("metrics").mkdir()
                suite = {"schema": 1, "settings": {"suite": "amplitude"}, "seeds": [42, 52, 3047, 3407], "variants": {}, "design": {"variants": {}}}
                for name, rows in arms.items():
                    source = Path("runs") / (name + ".json")
                    signature = hashlib.sha256(name.encode()).hexdigest()
                    source.write_text(json.dumps({"schema": 1, "seeds": suite["seeds"], "rows": inference[name], "controlsignature": {"sha256": signature}}), encoding="utf-8")
                    suite["variants"][name] = {"path": source.as_posix()}
                    suite["design"]["variants"][name] = {"sha256": signature, "settings": {"fieldfactor": RADIAL.get(name, 1.)}}
                    scored = {"schema": 1, "seeds": suite["seeds"], "protocol": "PIE external metric adapter", "inputsha256": digest(source), "mappingsha256": "c" * 64, "backend": {"complete": True, "sha256": "a" * 64}, "rows": [{"key": row["key"], "seed": row["seed"], "table": row["metrics"]} for row in rows]}
                    Path("metrics", name + ".json").write_text(json.dumps(scored), encoding="utf-8")
                Path("suite.json").write_text(json.dumps(suite), encoding="utf-8")
                result = load("suite.json", "metrics", repeats=10)
                self.assertEqual(len(result["curve"]), 7)
                secondary = json.loads(json.dumps(suite))
                secondary["settings"]["suite"] = "conditional"
                names = {"full", "anchor", "expanded", "plain", "residual", "feedback", "matched"}
                for name in ("expanded", "plain", "residual"):
                    source = Path("runs") / (name + ".json")
                    signature = hashlib.sha256(name.encode()).hexdigest()
                    payload = json.loads(Path("runs/full.json").read_text(encoding="utf-8"))
                    payload["controlsignature"]["sha256"] = signature
                    source.write_text(json.dumps(payload), encoding="utf-8")
                    secondary["variants"][name] = {"path": source.as_posix()}
                    secondary["design"]["variants"][name] = {"sha256": signature, "settings": {"fieldfactor": 1.}}
                    scored = json.loads(Path("metrics/full.json").read_text(encoding="utf-8"))
                    scored["inputsha256"] = digest(source)
                    Path("metrics", name + ".json").write_text(json.dumps(scored), encoding="utf-8")
                secondary["variants"] = {name: value for name, value in secondary["variants"].items() if name in names}
                Path("conditional.json").write_text(json.dumps(secondary), encoding="utf-8")
                result = load("conditional.json", "metrics", repeats=10)
                self.assertEqual(set(result["arms"]), names)
                source = Path("runs/full.json")
                source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "exact inference"):
                    load("suite.json", "metrics", repeats=10)
            finally:
                os.chdir(original)

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
