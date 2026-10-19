# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import json
import unittest
import numpy as np
from engine.endpoint import design
from engine.native import Native
from engine.path import Path
from session import editor
from PIL import Image

class EndpointTests(unittest.TestCase):
    def testdesignranks(self):
        for names, rank in ((("targetzero", "targetfield"), 2), (("sourcezero", "targetzero", "targetfield"), 3), (("sourcezero", "targetzero", "sourcefield", "targetfield"), 4)):
            result = design(names)
            self.assertEqual(result["rank"], rank)
            self.assertEqual(result["nullity"], 4 - rank)
            matrix, kernel = np.asarray(result["matrix"]), np.asarray(result["kernel"]).reshape(-1, 4)
            self.assertLessEqual(np.linalg.norm(matrix @ kernel.T), 1e-12)
            json.dumps(result, allow_nan=False)

    def testtwoconfoundingdirections(self):
        result = design(("targetzero", "targetfield"))
        matrix = np.asarray(result["matrix"])
        self.assertTrue(np.array_equal(matrix @ [1, -1, 0, 0], [0, 0]))
        self.assertTrue(np.array_equal(matrix @ [0, 0, 1, -1], [0, 0]))
        self.assertFalse(result["identifiable"]["bias"])
        self.assertFalse(result["identifiable"]["condition"])
        self.assertTrue(result["identifiable"]["biascondition"])
        self.assertTrue(result["identifiable"]["fieldinteraction"])

    def testthreeendpointseparation(self):
        result = design()
        self.assertTrue(result["identifiable"]["bias"])
        self.assertTrue(result["identifiable"]["condition"])
        self.assertFalse(result["identifiable"]["field"])
        self.assertFalse(result["identifiable"]["interaction"])

    def testfullbijection(self):
        result = design(("sourcezero", "targetzero", "sourcefield", "targetfield"))
        matrix, inverse = np.asarray(result["matrix"]), np.asarray(result["fullinverse"])
        self.assertTrue(np.array_equal(matrix @ inverse, np.eye(4)))
        self.assertTrue(all(result["identifiable"].values()))
        effects = np.random.default_rng(20261010).normal(size=(4, 12))
        self.assertLess(np.linalg.norm(inverse @ matrix @ effects - effects), 1e-12)

    def testnonlineareffectsremainexact(self):
        source = np.array([.4])
        endpoints = np.array([[.5], [.8], [.7], [1.6]])
        result = design(("sourcezero", "targetzero", "sourcefield", "targetfield"))
        effects = np.asarray(result["fullinverse"]) @ (endpoints - source)
        self.assertAlmostEqual(float(effects[3, 0]), .6)
        self.assertTrue(np.allclose(np.asarray(result["matrix"]) @ effects + source, endpoints))

    def testemptyandinvalidobservations(self):
        self.assertEqual(design(())["nullity"], 4)
        for observed in (("unknown",), ("targetzero", "targetzero"), "targetzero"):
            with self.subTest(observed=observed), self.assertRaises(ValueError):
                design(observed)

    def testpaidendpointlogging(self):
        current = editor().prepare(Image.new("RGB", (16, 16)), "source", "target")
        path = Path(Native(current), current.field(.9, .15))
        path.initialize()
        before = current.samples
        self.assertEqual(path.metadata()["design"]["rank"], 3)
        self.assertEqual(current.samples, before)
        result = path.mechanism()
        self.assertEqual(result["design"]["rank"], 4)

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu 2026
# Email: wlzchina05@gmail.com
