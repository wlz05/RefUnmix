# -*- coding: utf-8 -*-
# RefUnmix: Reference-Grounded Response Unmixing for One-Step Image Editing

import copy
import unittest
from aggregate import equal, finite

class AggregationTests(unittest.TestCase):
    def testmetricwiseexclusion(self):
        rows = [{"table": {"psnr": None, "mse": 0.}, "status": {"psnr": "positive infinity", "mse": "finite"}}, {"table": {"psnr": 30., "mse": .1}, "status": {"psnr": "finite", "mse": "finite"}}]
        original = copy.deepcopy(rows)
        result = finite(rows, "table")
        self.assertEqual(result["psnr"], {"mean": 30., "count": 1, "excludednonfinite": 1})
        self.assertEqual(result["mse"]["mean"], .05)
        self.assertEqual(result["mse"]["count"], 2)
        self.assertEqual(rows, original)

    def testequalseedweights(self):
        reports = {"42": {"table": finite([{"table": {"psnr": 10.}}], "table")}, "52": {"table": finite([{"table": {"psnr": 30.}}, {"table": {"psnr": 30.}}], "table")}}
        self.assertEqual(equal(reports)["psnr"]["mean"], 20.)
        self.assertEqual(equal(reports)["psnr"]["count"], 3)
        reports["52"]["table"]["psnr"]["mean"] = None
        self.assertIsNone(equal(reports)["psnr"]["mean"])

    def testnonfiniteandempty(self):
        rows = [{"table": {"psnr": value}} for value in (float("inf"), float("nan"), None)]
        result = finite(rows, "table")["psnr"]
        self.assertIsNone(result["mean"])
        self.assertEqual(result["count"], 0)
        self.assertEqual(result["excludednonfinite"], 2)

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
