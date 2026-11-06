# -*- coding: utf-8 -*-
# RefUnmix: Reference-Grounded Response Unmixing for One-Step Image Editing

import argparse
import importlib.util
import os
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from run import common, validate

spec = importlib.util.spec_from_file_location("latencyentry", Path(__file__).resolve().parents[1] / "timing.py")
timing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timing)

class TimingTests(unittest.TestCase):
    def population(self):
        return [SimpleNamespace(key=str(group) + str(index).zfill(3), category=str(group)) for group in range(10) for index in range(70)]

    def test_fixed_balanced_selection(self):
        population = self.population()
        selected = timing.selection(population)
        shuffled = list(population)
        random.Random(52).shuffle(shuffled)
        self.assertEqual([item.key for item in selected], [item.key for item in timing.selection(shuffled)])
        self.assertEqual(len(selected), 50)
        self.assertEqual({item.category for item in selected}, set("0123456789"))
        for group in "0123456789":
            self.assertEqual(sum(item.category == group for item in selected), 5)

    def test_population_identity_and_coverage(self):
        for items in [self.population()[:-1], self.population()[:-1] + [self.population()[0]], [SimpleNamespace(key=str(index), category="0") for index in range(700)]]:
            with self.assertRaises(ValueError):
                timing.selection(items)

    def test_complete_duration_audit(self):
        selected = timing.selection(self.population())
        rows = [{"key": item.key, "seed": 42, "seconds": .25} for item in selected]
        self.assertEqual(timing.verify(rows, selected, 42), .25)
        for changed in [rows[:-1], rows[::-1], [dict(row, seed=52) for row in rows], [dict(row, seconds=float("nan")) for row in rows], [dict(row, seconds=0) for row in rows], [dict(row, seconds=True) for row in rows]]:
            with self.assertRaises(ValueError):
                timing.verify(changed, selected, 42)

    def test_gpu_exclusivity(self):
        for output in ["", str(os.getpid()) + "\n"]:
            with patch("subprocess.check_output", return_value=output):
                timing.exclusive()
        with patch("subprocess.check_output", return_value=str(os.getpid() + 1)):
            with self.assertRaises(RuntimeError):
                timing.exclusive()

    def test_field_and_context_boundaries(self):
        for flag in ["--targetpull", "--retainedcontext"]:
            for value in ["0", "1"]:
                validate(common(argparse.ArgumentParser()).parse_args(["--clip", "clip", flag, value]))
            for value in ["nan", "inf", "-.01", "1.01"]:
                with self.assertRaises(ValueError):
                    validate(common(argparse.ArgumentParser()).parse_args(["--clip", "clip", flag, value]))

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
