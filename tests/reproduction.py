# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import conduct

class ReproductionTests(unittest.TestCase):
    def test_schedule(self):
        args = conduct.parser().parse_args([])
        self.assertEqual(args.seeds, [42, 52, 3047, 3407])
        self.assertEqual(args.batch, 3)
        self.assertEqual(args.profile, 'main')

    def test_containing_workspace(self):
        root = Path(conduct.__file__).resolve().parent
        args = conduct.parser().parse_args([])
        before = conduct.source()
        with patch('conduct.Path.cwd', return_value=root.parent):
            command = conduct.generation(args, 42, Path('outputs/main/42'))
            self.assertEqual(command[2], root.name + '/evaluate.py')
            self.assertEqual(conduct.scoring(args, 42, Path('outputs/main/42'))[2], root.name + '/assess.py')
            self.assertEqual(conduct.source(), before)

    def test_registered_profiles(self):
        import argparse
        from evaluate import datasetargs
        from run import common
        for profile in conduct.PROFILES:
            with self.subTest(profile=profile):
                args = conduct.parser().parse_args(['--profile', profile])
                command = conduct.generation(args, 42, Path('outputs') / profile / '42')
                settings = datasetargs(common(argparse.ArgumentParser())).parse_args(command[3:])
                self.assertTrue(settings.single)
                self.assertEqual(settings.seed, 42)
                self.assertEqual(settings.seeds, [42])
                self.assertEqual(settings.singlegain, 1.5)
                self.assertEqual(settings.conditiongain, 1. if profile == 'condition' else 1.25)
                self.assertEqual(settings.innovationgain, 1. if profile == 'amplification' else 2.)
                self.assertEqual(settings.singlemode, 'recenter' if profile == 'calibration' else 'contrast')
                self.assertEqual(settings.recenter, .5 if profile == 'calibration' else 0.)
                self.assertEqual(settings.singlebias, profile in {'calibration', 'bias', 'locality'})
                self.assertEqual(settings.biasgeometry, 'global' if profile == 'locality' else 'local')
                self.assertFalse(settings.control)
                self.assertFalse(settings.cfg)
                self.assertFalse(settings.compensation)
                self.assertEqual(settings.poststeps, 1)
                self.assertEqual(settings.delta, 0.)

    def test_escaping_paths(self):
        for path in ('../models', '/models', 'D:/models'):
            with self.subTest(path=path):
                with self.assertRaises(Exception):
                    conduct.relative(path)

    def test_seed_completeness(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / 'report.json'
            report.write_text(json.dumps({'rows': [{'key': str(i), 'seed': 42} for i in range(700)]}))
            self.assertEqual(len(conduct.complete(report, 42)['rows']), 700)
            with self.assertRaises(ValueError):
                conduct.complete(report, 52)
            report.write_text(json.dumps({'rows': [{'key': 'same', 'seed': 42}] * 700}))
            with self.assertRaises(ValueError):
                conduct.complete(report, 42)

if __name__ == '__main__':
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
