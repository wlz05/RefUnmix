# -*- coding: utf-8 -*-
# RefUnmix: Reference-Grounded Response Unmixing for One-Step Image Editing

import argparse
import contextlib
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from PIL import Image
import torch
import conduct
from engine import bench
from engine.bridge import check
from engine.flowedit import FlowEdit, implementation, settings
from engine.serial import read, write
from engine.sources import RELEASES
from evaluate import datasetargs
from run import common, run, validate
from session import editor
from intervention import Collector

@contextlib.contextmanager
def workspace():
    previous = Path.cwd()
    with tempfile.TemporaryDirectory(dir=previous) as directory:
        os.chdir(directory)
        try:
            yield Path(directory)
        finally:
            os.chdir(previous)

class BenchmarkTests(unittest.TestCase):
    def options(self, *values):
        return bench.options(bench.parser().parse_args(list(values)))

    def testplanswithoutmodels(self):
        for method in bench.METHODS:
            args = self.options('--method', method, '--seeds', '42')
            value = bench.plan(args)
            self.assertEqual(value['requestsperseed'], 700)
            self.assertEqual(value['seeds'], [42])
            self.assertNotIn('--worker', value['commands'][0])
            self.assertEqual(value['action'], 'plan only')
        self.assertEqual(len(bench.METHODS), 11)
        self.assertTrue(all(len(value[1]) == 40 for value in RELEASES.values()))

    def testinvalidcombinationsandpaths(self):
        for flags in [('--method', 'swiftedit', '--backbone', 'instaflow'), ('--method', 'flowedit', '--backbone', 'sdturbo'), ('--method', 'chord', '--batch', '3'), ('--method', 'chord', '--seeds', '52', '42'), ('--method', 'chord', '--output', 'models/result'), ('--method', 'chord', '--input', 'outputs/comparison'), ('--method', 'chord', '--output', '../elsewhere')]:
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                self.options(*flags)

    def testisolatedworkerrequiresowner(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            bench.main(['--method', 'chord', '--action', 'generate', '--worker', '--seeds', '42'])

    def testaggregationdestination(self):
        subset = self.options('--method', 'chord', '--seeds', '42')
        complete = self.options('--method', 'chord')
        self.assertEqual(bench.meanpath(subset).name, 'mean42.json')
        self.assertEqual(bench.meanpath(complete).name, 'mean.json')

    def testimportchecksallfilesbeforewriting(self):
        with workspace():
            Path('external').mkdir()
            Image.new('RGB', (512, 512)).save('external/a.png')
            items = [SimpleNamespace(key=key, relative=Path(key + '.png'), source='source', target='target') for key in ('a', 'b')]
            args = self.options('--method', 'flowedit', '--seeds', '42', '--input', 'external')
            with self.assertRaises(FileNotFoundError):
                bench.imported(args, items, 42)
            self.assertFalse(bench.root(args).exists())
            Image.new('RGB', (512, 512)).save('external/b.png')
            bench.imported(args, items, 42)
            summary = bench.root(args) / '42/summary.json'
            value = bench.verify(summary, items, 42)
            self.assertEqual(len(value['rows']), 2)
            target = summary.parent / value['rows'][0]['image']
            target.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                bench.verify(summary, items, 42)

    def testmanifestduplicates(self):
        with workspace():
            row = {'key': 'a', 'seed': 42, 'image': 'a.png'}
            write('import.json', {'rows': [row, row]})
            args = self.options('--method', 'flowedit', '--seeds', '42', '--input-manifest', 'import.json')
            with self.assertRaises(ValueError):
                bench.imported(args, [SimpleNamespace(key='a')], 42)

    def testnativeflowalgebra(self):
        with contextlib.redirect_stdout(io.StringIO()):
            check()

class Transformer(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states

class FlowEditTests(unittest.TestCase):
    def testreleasedparameters(self):
        self.assertEqual((settings('sd3')['T_steps'], settings('sd3')['n_max']), (50, 33))
        self.assertEqual((settings('flux')['T_steps'], settings('flux')['n_max']), (28, 24))
        for family, changes in [('other', None), ('sd3', {'n_min': 1}), ('sd3', {'T_steps': True}), ('sd3', {'tar_guidance_scale': float('nan')}), ('flux', {'n_max': 30}), ('flux', {'unknown': 2})]:
            with self.subTest(family=family, changes=changes), self.assertRaises(ValueError):
                settings(family, changes)

    def testofficialdispatchandvaenormalization(self):
        decoded, calls = [], []
        latent = torch.full((1, 4, 2, 2), .7)
        vae = SimpleNamespace(dtype=torch.float32, config=SimpleNamespace(scaling_factor=2., shift_factor=.4), encode=lambda pixels: SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latent.clone())), decode=lambda value, **kw: decoded.append(value.clone()) or (torch.zeros(1, 3, 16, 16),))
        processor = SimpleNamespace(preprocess=lambda image: torch.zeros(1, 3, 16, 16), postprocess=lambda pixels: [Image.new('RGB', (512, 512))])
        pipe = SimpleNamespace(_execution_device='cpu', vae=vae, scheduler=object(), image_processor=processor, transformer=Transformer())
        def official(pipeline, scheduler, value, source, target, negative, **parameters):
            calls.append((value.clone(), source, target, negative, parameters))
            pipeline.transformer(hidden_states=value)
            return value + .2
        result, report = FlowEdit(pipe, official, 'sd3')(SimpleNamespace(source='before', target='after'), Image.new('RGB', (512, 512)), 42)
        torch.testing.assert_close(calls[0][0], (latent - .4) * 2)
        torch.testing.assert_close(decoded[0], latent + .1)
        self.assertEqual(calls[0][1:4], ('before', 'after', ''))
        self.assertEqual(report['calls'], 1)
        self.assertEqual(result.size, (512, 512))

    def testinvalidmodulefamily(self):
        with self.assertRaises(ValueError):
            implementation('missing', 'other')

class AblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def options(self, profile):
        preset = conduct.parser().parse_args(['--profile', profile, '--batch', '1'])
        command = conduct.generation(preset, 42, Path('outputs') / profile / '42')
        args = datasetargs(common(argparse.ArgumentParser())).parse_args(command[3:])
        args.size, args.device = 16, 'cpu'
        validate(args)
        return args

    def testactualendpointqueryskips(self):
        expected = {'main': (5, 2, 3), 'assignment': (5, 2, 3), 'reference': (5, 2, 2), 'amplification': (5, 2, 3), 'gating': (5, 2, 3), 'target': (3, 2, 1)}
        with patch('engine.attention.Collector', Collector), patch('engine.attention.tokens', return_value=((1,), (1,))):
            for profile, cost in expected.items():
                with self.subTest(profile=profile):
                    args = self.options(profile)
                    pixels, report, session = run(editor(channels=4), None, Image.new('RGB', (16, 16), (90, 130, 160)), 'source', 'target long', args)
                    self.assertEqual((session.samples, session.calls, session.decodes), cost)
                    self.assertEqual(len(session.queries), session.calls)
                    self.assertTrue(torch.isfinite(pixels).all())
                    self.assertFalse(args.singlebias)
                    self.assertEqual(args.recenter, 0)

    def testcalibrationisaddback(self):
        main, restored = self.options('main'), self.options('calibration')
        self.assertEqual((main.singlemode, main.singlebias, main.recenter), ('contrast', False, 0))
        self.assertEqual((restored.singlemode, restored.singlebias, restored.recenter), ('recenter', True, .5))
        self.assertEqual((main.conditiongain, main.singlegain, main.innovationgain), (1.25, 1., 3.))

# Author: Weiliang Zhu; Other main author: Yeming Dai, Yaming Lv
# Email: wlzchina05@gmail.com
