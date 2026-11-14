# -*- coding: utf-8 -*-
# RefUnmix: Reference-Grounded Response Unmixing for One-Step Image Editing

import unittest
import torch
from engine.memory import active, install, route
from session import editor

class Attention(torch.nn.Module):
    def __init__(self, cross=False):
        super().__init__()
        self.is_cross_attention = cross
        self.to_q = torch.nn.Identity()
        self.to_k = torch.nn.Identity()
        self.to_v = torch.nn.Identity()

    def forward(self, value):
        query, key, values = self.to_q(value), self.to_k(value), self.to_v(value)
        return query, key, values

class Model(torch.nn.Module):
    def __init__(self, bypass=False):
        super().__init__()
        self.attn1 = Attention()
        self.attn2 = Attention(True)
        self.bypass = bypass

    def forward(self, sample, timestep, encoder_hidden_states, return_dict=False):
        value = sample.flatten(2).transpose(1, 2)
        if self.bypass:
            return (sample,)
        _, key, values = self.attn1(value)
        return ((key + values).transpose(1, 2).reshape_as(sample),)

class ReferenceTests(unittest.TestCase):
    def testpaired(self):
        model = Model()
        self.assertEqual(install(model), 2)
        value = torch.arange(24, dtype=torch.float32).reshape(4, 3, 2)
        with route(.5, 0, 2) as counts:
            query, key, values = model.attn1(value)
            self.assertTrue(torch.equal(query, value))
            self.assertTrue(torch.equal(values, value))
            self.assertTrue(torch.equal(key[0::2], value[0::2]))
            self.assertTrue(torch.equal(key[1::2], (value[0::2] + value[1::2]) / 2))
            self.assertEqual(counts, [1, 0])
            self.assertTrue(torch.equal(model.attn2(value)[1], value))
        self.assertEqual(active.get(), (0., 0., 2))

    def testtriplet(self):
        model = Model()
        install(model)
        value = torch.arange(36, dtype=torch.float32).reshape(6, 3, 2)
        with route(1, 1, 3) as counts:
            query, key, values = model.attn1(value)
            self.assertTrue(torch.equal(query, value))
            for result in (key, values):
                self.assertTrue(torch.equal(result[0::3], value[0::3]))
                self.assertTrue(torch.equal(result[1::3], value[1::3]))
                self.assertTrue(torch.equal(result[2::3], value[1::3]))
            self.assertEqual(counts, [1, 1])

    def testnestedcleanup(self):
        model = Model()
        install(model)
        value = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
        with route(.5, 0) as outer:
            with self.assertRaisesRegex(RuntimeError, 'intentional'):
                with route(0, 1):
                    raise RuntimeError('intentional')
            self.assertEqual(active.get(), (.5, 0, 2))
            model.attn1(value)
            self.assertEqual(outer, [1, 0])
        self.assertEqual(active.get(), (0., 0., 2))
        for result in model.attn1(value):
            self.assertTrue(torch.equal(result, value))

    def testlazyinstallation(self):
        session = editor()
        session.unet = Model()
        anchor = torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2) / 24
        noise = torch.zeros_like(anchor)
        embedding = torch.zeros(2, 1, 1)
        original = session.predict(anchor, noise, 50, embedding)
        self.assertFalse(hasattr(session.unet, 'referencehandles'))
        changed = session.predict(anchor, noise, 50, embedding, keys=.5)
        self.assertTrue(torch.equal(changed[0], original[0]))
        self.assertFalse(torch.equal(changed[1], original[1]))
        self.assertEqual(len(session.unet.referencehandles), 2)
        again = session.predict(anchor, noise, 50, embedding, keys=.5)
        self.assertTrue(torch.equal(changed, again))
        self.assertEqual(len(session.unet.referencehandles), 2)
        self.assertTrue(torch.equal(original, session.predict(anchor, noise, 50, embedding)))

    def testbackendguard(self):
        session = editor()
        session.unet = Model(True)
        anchor = torch.zeros(2, 3, 2, 2)
        with self.assertRaisesRegex(RuntimeError, 'bypassed'):
            session.predict(anchor, anchor, 50, torch.zeros(2, 1, 1), keys=.5)
        self.assertEqual(active.get(), (0., 0., 2))

    def testunsupported(self):
        session = editor()
        anchor = torch.zeros(2, 3, 2, 2)
        with self.assertRaisesRegex(ValueError, 'no compatible'):
            session.predict(anchor, anchor, 50, torch.zeros(2, 1, 1), keys=.5)
        model = Model()
        install(model)
        with route(.5, 0):
            with self.assertRaisesRegex(ValueError, 'complete endpoint'):
                model.attn1(torch.zeros(3, 2, 4))

if __name__ == '__main__':
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yeming Dai, Yaming Lv
# Email: wlzchina05@gmail.com
