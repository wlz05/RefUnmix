# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import unittest
from types import SimpleNamespace
import torch
from engine.attention import Collector, tokens

class Tokenizer:
    all_special_ids = [0, 1]

    def __call__(self, values, **kwargs):
        words = {"a": 2, "blue": 3, "striped": 4, "shirt": 5, "on": 6, "man": 7}
        rows = [[words[word] for word in value.split()] for value in values]
        if kwargs.get("add_special_tokens", True):
            rows = [[0, *row, 1] for row in rows]
        width = max(map(len, rows))
        return SimpleNamespace(input_ids=torch.tensor([row + [1] * (width - len(row)) for row in rows]))

class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = torch.nn.Linear(4, 4, bias=False)
        self.to_k = torch.nn.Linear(4, 4, bias=False)
        self.heads, self.scale = 2, 2 ** -.5

    def forward(self, hidden_states, encoder_hidden_states=None):
        return hidden_states + encoder_hidden_states.mean(1, keepdim=True)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn2 = Attention()

class AttentionTests(unittest.TestCase):
    def testtokens(self):
        result = tokens(Tokenizer(), "a blue shirt on man", "a striped shirt on man", 8)
        self.assertIn(2, result[0])
        self.assertIn(3, result[0])
        self.assertNotIn(0, result[0])
        self.assertEqual(tokens(Tokenizer(), "a blue shirt", "a striped shirt", 8, "shirt"), [[3], [3]])

    def testidenticaltokens(self):
        self.assertEqual(tokens(Tokenizer(), "a blue shirt", "a blue shirt", 8), [[], []])

    def testcaptureunchanged(self):
        model = Model()
        hidden, context = torch.randn(2, 4, 4), torch.randn(2, 5, 4)
        original = model.attn2(hidden, encoder_hidden_states=context)
        collector = Collector(model, [[1], [2]], (4, 4))
        with collector:
            observed = model.attn2(hidden, encoder_hidden_states=context)
        torch.testing.assert_close(observed, original, atol=0, rtol=0)
        self.assertEqual(collector.result().shape, (1, 1, 4, 4))
        self.assertEqual(collector.report()["maps"], 1)
        self.assertEqual(len(model.attn2._forward_pre_hooks), 0)

    def testexceptioncleanup(self):
        model = Model()
        collector = Collector(model, [[1], [1]], (4, 4))
        with self.assertRaises(RuntimeError):
            with collector:
                raise RuntimeError("probe failure")
        self.assertEqual(len(model.attn2._forward_pre_hooks), 0)

    def testunsupported(self):
        model = Model()
        collector = Collector(model, [[1], [1]], (4, 4), maximum=2)
        with collector:
            model.attn2(torch.zeros(2, 4, 4), encoder_hidden_states=torch.zeros(2, 5, 4))
        self.assertIsNone(collector.result())
        self.assertTrue(collector.report()["skipped"])

    def testmultipleobservations(self):
        model = Model()
        collector = Collector(model, [[1], [1]], (4, 4), count=2)
        with collector:
            model.attn2(torch.zeros(4, 4, 4), encoder_hidden_states=torch.zeros(4, 5, 4))
            model.attn2(torch.zeros(4, 4, 4), encoder_hidden_states=torch.zeros(4, 5, 4))
        self.assertEqual(collector.report()["maps"], 2)
        torch.testing.assert_close(collector.result(), torch.full((1, 1, 4, 4), .2))

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu 2026
# Email: wlzchina05@gmail.com
