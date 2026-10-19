# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from __future__ import annotations
import math
import unittest
import torch
from PIL import Image
from engine.clip import ClipJudge
from engine.timing import Metric

class ClipTest(unittest.TestCase):
    def testinterface(self) -> None:
        from transformers import CLIPConfig, CLIPModel, CLIPImageProcessor
        torch.manual_seed(1)
        config = CLIPConfig(text_config={"vocab_size": 16, "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2, "max_position_embeddings": 8, "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0}, vision_config={"hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2, "image_size": 32, "patch_size": 16}, projection_dim=8)

        class Processor:
            def __call__(self, images=None, text=None, **kwargs):
                if images is not None:
                    return CLIPImageProcessor(size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32})(images=images, return_tensors="pt")
                tokens = torch.zeros(len(text), 8, dtype=torch.long)
                tokens[:, 0] = 1
                for index, value in enumerate(text):
                    tokens[index, 1] = 3 + sum(map(ord, value)) % 13
                tokens[:, 2] = 2
                return {"input_ids": tokens, "attention_mask": (tokens != 0).long()}

        judge = ClipJudge(CLIPModel(config), Processor())
        source = Image.new("RGB", (64, 32), (127, 127, 127))
        mask = torch.ones(1, 1, 32, 64)
        mask[..., 8:24, 16:48] = 0
        evaluate = judge.prepare(source, "gray room", "snowy forest", mask, "snowy trees")
        unchanged = evaluate(source)
        self.assertAlmostEqual(unchanged["global"], 0.0, places=6)
        self.assertAlmostEqual(unchanged["local"], 0.0, places=6)
        self.assertAlmostEqual(unchanged["direction"], 0.0, places=6)
        changed = evaluate(Image.new("RGB", source.size, (240, 240, 255)))
        self.assertTrue(all(math.isfinite(value) for value in changed.values()))
        self.assertEqual(changed["localviews"], 4)
        with self.assertRaises(ValueError):
            evaluate(Image.new("RGB", (16, 16)))
        metric = Metric(judge, source, "snowy forest", Image.new("L", source.size, 255))
        scores = metric(source)
        self.assertTrue(math.isfinite(scores["semantic"]))
        self.assertIsNone(scores["psnr"])

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu 2026
# Email: wlzchina05@gmail.com
