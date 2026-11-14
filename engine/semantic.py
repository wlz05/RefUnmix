# -*- coding: utf-8 -*-
# RefUnmix: Reference-Grounded Response Unmixing for One-Step Image Editing

import torch
import torch.nn.functional as functional
from engine.judge import ClipJudge
from engine.session import relative

class SemanticJudge(ClipJudge):
    @classmethod
    def load(cls, root="semantic", device="cuda:0"):
        # Load an independent semantic proxy without evaluator weights.
        root = relative(root)
        from transformers import AutoModel, AutoProcessor
        from engine.checkpoint import signature
        model = AutoModel.from_pretrained(root, local_files_only=True)
        processor = AutoProcessor.from_pretrained(root, local_files_only=True, use_fast=False)
        result = cls(model, processor, device)
        result.metadata = {"family": "siglip2", "repository": "google/siglip2-base-patch16-224", "revision": "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2", "weights": signature(result.model)}
        return result

    @torch.no_grad()
    def texts(self, values):
        # Match the fixed-length padding used by the semantic encoder.
        length = self.model.config.text_config.max_position_embeddings
        encoded = self.processor(text=values, padding="max_length", max_length=length, truncation=True, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items() if key in {"input_ids", "attention_mask"}}
        features = self.model.get_text_features(**encoded).float()
        self.evaluations += 1
        return functional.normalize(features, dim=-1)

def feature(judge, values):
    # Preserve differentiable image features for either frozen proxy.
    if hasattr(judge.model, "visual_projection"):
        return judge.model.visual_projection(judge.model.vision_model(pixel_values=values).pooler_output).float()
    return judge.model.get_image_features(pixel_values=values).float()

# Author: Weiliang Zhu; Other main author: Yeming Dai, Yaming Lv
# Email: wlzchina05@gmail.com
