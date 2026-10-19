# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image, ImageOps
from data import clean, locate

class ClipJudge:
    def __init__(self, model, processor, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.processor = processor

    @classmethod
    def load(cls, root: str, device: str = "cpu") -> "ClipJudge":
        path = locate(Path("."), clean(root, "clip"))
        if not path.is_dir():
            raise FileNotFoundError(path)
        from transformers import CLIPModel, CLIPProcessor
        return cls(CLIPModel.from_pretrained(path, local_files_only=True), CLIPProcessor.from_pretrained(path, local_files_only=True), device)

    def images(self, images: list[Image.Image]) -> torch.Tensor:
        size = int(self.model.config.vision_config.image_size)
        squares = [ImageOps.pad(value.convert("RGB"), (size, size), color=(127, 127, 127)) for value in images]
        values = self.processor(images=squares, return_tensors="pt")["pixel_values"].to(self.device)
        features = self.model.visual_projection(self.model.vision_model(pixel_values=values).pooler_output)
        return functional.normalize(features.float(), dim=-1)

    def texts(self, values: list[str]) -> torch.Tensor:
        tokens = self.processor(text=values, padding=True, truncation=True, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in tokens.items() if key in {"input_ids", "attention_mask"}}
        features = self.model.text_projection(self.model.text_model(**inputs).pooler_output)
        return functional.normalize(features.float(), dim=-1)

    @torch.no_grad()
    def prepare(self, source: Image.Image, original: str, target: str, mask: torch.Tensor, localtext: str | None = None):
        # Keep identical crop geometry across all candidates.
        height, width = source.height, source.width
        area = functional.interpolate(mask.detach().float().cpu(), (height, width), mode="bilinear", align_corners=False)[0, 0].clamp(0.0, 1.0).numpy()
        boxes = []
        weights = []
        tiles = [(0, 0, width, height)]
        if float(area.mean()) > 0.4 and min(height, width) >= 4:
            tiles = [(x, y, right, bottom) for x, right in ((0, width // 2), (width // 2, width)) for y, bottom in ((0, height // 2), (height // 2, height))]
        for left, top, right, bottom in tiles:
            rows, columns = np.nonzero(area[top:bottom, left:right] > 0.05)
            if not rows.size:
                continue
            box = (left + int(columns.min()), top + int(rows.min()), left + int(columns.max()) + 1, top + int(rows.max()) + 1)
            boxes.append(box)
            weights.append(float(area[box[1]:box[3], box[0]:box[2]].sum()))

        def views(image: Image.Image) -> list[Image.Image]:
            if image.size != source.size:
                raise ValueError("CLIP candidates must share the source image size.")
            pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
            focused = Image.fromarray(np.clip(pixels * area[..., None] + 127.0 * (1.0 - area[..., None]), 0, 255).astype(np.uint8))
            return [image] + [focused.crop(box) for box in boxes]

        reference = self.images(views(source))
        text = self.texts([original, target, localtext or target])
        direction = text[1] - text[0]
        regionweights = torch.tensor(weights, device=self.device)
        regionweights = regionweights / regionweights.sum().clamp_min(1e-8)

        @torch.no_grad()
        def evaluate(image: Image.Image) -> dict[str, float]:
            features = self.images(views(image))
            change = features[0] - reference[0]
            globalgain = torch.dot(change, direction)
            localgain = ((features[1:] - reference[1:]) @ text[2] * regionweights).sum() if boxes else torch.zeros((), device=self.device)
            directional = torch.dot(change, direction) / (change.norm() * direction.norm()).clamp_min(1e-6)
            directional = directional * (change.norm() / 0.05).clamp(max=1.0)
            return {"global": float(globalgain), "local": float(localgain), "direction": float(directional), "localviews": len(boxes)}

        return evaluate

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
