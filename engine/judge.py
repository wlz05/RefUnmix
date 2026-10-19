# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from __future__ import annotations
from weakref import ref
import torch
import torch.nn.functional as functional
from engine.session import relative, tensor

class ClipJudge:
    def __init__(self, model, processor, device="cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        if hasattr(self.model, "requires_grad_"):
            self.model.requires_grad_(False)
        self.processor = processor
        self.metadata = {"family": "clip", "weights": None}
        self.evaluations, self.imagecalls, self.imagesamples = 0, 0, 0
        config = getattr(processor, "image_processor", getattr(processor, "feature_extractor", processor))
        self.size = int(model.config.vision_config.image_size)
        sizing = getattr(config, "size", {"shortest_edge": self.size})
        self.short = int(sizing.get("shortest_edge", self.size) if isinstance(sizing, dict) else sizing)
        cropping = getattr(config, "crop_size", {"height": self.size, "width": self.size})
        self.crop = (int(cropping["height"]), int(cropping["width"])) if isinstance(cropping, dict) else (int(cropping), int(cropping))
        if min(self.size, self.short, *self.crop) < 1 or self.crop != (self.size, self.size):
            raise ValueError("CLIP resize and crop metadata are incompatible.")
        self.mean = torch.tensor(getattr(config, "image_mean", [0.48145466, 0.4578275, 0.40821073]), device=self.device).reshape(1, 3, 1, 1)
        self.std = torch.tensor(getattr(config, "image_std", [0.26862954, 0.26130258, 0.27577711]), device=self.device).reshape(1, 3, 1, 1)
        if not torch.isfinite(self.mean).all().item() or not torch.isfinite(self.std).all().item() or torch.any(self.std <= 0).item():
            raise ValueError("CLIP normalization metadata are invalid.")

    @classmethod
    def load(cls, root="./clip", device="cpu"):
        path = relative(root)
        if not path.is_dir():
            raise FileNotFoundError(path)
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as error:
            raise RuntimeError("CLIP loading requires transformers.") from error
        from engine.checkpoint import configuration, signature
        result = cls(CLIPModel.from_pretrained(path, local_files_only=True), CLIPProcessor.from_pretrained(path, local_files_only=True), device)
        result.metadata.update({"weights": signature(result.model), "configuration": configuration(path)})
        return result

    def pixels(self, image):
        # Match CLIP geometry without quantizing response probes.
        image = tensor(image, 3).to(self.device, torch.float32)
        if torch.any((image < 0) | (image > 1)).item():
            raise ValueError("CLIP images must be in [0, 1].")
        height, width = image.shape[-2:]
        scale = self.short / min(height, width)
        shape = (max(1, int(height * scale)), max(1, int(width * scale)))
        image = functional.interpolate(image, shape, mode="bicubic", align_corners=False, antialias=True)
        cropheight, cropwidth = self.crop
        vertical, horizontal = max(0, cropheight - shape[0]), max(0, cropwidth - shape[1])
        if vertical or horizontal:
            image = functional.pad(image, (horizontal // 2, horizontal - horizontal // 2, vertical // 2, vertical - vertical // 2))
        top, left = (image.shape[-2] - cropheight) // 2, (image.shape[-1] - cropwidth) // 2
        return (image[..., top:top + cropheight, left:left + cropwidth] - self.mean) / self.std

    @torch.no_grad()
    def images(self, images):
        if not images:
            raise ValueError("At least one CLIP view is required.")
        values = torch.cat([self.pixels(image) for image in images])
        self.imagecalls += 1
        self.imagesamples += len(images)
        features = self.model.visual_projection(self.model.vision_model(pixel_values=values).pooler_output).float()
        if not torch.isfinite(features).all().item():
            raise RuntimeError("CLIP image features contain nonfinite values.")
        return functional.normalize(features, dim=-1)

    @torch.no_grad()
    def texts(self, values):
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError("CLIP prompts must be nonempty strings.")
        tokens = self.processor(text=values, padding=True, truncation=True, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in tokens.items() if key in {"input_ids", "attention_mask"}}
        features = self.model.text_projection(self.model.text_model(**inputs).pooler_output).float()
        if not torch.isfinite(features).all().item():
            raise RuntimeError("CLIP text features contain nonfinite values.")
        return functional.normalize(features, dim=-1)

    @torch.no_grad()
    def prepare(self, origin, source, target, mask=None, localtext=None, attributes=None, retain=None):
        # Freeze crops and reference features across candidates.
        origin = tensor(origin, 3).detach().to(self.device, torch.float32)
        height, width = origin.shape[-2:]
        if mask is None:
            area = torch.ones((1, 1, height, width), device=self.device)
        else:
            tensor(mask, 1)
            if torch.any((mask < 0) | (mask > 1)).item():
                raise ValueError("CLIP edit masks must be in [0, 1].")
            area = functional.interpolate(mask.detach().to(self.device, torch.float32), (height, width), mode="bilinear", align_corners=False)
        tiles = [(0, 0, width, height)]
        if float(area.mean()) > 0.4 and min(height, width) >= 4:
            tiles = [(left, top, right, bottom) for left, right in ((0, width // 2), (width // 2, width)) for top, bottom in ((0, height // 2), (height // 2, height))]
        boxes, weights = [], []
        for left, top, right, bottom in tiles:
            positions = (area[0, 0, top:bottom, left:right] > 0.05).nonzero()
            if positions.numel() == 0:
                continue
            rows, columns = positions[:, 0], positions[:, 1]
            box = (left + int(columns.min()), top + int(rows.min()), left + int(columns.max()) + 1, top + int(rows.max()) + 1)
            boxes.append(box)
            weights.append(area[0, 0, box[1]:box[3], box[0]:box[2]].sum())
        regionweights = torch.stack(weights) if weights else torch.empty(0, device=self.device)
        regionweights = regionweights / regionweights.sum().clamp_min(1e-8)

        def views(pixels):
            pixels = tensor(pixels, 3).to(self.device, torch.float32)
            if pixels.shape != origin.shape:
                raise ValueError("CLIP candidates must share the source image shape.")
            focused = pixels * area + 0.5 * (1 - area)
            return [pixels] + [focused[..., top:bottom, left:right] for left, top, right, bottom in boxes]

        reference = self.images(views(origin))
        descriptions = list(attributes or []) + list(retain or [])
        text = self.texts([source, target, localtext or target] + descriptions)
        componentnames = ["attribute" + str(index) for index in range(len(attributes or []))] + ["retain" + str(index) for index in range(len(retain or []))]
        direction = text[1] - text[0]

        @torch.no_grad()
        def evaluate(pixels):
            features = self.images(views(pixels))
            change = features[0] - reference[0]
            globalgain = torch.dot(change, direction)
            localgain = ((features[1:] - reference[1:]) @ text[2] * regionweights).sum() if boxes else globalgain
            score = (globalgain + localgain) * 0.5
            directional = torch.dot(change, direction) / (change.norm() * direction.norm()).clamp_min(1e-6)
            directional = directional * (change.norm() / 0.05).clamp(max=1)
            self.evaluations += 1
            active().evaluations += 1
            individual = {"local" + str(index): float(value) for index, value in enumerate((features[1:] - reference[1:]) @ text[2])}
            components = {key: float(features[0] @ vector) for key, vector in zip(componentnames, text[3:])}
            return {"score": float(score), "global": float(globalgain), "local": float(localgain), "target": float(features[0] @ text[1]), "direction": float(directional), "localviews": len(boxes), **individual, **components}

        evaluate.evaluations = 0
        evaluate.boxes = tuple(boxes)
        active = ref(evaluate)
        return evaluate

    def stats(self):
        return {"evaluations": self.evaluations, "imagecalls": self.imagecalls, "imagesamples": self.imagesamples}

# Author: Weiliang Zhu 2026
# Email: wlzchina05@gmail.com
