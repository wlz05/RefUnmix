# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from __future__ import annotations
from numbers import Integral
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from data import Record, records
from split import select
from engine.format import fit, orient, read, runs
from engine.reproduce import worker

def annotation(item: Record, size: int | None = None, border: bool = False) -> Image.Image | None:
    # Load raster or zero-based row-major masks.
    if size is not None and (not isinstance(size, Integral) or isinstance(size, bool) or size < 1):
        raise ValueError("Mask fitting requires a positive integer size.")
    if not isinstance(border, bool):
        raise ValueError("Mask border control must be boolean.")
    if item.maskpath is not None:
        image = read(item.maskpath, "L")
        if image.size != (item.masksize[1], item.masksize[0]):
            raise ValueError("Raster mask geometry differs from its metadata.")
        if border:
            values = np.asarray(image, dtype=np.uint8).copy()
            values[0, :] = values[-1, :] = 255
            values[:, 0] = values[:, -1] = 255
            image = Image.fromarray(values)
        return ImageOps.fit(image, (size, size), method=Image.Resampling.NEAREST) if size is not None else image
    if item.mask is None:
        return None
    height, width = item.masksize
    values = np.zeros(runs(item.mask, item.masksize), dtype=np.uint8)
    for start, length in zip(item.mask[::2], item.mask[1::2]):
        values[start:start + length] = 255
    values = values.reshape(height, width)
    if border:
        values[0, :] = values[-1, :] = 255
        values[:, 0] = values[:, -1] = 255
    image = Image.fromarray(values)
    return ImageOps.fit(image, (size, size), method=Image.Resampling.NEAREST) if size is not None else image

class EditDataset(Dataset):
    def __init__(self, items: list[Record], size: int = 512, paired: bool = False) -> None:
        if not isinstance(size, Integral) or isinstance(size, bool) or size < 8 or size % 8:
            raise ValueError("size must be a positive multiple of eight.")
        if not items:
            raise ValueError("The selected dataset is empty.")
        if paired and any(item.edited is None for item in items):
            raise ValueError("Paired training requires real target images for every record.")
        self.items = list(items)
        self.size = size
        self.paired = paired

    def __len__(self) -> int:
        return len(self.items)

    def tensor(self, path) -> torch.Tensor:
        fitted = fit(read(path), (self.size, self.size))
        return torch.from_numpy(np.asarray(fitted, dtype=np.float32).copy() / 127.5 - 1.0).permute(2, 0, 1)

    def __getitem__(self, index: int) -> dict:
        item = self.items[index]
        source = self.tensor(item.image)
        mask = annotation(item)
        if mask is not None:
            original = read(item.image)
            if item.maskpath is not None and mask.size != original.size:
                raise ValueError("Raster masks must match raw source geometry.")
            mask = mask.resize(original.size, Image.Resampling.NEAREST)
            _, mask = orient(original, mask)
            mask = ImageOps.fit(mask, (self.size, self.size), method=Image.Resampling.NEAREST)
        region = torch.zeros(1, self.size, self.size) if mask is None else torch.from_numpy(np.asarray(mask, dtype=np.float32).copy() / 255.0).unsqueeze(0)
        result = {"key": item.key, "image": source, "source": item.source, "target": item.target, "instruction": item.instruction, "category": item.category, "mask": region, "hasmask": mask is not None, "haspair": item.edited is not None}
        if self.paired:
            result["edited"] = self.tensor(item.edited)
        return result

def training(dataset: str, manifest: str, subset: str = "train", mapping: str = "./mapping_file.json", images: str = "./annotation_images", size: int = 512, paired: bool = False) -> EditDataset:
    if subset.strip().casefold() not in {"train", "val"}:
        raise ValueError("Training interfaces accept only train or val subsets.")
    return EditDataset(select(records(dataset, mapping, images), manifest, subset), size, paired)

def loader(dataset: EditDataset, batch: int = 1, shuffle: bool = True, seed: int = 17, workers: int = 0) -> DataLoader:
    if type(batch) is not int or batch < 1 or type(workers) is not int or workers < 0:
        raise ValueError("batch must be positive and workers must be nonnegative.")
    if not isinstance(shuffle, bool) or type(seed) is not int or not -(2 ** 63) <= seed < 2 ** 64:
        raise ValueError("shuffle must be boolean and seed must be a supported integer.")
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=workers, generator=torch.Generator().manual_seed(seed), worker_init_fn=worker)

# Author: Weiliang Zhu 2026
# Email: wlzchina05@gmail.com
