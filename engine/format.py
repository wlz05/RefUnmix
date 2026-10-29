# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from numbers import Integral
import numpy as np
from PIL import Image, ImageOps

POLICY = {"version": 1, "color": "8-bit RGB; Pillow conversion without ICC transforms", "alpha": "white RGB matte", "frames": "single frame only", "orientation": "apply source EXIF to source and mask together", "resize": "center crop with RGB Lanczos and label-mask nearest", "controlmask": "bilinear pixel fit followed by latent area reduction", "mask": "8-bit grayscale; binary 0/1 expands to 0/255", "rle": "zero-based row-major start/length pairs; no clipping", "training": "float32 CHW in [-1,1]", "inference": "float32 NCHW in [0,1]"}
TRANSFORMS = {2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180, 4: Image.Transpose.FLIP_TOP_BOTTOM, 5: Image.Transpose.TRANSPOSE, 6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE, 8: Image.Transpose.ROTATE_90}

def check(image):
    if not isinstance(image, Image.Image) or min(image.size) < 1:
        raise ValueError("Expected a nonempty PIL image.")
    if getattr(image, "n_frames", 1) != 1:
        raise ValueError("Animated or multipage images require explicit frame export.")
    if image.mode not in {"1", "L", "LA", "P", "RGB", "RGBA", "CMYK", "YCbCr"}:
        raise ValueError("Convert high-depth or floating images to explicit 8-bit pixels first.")
    orientation = image.getexif().get(274, 1)
    if type(orientation) is not int or orientation not in {1, *TRANSFORMS}:
        raise ValueError("Unsupported EXIF image orientation.")
    image.load()
    return orientation

def rgb(image):
    # Preserve orientation until source and mask are aligned.
    orientation = check(image)
    transparent = "A" in image.getbands() or "transparency" in image.info
    pixels = Image.alpha_composite(Image.new("RGBA", image.size, (255, 255, 255, 255)), image.convert("RGBA")).convert("RGB") if transparent else image.convert("RGB")
    pixels.info.clear()
    if orientation != 1:
        pixels.getexif()[274] = orientation
    return pixels

def mask(image):
    # Reject ambiguous labels instead of guessing their meaning.
    if check(image) != 1:
        raise ValueError("Masks must use raw source coordinates without EXIF tags.")
    pixels = image.convert("RGBA")
    values = np.asarray(pixels, dtype=np.uint8)
    if np.any(values[..., 3] != 255) or np.any(values[..., 0] != values[..., 1]) or np.any(values[..., 0] != values[..., 2]):
        raise ValueError("Masks require opaque grayscale pixels, not colors or transparency.")
    values = values[..., 0].copy()
    if int(values.max()) <= 1:
        values *= 255
    return Image.fromarray(values)

def read(value, mode="RGB"):
    if mode not in {"RGB", "L"}:
        raise ValueError("Image loading supports RGB sources and grayscale masks.")
    with Image.open(value) as image:
        return rgb(image) if mode == "RGB" else mask(image)

def orient(image, region=None):
    orientation = check(image)
    if region is not None and (not isinstance(region, Image.Image) or region.size != image.size or check(region) != 1):
        raise ValueError("Masks must match raw source geometry without orientation tags.")
    region = mask(region) if region is not None else None
    output = ImageOps.exif_transpose(rgb(image))
    aligned = region.transpose(TRANSFORMS[orientation]) if region is not None and orientation in TRANSFORMS else region
    return output, aligned

def fit(image, size):
    if not isinstance(size, (list, tuple)) or len(size) != 2 or any(not isinstance(value, Integral) or isinstance(value, bool) or value < 1 for value in size):
        raise ValueError("Image sizes require positive width and height.")
    return ImageOps.fit(orient(image)[0], tuple(int(value) for value in size), method=Image.Resampling.LANCZOS)

def runs(encoded, shape):
    if not isinstance(shape, (tuple, list)) or len(shape) != 2 or any(not isinstance(value, Integral) or isinstance(value, bool) or value < 1 for value in shape):
        raise ValueError("Mask sizes require positive height and width.")
    if not isinstance(encoded, (tuple, list)) or len(encoded) % 2 or any(not isinstance(value, Integral) or isinstance(value, bool) or value < 0 for value in encoded):
        raise ValueError("RLE masks require nonnegative integer pairs.")
    total = int(shape[0]) * int(shape[1])
    if any(int(start) + int(length) > total for start, length in zip(encoded[::2], encoded[1::2])):
        raise ValueError("RLE mask runs exceed their declared geometry.")
    return total

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
