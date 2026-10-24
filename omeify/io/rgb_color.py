"""Canonical RGB color-management boundary for all output writers.

Untrusted ICC bytes may be used as an input color transform, but are never embedded
in generated images. A single audited sRGB ICC is emitted in both TIFF and Zarr.
Color transforms are applied regionally before pyramid calculation and compression.
"""
from __future__ import annotations

import base64
import hashlib
import io
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace

import numpy as np
from PIL import Image as PILImage
from PIL import ImageCms

from .base import Image, RGBImage
from .image_views import BorrowedSource

# Generated from LittleCMS's standard sRGB profile, then fixed to a known creation
# timestamp/platform/profile-ID. Reviewed, packaged bytes; never build a profile
# from arbitrary vendor metadata. Altering this constant is a contract change.
_CANONICAL_BASE64 = (
    'AAACTGxjbXMEQAAAbW50clJHQiBYWVogB+QAAQABAAAAAAAAYWNzcAAAAAAAAAAAAAAAAAAAAAAA'
    'AAAAAAAAAAAAAAAAAPbWAAEAAAAA0y1sY21zAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
    'AAAAAAAAAAAAAAAAAAAAAAALZGVzYwAAAQgAAAA2Y3BydAAAAUAAAABMd3RwdAAAAYwAAAAUY2hh'
    'ZAAAAaAAAAAsclhZWgAAAcwAAAAUYlhZWgAAAeAAAAAUZ1hZWgAAAfQAAAAUclRSQwAAAggAAAAg'
    'Z1RSQwAAAggAAAAgYlRSQwAAAggAAAAgY2hybQAAAigAAAAkbWx1YwAAAAAAAAABAAAADGVuVVMA'
    'AAAaAAAAHABzAFIARwBCACAAYgB1AGkAbAB0AC0AaQBuAABtbHVjAAAAAAAAAAEAAAAMZW5VUwAA'
    'ADAAAAAcAE4AbwAgAGMAbwBwAHkAcgBpAGcAaAB0ACwAIAB1AHMAZQAgAGYAcgBlAGUAbAB5WFla'
    'IAAAAAAAAPbWAAEAAAAA0y1zZjMyAAAAAAABDEIAAAXe///zJQAAB5MAAP2Q///7of///aIAAAPc'
    'AADAblhZWiAAAAAAAABvoAAAOPUAAAOQWFlaIAAAAAAAACSfAAAPhAAAtsNYWVogAAAAAAAAYpcA'
    'ALeHAAAY2XBhcmEAAAAAAAMAAAACZmYAAPKnAAANWQAAE9AAAApbY2hybQAAAAAAAwAAAACj1wAA'
    'VHsAAEzNAACZmgAAJmYAAA9c'
)
SRGB_ICC = base64.b64decode(_CANONICAL_BASE64)
SRGB_SHA256 = "73218bf58875dd3ada4c5dd253f87df83eb5960bb9c675a066ad02240364ea79"
MAX_SOURCE_ICC_BYTES = 1024 * 1024


def _source_profile(value: bytes) -> ImageCms.ImageCmsProfile:
    # Preflight *before* giving untrusted bytes to LittleCMS. The ICC header
    # declares its own length and has a required acsp signature.
    if (not 128 <= len(value) <= MAX_SOURCE_ICC_BYTES
            or int.from_bytes(value[:4], "big") != len(value)
            or value[36:40] != b"acsp"):
        raise ValueError("Invalid or oversized source RGB ICC profile")
    try:
        profile = ImageCms.ImageCmsProfile(io.BytesIO(value))
        if profile.profile.xcolor_space.strip() != "RGB":
            raise ValueError("Non-RGB ICC")
        return profile
    except (ValueError, OSError, TypeError, ImageCms.PyCMSError):
        raise ValueError("Unsupported or invalid source RGB ICC profile") from None


class _CanonicalRGBSource(BorrowedSource):
    """Borrow a fixed RGB observation, mapping source colors to sRGB on demand."""

    def __init__(self, image: Image) -> None:
        if image.image_type != "rgb":
            raise TypeError("RGB color management requires an RGB image")
        original = image.icc_profile
        if original is None:
            source_mode = "assumed_srgb"
        elif original == SRGB_ICC:
            source_mode = "canonical_srgb"
        else:
            source_mode = "converted_icc"
        self._transform = None
        if source_mode == "converted_icc":
            source = _source_profile(original)
            try:
                destination = ImageCms.ImageCmsProfile(io.BytesIO(SRGB_ICC))
                self._transform = ImageCms.buildTransformFromOpenProfiles(
                    source, destination, "RGB", "RGB",
                    renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                )
            except (OSError, ValueError, TypeError, ImageCms.PyCMSError):
                raise ValueError("Source RGB ICC color transform is unsupported") from None
        self.source_mode = source_mode
        self._image = image
        super().__init__(replace(image._metadata, icc_profile=SRGB_ICC), (image,))

    def _read_region(self, y0: int, y1: int, x0: int, x1: int, *, level: int,
                     channels: tuple[int, ...]) -> np.ndarray:
        data = self._image.read_region(y0, y1, x0, x1, level=level, channels=channels)
        if self._transform is None or not data.size:
            return data
        # ImageSource reads run on the calling thread; tile encoders receive
        # detached pixels. Never change the input image or entire slide eagerly.
        block = PILImage.fromarray(np.ascontiguousarray(data))
        try:
            converted = ImageCms.applyTransform(block, self._transform)
        except (OSError, ValueError, TypeError, ImageCms.PyCMSError):
            # Do not let an untrusted profile's descriptive text enter an error.
            raise ValueError("Source RGB ICC color conversion failed") from None
        return np.asarray(converted).copy()


def color_management_report(image: Image) -> dict[str, object] | None:
    if image.image_type != "rgb":
        return None
    source = image._source
    if not isinstance(source, _CanonicalRGBSource):
        raise RuntimeError("RGB output bypassed canonical color management")
    return {
        "output_space": "sRGB",
        "output_profile_sha256": SRGB_SHA256,
        "source_profile": source.source_mode,
        "source_profile_present": source.source_mode != "assumed_srgb",
        "pixels_transformed": source.source_mode == "converted_icc",
        "rendering_intent": (
            "relative_colorimetric" if source.source_mode == "converted_icc" else None
        ),
    }


@contextmanager
def canonical_rgb(image: Image) -> Iterator[Image]:
    """Normalize an Image for writing; return non-RGB images unchanged."""
    image._ensure_open()
    if image.image_type != "rgb":
        yield image
    else:
        with RGBImage(_CanonicalRGBSource(image)) as normalized:
            yield normalized


def check_canonical_icc(value: bytes | None) -> None:
    """Defense in depth on serialized RGB metadata, beyond public writers."""
    if value != SRGB_ICC:
        raise ValueError("RGB output must contain Omeify's canonical sRGB ICC profile")


if hashlib.sha256(SRGB_ICC).hexdigest() != SRGB_SHA256:
    raise RuntimeError("Bundled sRGB profile does not match its pinned SHA-256")
