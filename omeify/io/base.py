from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import numpy as np


class Image(ABC):
    """Minimal interface shared by omeify image readers."""

    @property
    @abstractmethod
    def path(self) -> Path:
        """Path backing the image."""

    @property
    @abstractmethod
    def axes(self) -> str:
        """Axis order used by the selected image series."""

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Shape of the selected full-resolution image series."""

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """Native pixel dtype."""

    @abstractmethod
    def asarray(self, *, level: int = 0) -> np.ndarray:
        """Materialize one pyramid level as a NumPy array."""

    @abstractmethod
    def read_region(
        self,
        y0: int,
        y1: int,
        x0: int,
        x1: int,
        *,
        level: int = 0,
    ) -> np.ndarray:
        """Read a spatial region without materializing the whole image."""


class MultichannelImage(Image):
    """Generic interface for images with one or more logical OME channels.

    Logical channels and stored samples are deliberately distinct. A planar
    multiplex image normally has one sample per logical channel. RGB has one
    logical OME channel with three samples per pixel.
    """

    @property
    @abstractmethod
    def channel_names(self) -> tuple[str, ...]:
        """Logical OME channel names in image order."""

    @property
    def logical_channel_names(self) -> tuple[str, ...]:
        return self.channel_names

    @property
    def logical_channel_count(self) -> int:
        return len(self.logical_channel_names)

    @property
    def channel_count(self) -> int:
        """Compatibility alias for :attr:`logical_channel_count`."""

        return self.logical_channel_count

    @property
    def samples_per_pixel(self) -> int:
        return 1

    @property
    def sample_count(self) -> int:
        return self.logical_channel_count * self.samples_per_pixel

    @property
    def sample_names(self) -> tuple[str, ...]:
        if self.samples_per_pixel == 1:
            return self.logical_channel_names
        return tuple(f"Sample {index + 1}" for index in range(self.sample_count))


class RGBImage(MultichannelImage):
    """Generic interface for contiguous three-sample RGB images."""

    channel_names: tuple[str] = ("RGB",)
    sample_names: tuple[str, str, str] = ("Red", "Green", "Blue")
    samples_per_pixel: int = 3


class LabelImage(Image):
    """Generic interface for one categorical label raster.

    The interface intentionally exposes image access only. It does not infer
    cell-versus-tissue semantics, compute region properties, or promise a
    label count. An exact generic label count requires scanning the raster and
    cannot be obtained reliably from ``max(label)`` when IDs are sparse.
    """

    background_label: int = 0

    def validate_label_dtype(self) -> None:
        """Raise when the image dtype cannot represent integer labels."""

        if not np.issubdtype(self.dtype, np.integer):
            raise TypeError(f"Label images require an integer dtype, found {self.dtype}")


ChannelSelection = int | Sequence[int] | None
