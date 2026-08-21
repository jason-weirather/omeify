from __future__ import annotations

from abc import ABC, abstractmethod
from numbers import Integral
from pathlib import Path
from typing import Sequence

import numpy as np

from .channel import Channel
from .pixel_size import PixelSize


class Image(ABC):
    """Minimal interface shared by omeify image readers."""

    @property
    @abstractmethod
    def path(self) -> Path:
        """Path backing the image."""

    @property
    @abstractmethod
    def axes(self) -> str:
        """Axis order used by the selected full-resolution image series."""

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Shape of the selected full-resolution image series."""

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """Native pixel dtype."""

    @property
    @abstractmethod
    def pixel_size(self) -> PixelSize | None:
        """Physical X/Y pixel size, or None when the source is uncalibrated."""

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
    """Generic interface for images with one or more logical OME channels."""

    @property
    @abstractmethod
    def channels(self) -> tuple[Channel, ...]:
        """Ordered lazy logical channels."""

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(channel.name for channel in self.channels)

    @property
    def logical_channel_names(self) -> tuple[str, ...]:
        return self.channel_names

    @property
    def logical_channel_count(self) -> int:
        return len(self.channels)

    @property
    def channel_count(self) -> int:
        return self.logical_channel_count

    @property
    def samples_per_pixel(self) -> int:
        values = {channel.samples_per_pixel for channel in self.channels}
        return values.pop() if len(values) == 1 else 1

    @property
    def sample_count(self) -> int:
        return sum(channel.samples_per_pixel for channel in self.channels)

    @property
    def sample_names(self) -> tuple[str, ...]:
        if all(channel.samples_per_pixel == 1 for channel in self.channels):
            return self.channel_names
        return tuple(f"Sample {index + 1}" for index in range(self.sample_count))

    def __getitem__(self, index: int) -> Channel:
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise TypeError("Image channel lookup requires an integer index")
        normalized = int(index)
        if normalized < 0 or normalized >= len(self.channels):
            raise IndexError(
                f"Channel {normalized} is outside the available range 0..{len(self.channels) - 1}"
            )
        return self.channels[normalized]

    def get_by_name(self, name: str) -> Channel:
        matches = [channel for channel in self.channels if channel.name == name]
        if not matches:
            raise KeyError(f"No channel has name {name!r}")
        if len(matches) > 1:
            indices = ", ".join(str(channel.index) for channel in matches)
            raise ValueError(
                f"Channel name {name!r} is ambiguous; it occurs at indices {indices}"
            )
        return matches[0]

    def get_by_id(self, channel_id: str) -> Channel:
        matches = [channel for channel in self.channels if channel.id == channel_id]
        if not matches:
            raise KeyError(f"No channel has ID {channel_id!r}")
        if len(matches) > 1:
            raise ValueError(f"Channel ID {channel_id!r} is not unique")
        return matches[0]


class RGBImage(MultichannelImage):
    """Generic interface for one contiguous three-sample RGB logical channel."""

    # Class-level constants preserve useful introspection on the abstract type.
    channel_names: tuple[str] = ("RGB",)
    sample_names: tuple[str, str, str] = ("Red", "Green", "Blue")
    samples_per_pixel: int = 3


class LabelImage(Image):
    """Generic interface for one categorical label raster.

    The interface intentionally exposes image access only. It does not infer
    cell-versus-tissue semantics, compute region properties, or promise a label
    count. An exact generic label count requires scanning the raster and cannot
    be obtained reliably from ``max(label)`` when IDs are sparse.
    """

    background_label: int = 0

    def validate_label_dtype(self) -> None:
        if not np.issubdtype(self.dtype, np.integer):
            raise TypeError(f"Label images require an integer dtype, found {self.dtype}")


ChannelSelection = int | Sequence[int] | None
