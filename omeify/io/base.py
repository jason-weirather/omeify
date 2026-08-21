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
    """Generic interface for images with one or more logical channels."""

    @property
    @abstractmethod
    def channel_names(self) -> tuple[str, ...]:
        """Logical channel names in image order."""

    @property
    def channel_count(self) -> int:
        return len(self.channel_names)


class RGBImage(Image):
    """Generic interface for contiguous three-sample RGB images."""

    channel_names: tuple[str, str, str] = ("Red", "Green", "Blue")
    samples_per_pixel: int = 3


class LabelImage(Image):
    """Generic interface for categorical label rasters."""

    background_label: int = 0

    def validate_label_dtype(self) -> None:
        """Raise when the image dtype cannot represent integer labels."""

        if not np.issubdtype(self.dtype, np.integer):
            raise TypeError(f"Label images require an integer dtype, found {self.dtype}")


ChannelSelection = int | Sequence[int] | None
