"""One semantic image interface, independent of its pixel provider."""
from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from pathlib import Path
from typing import TypeVar

import numpy as np

from .channel import Channel
from .image_metadata import ImageLevel, ImageMetadata
from .image_source import ChannelSelection, ImageSource
from .pixel_size import PixelSize
from .spec import ImageType

ImageT = TypeVar("ImageT", bound="Image")


class Image:
    """A read-only observation with an explicit resource lifetime.

    Construction does not open sources or read pixels. Enter a context or call
    open(). By default the image owns its source; a borrowed source must already
    be open. Metadata is exposed on the image, not through a second public record.
    """

    _accepted_types = frozenset({"multichannel", "rgb", "label"})
    _array_type = "multichannel"

    def __init__(self, source: ImageSource, *, owns_source: bool = True) -> None:
        if not isinstance(source, ImageSource):
            raise TypeError("source must implement ImageSource")
        if not isinstance(owns_source, bool):
            raise TypeError("owns_source must be a boolean")
        self._source = source
        self._owns_source = owns_source
        self._active = False
        self._context_active = False
        self._generation = 0
        self._source_generation = -1
        self._channels: tuple[Channel, ...] | None = None

    @property
    def _metadata(self) -> ImageMetadata:
        self._ensure_open()
        return self._source.metadata

    @property
    def axes(self) -> str:
        return self._metadata.axes

    @property
    def shape(self) -> tuple[int, ...]:
        return self._metadata.shape

    @property
    def dtype(self) -> np.dtype:
        return self._metadata.dtype

    @property
    def image_type(self) -> ImageType:
        return self._metadata.image_type

    @property
    def pixel_size(self) -> PixelSize | None:
        return self._metadata.pixel_size

    @property
    def icc_profile(self) -> bytes | None:
        return self._metadata.icc_profile

    @property
    def levels(self) -> tuple[ImageLevel, ...]:
        """Backend-neutral resolution descriptors, including level zero."""
        return self._metadata.levels

    @property
    def channel_names(self) -> tuple[str, ...]:
        return self._metadata.channel_names

    @property
    def channel_count(self) -> int:
        """Number of logical channels; one for RGB, not three."""
        return self._metadata.logical_channel_count

    @property
    def samples_per_pixel(self) -> int:
        return self._metadata.samples_per_pixel

    @property
    def sample_count(self) -> int:
        return self._metadata.sample_count

    @property
    def sample_names(self) -> tuple[str, ...]:
        return ("Red", "Green", "Blue") if self.image_type == "rgb" else self.channel_names

    @property
    def backing_paths(self) -> tuple[Path, ...]:
        """Local pixel dependencies, used to prevent overwriting the source."""
        return self._source.backing_paths

    @property
    def is_open(self) -> bool:
        return (self._active and self._source.is_open
                and self._source.generation == self._source_generation)

    def open(self: ImageT) -> ImageT:
        if self._active:
            self._ensure_open()
            return self
        opened_here = not self._source.is_open
        if self._owns_source:
            self._source.open()
        elif not self._source.is_open:
            raise RuntimeError("Borrowed source must already be open")
        try:
            if self._source.metadata.image_type not in self._accepted_types:
                raise TypeError(
                    f"{type(self).__name__} cannot wrap {self._source.metadata.image_type!r} pixels"
                )
        except BaseException:
            if self._owns_source and opened_here:
                self._source.close()
            raise
        self._source_generation = self._source.generation
        self._active = True
        self._generation += 1
        return self

    def close(self) -> None:
        if self._active:
            self._active = False
            self._generation += 1
            self._channels = None
            if self._owns_source and self._source.generation == self._source_generation:
                self._source.close()

    def _ensure_open(self) -> None:
        if not self.is_open:
            raise RuntimeError(
                "Image is closed or its source session changed; use 'with' or open()",
            )

    def _ensure_session(self, generation: int) -> None:
        self._ensure_open()
        if self._generation != generation:
            raise RuntimeError("View belongs to an earlier image session; acquire it again")

    def read_region(
        self, y0: int, y1: int, x0: int, x1: int, *,
        level: int = 0, channels: ChannelSelection = None,
    ) -> np.ndarray:
        """Return owned pixels for half-open bounds in the selected level.

        Channels always mean logical channels. RGB channel zero includes all
        three samples. CYX reads retain their channel axis, including singleton C.
        """
        self._ensure_open()
        return self._source.read_region(y0, y1, x0, x1, level=level, channels=channels)

    def asarray(self, *, level: int = 0) -> np.ndarray:
        """Explicit whole-level materialization; ordinary metadata access is cheap."""
        height, width = self._metadata.level(level).spatial_shape
        return self.read_region(0, height, 0, width, level=level)

    def clear_cache(self) -> None:
        self._ensure_open()
        self._source.clear_cache()

    @property
    def channels(self) -> tuple[Channel, ...]:
        """Lazy logical-channel handles tied to this open image session."""
        self._ensure_open()
        if self._channels is None:
            self._channels = tuple(Channel(self, i) for i in range(self.channel_count))
        return self._channels

    def __getitem__(self, index: int) -> Channel:
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral):
            raise TypeError("Channel index must be an integer")
        index = int(index)
        if index < 0 or index >= self.channel_count:
            raise IndexError(f"Channel {index} is outside 0..{self.channel_count - 1}")
        return self.channels[index]

    def get_by_name(self, name: str) -> Channel:
        matches = [channel for channel in self.channels if channel.name == name]
        if not matches:
            raise KeyError(f"No channel has name {name!r}")
        if len(matches) != 1:
            raise ValueError(f"Channel name {name!r} is ambiguous; use an index or ID")
        return matches[0]

    def get_by_id(self, channel_id: str) -> Channel:
        for channel in self.channels:
            if channel.id == channel_id:
                return channel
        raise KeyError(f"No channel has ID {channel_id!r}")

    @classmethod
    def from_array(
        cls: type[ImageT], array: np.ndarray, *, axes: str | None = None,
        channel_names: Sequence[str] | None = None, pixel_size: PixelSize | None = None,
        copy: bool = False, icc_profile: bytes | None = None, background_label: int = 0,
        channel_ids: Sequence[str | None] | None = None,
        level_arrays: Sequence[np.ndarray] = (), levels: Sequence[ImageLevel] = (),
    ) -> ImageT:
        """Construct an unopened image around existing array storage.

        Borrowing is deliberate for large images. Keep borrowed arrays unchanged
        while the image is in use; copy=True takes an independent full snapshot.
        For multichannel data declare YX or CYX explicitly. RGB/labels fix their
        layout to YXS/YX. Supplied pyramids require explicit level descriptors.
        """
        from .array_source import ArraySource
        if axes is None:
            if cls._array_type == "multichannel":
                raise TypeError("Multichannel array construction requires axes='YX' or 'CYX'")
            axes = "YXS" if cls._array_type == "rgb" else "YX"
        return cls(ArraySource(
            array, axes=axes, image_type=cls._array_type, channel_names=channel_names,
            pixel_size=pixel_size, copy=copy, icc_profile=icc_profile,
            background_label=background_label, channel_ids=channel_ids,
            level_arrays=level_arrays, levels=levels,
        ))

    def with_metadata(
        self, *, channel_names: Sequence[str] | None = None,
        pixel_size: PixelSize | None = None, icc_profile: bytes | None = None,
    ) -> Image:
        """Create an unopened, borrowed observation with explicit metadata overrides.

        Pixels are unchanged. None keeps the existing field. A calibration override
        recalibrates levels with known sampling scales; unknown scales stay unknown.
        The parent must remain open until the returned image is finished.
        """
        from .image_views import MetadataSource
        return _semantic_image(MetadataSource(
            self, channel_names=channel_names, pixel_size=pixel_size, icc_profile=icc_profile,
        ))

    def __enter__(self: ImageT) -> ImageT:
        if self._context_active:
            raise RuntimeError("Nested contexts on the same image are not supported")
        self.open()
        self._context_active = True
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._context_active = False
        self.close()

    def __repr__(self) -> str:
        if not self.is_open:
            return f"{type(self).__name__}(is_open=False)"
        return (f"{type(self).__name__}(axes={self.axes!r}, shape={self.shape}, "
                f"dtype={self.dtype}, channels={self.channel_names}, "
                f"levels={len(self.levels)}, pixel_size={self.pixel_size!r})")


class MultichannelImage(Image):
    """An intensity image, including the single three-sample RGB logical channel."""

    _accepted_types = frozenset({"multichannel", "rgb"})

    @classmethod
    def from_channels(
        cls, channels: Sequence[Channel], *, level: int = 0,
        channel_names: Sequence[str] | None = None, pixel_size: PixelSize | None = None,
    ) -> MultichannelImage:
        """Assemble scalar lazy channels without reading their full arrays.

        All parent images must remain open. Selected channels must have matching
        dtype, spatial shape, and calibration unless explicitly recalibrated.
        The result is CYX with a single level, including a single selected channel.
        """
        from .image_views import ChannelSource
        return cls(ChannelSource(
            channels, level=level, channel_names=channel_names, pixel_size=pixel_size,
        ))


class RGBImage(MultichannelImage):
    """Three interleaved RGB samples, not independently stained channels."""

    _accepted_types = frozenset({"rgb"})
    _array_type = "rgb"


class LabelImage(Image):
    """One categorical integer raster; no inferred object-count or biology."""

    _accepted_types = frozenset({"label"})
    _array_type = "label"

    @property
    def background_label(self) -> int:
        return self._metadata.background_label


def _semantic_image(source: ImageSource) -> Image:
    cls = {"multichannel": MultichannelImage, "rgb": RGBImage, "label": LabelImage}
    return cls[source.metadata.image_type](source)
