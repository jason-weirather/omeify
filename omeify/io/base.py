"""Semantic images backed by any ImageSource, plus the existing reader interfaces."""
from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from pathlib import Path
from typing import Any, TypeVar, cast

import numpy as np

from .channel import Channel
from .image_metadata import ImageLevel, ImageMetadata
from .image_source import ImageSource
from .pixel_size import PixelSize
from .tiff import PlaneReader

ImageT = TypeVar("ImageT", bound="Image")


class Image:
    """Storage-independent image facade. No path or display state is required.

    Use ``with Image(source)`` or ``open()`` for source-backed images. Sources
    are owned by default; ``owns_source=False`` borrows an already-open source.
    Readers retain their established open/close API. ``from_array`` returns an
    already-open image for ordinary notebook use; it also supports ``with``.
    """

    _accepted_types = frozenset({"multichannel", "rgb", "label"})
    _array_type = "multichannel"

    def __init__(self, source: ImageSource, *, owns_source: bool = True) -> None:
        if not isinstance(source, ImageSource):
            raise TypeError("source must implement the ImageSource base contract")
        if not isinstance(owns_source, bool):
            raise TypeError("owns_source must be a boolean")
        self._wrapped_source = source
        self._owns_source = owns_source
        self._image_active = False
        self._image_context = False
        self._image_epoch = 0
        self._source_epoch = -1
        self._image_channels: tuple[Channel, ...] | None = None

    @property
    def source(self) -> ImageSource:
        """Pixel backend; file readers expose a session-bound borrowing adapter."""
        return self._wrapped_source

    @property
    def metadata(self) -> ImageMetadata:
        self._ensure_available()
        return self.source.metadata

    @property
    def axes(self) -> str:
        return self.metadata.axes

    @property
    def shape(self) -> tuple[int, ...]:
        return self.metadata.shape

    @property
    def output_axes(self) -> str:
        return self.metadata.axes

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self.metadata.shape

    @property
    def size_c(self) -> int:
        return self.metadata.sample_count

    @property
    def dtype(self) -> np.dtype:
        return self.metadata.dtype

    @property
    def image_type(self) -> str:
        return self.metadata.image_type

    @property
    def is_rgb(self) -> bool:
        return self.image_type == "rgb"

    @property
    def is_label(self) -> bool:
        return self.image_type == "label"

    @property
    def pixel_size(self) -> PixelSize | None:
        return self.metadata.pixel_size

    @property
    def icc_profile(self) -> bytes | None:
        return self.metadata.icc_profile

    @property
    def level_descriptors(self) -> tuple[ImageLevel, ...]:
        """Backend-neutral levels. Legacy reader.levels keeps its TIFF objects."""
        return self.metadata.levels

    @property
    def levels(self) -> tuple[ImageLevel, ...]:
        return self.level_descriptors

    @property
    def level_count(self) -> int:
        return len(self.level_descriptors)

    @property
    def is_pyramidal(self) -> bool:
        return self.level_count > 1

    def level_shape(self, level: int = 0) -> tuple[int, ...]:
        return self.metadata.level(level).shape

    def pixel_size_at_level(self, level: int = 0) -> PixelSize | None:
        return self.metadata.level(level).pixel_size

    def level_downsample(self, level: int = 0) -> tuple[float, float] | None:
        """Known Y/X sampling scale, or None; never inferred from rounded shapes."""
        return self.metadata.level(level).downsample_yx

    @property
    def backing_paths(self) -> tuple[Path, ...]:
        return self.source.backing_paths

    @property
    def is_open(self) -> bool:
        return (self._image_active and self.source.is_open
                and self.source.generation == self._source_epoch)

    @property
    def closed(self) -> bool:
        return not self.is_open

    def open(self: ImageT) -> ImageT:
        if self._image_active:
            self._ensure_available()
            return self
        opened_here = not self.source.is_open
        if self._owns_source:
            self.source.open()
        elif not self.source.is_open:
            raise RuntimeError("Borrowed source must already be open")
        try:
            metadata = self.source.metadata
            if metadata.image_type not in self._accepted_types:
                raise TypeError(f"{type(self).__name__} cannot wrap {metadata.image_type!r} pixels")
        except BaseException:
            if self._owns_source and opened_here:
                self.source.close()
            raise
        self._source_epoch = self.source.generation
        self._image_active = True
        self._image_epoch += 1
        if isinstance(self, LabelImage):
            self.background_label = metadata.background_label
        if isinstance(self, RGBImage):
            # Preserve old class constants without discarding a supplied logical name.
            self.channel_names = metadata.channel_names
        return self

    def close(self) -> None:
        if not self._image_active:
            return
        if self._image_channels is not None:
            for channel in self._image_channels:
                channel.clear_cache()
        self._image_channels = None
        self._image_active = False
        self._image_epoch += 1
        if self._owns_source and self.source.generation == self._source_epoch:
            self.source.close()

    def _ensure_available(self) -> None:
        if not self.is_open:
            raise RuntimeError(
                "Image is closed or its source session changed; use 'with' or open()"
            )

    def _ensure_channel_session(self, epoch: int) -> None:
        self._ensure_available()
        if self._image_epoch != epoch:
            raise RuntimeError("Channel belongs to an earlier image session; acquire it again")

    def read_region(
        self, y0: int, y1: int, x0: int, x1: int, *,
        level: int = 0, channels: ChannelSelection = None,
    ) -> np.ndarray:
        self._ensure_available()
        return self.source.read_region(y0, y1, x0, x1, level=level, channels=channels)

    def asarray(self, *, level: int = 0) -> np.ndarray:
        """Explicitly materialize a whole level. No implicit __array__ conversion."""
        height, width = self.metadata.level(level).spatial_shape
        return self.read_region(0, height, 0, width, level=level)

    def clear_cache(self) -> None:
        self._ensure_available()
        self.source.clear_cache()

    def plane_readers(self, *, cache_mib: int = 64) -> list[PlaneReader]:
        """Borrowed adapters for the established streaming writer boundary."""
        from .image_planes import ImagePlaneSource
        return ImagePlaneSource(self).plane_readers(cache_mib=cache_mib)

    def as_image(self) -> Image:
        """Return an opened semantic facade borrowing this already-open image/reader."""
        from .reader_source import ReaderImageSource
        return Image.from_source(ReaderImageSource(self, owns_reader=False))

    @classmethod
    def from_source(
        cls: type[ImageT], source: ImageSource, *, owns_source: bool = True,
    ) -> ImageT:
        """Open a source without reading pixels; select semantic type when cls is Image."""
        if not isinstance(source, ImageSource):
            raise TypeError("source must implement ImageSource")
        if not isinstance(owns_source, bool):
            raise TypeError("owns_source must be a boolean")
        opened_here = not source.is_open
        if owns_source:
            source.open()
        elif not source.is_open:
            raise RuntimeError("Borrowed source must already be open")
        try:
            target = cls
            if cls is Image:
                target = {"multichannel": MultichannelImage, "rgb": RGBImage,
                          "label": LabelImage}[source.metadata.image_type]
            return cast(ImageT, target(source, owns_source=owns_source).open())
        except BaseException:
            if owns_source and opened_here:
                source.close()
            raise

    @classmethod
    def from_array(
        cls: type[ImageT], array: np.ndarray, *, axes: str | None = None,
        channel_names: Sequence[str] | None = None, pixel_size: PixelSize | None = None,
        copy: bool = False, **source_options: Any,
    ) -> ImageT:
        """Wrap a NumPy array. A 3D multichannel array is CYX, never guessed RGB."""
        from .array_source import ArraySource
        if not isinstance(array, np.ndarray):
            raise TypeError("from_array requires a NumPy ndarray")
        if axes is None:
            axes = "YXS" if cls._array_type == "rgb" else "YX" if array.ndim == 2 else "CYX"
        source = ArraySource(
            array, axes=axes, image_type=cls._array_type, channel_names=channel_names,
            pixel_size=pixel_size, copy=copy, **source_options,
        )
        return cls.from_source(source)

    def __enter__(self: ImageT) -> ImageT:
        if self._image_context:
            raise RuntimeError("Nested contexts on the same image are not supported")
        self.open()
        self._image_context = True
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._image_context = False
        self.close()

    def __repr__(self) -> str:
        if not self.is_open:
            return f"{type(self).__name__}(closed=True)"
        return (f"{type(self).__name__}(axes={self.axes!r}, shape={self.shape}, "
                f"dtype={self.dtype}, levels={self.level_count}, pixel_size={self.pixel_size!r})")


class MultichannelImage(Image):
    """Logical channels over any compatible source, including interleaved RGB."""

    _accepted_types = frozenset({"multichannel", "rgb"})

    @property
    def channels(self) -> tuple[Channel, ...]:
        self._ensure_available()
        if self._image_channels is None:
            from .image_planes import ImagePlaneReader
            metadata = self.metadata
            height, width = metadata.levels[0].spatial_shape
            shape = (height, width, 3) if self.is_rgb else (height, width)
            epoch = self._image_epoch
            self._image_channels = tuple(
                Channel(
                    index=i, channel_id=cid if cid is not None else f"Channel:0:{i}", name=name,
                    dtype=metadata.dtype, shape=shape, samples_per_pixel=metadata.samples_per_pixel,
                    plane_reader_factory=lambda level, i=i: ImagePlaneReader(self, i, level=level),
                    ensure_available=lambda: self._ensure_channel_session(epoch),
                    id_is_generated=cid is None, source_id=cid,
                ) for i, (name, cid) in enumerate(zip(metadata.channel_names, metadata.channel_ids))
            )
        return self._image_channels

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
        if self.is_rgb:
            return ("Red", "Green", "Blue")
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
    """Three interleaved RGB samples, not three independently stained channels."""

    _accepted_types = frozenset({"rgb"})
    _array_type = "rgb"
    channel_names: tuple[str, ...] = ("RGB",)
    sample_names: tuple[str, str, str] = ("Red", "Green", "Blue")
    samples_per_pixel: int = 3


class LabelImage(Image):
    """One categorical integer raster, without biological/object-count assumptions."""

    _accepted_types = frozenset({"label"})
    _array_type = "label"
    background_label: int = 0

    @property
    def image_type(self) -> str:
        return "label"

    def validate_label_dtype(self) -> None:
        if not np.issubdtype(self.dtype, np.integer):
            raise TypeError(f"Label images require an integer dtype, found {self.dtype}")


ChannelSelection = int | Sequence[int] | None
