from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from omeify.io._writer import (
    ArraySource,
    DownsampleMethod,
    PlaneReaderSource,
    resolve_downsample,
)
from omeify.io.base import Image
from omeify.io.image_planes import ImagePlaneSource
from omeify.io.pixel_size import PixelSize
from omeify.io.spec import ImageType, OMEImageSpec


@dataclass(frozen=True, slots=True)
class OMEImageSeries:
    """One named OME Image and the streaming source that supplies its pixels.

    A multi-series output may combine different scalar dtypes and image types.
    The OME ``Image/@Name`` is the canonical series identity used for reader and
    viewer navigation. The underlying :class:`OMEImageSpec` remains the single
    authoritative description of axes, shape, dtype, channels, and calibration.
    """

    name: str
    source: PlaneReaderSource
    spec: OMEImageSpec
    downsample: DownsampleMethod
    compression: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("OME series name must be a non-empty string")
        object.__setattr__(self, "name", self.name.strip())
        if not isinstance(self.source, PlaneReaderSource):
            raise TypeError("source must implement PlaneReaderSource")
        if not isinstance(self.spec, OMEImageSpec):
            raise TypeError("spec must be an OMEImageSpec")
        object.__setattr__(
            self,
            "downsample",
            resolve_downsample(self.spec.image_type, self.downsample),
        )
        if self.compression is not None:
            if not isinstance(self.compression, str) or not self.compression.strip():
                raise ValueError("series compression must be a non-empty string or None")
            object.__setattr__(self, "compression", self.compression.strip())

    @classmethod
    def from_array(
        cls,
        name: str,
        image: np.ndarray,
        *,
        image_type: ImageType,
        channel_names: Sequence[str] | None,
        pixel_size: PixelSize,
        downsample: DownsampleMethod | None = None,
        compression: str | None = None,
        icc_profile: bytes | None = None,
    ) -> OMEImageSeries:
        """Create one series from a canonical in-memory or memory-mapped array."""

        array = np.asarray(image)
        if image_type == "rgb":
            axes = "YXS"
        elif image_type == "label":
            axes = "YX"
        elif array.ndim == 2:
            axes = "YX"
        elif array.ndim == 3:
            axes = "CYX"
        else:
            raise ValueError(
                "Multichannel series arrays must use canonical YX or CYX shape; "
                f"received {array.shape}"
            )
        spec = OMEImageSpec.from_shape(
            image_type=image_type,
            axes=axes,
            shape=array.shape,
            dtype=array.dtype,
            channel_names=channel_names,
            pixel_size=pixel_size,
            icc_profile=icc_profile,
        )
        return cls(
            name=name,
            source=ArraySource(array, spec),
            spec=spec,
            downsample=resolve_downsample(image_type, downsample),
            compression=compression,
        )

    @classmethod
    def from_image(
        cls, name: str, image: Image, *, level: int = 0,
        channel_names: Sequence[str] | None = None, pixel_size: PixelSize | None = None,
        downsample: DownsampleMethod | None = None, compression: str | None = None,
        icc_profile: bytes | None = None,
    ) -> OMEImageSeries:
        """Borrow any open semantic image for one heterogeneous output series.

        No pixel reads occur here. Keep the image open until write() completes.
        All semantic and calibrated metadata comes from the selected image level.
        """
        source = ImagePlaneSource(image, level=level)
        spec = source.output_spec(
            channel_names=channel_names, pixel_size=pixel_size, icc_profile=icc_profile,
        )
        return cls(name, source, spec, resolve_downsample(spec.image_type, downsample), compression)

    @classmethod
    def from_source(
        cls,
        name: str,
        source: PlaneReaderSource,
        *,
        image_type: ImageType,
        axes: str,
        shape: Sequence[int],
        dtype: np.dtype | str | type,
        channel_names: Sequence[str] | None,
        pixel_size: PixelSize,
        downsample: DownsampleMethod | None = None,
        compression: str | None = None,
        icc_profile: bytes | None = None,
    ) -> OMEImageSeries:
        """Create one series from an explicit random-access plane source."""

        spec = OMEImageSpec.from_shape(
            image_type=image_type,
            axes=axes,
            shape=shape,
            dtype=dtype,
            channel_names=channel_names,
            pixel_size=pixel_size,
            icc_profile=icc_profile,
        )
        return cls(
            name=name,
            source=source,
            spec=spec,
            downsample=resolve_downsample(image_type, downsample),
            compression=compression,
        )
