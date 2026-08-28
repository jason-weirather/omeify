from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from omeify.io.ome_tiff_writer import DownsampleMethod, PlaneReaderSource
from omeify.io.pixel_size import PixelSize
from omeify.io.spec import ImageType, OMEImageSpec
from omeify.io.tiff import ArrayPlaneReader, PlaneReader


class _ArraySeriesSource:
    def __init__(self, array: np.ndarray, spec: OMEImageSpec) -> None:
        self._array = np.asarray(array)
        self._spec = spec

    def plane_readers(self, *, cache_mib: int = 64) -> list[PlaneReader]:
        del cache_mib
        if self._spec.axes == "CYX":
            return [
                ArrayPlaneReader(self._array[index])
                for index in range(self._spec.plane_count)
            ]
        return [ArrayPlaneReader(self._array)]


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

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("OME series name must be a non-empty string")
        object.__setattr__(self, "name", self.name.strip())
        if not isinstance(self.source, PlaneReaderSource):
            raise TypeError("source must implement PlaneReaderSource")
        if not isinstance(self.spec, OMEImageSpec):
            raise TypeError("spec must be an OMEImageSpec")
        if self.downsample not in {"mean", "nearest"}:
            raise ValueError("downsample must be 'mean' or 'nearest'")
        if self.spec.is_label and self.downsample != "nearest":
            raise ValueError("Label-image series require nearest-neighbor downsampling")

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
        effective_downsample = downsample or (
            "nearest" if image_type == "label" else "mean"
        )
        return cls(
            name=name,
            source=_ArraySeriesSource(array, spec),
            spec=spec,
            downsample=effective_downsample,
        )

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
        effective_downsample = downsample or (
            "nearest" if image_type == "label" else "mean"
        )
        return cls(
            name=name,
            source=source,
            spec=spec,
            downsample=effective_downsample,
        )
