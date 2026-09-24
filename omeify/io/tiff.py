from __future__ import annotations

import math
import threading
from collections import OrderedDict
from typing import Protocol, runtime_checkable

import numpy as np
import tifffile


@runtime_checkable
class PlaneReader(Protocol):
    """Minimal random-access contract consumed by :class:`OMETiffWriter`."""

    height: int
    width: int
    dtype: np.dtype
    samples_per_pixel: int

    def read_region(self, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """Read a Y/X region from one grayscale or interleaved plane."""

    def clear_cache(self) -> None:
        """Release optional decoded-segment caches."""


class TiffPlaneReader:
    """Random-access reader for one grayscale or interleaved RGB TIFF page.

    Only intersecting tiles or strips are decoded. A small LRU cache prevents
    repeated decoding when the requested output tile grid does not exactly
    match the source grid.
    """

    def __init__(
        self,
        page: tifffile.TiffPage,
        *,
        lock: threading.RLock | None = None,
        cache_mib: int = 64,
    ) -> None:
        self.samples_per_pixel = int(page.samplesperpixel)
        if self.samples_per_pixel not in {1, 3}:
            raise ValueError(
                "omeify supports one grayscale sample or three interleaved RGB samples "
                f"per TIFF page; found SamplesPerPixel={self.samples_per_pixel}."
            )
        if int(getattr(page, "imagedepth", 1)) != 1:
            raise ValueError("TiffPlaneReader requires a two-dimensional TIFF plane")
        if self.samples_per_pixel > 1 and int(page.planarconfig) != 1:
            raise ValueError(
                "RGB input must use contiguous samples (PlanarConfiguration=1); "
                f"found PlanarConfiguration={int(page.planarconfig)}."
            )

        self.page = page
        self.filehandle = page.parent.filehandle
        self.lock = lock or threading.RLock()
        self.height = int(page.imagelength)
        self.width = int(page.imagewidth)
        self.dtype = np.dtype(page.dtype).newbyteorder("=")
        self._cache: OrderedDict[int, tuple[np.ndarray, int, int]] = OrderedDict()

        if page.is_tiled:
            self.segment_height = int(page.tilelength)
            self.segment_width = int(page.tilewidth)
            self.segments_across = math.ceil(self.width / self.segment_width)
        else:
            self.segment_height = int(page.rowsperstrip or self.height)
            self.segment_width = self.width
            self.segments_across = 1

        nominal_segment_bytes = max(
            1,
            self.segment_height
            * self.segment_width
            * self.samples_per_pixel
            * max(1, self.dtype.itemsize),
        )
        self.cache_capacity = max(
            1,
            min(128, (max(1, cache_mib) * 1024 * 1024) // nominal_segment_bytes),
        )

    @property
    def shape(self) -> tuple[int, ...]:
        if self.samples_per_pixel == 1:
            return self.height, self.width
        return self.height, self.width, self.samples_per_pixel

    def clear_cache(self) -> None:
        self._cache.clear()

    def _segment_index(self, segment_y: int, segment_x: int) -> int:
        return segment_y * self.segments_across + segment_x

    def _normalize_decoded(self, decoded: np.ndarray) -> np.ndarray:
        array = np.asarray(decoded)
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]

        if self.samples_per_pixel == 1:
            # tifffile decodes segments as (depth, Y, X, samples). After
            # removing depth above, a grayscale sample axis is ALWAYS last.
            # A singleton Y is real image geometry, notably one-row strips.
            if array.ndim == 3 and array.shape[-1] == 1:
                array = array[..., 0]
            if array.ndim != 2:
                raise ValueError(f"Unsupported decoded grayscale TIFF segment shape {array.shape}")
            return array

        if array.ndim != 3 or array.shape[-1] != self.samples_per_pixel:
            raise ValueError(f"Unsupported decoded RGB TIFF segment shape {array.shape}")
        return array

    def _empty_segment(self, height: int, width: int) -> np.ndarray:
        shape = (height, width)
        if self.samples_per_pixel > 1:
            shape += (self.samples_per_pixel,)
        return np.zeros(shape, dtype=self.dtype)

    def _decode_segment(self, index: int) -> tuple[np.ndarray, int, int]:
        cached = self._cache.get(index)
        if cached is not None:
            self._cache.move_to_end(index)
            return cached

        try:
            offset = int(self.page.dataoffsets[index])
            bytecount = int(self.page.databytecounts[index])
        except IndexError as exc:
            raise IndexError(
                f"TIFF segment index {index} is outside page segment table "
                f"({len(self.page.dataoffsets)} segments)"
            ) from exc

        encoded: bytes | None
        if offset == 0 or bytecount == 0:
            encoded = None
        else:
            with self.lock:
                self.filehandle.seek(offset)
                encoded = self.filehandle.read(bytecount)
            if len(encoded) != bytecount:
                raise OSError(
                    f"Short read for TIFF segment {index}: expected {bytecount} bytes, "
                    f"received {len(encoded)}"
                )

        decoded, indices, _shape = self.page.decode(
            encoded,
            index,
            jpegtables=self.page.jpegtables,
        )

        if indices is not None and len(indices) >= 4:
            origin_y = int(indices[2])
            origin_x = int(indices[3])
        else:
            segment_y, segment_x = divmod(index, self.segments_across)
            origin_y = segment_y * self.segment_height
            origin_x = segment_x * self.segment_width

        segment_y, segment_x = divmod(index, self.segments_across)
        expected_origin = (segment_y * self.segment_height, segment_x * self.segment_width)
        if (origin_y, origin_x) != expected_origin:
            raise ValueError(f"TIFF segment {index} has an unexpected spatial origin")
        valid_h = min(self.segment_height, self.height - origin_y)
        valid_w = min(self.segment_width, self.width - origin_x)
        if valid_h <= 0 or valid_w <= 0:
            raise ValueError(f"TIFF segment {index} lies outside the plane")
        if decoded is None:
            if encoded is not None:
                raise ValueError(f"TIFF segment {index} has data but decoded to no pixels")
            array = self._empty_segment(valid_h, valid_w)
        else:
            array = self._normalize_decoded(decoded)
            # Never silently leave zeros where a truncated/misinterpreted
            # segment failed to cover its promised in-bounds rectangle.
            if array.shape[0] < valid_h or array.shape[1] < valid_w:
                raise ValueError(
                    f"TIFF segment {index} decoded shape {array.shape} does not cover "
                    f"its in-bounds rectangle {(valid_h, valid_w)}"
                )
            if array.dtype != self.dtype:
                array = array.astype(self.dtype, copy=False)
            array = np.ascontiguousarray(array[:valid_h, :valid_w, ...])

        value = (array, origin_y, origin_x)
        self._cache[index] = value
        self._cache.move_to_end(index)
        while len(self._cache) > self.cache_capacity:
            self._cache.popitem(last=False)
        return value

    def read_region(self, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        if not (0 <= y0 <= y1 <= self.height and 0 <= x0 <= x1 <= self.width):
            raise ValueError(
                f"Requested region {(y0, y1, x0, x1)} is outside page shape {self.shape}"
            )
        output = self._empty_segment(y1 - y0, x1 - x0)
        if output.size == 0:
            return output

        first_segment_y = y0 // self.segment_height
        last_segment_y = (y1 - 1) // self.segment_height
        first_segment_x = x0 // self.segment_width
        last_segment_x = (x1 - 1) // self.segment_width

        for segment_y in range(first_segment_y, last_segment_y + 1):
            for segment_x in range(first_segment_x, last_segment_x + 1):
                index = self._segment_index(segment_y, segment_x)
                segment, seg_y0, seg_x0 = self._decode_segment(index)
                seg_y1 = seg_y0 + segment.shape[0]
                seg_x1 = seg_x0 + segment.shape[1]

                copy_y0 = max(y0, seg_y0)
                copy_y1 = min(y1, seg_y1)
                copy_x0 = max(x0, seg_x0)
                copy_x1 = min(x1, seg_x1)
                if copy_y0 >= copy_y1 or copy_x0 >= copy_x1:
                    continue

                output[
                    copy_y0 - y0 : copy_y1 - y0,
                    copy_x0 - x0 : copy_x1 - x0,
                    ...,
                ] = segment[
                    copy_y0 - seg_y0 : copy_y1 - seg_y0,
                    copy_x0 - seg_x0 : copy_x1 - seg_x0,
                    ...,
                ]
        return output
