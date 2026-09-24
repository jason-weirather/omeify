from __future__ import annotations

from dataclasses import replace

import numpy as np

from omeify.io.spec import OMEImageSpec
from omeify.io.tiff import PlaneReader

from .model import PlaneReaderSource

_FLOAT32_FRACTION_BITS = 23
_FLOAT32_NONFRACTION_BITS = 9  # sign + exponent


def float32_significant_bits(mantissa_bits: int) -> int:
    """Return OME SignificantBits for float32 with retained fraction bits."""

    if isinstance(mantissa_bits, bool) or not isinstance(mantissa_bits, int):
        raise TypeError("mantissa_bits must be an integer")
    if mantissa_bits < 0 or mantissa_bits > _FLOAT32_FRACTION_BITS:
        raise ValueError("mantissa_bits must be between 0 and 23")
    return _FLOAT32_NONFRACTION_BITS + mantissa_bits


def round_float32_mantissa(values: np.ndarray, mantissa_bits: int) -> np.ndarray:
    """Round float32 values to a fixed number of stored fraction bits.

    Rounding is nearest with ties to even. NaN and infinity bit patterns are
    retained unchanged. The returned array remains float32 and therefore keeps
    the float32 exponent range even when its fractional precision is reduced.
    If rounding a finite value would produce infinity, raise OverflowError
    rather than silently introducing nonfinite pixels or clipping the value.
    Non-native float32 input is byte-swapped without changing NaN payloads.
    """

    if isinstance(mantissa_bits, bool) or not isinstance(mantissa_bits, int):
        raise TypeError("mantissa_bits must be an integer")
    bits_to_keep = mantissa_bits
    if bits_to_keep < 0 or bits_to_keep > _FLOAT32_FRACTION_BITS:
        raise ValueError("mantissa_bits must be between 0 and 23")

    source = np.asarray(values)
    if source.dtype.newbyteorder("=") != np.dtype("float32"):
        raise TypeError(
            f"float mantissa trimming requires float32 pixels, found {source.dtype}"
        )
    if not source.dtype.isnative:
        source = source.byteswap().view(np.dtype("float32"))
    if bits_to_keep == _FLOAT32_FRACTION_BITS:
        return np.ascontiguousarray(source)

    output = np.ascontiguousarray(source).copy()
    raw = output.view(np.uint32)
    sign = raw & np.uint32(0x80000000)
    magnitude = raw & np.uint32(0x7FFFFFFF)
    exponent = magnitude & np.uint32(0x7F800000)
    finite = exponent != np.uint32(0x7F800000)

    discarded = _FLOAT32_FRACTION_BITS - bits_to_keep
    mask = np.uint32((1 << discarded) - 1)
    halfway = np.uint32(1 << (discarded - 1))
    remainder = magnitude & mask
    retained_lsb = (magnitude >> np.uint32(discarded)) & np.uint32(1)
    increment = (remainder > halfway) | (
        (remainder == halfway) & (retained_lsb == np.uint32(1))
    )
    rounded = (magnitude & ~mask) + (
        increment.astype(np.uint32, copy=False) << np.uint32(discarded)
    )
    if np.any(finite & (rounded >= np.uint32(0x7F800000))):
        raise OverflowError(
            f"Rounding float32 to {bits_to_keep} fraction bits would turn a finite "
            "pixel into infinity; retain more bits or explicitly rescale upstream"
        )
    magnitude[finite] = rounded[finite]
    raw[...] = sign | magnitude
    return output


class _FloatMantissaPlaneReader:
    def __init__(self, reader: PlaneReader, mantissa_bits: int) -> None:
        if np.dtype(reader.dtype).newbyteorder("=") != np.dtype("float32"):
            raise TypeError(
                "float mantissa trimming requires float32 source planes"
            )
        self._reader = reader
        self._mantissa_bits = mantissa_bits
        self.height = int(reader.height)
        self.width = int(reader.width)
        self.samples_per_pixel = int(reader.samples_per_pixel)
        self.dtype = np.dtype("float32")

    def clear_cache(self) -> None:
        self._reader.clear_cache()

    def read_region(self, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        return round_float32_mantissa(
            self._reader.read_region(y0, y1, x0, x1),
            self._mantissa_bits,
        )


class FloatMantissaSource:
    """Streaming source that rounds every float32 region before writing."""

    def __init__(self, source: PlaneReaderSource, mantissa_bits: int) -> None:
        self._source = source
        self._mantissa_bits = mantissa_bits

    def plane_readers(self) -> list[_FloatMantissaPlaneReader]:
        return [
            _FloatMantissaPlaneReader(reader, self._mantissa_bits)
            for reader in self._source.plane_readers()
        ]


def prepare_float_precision(
    source: PlaneReaderSource,
    spec: OMEImageSpec,
    mantissa_bits: int | None,
) -> tuple[PlaneReaderSource, OMEImageSpec, int | None]:
    """Apply optional float32 precision trimming to one writer input."""

    if mantissa_bits is None:
        return source, spec, None
    if spec.dtype != np.dtype("float32"):
        return source, spec, None
    significant_bits = float32_significant_bits(mantissa_bits)
    effective_spec = replace(
        spec,
        significant_bits_override=significant_bits,
    )
    if mantissa_bits == _FLOAT32_FRACTION_BITS:
        return source, effective_spec, mantissa_bits
    return FloatMantissaSource(source, mantissa_bits), effective_spec, mantissa_bits
