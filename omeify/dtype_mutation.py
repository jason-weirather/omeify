from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import Literal, Sequence

import numpy as np

from .io.base import Image
from .io.image_planes import ImagePlaneSource
from .io.image_views import BorrowedSource
from .io.tiff import PlaneReader
from .progress import ProgressLogger

LOGGER = logging.getLogger(__name__)

TargetDType = Literal["uint8", "uint16"]
RangeMode = Literal["auto", "preserve", "full"]
TARGET_DTYPES: tuple[str, ...] = ("uint8", "uint16")
RANGE_MODES: tuple[str, ...] = ("auto", "preserve", "full")
DTYPE_MUTATION_PROTOCOL_VERSION = "1.0"

DEFAULT_SAMPLE_PIXELS_PER_CHANNEL = 262_144
DEFAULT_AUTO_MAX_NORMALIZED_RMSE = 0.005
_SCAN_TILE_SIZE = 2048
_INTEGER_TOLERANCE = 0.01
_INTEGER_MODERATE_NEAR_FRACTION = 0.10
_INTEGER_MODERATE_ENRICHMENT = 5.0
_INTEGER_STRONG_NEAR_FRACTION = 0.50
_INTEGER_STRONG_ENRICHMENT = 10.0
_MIN_INTEGER_EVIDENCE_SAMPLES = 100


def _target_dtype(value: str | np.dtype | type) -> np.dtype:
    dtype = np.dtype(value).newbyteorder("=")
    if dtype.name not in TARGET_DTYPES:
        raise TypeError("dtype mutation currently supports only uint8 and uint16 output")
    return dtype


def _quantile(values: np.ndarray, probability: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.quantile(values, probability))


def _smallest_unsigned_dtype(minimum: float, maximum: float) -> str | None:
    if minimum < 0:
        return None
    rounded_maximum = int(np.rint(maximum))
    if rounded_maximum <= np.iinfo(np.uint8).max:
        return "uint8"
    if rounded_maximum <= np.iinfo(np.uint16).max:
        return "uint16"
    if rounded_maximum <= np.iinfo(np.uint32).max:
        return "uint32"
    return None


def _integer_bits(minimum: float, maximum: float) -> int | None:
    if minimum < 0:
        return None
    lower = int(np.rint(minimum))
    upper = int(np.rint(maximum))
    span = upper - lower + 1
    if span < 1:
        return None
    return max(1, int(math.ceil(math.log2(span))))


@dataclass(frozen=True, slots=True)
class _PlaneScan:
    channel_index: int
    channel_name: str
    total_pixels: int
    finite_pixels: int
    nan_pixels: int
    positive_infinity_pixels: int
    negative_infinity_pixels: int
    negative_pixels: int
    zero_pixels: int
    nonzero_pixels: int
    near_integer_nonzero_pixels: int
    unit_rounding_absolute_error_sum: float
    unit_rounding_squared_error_sum: float
    unit_rounding_maximum_absolute_error: float
    minimum: float
    maximum: float
    sample_stride: int
    sample: np.ndarray


@dataclass(frozen=True, slots=True)
class DTypeChannelPlan:
    """One deterministic per-channel float-to-unsigned-integer mapping."""

    channel_index: int
    channel_name: str
    source_dtype: np.dtype
    target_dtype: np.dtype
    mapping: str
    offset: float
    quantum: float
    source_lower: float
    source_upper: float
    reason: str
    report: dict[str, object]

    def quantize(self, values: np.ndarray) -> np.ndarray:
        """Apply the planned mapping using nearest-even rounding."""

        source = np.asarray(values)
        scaled = (source.astype(np.float64, copy=False) - self.offset) / self.quantum
        rounded = np.rint(scaled)
        limits = np.iinfo(self.target_dtype)
        clipped = np.clip(rounded, limits.min, limits.max)
        return np.ascontiguousarray(clipped.astype(self.target_dtype, copy=False))


class DTypeMutationSource(BorrowedSource):
    """One fixed, per-channel dtype mapping applied on demand to an Image."""

    def __init__(self, image: Image, plans: Sequence[DTypeChannelPlan]) -> None:
        plans = tuple(plans)
        if not plans or len(plans) != image.channel_count:
            raise ValueError("dtype plans must match the image channels")
        if image.image_type != "multichannel":
            raise ValueError("dtype mutation requires scalar intensity channels")
        if any(p.channel_index != i or p.source_dtype != image.dtype for i, p in enumerate(plans)):
            raise ValueError("dtype plans do not match source channel indices or dtype")
        dtype = plans[0].target_dtype
        if any(p.target_dtype != dtype for p in plans):
            raise ValueError("All dtype plans must produce the same dtype")
        # A mutation is evaluated at base resolution; its output pyramids are rebuilt.
        metadata = replace(image._metadata, dtype=dtype, levels=(image.levels[0],))
        super().__init__(metadata, (image,))
        self._image = image
        self._plans = plans

    def _read_region(
        self, y0: int, y1: int, x0: int, x1: int, *, level: int, channels: tuple[int, ...],
    ) -> np.ndarray:
        values = self._image.read_region(y0, y1, x0, x1, channels=channels)
        if self.metadata.axes == "YX":
            return self._plans[0].quantize(values)
        return np.stack([self._plans[c].quantize(values[i]) for i, c in enumerate(channels)])


def _scan_plane(
    reader: PlaneReader,
    *,
    channel_index: int,
    channel_name: str,
    sample_pixels: int,
    progress_label: str | None = None,
) -> _PlaneScan:
    if int(reader.samples_per_pixel) != 1:
        raise ValueError("dtype mutation requires planar grayscale channel pages")
    if not np.issubdtype(reader.dtype, np.floating):
        raise TypeError(f"dtype mutation requires floating input, found {reader.dtype}")

    height = int(reader.height)
    width = int(reader.width)
    total_pixels = height * width
    stride = max(1, int(math.ceil(math.sqrt(total_pixels / sample_pixels))))
    finite_pixels = 0
    nan_pixels = 0
    positive_infinity_pixels = 0
    negative_infinity_pixels = 0
    negative_pixels = 0
    zero_pixels = 0
    nonzero_pixels = 0
    near_integer_nonzero_pixels = 0
    unit_rounding_absolute_error_sum = 0.0
    unit_rounding_squared_error_sum = 0.0
    unit_rounding_maximum_absolute_error = 0.0
    minimum = math.inf
    maximum = -math.inf
    sampled: list[np.ndarray] = []

    blocks_y = math.ceil(height / _SCAN_TILE_SIZE)
    blocks_x = math.ceil(width / _SCAN_TILE_SIZE)
    progress = (
        None
        if progress_label is None
        else ProgressLogger(
            LOGGER,
            progress_label,
            blocks_y * blocks_x,
            unit="blocks",
        )
    )
    completed_blocks = 0
    try:
        for y0 in range(0, height, _SCAN_TILE_SIZE):
            y1 = min(height, y0 + _SCAN_TILE_SIZE)
            for x0 in range(0, width, _SCAN_TILE_SIZE):
                x1 = min(width, x0 + _SCAN_TILE_SIZE)
                block = np.asarray(reader.read_region(y0, y1, x0, x1))
                if block.ndim != 2:
                    raise ValueError(
                        f"Channel {channel_index} did not read as one YX plane: {block.shape}"
                    )

                nan_pixels += int(np.count_nonzero(np.isnan(block)))
                positive_infinity_pixels += int(np.count_nonzero(np.isposinf(block)))
                negative_infinity_pixels += int(np.count_nonzero(np.isneginf(block)))
                finite = np.isfinite(block)
                finite_values = block[finite]
                finite_pixels += int(finite_values.size)
                if finite_values.size:
                    minimum = min(minimum, float(np.min(finite_values)))
                    maximum = max(maximum, float(np.max(finite_values)))
                    negative_pixels += int(np.count_nonzero(finite_values < 0))
                    zero_pixels += int(np.count_nonzero(finite_values == 0))
                    nonzero_values = finite_values[finite_values != 0]
                    nonzero_pixels += int(nonzero_values.size)
                    if nonzero_values.size:
                        unit_residual = np.abs(nonzero_values - np.rint(nonzero_values))
                        near_integer_nonzero_pixels += int(
                            np.count_nonzero(unit_residual <= _INTEGER_TOLERANCE)
                        )
                        unit_rounding_absolute_error_sum += float(np.sum(unit_residual))
                        unit_rounding_squared_error_sum += float(
                            np.sum(np.square(unit_residual, dtype=np.float64))
                        )
                        unit_rounding_maximum_absolute_error = max(
                            unit_rounding_maximum_absolute_error,
                            float(np.max(unit_residual)),
                        )

                sample_y0 = (-y0) % stride
                sample_x0 = (-x0) % stride
                sample_block = block[sample_y0::stride, sample_x0::stride]
                finite_sample = sample_block[np.isfinite(sample_block)]
                if finite_sample.size:
                    sampled.append(finite_sample.astype(np.float64, copy=False).reshape(-1))

                completed_blocks += 1
                if progress is not None:
                    progress.update(completed_blocks)
    finally:
        reader.clear_cache()

    if progress is not None:
        progress.finish()
    if finite_pixels == 0:
        raise ValueError(f"Channel {channel_index} contains no finite pixel values")
    sample = np.concatenate(sampled) if sampled else np.empty(0, dtype=np.float64)
    if sample.size > sample_pixels:
        positions = np.linspace(0, sample.size - 1, sample_pixels, dtype=np.int64)
        sample = sample[positions]
    return _PlaneScan(
        channel_index=channel_index,
        channel_name=channel_name,
        total_pixels=total_pixels,
        finite_pixels=finite_pixels,
        nan_pixels=nan_pixels,
        positive_infinity_pixels=positive_infinity_pixels,
        negative_infinity_pixels=negative_infinity_pixels,
        negative_pixels=negative_pixels,
        zero_pixels=zero_pixels,
        nonzero_pixels=nonzero_pixels,
        near_integer_nonzero_pixels=near_integer_nonzero_pixels,
        unit_rounding_absolute_error_sum=unit_rounding_absolute_error_sum,
        unit_rounding_squared_error_sum=unit_rounding_squared_error_sum,
        unit_rounding_maximum_absolute_error=unit_rounding_maximum_absolute_error,
        minimum=float(minimum),
        maximum=float(maximum),
        sample_stride=stride,
        sample=np.ascontiguousarray(sample),
    )


def _integer_evidence(scan: _PlaneScan) -> dict[str, object]:
    sample_values = np.asarray(scan.sample, dtype=np.float64)
    sampled_nonzero = sample_values[sample_values != 0]
    background_excluded = int(sample_values.size - sampled_nonzero.size)

    robust_lower = _quantile(sampled_nonzero, 0.001)
    robust_upper = _quantile(sampled_nonzero, 0.999)
    robust_span = (
        None
        if robust_lower is None or robust_upper is None
        else float(robust_upper - robust_lower)
    )

    phase_values = sampled_nonzero
    if phase_values.size >= 1000:
        lower, upper = np.quantile(phase_values, [0.005, 0.995])
        interior = phase_values[(phase_values >= lower) & (phase_values <= upper)]
        if interior.size >= _MIN_INTEGER_EVIDENCE_SAMPLES:
            phase_values = interior
    if phase_values.size >= 2 * _MIN_INTEGER_EVIDENCE_SAMPLES:
        # Integer-valued background or near-zero clipping can otherwise create
        # a persuasive peak at fractional phase zero. Requiring the same signal
        # in the upper half of sampled nonzero intensities makes the ancestry
        # diagnostic conservative without affecting the exact unit-loss scan.
        median = float(np.median(phase_values))
        upper_half = phase_values[phase_values >= median]
        if upper_half.size >= _MIN_INTEGER_EVIDENCE_SAMPLES:
            phase_values = upper_half

    rounded_code_span = int(np.rint(scan.maximum) - np.rint(scan.minimum) + 1)
    expected = 2 * _INTEGER_TOLERANCE
    full_near_fraction = (
        None
        if scan.nonzero_pixels == 0
        else scan.near_integer_nonzero_pixels / scan.nonzero_pixels
    )
    exact_mae = (
        None
        if scan.nonzero_pixels == 0
        else scan.unit_rounding_absolute_error_sum / scan.nonzero_pixels
    )
    exact_rmse = (
        None
        if scan.nonzero_pixels == 0
        else math.sqrt(scan.unit_rounding_squared_error_sum / scan.nonzero_pixels)
    )
    normalized_rmse = (
        None
        if exact_rmse is None or robust_span in {None, 0.0}
        else float(exact_rmse / robust_span)
    )

    if phase_values.size:
        sample_residual = np.abs(phase_values - np.rint(phase_values))
        sample_near_pixels = int(
            np.count_nonzero(sample_residual <= _INTEGER_TOLERANCE)
        )
        sample_near_fraction = sample_near_pixels / int(phase_values.size)
        sample_enrichment = sample_near_fraction / expected
        fractional = np.remainder(phase_values, 1.0)
        cosine = float(np.mean(np.cos(2 * math.pi * fractional)))
        sine = float(np.mean(np.sin(2 * math.pi * fractional)))
        concentration = math.hypot(cosine, sine)
        sample_p99 = float(np.quantile(sample_residual, 0.99))
    else:
        sample_near_pixels = 0
        sample_near_fraction = None
        sample_enrichment = None
        concentration = None
        sample_p99 = None

    if phase_values.size < _MIN_INTEGER_EVIDENCE_SAMPLES:
        classification = "inconclusive"
    elif rounded_code_span < 4:
        # A narrow continuous range close to one integer can otherwise look
        # deceptively lattice-like. It does not provide enough repeated
        # integer intervals to support unit-scale ancestry.
        classification = "weak"
    elif (
        sample_near_fraction is not None
        and sample_enrichment is not None
        and sample_near_fraction >= _INTEGER_STRONG_NEAR_FRACTION
        and sample_enrichment >= _INTEGER_STRONG_ENRICHMENT
    ):
        classification = "strong"
    elif (
        sample_near_fraction is not None
        and sample_enrichment is not None
        and sample_near_fraction >= _INTEGER_MODERATE_NEAR_FRACTION
        and sample_enrichment >= _INTEGER_MODERATE_ENRICHMENT
    ):
        classification = "moderate"
    else:
        classification = "weak"

    return {
        "classification": classification,
        "interpretation": (
            "Compatibility evidence only; the stitched float raster cannot prove the "
            "acquisition dtype or bit depth."
        ),
        "classification_basis": (
            "deterministic spatial sample of finite nonzero values, central 99 percent when "
            "large enough, then the upper intensity half to resist background-driven peaks"
        ),
        "evidence_sample_pixels": int(phase_values.size),
        "evidence_near_integer_pixels": sample_near_pixels,
        "near_integer_tolerance": _INTEGER_TOLERANCE,
        "rounded_code_span": rounded_code_span,
        "evidence_near_integer_fraction": sample_near_fraction,
        "uniform_fraction_expectation": expected,
        "evidence_near_integer_enrichment": sample_enrichment,
        "full_resolution_nonzero_pixels": scan.nonzero_pixels,
        "full_resolution_near_integer_pixels": scan.near_integer_nonzero_pixels,
        "full_resolution_near_integer_fraction": full_near_fraction,
        "classification_thresholds": {
            "moderate_near_integer_fraction": _INTEGER_MODERATE_NEAR_FRACTION,
            "moderate_enrichment": _INTEGER_MODERATE_ENRICHMENT,
            "strong_near_integer_fraction": _INTEGER_STRONG_NEAR_FRACTION,
            "strong_enrichment": _INTEGER_STRONG_ENRICHMENT,
        },
        "unit_rounding_mae": exact_mae,
        "unit_rounding_rmse": exact_rmse,
        "unit_rounding_normalized_rmse_of_nonzero_p0_1_to_p99_9_span": normalized_rmse,
        "unit_rounding_maximum_absolute_error": (
            None
            if scan.nonzero_pixels == 0
            else scan.unit_rounding_maximum_absolute_error
        ),
        "sample_pixels_before_background_exclusion": int(sample_values.size),
        "background_pixels_excluded_from_sample": background_excluded,
        "sample_nonzero_robust_lower_p0_1": robust_lower,
        "sample_nonzero_robust_upper_p99_9": robust_upper,
        "sample_nonzero_robust_span": robust_span,
        "fractional_phase_concentration": concentration,
        "unit_rounding_sample_p99_absolute_error": sample_p99,
    }


def _mapping_for_scan(
    scan: _PlaneScan,
    *,
    source_dtype: np.dtype,
    target_dtype: np.dtype,
    range_mode: RangeMode,
    auto_max_normalized_rmse: float,
) -> DTypeChannelPlan:
    if scan.nan_pixels or scan.positive_infinity_pixels or scan.negative_infinity_pixels:
        raise ValueError(
            f"Channel {scan.channel_index} {scan.channel_name!r} contains values that cannot "
            "be represented by an unsigned integer: "
            f"NaN={scan.nan_pixels}, +Inf={scan.positive_infinity_pixels}, "
            f"-Inf={scan.negative_infinity_pixels}"
        )
    if range_mode in {"auto", "preserve"} and scan.negative_pixels:
        raise ValueError(
            f"Channel {scan.channel_index} {scan.channel_name!r} contains "
            f"{scan.negative_pixels} negative pixels (minimum {scan.minimum:g}). "
            "Zero-preserving unsigned conversion will not silently shift or clip negative "
            "values; use range_mode='full' only when mapping the observed minimum to zero "
            "is scientifically intended."
        )

    evidence = _integer_evidence(scan)
    limits = np.iinfo(target_dtype)
    rounded_minimum = int(np.rint(scan.minimum))
    rounded_maximum = int(np.rint(scan.maximum))
    identity_fits = rounded_minimum >= limits.min and rounded_maximum <= limits.max
    unit_normalized_rmse = evidence[
        "unit_rounding_normalized_rmse_of_nonzero_p0_1_to_p99_9_span"
    ]
    unit_error_is_small = (
        unit_normalized_rmse is not None
        and float(unit_normalized_rmse) <= auto_max_normalized_rmse
    )
    strong_lattice = evidence["classification"] == "strong"

    if range_mode == "preserve":
        if not identity_fits:
            raise ValueError(
                f"Channel {scan.channel_index} nearest-integer range "
                f"{rounded_minimum}..{rounded_maximum} does not fit {target_dtype.name}; "
                "use range_mode='auto' or 'full' to rescale it explicitly"
            )
        mapping = "identity"
        offset = 0.0
        quantum = 1.0
        source_lower = scan.minimum
        source_upper = scan.maximum
        reason = (
            "Preserve mode was requested, so the source numeric scale is retained and values "
            "are rounded to the nearest integer without stretching the channel."
        )
    elif range_mode == "auto":
        if scan.minimum == scan.maximum:
            if not identity_fits:
                raise ValueError(
                    f"Channel {scan.channel_index} is constant at {scan.minimum:g}, which does "
                    f"not fit {target_dtype.name}; automatic scaling of a zero-width source "
                    "range would be arbitrary"
                )
            mapping = "identity"
            offset = 0.0
            quantum = 1.0
            source_lower = scan.minimum
            source_upper = scan.maximum
            reason = (
                "The channel is constant and its nearest integer fits the target dtype. "
                "Automatic range stretching would be arbitrary, so the source scale is retained."
            )
        elif identity_fits and (strong_lattice or unit_error_is_small):
            mapping = "identity"
            offset = 0.0
            quantum = 1.0
            source_lower = scan.minimum
            source_upper = scan.maximum
            if strong_lattice:
                reason = (
                    "The non-background values contain strong evidence of a unit-spaced integer "
                    "lattice, and nearest-integer values fit the target dtype. The source numeric "
                    "scale is retained instead of stretching the channel."
                )
            else:
                reason = (
                    "Nearest-integer values fit the target dtype and the measured unit-rounding "
                    f"RMSE is {float(unit_normalized_rmse):.6g} of the sampled nonzero robust "
                    f"intensity span, within the automatic loss limit "
                    f"{auto_max_normalized_rmse:.6g}. The source numeric scale is retained."
                )
        else:
            mapping = "zero_anchored_linear"
            offset = 0.0
            source_lower = 0.0
            source_upper = scan.maximum
            if source_upper <= 0:
                raise ValueError(
                    f"Channel {scan.channel_index} has no positive range available for "
                    "zero-anchored unsigned conversion"
                )
            quantum = source_upper / float(limits.max)
            reason_bits: list[str] = []
            if not identity_fits:
                reason_bits.append("unit-scale rounded values do not fit the target dtype")
            elif not strong_lattice and not unit_error_is_small:
                if unit_normalized_rmse is None:
                    reason_bits.append(
                        "unit-rounding loss could not be normalized to a nonzero robust span"
                    )
                else:
                    reason_bits.append(
                        "unit-rounding normalized RMSE "
                        f"{float(unit_normalized_rmse):.6g} exceeds the automatic loss limit "
                        f"{auto_max_normalized_rmse:.6g}"
                    )
            reason = (
                "; ".join(reason_bits).capitalize()
                + ". A zero-anchored linear mapping uses the available output codes without "
                "clipping any finite source value."
            )
    elif range_mode == "full":
        if scan.minimum == scan.maximum:
            if identity_fits:
                mapping = "identity"
                offset = 0.0
                quantum = 1.0
                source_lower = scan.minimum
                source_upper = scan.maximum
                reason = (
                    "The channel is constant and already fits the target dtype; stretching a "
                    "zero-width range is undefined, so the constant value is preserved by "
                    "nearest-integer rounding."
                )
            else:
                raise ValueError(
                    f"Channel {scan.channel_index} is constant at {scan.minimum:g}, which does "
                    f"not fit {target_dtype.name}; full-range mapping is undefined for a "
                    "zero-width source range"
                )
        else:
            mapping = "full_range_linear"
            offset = scan.minimum
            source_lower = scan.minimum
            source_upper = scan.maximum
            quantum = (scan.maximum - scan.minimum) / float(limits.max)
            reason = (
                "Full-range mode was requested, so the exact observed minimum and maximum are "
                "mapped to the first and last output codes."
            )
    else:
        raise ValueError("range_mode must be 'auto', 'preserve', or 'full'")

    if not math.isfinite(quantum) or quantum <= 0:
        raise ValueError(
            f"Channel {scan.channel_index} produced an invalid quantization quantum {quantum!r}"
        )

    provisional = DTypeChannelPlan(
        channel_index=scan.channel_index,
        channel_name=scan.channel_name,
        source_dtype=source_dtype,
        target_dtype=target_dtype,
        mapping=mapping,
        offset=float(offset),
        quantum=float(quantum),
        source_lower=float(source_lower),
        source_upper=float(source_upper),
        reason=reason,
        report={},
    )
    exact_bound_codes = provisional.quantize(
        np.asarray([scan.minimum, scan.maximum], dtype=np.float64)
    )
    exact_output_minimum = int(exact_bound_codes[0])
    exact_output_maximum = int(exact_bound_codes[1])
    exact_output_span = exact_output_maximum - exact_output_minimum + 1
    sample_codes = provisional.quantize(scan.sample)
    reconstructed = sample_codes.astype(np.float64) * quantum + offset
    absolute_error = np.abs(reconstructed - scan.sample)
    sample_nonzero = scan.sample != 0
    nonzero_absolute_error = absolute_error[sample_nonzero]
    nonzero_signed_error = (reconstructed - scan.sample)[sample_nonzero]
    robust_lower = _quantile(scan.sample, 0.001)
    robust_upper = _quantile(scan.sample, 0.999)
    robust_span = (
        None
        if robust_lower is None or robust_upper is None
        else float(robust_upper - robust_lower)
    )
    rmse = float(np.sqrt(np.mean(np.square(absolute_error)))) if absolute_error.size else 0.0
    normalized_rmse = (
        None if robust_span in {None, 0.0} else float(rmse / robust_span)
    )
    output_minimum = int(np.min(sample_codes)) if sample_codes.size else None
    output_maximum = int(np.max(sample_codes)) if sample_codes.size else None
    output_code_span = (
        None
        if output_minimum is None or output_maximum is None
        else output_maximum - output_minimum + 1
    )
    error_tolerance = max(
        1e-12,
        8 * np.finfo(np.float64).eps * max(1.0, abs(scan.minimum), abs(scan.maximum)),
    )
    changed_fraction = (
        float(np.mean(absolute_error > error_tolerance)) if absolute_error.size else 0.0
    )
    signed_error = reconstructed - scan.sample
    nonzero_changed_fraction = (
        float(np.mean(nonzero_absolute_error > error_tolerance))
        if nonzero_absolute_error.size
        else None
    )
    minimum_code_fraction = (
        float(np.mean(sample_codes == limits.min)) if sample_codes.size else 0.0
    )
    maximum_code_fraction = (
        float(np.mean(sample_codes == limits.max)) if sample_codes.size else 0.0
    )
    sampled_nonzero_values = scan.sample[sample_nonzero]
    lattice_robust_lower = evidence["sample_nonzero_robust_lower_p0_1"]
    lattice_robust_upper = evidence["sample_nonzero_robust_upper_p99_9"]
    robust_unsigned_dtype = (
        None
        if lattice_robust_lower is None or lattice_robust_upper is None
        else _smallest_unsigned_dtype(
            float(lattice_robust_lower),
            float(lattice_robust_upper),
        )
    )
    report: dict[str, object] = {
        "channel_index": scan.channel_index,
        "channel_name": scan.channel_name,
        "source": {
            "dtype": source_dtype.name,
            "total_pixels": scan.total_pixels,
            "finite_pixels": scan.finite_pixels,
            "nan_pixels": scan.nan_pixels,
            "positive_infinity_pixels": scan.positive_infinity_pixels,
            "negative_infinity_pixels": scan.negative_infinity_pixels,
            "negative_pixels": scan.negative_pixels,
            "zero_pixels": scan.zero_pixels,
            "nonzero_pixels": scan.nonzero_pixels,
            "minimum": scan.minimum,
            "maximum": scan.maximum,
            "nearest_integer_minimum": rounded_minimum,
            "nearest_integer_maximum": rounded_maximum,
            "sample_pixels": int(scan.sample.size),
            "sample_stride": scan.sample_stride,
            "sample_percentiles": {
                "p0_1": _quantile(scan.sample, 0.001),
                "p1": _quantile(scan.sample, 0.01),
                "p50": _quantile(scan.sample, 0.5),
                "p99": _quantile(scan.sample, 0.99),
                "p99_9": _quantile(scan.sample, 0.999),
            },
            "sample_nonzero_percentiles": {
                "p0_1": _quantile(sampled_nonzero_values, 0.001),
                "p1": _quantile(sampled_nonzero_values, 0.01),
                "p50": _quantile(sampled_nonzero_values, 0.5),
                "p99": _quantile(sampled_nonzero_values, 0.99),
                "p99_9": _quantile(sampled_nonzero_values, 0.999),
            },
            "smallest_unsigned_dtype_after_unit_rounding": _smallest_unsigned_dtype(
                scan.minimum,
                scan.maximum,
            ),
            "sample_nonzero_p0_1_to_p99_9_smallest_unsigned_dtype_after_unit_rounding": (
                robust_unsigned_dtype
            ),
            "integer_code_span_bits": _integer_bits(scan.minimum, scan.maximum),
        },
        "integer_lattice_evidence": evidence,
        "automatic_preserve_assessment": {
            "nearest_integer_range_fits_target": identity_fits,
            "strong_integer_lattice_evidence": strong_lattice,
            "unit_rounding_normalized_rmse": unit_normalized_rmse,
            "maximum_normalized_rmse": auto_max_normalized_rmse,
            "unit_rounding_loss_within_limit": unit_error_is_small,
            "preserve_source_scale": mapping == "identity",
        },
        "mapping": {
            "name": mapping,
            "formula": "output = clip(rint((source - offset) / quantum), dtype range)",
            "rounding": "nearest, ties to even",
            "offset_source_units": float(offset),
            "quantum_source_units_per_output_code": float(quantum),
            "source_lower": float(source_lower),
            "source_upper": float(source_upper),
            "target_dtype": target_dtype.name,
            "target_code_minimum": int(limits.min),
            "target_code_maximum": int(limits.max),
            "exact_output_code_minimum_from_source_bounds": exact_output_minimum,
            "exact_output_code_maximum_from_source_bounds": exact_output_maximum,
            "exact_output_code_span_from_source_bounds": exact_output_span,
            "exact_output_code_span_bits": max(
                1,
                int(math.ceil(math.log2(exact_output_span))),
            ),
            "target_code_span_fraction_used_by_source_bounds": (
                exact_output_span / (int(limits.max) - int(limits.min) + 1)
            ),
            "reason": reason,
        },
        "anticipated_loss": {
            "basis": "deterministic spatial sample",
            "sample_pixels": int(scan.sample.size),
            "clipped_finite_pixels": 0,
            "theoretical_maximum_absolute_error_source_units": float(quantum / 2),
            "sample_mean_signed_error_source_units": (
                float(np.mean(signed_error)) if signed_error.size else 0.0
            ),
            "sample_mean_absolute_error_source_units": (
                float(np.mean(absolute_error)) if absolute_error.size else 0.0
            ),
            "sample_root_mean_square_error_source_units": rmse,
            "sample_p99_absolute_error_source_units": (
                float(np.quantile(absolute_error, 0.99)) if absolute_error.size else 0.0
            ),
            "sample_maximum_absolute_error_source_units": (
                float(np.max(absolute_error)) if absolute_error.size else 0.0
            ),
            "sample_normalized_rmse_of_p0_1_to_p99_9_span": normalized_rmse,
            "sample_changed_fraction": changed_fraction,
            "sample_exactly_represented_fraction": 1.0 - changed_fraction,
            "sample_nonzero_mean_signed_error_source_units": (
                float(np.mean(nonzero_signed_error)) if nonzero_signed_error.size else None
            ),
            "sample_nonzero_mean_absolute_error_source_units": (
                float(np.mean(nonzero_absolute_error))
                if nonzero_absolute_error.size
                else None
            ),
            "sample_nonzero_root_mean_square_error_source_units": (
                float(np.sqrt(np.mean(np.square(nonzero_absolute_error))))
                if nonzero_absolute_error.size
                else None
            ),
            "sample_nonzero_p99_absolute_error_source_units": (
                float(np.quantile(nonzero_absolute_error, 0.99))
                if nonzero_absolute_error.size
                else None
            ),
            "sample_nonzero_changed_fraction": nonzero_changed_fraction,
            "sample_nonzero_exactly_represented_fraction": (
                None
                if nonzero_changed_fraction is None
                else 1.0 - nonzero_changed_fraction
            ),
            "sample_target_minimum_code_fraction": minimum_code_fraction,
            "sample_target_maximum_code_fraction": maximum_code_fraction,
            "sample_output_code_minimum": output_minimum,
            "sample_output_code_maximum": output_maximum,
            "sample_output_code_span": output_code_span,
            "sample_distinct_output_codes": (
                int(np.unique(sample_codes).size) if sample_codes.size else 0
            ),
        },
    }
    return DTypeChannelPlan(
        channel_index=scan.channel_index,
        channel_name=scan.channel_name,
        source_dtype=source_dtype,
        target_dtype=target_dtype,
        mapping=mapping,
        offset=float(offset),
        quantum=float(quantum),
        source_lower=float(source_lower),
        source_upper=float(source_upper),
        reason=reason,
        report=report,
    )


def analyze_dtype_mutation(
    image: Image,
    *,
    dtype: TargetDType | np.dtype | type,
    range_mode: RangeMode = "auto",
    sample_pixels_per_channel: int = DEFAULT_SAMPLE_PIXELS_PER_CHANNEL,
    auto_max_normalized_rmse: float = DEFAULT_AUTO_MAX_NORMALIZED_RMSE,
) -> tuple[DTypeChannelPlan, ...]:
    """Scan every source plane and create one fixed per-channel dtype mapping.

    The full raster is scanned for exact bounds and unrepresentable values. A
    bounded deterministic spatial sample is retained for integer-lattice
    diagnostics and anticipated quantization-error estimates.
    """

    if range_mode not in RANGE_MODES:
        raise ValueError("range_mode must be 'auto', 'preserve', or 'full'")
    sample_pixels = int(sample_pixels_per_channel)
    if sample_pixels < 1024:
        raise ValueError("sample_pixels_per_channel must be at least 1024")
    normalized_auto_loss = float(auto_max_normalized_rmse)
    if not math.isfinite(normalized_auto_loss) or normalized_auto_loss < 0:
        raise ValueError("auto_max_normalized_rmse must be a finite value of zero or greater")
    source_dtype = image.dtype
    channel_names = image.channel_names
    normalized_source_dtype = np.dtype(source_dtype).newbyteorder("=")
    if not np.issubdtype(normalized_source_dtype, np.floating):
        raise TypeError(
            f"dtype mutation requires float32 or float64 input, found {normalized_source_dtype}"
        )
    normalized_target_dtype = _target_dtype(dtype)
    names = tuple(str(item) for item in channel_names)
    readers = ImagePlaneSource(image).plane_readers()
    if len(readers) != len(names):
        for reader in readers:
            reader.clear_cache()
        raise ValueError(
            f"Source supplied {len(readers)} planes, but {len(names)} channel names were provided"
        )

    LOGGER.info(
        "Mutation analysis: scanning %s full-resolution channel(s) for exact bounds, "
        "non-finite values, and quantization diagnostics",
        len(readers),
    )
    plans: list[DTypeChannelPlan] = []
    try:
        for index, reader in enumerate(readers):
            name = names[index]
            scan = _scan_plane(
                reader,
                channel_index=index,
                channel_name=name,
                sample_pixels=sample_pixels,
                progress_label=(
                    f"Scanning mutation channel {index + 1}/{len(readers)} {name!r}"
                ),
            )
            LOGGER.info(
                "Mutation scan %s/%s complete for %r: range=%g..%g, finite=%s, "
                "nonzero=%s, NaN=%s, +Inf=%s, -Inf=%s",
                index + 1,
                len(readers),
                name,
                scan.minimum,
                scan.maximum,
                f"{scan.finite_pixels:,}",
                f"{scan.nonzero_pixels:,}",
                f"{scan.nan_pixels:,}",
                f"{scan.positive_infinity_pixels:,}",
                f"{scan.negative_infinity_pixels:,}",
            )
            plan = _mapping_for_scan(
                scan,
                source_dtype=normalized_source_dtype,
                target_dtype=normalized_target_dtype,
                range_mode=range_mode,
                auto_max_normalized_rmse=normalized_auto_loss,
            )
            plans.append(plan)
            LOGGER.info(
                "Mutation plan %s/%s for channel [%s] %r: "
                "mapping=%s, offset=%g, quantum=%g",
                index + 1,
                len(readers),
                index,
                name,
                plan.mapping,
                plan.offset,
                plan.quantum,
            )
            LOGGER.info("  Mapping decision: %s", plan.reason)
    finally:
        for reader in readers:
            reader.clear_cache()

    LOGGER.info("Mutation analysis complete: planned %s channel(s)", len(plans))
    return tuple(plans)
