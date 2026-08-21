from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


# Conversion factors are deliberately kept outside PixelSize validation. PixelSize
# is a generic immutable value object; format-specific code decides whether a unit
# is acceptable for serialization.
_LENGTH_TO_METERS: dict[str, float] = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "metre": 1.0,
    "metres": 1.0,
    "cm": 1e-2,
    "centimeter": 1e-2,
    "centimeters": 1e-2,
    "centimetre": 1e-2,
    "centimetres": 1e-2,
    "mm": 1e-3,
    "millimeter": 1e-3,
    "millimeters": 1e-3,
    "millimetre": 1e-3,
    "millimetres": 1e-3,
    "µm": 1e-6,
    "μm": 1e-6,
    "um": 1e-6,
    "micrometer": 1e-6,
    "micrometers": 1e-6,
    "micrometre": 1e-6,
    "micrometres": 1e-6,
    "nm": 1e-9,
    "nanometer": 1e-9,
    "nanometers": 1e-9,
    "nanometre": 1e-9,
    "nanometres": 1e-9,
    "pm": 1e-12,
    "picometer": 1e-12,
    "picometers": 1e-12,
    "picometre": 1e-12,
    "picometres": 1e-12,
    "å": 1e-10,
    "angstrom": 1e-10,
    "angstroms": 1e-10,
    "in": 0.0254,
    "inch": 0.0254,
    "inches": 0.0254,
}

_CANONICAL_LENGTH_UNITS: dict[str, str] = {
    "m": "m",
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "metres": "m",
    "cm": "cm",
    "centimeter": "cm",
    "centimeters": "cm",
    "centimetre": "cm",
    "centimetres": "cm",
    "mm": "mm",
    "millimeter": "mm",
    "millimeters": "mm",
    "millimetre": "mm",
    "millimetres": "mm",
    "µm": "µm",
    "μm": "µm",
    "um": "µm",
    "micrometer": "µm",
    "micrometers": "µm",
    "micrometre": "µm",
    "micrometres": "µm",
    "nm": "nm",
    "nanometer": "nm",
    "nanometers": "nm",
    "nanometre": "nm",
    "nanometres": "nm",
    "pm": "pm",
    "picometer": "pm",
    "picometers": "pm",
    "picometre": "pm",
    "picometres": "pm",
    "å": "Å",
    "angstrom": "Å",
    "angstroms": "Å",
    "in": "in",
    "inch": "in",
    "inches": "in",
}


def _unit_key(unit: str) -> str:
    return str(unit).strip().lower().replace("μ", "µ")


def canonical_length_unit(unit: str) -> str:
    """Return a compact canonical spelling for a supported length unit."""

    key = _unit_key(unit)
    try:
        return _CANONICAL_LENGTH_UNITS[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported physical-size unit {unit!r}") from exc


def length_unit_is_supported(unit: str) -> bool:
    return _unit_key(unit) in _LENGTH_TO_METERS


@dataclass(frozen=True, slots=True)
class PixelSize:
    """Immutable physical size of one pixel along X and Y.

    The object itself does not impose an OME- or MITI-specific unit policy.
    Serializers may require a supported physical length unit at their boundary.
    """

    x: float
    y: float
    unit: str

    def __post_init__(self) -> None:
        x = float(self.x)
        y = float(self.y)
        unit = str(self.unit).strip()
        if not math.isfinite(x) or x <= 0:
            raise ValueError(f"PixelSize.x must be a positive finite value, found {self.x!r}")
        if not math.isfinite(y) or y <= 0:
            raise ValueError(f"PixelSize.y must be a positive finite value, found {self.y!r}")
        if not unit:
            raise ValueError("PixelSize.unit must be a non-empty string")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "unit", unit)

    def to_tuple(self) -> tuple[float, float, str]:
        return self.x, self.y, self.unit

    @classmethod
    def from_tuple(cls, value: Sequence[object]) -> "PixelSize":
        if len(value) != 3:
            raise ValueError("PixelSize tuple representation must contain exactly (x, y, unit)")
        return cls(float(value[0]), float(value[1]), str(value[2]))

    def scaled(self, x_factor: float, y_factor: float | None = None) -> "PixelSize":
        """Return the pixel size after spatial downsampling or rescaling.

        ``scaled(2)`` represents a 2x downsample in both dimensions: each new
        pixel spans twice the physical distance of an original pixel.
        """

        x_scale = float(x_factor)
        y_scale = x_scale if y_factor is None else float(y_factor)
        if not math.isfinite(x_scale) or x_scale <= 0:
            raise ValueError("x_factor must be a positive finite value")
        if not math.isfinite(y_scale) or y_scale <= 0:
            raise ValueError("y_factor must be a positive finite value")
        return PixelSize(self.x * x_scale, self.y * y_scale, self.unit)

    def converted_to(self, unit: str) -> "PixelSize":
        """Return an equivalent PixelSize expressed in another known unit."""

        source_key = _unit_key(self.unit)
        target_key = _unit_key(unit)
        try:
            source_factor = _LENGTH_TO_METERS[source_key]
            target_factor = _LENGTH_TO_METERS[target_key]
        except KeyError as exc:
            unsupported = self.unit if source_key not in _LENGTH_TO_METERS else unit
            raise ValueError(f"Unsupported physical-size unit {unsupported!r}") from exc
        return PixelSize(
            self.x * source_factor / target_factor,
            self.y * source_factor / target_factor,
            canonical_length_unit(unit),
        )


def normalize_ome_pixel_size(pixel_size: PixelSize) -> PixelSize:
    """Validate and canonicalize a PixelSize at the OME serialization boundary."""

    if not isinstance(pixel_size, PixelSize):
        raise TypeError("pixel_size must be a PixelSize instance")
    canonical = canonical_length_unit(pixel_size.unit)
    if canonical == pixel_size.unit:
        return pixel_size
    return pixel_size.converted_to(canonical)


def pixel_size_from_xy_units(
    x: float,
    x_unit: str,
    y: float,
    y_unit: str,
) -> PixelSize:
    """Construct one PixelSize from independently unit-tagged X and Y values."""

    x_size = PixelSize(float(x), float(x), str(x_unit))
    if _unit_key(x_unit) == _unit_key(y_unit):
        # Readers should preserve an unfamiliar but internally consistent OME
        # unit rather than imposing the writer's supported-unit policy.
        try:
            unit = canonical_length_unit(x_unit)
        except ValueError:
            unit = str(x_unit).strip()
        return PixelSize(float(x), float(y), unit)
    y_size = PixelSize(float(y), float(y), str(y_unit)).converted_to(x_unit)
    return PixelSize(x_size.x, y_size.y, canonical_length_unit(x_unit))
