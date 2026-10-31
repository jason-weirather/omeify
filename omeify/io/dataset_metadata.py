"""Source-axis descriptors. ImageMetadata deliberately remains two-dimensional."""
from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

from .image_metadata import ImageMetadata, integer


@dataclass(frozen=True, slots=True)
class AxisInfo:
    """Regular source sampling; unknown origin is not an implicit physical zero.

    Units are explicit strings from the format adapter. ``scale`` is a sampling
    interval, not a measured acquisition position. Per-plane acquisition records
    are exposed separately by the TIFF adapter.
    """
    name: str
    size: int
    kind: str
    unit: str | None = None
    scale: float | None = None
    origin: float | None = None

    def __post_init__(self):
        if self.name not in "TCZYXS" or len(self.name) != 1:
            raise ValueError("Source axes must be named T, C, Z, Y, X, or S")
        object.__setattr__(self, "size", integer(self.size, "axis size", minimum=1))
        expected = "time" if self.name == "T" else "channel" if self.name in "CS" else "space"
        if self.kind != expected:
            raise ValueError("Axis role conflicts with its name")
        for key in ("scale", "origin"):
            value = getattr(self, key)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                    raise ValueError(f"Axis {key} must be finite or unknown")
                if key == "scale" and value <= 0:
                    raise ValueError("Axis scale must be positive")
                object.__setattr__(self, key, float(value))
        if self.unit is not None and (not isinstance(self.unit, str) or not self.unit):
            raise ValueError("Axis unit must be a nonempty string or None")

    def coordinate(self, index: int) -> float | None:
        index = integer(index, "axis index")
        if index >= self.size:
            raise IndexError(f"{self.name} index {index} outside 0..{self.size - 1}")
        if self.scale is None or self.origin is None:
            return None
        coordinate = self.origin + index * self.scale
        if not math.isfinite(coordinate):
            raise ValueError("Axis coordinate overflowed")
        return coordinate


@dataclass(frozen=True, slots=True)
class DatasetLevel:
    """One stored multidimensional level and its ordinary 2D plane description."""
    index: int
    axes: tuple[AxisInfo, ...]
    plane_metadata: ImageMetadata
    plane_preserving: bool = True

    def __post_init__(self):
        object.__setattr__(self, "index", integer(self.index, "source level"))
        object.__setattr__(self, "axes", tuple(self.axes))
        names = tuple(a.name for a in self.axes)
        if len(names) != len(set(names)) or not {"Y", "X"}.issubset(names):
            raise ValueError("Source axes must be unique and include Y/X")
        if not isinstance(self.plane_metadata, ImageMetadata):
            raise TypeError("plane_metadata must be ImageMetadata")

    @property
    def axis_names(self) -> str:
        return "".join(a.name for a in self.axes)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(a.size for a in self.axes)

    def axis(self, name: str) -> AxisInfo | None:
        return next((a for a in self.axes if a.name == name.upper()), None)


def selected_index(info: AxisInfo | None, index, *, name: str, default=False):
    if info is None:
        if index is not None:
            raise ValueError(f"The source has no {name.lower()} axis to select")
        return None
    if index is None:
        if info.size != 1 and not default:
            raise ValueError(f"Non-singleton {name.lower()}={info.size} requires an explicit {name.lower()} index")
        return 0
    index = integer(index, f"{name.lower()} index")
    if index >= info.size:
        raise IndexError(f"{name} index {index} outside 0..{info.size - 1}")
    return index


def time_unit_seconds(unit: str) -> float:
    """Return seconds per declared OME/NGFF time unit without case folding SI symbols."""
    factors = {"s": 1., "second": 1., "min": 60., "minute": 60.,
               "h": 3600., "hour": 3600., "d": 86400., "day": 86400.}
    prefixes = (("yotta", "Y", 24), ("zetta", "Z", 21), ("exa", "E", 18),
                ("peta", "P", 15), ("tera", "T", 12), ("giga", "G", 9),
                ("mega", "M", 6), ("kilo", "k", 3), ("hecto", "h", 2),
                ("deca", "da", 1), ("deci", "d", -1), ("centi", "c", -2),
                ("milli", "m", -3), ("micro", "µ", -6), ("nano", "n", -9),
                ("pico", "p", -12), ("femto", "f", -15), ("atto", "a", -18),
                ("zepto", "z", -21), ("yocto", "y", -24))
    for name, symbol, exponent in prefixes:
        factors[name + "second"] = factors[symbol + "s"] = 10. ** exponent
    factors["us"] = factors["μs"] = 1e-6
    try:
        return factors[unit]
    except (KeyError, TypeError) as error:
        raise NotImplementedError(f"Unsupported time unit {unit!r}") from error


def canonical_axis(info: AxisInfo) -> AxisInfo:
    """Normalize declared physical/time quantities, without inventing missing values."""
    from dataclasses import replace
    from .pixel_size import PixelSize

    if info.unit is None or info.name in "CS":
        return info
    if info.name == "T":
        factor, unit = time_unit_seconds(info.unit), "s"
    else:
        factor, unit = PixelSize(1., 1., info.unit).converted_to("µm").x, "µm"
    return replace(info, unit=unit,
                   scale=None if info.scale is None else info.scale * factor,
                   origin=None if info.origin is None else info.origin * factor)
