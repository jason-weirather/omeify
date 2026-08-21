from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal, Sequence
from xml.etree import ElementTree

import numpy as np
import tifffile

from .pixel_size import PixelSize

ChannelNameField = Literal["name", "biomarker", "auto"]
_XML_ENCODING_DECLARATION = re.compile(
    r"(<\?xml\b[^>]*?)\s+encoding=(['\"])[^'\"]+\2",
    flags=re.IGNORECASE,
)


def _local_name(tag: str) -> str:
    return str(tag).rsplit("}", 1)[-1]


def _parse_description(description: str | bytes | None) -> ElementTree.Element | None:
    if not description:
        return None
    if isinstance(description, bytes):
        payload: str | bytes = description
    else:
        # tifffile has already decoded the tag. Remove a stale declaration such
        # as encoding="utf-16" before parsing the Python string as UTF-8 text.
        payload = _XML_ENCODING_DECLARATION.sub(r"\1", str(description), count=1)
    try:
        return ElementTree.fromstring(payload)
    except (ElementTree.ParseError, TypeError, ValueError):
        return None


def _direct_child(element: ElementTree.Element, name: str) -> ElementTree.Element | None:
    return next((child for child in element if _local_name(child.tag) == name), None)


def _direct_child_text(element: ElementTree.Element, name: str) -> str | None:
    child = _direct_child(element, name)
    if child is None or child.text is None:
        return None
    value = child.text.strip()
    return value or None


def _path_text(root: ElementTree.Element, path: Sequence[str]) -> str | None:
    current = root
    for name in path:
        child = _direct_child(current, name)
        if child is None:
            return None
        current = child
    if current.text is None:
        return None
    value = current.text.strip()
    return value or None


@dataclass(frozen=True, slots=True)
class AkoyaQPIChannelMetadata:
    name: str | None
    biomarker: str | None
    pixel_size_microns: float | None

    def as_dict(self) -> dict[str, str | float | None]:
        return {
            "name": self.name,
            "biomarker": self.biomarker,
            "pixel_size_microns": self.pixel_size_microns,
        }


def parse_akoya_qpi_channel_metadata(
    description: str | bytes | None,
    *,
    channel_index: int | None = None,
) -> AkoyaQPIChannelMetadata:
    root = _parse_description(description)
    prefix = f"Akoya channel {channel_index}" if channel_index is not None else "Akoya channel"
    if root is None or _local_name(root.tag) != "PerkinElmer-QPI-ImageDescription":
        raise ValueError(
            f"{prefix} does not contain a readable PerkinElmer-QPI-ImageDescription"
        )

    name = _direct_child_text(root, "Name")
    biomarker = _direct_child_text(root, "Biomarker")
    raw_pixel_size = _path_text(
        root,
        ("ScanProfile", "root", "ScanResolution", "PixelSizeMicrons"),
    )
    pixel_size = None
    if raw_pixel_size is not None:
        try:
            pixel_size = float(raw_pixel_size)
        except ValueError as exc:
            raise ValueError(
                f"{prefix} has invalid PixelSizeMicrons={raw_pixel_size!r}"
            ) from exc
        if not math.isfinite(pixel_size) or pixel_size <= 0:
            raise ValueError(
                f"{prefix} has non-positive or non-finite PixelSizeMicrons={raw_pixel_size!r}"
            )
    return AkoyaQPIChannelMetadata(
        name=name,
        biomarker=biomarker,
        pixel_size_microns=pixel_size,
    )


def select_akoya_channel_name(
    metadata: AkoyaQPIChannelMetadata,
    field: ChannelNameField,
    *,
    channel_index: int,
) -> tuple[str, str]:
    if field not in {"name", "biomarker", "auto"}:
        raise ValueError("channel_name_field must be 'name', 'biomarker', or 'auto'")
    if field == "name":
        if metadata.name is None:
            raise ValueError(
                f"Akoya channel {channel_index} is missing Name metadata requested by "
                "channel_name_field='name'"
            )
        return metadata.name, "name"
    if field == "biomarker":
        if metadata.biomarker is None:
            raise ValueError(
                f"Akoya channel {channel_index} is missing Biomarker metadata requested by "
                "channel_name_field='biomarker'"
            )
        return metadata.biomarker, "biomarker"
    if metadata.biomarker is not None:
        return metadata.biomarker, "biomarker"
    if metadata.name is not None:
        return metadata.name, "name"
    return f"Channel {channel_index + 1}", "generated"


def _resolution_value(value: object) -> float:
    if isinstance(value, (tuple, list, np.ndarray)) and len(value) == 2:
        numerator, denominator = value
        return float(numerator) / float(denominator)
    return float(value)


def pixel_size_from_tiff_resolution(page: tifffile.TiffPage) -> PixelSize | None:
    try:
        x_ppu = _resolution_value(page.tags["XResolution"].value)
        y_ppu = _resolution_value(page.tags["YResolution"].value)
        unit_value = int(page.tags["ResolutionUnit"].value)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    if x_ppu <= 0 or y_ppu <= 0:
        return None
    if unit_value == 3:  # centimeter
        return PixelSize(1e4 / x_ppu, 1e4 / y_ppu, "µm")
    if unit_value == 2:  # inch
        return PixelSize(25400.0 / x_ppu, 25400.0 / y_ppu, "µm")
    return None


def consistent_akoya_pixel_size(
    metadata: Sequence[AkoyaQPIChannelMetadata],
    pages: Sequence[tifffile.TiffPage],
) -> PixelSize | None:
    if len(metadata) != len(pages):
        raise ValueError("Akoya channel metadata/page counts do not match")

    declared = [item.pixel_size_microns for item in metadata]
    present = [value is not None for value in declared]
    if any(present):
        if not all(present):
            missing = [str(index) for index, value in enumerate(declared) if value is None]
            raise ValueError(
                "Akoya PixelSizeMicrons is present on only some full-resolution channel "
                f"pages; missing on channels {', '.join(missing)}"
            )
        reference = float(declared[0])
        inconsistent = [
            (index, float(value))
            for index, value in enumerate(declared)
            if not math.isclose(float(value), reference, rel_tol=1e-9, abs_tol=1e-12)
        ]
        if inconsistent:
            values = ", ".join(f"channel {index}={value:g}" for index, value in inconsistent)
            raise ValueError(
                f"Akoya channel pages report inconsistent PixelSizeMicrons; "
                f"channel 0={reference:g}, {values}"
            )
        return PixelSize(reference, reference, "µm")

    fallback = [pixel_size_from_tiff_resolution(page) for page in pages]
    if not any(item is not None for item in fallback):
        return None
    if not all(item is not None for item in fallback):
        missing = [str(index) for index, item in enumerate(fallback) if item is None]
        raise ValueError(
            "TIFF resolution calibration is present on only some Akoya channel pages; "
            f"missing on channels {', '.join(missing)}"
        )
    reference = fallback[0]
    assert reference is not None
    for index, item in enumerate(fallback[1:], start=1):
        assert item is not None
        converted = item.converted_to(reference.unit)
        if not (
            math.isclose(converted.x, reference.x, rel_tol=1e-9, abs_tol=1e-12)
            and math.isclose(converted.y, reference.y, rel_tol=1e-9, abs_tol=1e-12)
        ):
            raise ValueError(
                "Akoya channel pages report inconsistent TIFF resolution calibration; "
                f"channel 0={reference.to_tuple()}, channel {index}={item.to_tuple()}"
            )
    return reference
