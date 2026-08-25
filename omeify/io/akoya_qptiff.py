from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal, Sequence
from xml.etree import ElementTree

import tifffile

from .pixel_size import PixelSize, consistent_tiff_resolution_pixel_size

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


def consistent_akoya_pixel_size(
    metadata: Sequence[AkoyaQPIChannelMetadata],
    pages: Sequence[tifffile.TiffPage],
) -> PixelSize | None:
    if len(metadata) != len(pages):
        raise ValueError("Akoya channel metadata/page counts do not match")

    declared = [item.pixel_size_microns for item in metadata]
    present = [value is not None for value in declared]
    if all(present):
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

    fallback = consistent_tiff_resolution_pixel_size(pages)
    if fallback is not None:
        fallback_microns = fallback.converted_to("µm")
        inconsistent = [
            (index, float(value))
            for index, value in enumerate(declared)
            if value is not None
            and not (
                math.isclose(float(value), fallback_microns.x, rel_tol=1e-6, abs_tol=1e-9)
                and math.isclose(float(value), fallback_microns.y, rel_tol=1e-6, abs_tol=1e-9)
            )
        ]
        if inconsistent:
            values = ", ".join(f"channel {index}={value:g}" for index, value in inconsistent)
            raise ValueError(
                "Akoya PixelSizeMicrons disagrees with TIFF resolution calibration; "
                f"TIFF={fallback_microns.to_tuple()}, {values}"
            )
        return fallback

    if any(present):
        missing = [str(index) for index, value in enumerate(declared) if value is None]
        raise ValueError(
            "Akoya PixelSizeMicrons is present on only some full-resolution channel "
            f"pages; missing on channels {', '.join(missing)}"
        )
    return None
