from __future__ import annotations

import math
import re
from dataclasses import dataclass
from xml.etree import ElementTree

import tifffile

_XML_ENCODING_DECLARATION = re.compile(
    r"(<\?xml\b[^>]*?)\s+encoding=(['\"])[^'\"]+\2",
    flags=re.IGNORECASE,
)


def _local_name(tag: str) -> str:
    return str(tag).rsplit("}", 1)[-1]


def _parse_xml(description: str | bytes | None) -> ElementTree.Element:
    if not description:
        raise ValueError("Indica mIF TIFF is missing its ImageDescription XML")
    payload: str | bytes
    if isinstance(description, bytes):
        payload = description
    else:
        payload = _XML_ENCODING_DECLARATION.sub(r"\1", str(description), count=1)
    try:
        root = ElementTree.fromstring(payload)
    except (ElementTree.ParseError, TypeError, ValueError) as exc:
        raise ValueError("Indica mIF TIFF has unreadable ImageDescription XML") from exc
    if _local_name(root.tag) != "indica":
        raise ValueError(
            "Indica mIF TIFF requires an ImageDescription with an <indica> root element"
        )
    return root


def _required_int(element: ElementTree.Element, name: str, *, minimum: int) -> int:
    raw = element.get(name)
    if raw is None:
        raise ValueError(f"Indica <{_local_name(element.tag)}> is missing {name!r}")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"Indica <{_local_name(element.tag)}> has invalid {name}={raw!r}"
        ) from exc
    if value < minimum:
        raise ValueError(
            f"Indica <{_local_name(element.tag)}> requires {name}>={minimum}, found {value}"
        )
    return value


def _optional_int(element: ElementTree.Element, name: str) -> int | None:
    raw = element.get(name)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(
            f"Indica <{_local_name(element.tag)}> has invalid {name}={raw!r}"
        ) from exc


def _optional_float(element: ElementTree.Element, name: str) -> float | None:
    raw = element.get(name)
    if raw is None or raw == "":
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"Indica <{_local_name(element.tag)}> has invalid {name}={raw!r}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            f"Indica <{_local_name(element.tag)}> has non-finite {name}={raw!r}"
        )
    return value


@dataclass(frozen=True, slots=True)
class IndicaMIFChannelMetadata:
    id: int
    name: str
    rgb: int | None
    minimum: float | None
    maximum: float | None

    def as_dict(self) -> dict[str, int | float | str | None]:
        return {
            "id": self.id,
            "name": self.name,
            "rgb": self.rgb,
            "min": self.minimum,
            "max": self.maximum,
        }


@dataclass(frozen=True, slots=True)
class IndicaMIFDimension:
    size_x: int
    size_y: int
    ifd: int
    channel: int
    level: int


@dataclass(frozen=True, slots=True)
class IndicaMIFMetadata:
    channels: tuple[IndicaMIFChannelMetadata, ...]
    dimensions: tuple[IndicaMIFDimension, ...]
    objective: float | None

    @property
    def channel_ids(self) -> tuple[int, ...]:
        return tuple(channel.id for channel in self.channels)

    def dimensions_by_level(self) -> dict[int, dict[int, IndicaMIFDimension]]:
        levels: dict[int, dict[int, IndicaMIFDimension]] = {}
        for dimension in self.dimensions:
            by_channel = levels.setdefault(dimension.level, {})
            if dimension.channel in by_channel:
                raise ValueError(
                    "Indica ImageDescription contains duplicate dimensions for "
                    f"level={dimension.level}, channel={dimension.channel}"
                )
            by_channel[dimension.channel] = dimension
        return levels


def parse_indica_mif_metadata(description: str | bytes | None) -> IndicaMIFMetadata:
    """Parse the channel and IFD mapping from an Indica Labs mIF description."""

    root = _parse_xml(description)
    image = next((item for item in root if _local_name(item.tag) == "image"), None)
    if image is None:
        raise ValueError("Indica ImageDescription does not contain an <image> element")

    channels_element = next(
        (item for item in image if _local_name(item.tag) == "channels"),
        None,
    )
    if channels_element is None:
        raise ValueError("Indica ImageDescription does not contain <image>/<channels>")

    channels: list[IndicaMIFChannelMetadata] = []
    seen_channel_ids: set[int] = set()
    for element in channels_element:
        if _local_name(element.tag) != "channel":
            continue
        channel_id = _required_int(element, "id", minimum=0)
        if channel_id in seen_channel_ids:
            raise ValueError(f"Indica channel ID {channel_id} occurs more than once")
        name = (element.get("name") or "").strip()
        if not name:
            raise ValueError(f"Indica channel {channel_id} is missing a non-empty name")
        seen_channel_ids.add(channel_id)
        channels.append(
            IndicaMIFChannelMetadata(
                id=channel_id,
                name=name,
                rgb=_optional_int(element, "rgb"),
                minimum=_optional_float(element, "min"),
                maximum=_optional_float(element, "max"),
            )
        )
    if not channels:
        raise ValueError("Indica ImageDescription does not define any channels")
    channels.sort(key=lambda item: item.id)

    pixels_element = next(
        (item for item in image if _local_name(item.tag) == "pixels"),
        None,
    )
    if pixels_element is None:
        raise ValueError("Indica ImageDescription does not contain <image>/<pixels>")

    dimensions: list[IndicaMIFDimension] = []
    for element in pixels_element:
        if _local_name(element.tag) != "dimension":
            continue
        dimensions.append(
            IndicaMIFDimension(
                size_x=_required_int(element, "sizeX", minimum=1),
                size_y=_required_int(element, "sizeY", minimum=1),
                ifd=_required_int(element, "ifd", minimum=0),
                channel=_required_int(element, "channel", minimum=0),
                level=_required_int(element, "level", minimum=0),
            )
        )
    if not dimensions:
        raise ValueError("Indica ImageDescription does not define any pixel dimensions")

    objective = None
    objective_element = next(
        (item for item in image if _local_name(item.tag) == "objective"),
        None,
    )
    if objective_element is not None:
        objective = _optional_float(objective_element, "value")
        if objective is not None and objective <= 0:
            raise ValueError(
                f"Indica objective value must be positive when present, found {objective}"
            )

    metadata = IndicaMIFMetadata(
        channels=tuple(channels),
        dimensions=tuple(dimensions),
        objective=objective,
    )
    levels = metadata.dimensions_by_level()
    if 0 not in levels:
        raise ValueError("Indica ImageDescription does not define full-resolution level 0")
    level_ids = sorted(levels)
    if level_ids != list(range(level_ids[-1] + 1)):
        raise ValueError(
            "Indica ImageDescription pyramid levels must be contiguous from level 0; "
            f"found {level_ids}"
        )

    expected_channels = set(metadata.channel_ids)
    for level, by_channel in sorted(levels.items()):
        observed_channels = set(by_channel)
        if observed_channels != expected_channels:
            missing = sorted(expected_channels - observed_channels)
            unknown = sorted(observed_channels - expected_channels)
            details: list[str] = []
            if missing:
                details.append("missing channels " + ", ".join(str(item) for item in missing))
            if unknown:
                details.append("unknown channels " + ", ".join(str(item) for item in unknown))
            raise ValueError(
                f"Indica pyramid level {level} does not match the declared channel set: "
                + "; ".join(details)
            )
    return metadata


def build_indica_mif_series(
    tiff: tifffile.TiffFile,
    metadata: IndicaMIFMetadata,
) -> tifffile.TiffPageSeries:
    """Build one pyramidal CYX series from the Indica ImageDescription IFD map."""

    levels_by_channel = metadata.dimensions_by_level()
    channel_ids = metadata.channel_ids
    used_ifds: set[int] = set()
    levels: list[tifffile.TiffPageSeries] = []

    for level_index, level in enumerate(sorted(levels_by_channel)):
        dimensions = levels_by_channel[level]
        level_pages: list[tifffile.TiffPage] = []
        expected_shape: tuple[int, int] | None = None
        expected_dtype = None

        for channel_id in channel_ids:
            dimension = dimensions[channel_id]
            if dimension.ifd in used_ifds:
                raise ValueError(f"Indica IFD {dimension.ifd} is mapped more than once")
            if dimension.ifd >= len(tiff.pages):
                raise ValueError(
                    f"Indica dimension references IFD {dimension.ifd}, but the TIFF contains "
                    f"only {len(tiff.pages)} top-level IFDs"
                )
            page = tiff.pages[dimension.ifd].aspage()
            actual_shape = (int(page.imagelength), int(page.imagewidth))
            declared_shape = (dimension.size_y, dimension.size_x)
            if actual_shape != declared_shape:
                raise ValueError(
                    f"Indica IFD {dimension.ifd} has shape {actual_shape}, but its "
                    f"ImageDescription dimension declares {declared_shape}"
                )
            if expected_shape is None:
                expected_shape = actual_shape
                expected_dtype = page.dtype
            elif actual_shape != expected_shape:
                raise ValueError(
                    f"Indica pyramid level {level} contains inconsistent channel shapes"
                )
            elif page.dtype != expected_dtype:
                raise TypeError(
                    f"Indica pyramid level {level} contains inconsistent channel dtypes"
                )
            if int(page.samplesperpixel) != 1:
                raise ValueError(
                    f"Indica mIF IFD {dimension.ifd} requires SamplesPerPixel=1, found "
                    f"{int(page.samplesperpixel)}"
                )
            used_ifds.add(dimension.ifd)
            level_pages.append(page)

        assert expected_shape is not None
        height, width = expected_shape
        shape: tuple[int, ...]
        axes: str
        if len(level_pages) == 1:
            shape = (height, width)
            axes = "YX"
        else:
            shape = (len(level_pages), height, width)
            axes = "CYX"
        levels.append(
            tifffile.TiffPageSeries(
                level_pages,
                shape,
                level_pages[0].dtype,
                axes,
                name="Baseline" if level_index == 0 else f"Level {level}",
                kind="indica",
            )
        )

    baseline = levels[0]
    baseline.levels = levels
    return baseline
