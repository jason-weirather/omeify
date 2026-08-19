from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Any

from jsonschema import Draft202012Validator
from lxml import etree

_OME_NAMESPACE = "http://www.openmicroscopy.org/Schemas/OME/2016-06"
_MITI_HEADER_SOURCE = (
    "miti-consortium/MITI yaml/05-ome-tiff-header.yaml@"
    "6bffc09c7673500cffcdd237d7fc630a81406da9"
)
_PROFILE_NAME = "omeify MITI OME-TIFF header profile"
_SCHEMA_RESOURCE = "omeify.schemas/miti_ome_tiff_header.schema.json"
_OME_PIXEL_BITS = {
    "int8": 8,
    "int16": 16,
    "int32": 32,
    "uint8": 8,
    "uint16": 16,
    "uint32": 32,
    "float": 32,
    "double": 64,
}


@dataclass(frozen=True)
class MITIHeaderValidation:
    """Result of validating omeify's normalized MITI header minimums.

    The upstream MITI YAML is used as the field-level source, but omeify treats
    its omission of valid scalar OME types such as ``uint8`` as an incomplete
    enumeration. The bundled JSON Schema therefore accepts the OME scalar
    numeric types that the current writers can preserve without casting.
    """

    is_valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    profile: str = _PROFILE_NAME
    source: str = _MITI_HEADER_SOURCE
    schema_resource: str = _SCHEMA_RESOURCE

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "source": self.source,
            "schema_resource": self.schema_resource,
            "is_valid": self.is_valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    resource = files("omeify.schemas").joinpath("miti_ome_tiff_header.schema.json")
    schema = json.loads(resource.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _parse_bool(raw: str | None, field: str, errors: list[str]) -> bool | None:
    if raw is None:
        errors.append(f"{field} is required")
        return None
    normalized = raw.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    errors.append(f"{field} must be true or false, found {raw!r}")
    return None


def _parse_int(raw: str | None, field: str, errors: list[str]) -> int | None:
    if raw is None or not raw.strip():
        errors.append(f"{field} is required")
        return None
    try:
        return int(raw)
    except ValueError:
        errors.append(f"{field} must be an integer, found {raw!r}")
        return None


def _parse_float(raw: str | None, field: str, errors: list[str]) -> float | None:
    if raw is None or not raw.strip():
        errors.append(f"{field} is required")
        return None
    try:
        return float(raw)
    except ValueError:
        errors.append(f"{field} must be numeric, found {raw!r}")
        return None


def _schema_path(path: Any) -> str:
    rendered = "$"
    for item in path:
        if isinstance(item, int):
            rendered += f"[{item}]"
        else:
            rendered += f".{item}"
    return rendered


def _deduplicate(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def validate_miti_ome_tiff_header(xml_string: str) -> MITIHeaderValidation:
    """Validate an omeify planar mIF or interleaved RGB OME-TIFF header.

    JSON Schema handles required fields, primitive types, and controlled
    values. Python checks relationships that JSON Schema cannot express
    cleanly, including channel/sample count, plane count, sample layout, and
    the requirement that SignificantBits equal the declared pixel type width.
    """

    errors: list[str] = []
    warnings: list[str] = []

    parser = etree.XMLParser(resolve_entities=False, no_network=True)
    try:
        root = etree.fromstring(xml_string.encode("utf-8"), parser=parser)
    except etree.XMLSyntaxError as exc:
        return MITIHeaderValidation(
            is_valid=False,
            errors=(f"OME-XML is not well formed: {exc}",),
            warnings=(),
        )

    ns = {"ome": _OME_NAMESPACE}
    images = root.xpath("./ome:Image", namespaces=ns)
    if not images:
        return MITIHeaderValidation(
            is_valid=False,
            errors=("OME-XML does not contain an Image element",),
            warnings=(),
        )
    if len(images) > 1:
        warnings.append(
            "Header validation checked only the first Image element; omeify normally emits "
            "exactly one"
        )
    image = images[0]

    pixels_nodes = image.xpath("./ome:Pixels", namespaces=ns)
    if not pixels_nodes:
        return MITIHeaderValidation(
            is_valid=False,
            errors=("Image does not contain a Pixels element",),
            warnings=tuple(warnings),
        )
    pixels = pixels_nodes[0]

    record: dict[str, Any] = {}
    optional_root_fields = {
        "UUID": "uuid",
        "Creator": "creator",
    }
    for xml_name, json_name in optional_root_fields.items():
        value = root.get(xml_name)
        if value is not None:
            record[json_name] = value

    record["image_id"] = image.get("ID") or ""
    record["pixels_id"] = pixels.get("ID") or ""

    big_endian = _parse_bool(pixels.get("BigEndian"), "Pixels BigEndian", errors)
    if big_endian is not None:
        record["big_endian"] = big_endian
    interleaved = _parse_bool(pixels.get("Interleaved"), "Pixels Interleaved", errors)
    if interleaved is not None:
        record["interleaved"] = interleaved

    record["dimension_order"] = pixels.get("DimensionOrder") or ""
    record["physical_size_x_unit"] = pixels.get("PhysicalSizeXUnit") or ""
    record["physical_size_y_unit"] = pixels.get("PhysicalSizeYUnit") or ""
    record["pixel_type"] = pixels.get("Type") or ""

    numeric_float_fields = {
        "PhysicalSizeX": "physical_size_x",
        "PhysicalSizeY": "physical_size_y",
    }
    for xml_name, json_name in numeric_float_fields.items():
        value = _parse_float(pixels.get(xml_name), f"Pixels {xml_name}", errors)
        if value is not None:
            record[json_name] = value

    numeric_int_fields = {
        "SignificantBits": "significant_bits",
        "SizeC": "size_c",
        "SizeT": "size_t",
        "SizeX": "size_x",
        "SizeY": "size_y",
        "SizeZ": "size_z",
    }
    for xml_name, json_name in numeric_int_fields.items():
        value = _parse_int(pixels.get(xml_name), f"Pixels {xml_name}", errors)
        if value is not None:
            record[json_name] = value

    if "PhysicalSizeZ" in pixels.attrib:
        value = _parse_float(pixels.get("PhysicalSizeZ"), "Pixels PhysicalSizeZ", errors)
        if value is not None:
            record["physical_size_z"] = value
    if "PhysicalSizeZUnit" in pixels.attrib:
        record["physical_size_z_unit"] = pixels.get("PhysicalSizeZUnit") or ""

    channels: list[dict[str, Any]] = []
    for index, channel in enumerate(pixels):
        if _local_name(channel.tag) != "Channel":
            continue
        samples = _parse_int(
            channel.get("SamplesPerPixel"),
            f"Channel {index} SamplesPerPixel",
            errors,
        )
        channel_record: dict[str, Any] = {
            "id": channel.get("ID") or "",
            "name": channel.get("Name") or "",
        }
        if samples is not None:
            channel_record["samples_per_pixel"] = samples
        channels.append(channel_record)
    record["channels"] = channels

    tiff_data: list[dict[str, Any]] = []
    for index, element in enumerate(pixels):
        if _local_name(element.tag) != "TiffData":
            continue
        ifd = _parse_int(element.get("IFD", "0"), f"TiffData {index} IFD", errors)
        plane_count = _parse_int(
            element.get("PlaneCount"),
            f"TiffData {index} PlaneCount",
            errors,
        )
        item: dict[str, Any] = {}
        if ifd is not None:
            item["ifd"] = ifd
        if plane_count is not None:
            item["plane_count"] = plane_count
        tiff_data.append(item)
    record["tiff_data"] = tiff_data

    for schema_error in sorted(
        _validator().iter_errors(record),
        key=lambda item: tuple(str(part) for part in item.path),
    ):
        errors.append(f"{_schema_path(schema_error.path)}: {schema_error.message}")

    size_c = record.get("size_c")
    size_z = record.get("size_z")
    size_t = record.get("size_t")
    pixel_type = record.get("pixel_type")
    significant_bits = record.get("significant_bits")

    samples_per_pixel = [channel.get("samples_per_pixel") for channel in channels]
    if isinstance(size_c, int) and all(isinstance(value, int) for value in samples_per_pixel):
        total_samples = sum(samples_per_pixel)
        if total_samples != size_c:
            errors.append(
                f"Pixels declares SizeC={size_c}, but Channel SamplesPerPixel values "
                f"sum to {total_samples}"
            )

    if interleaved is False and channels and any(value != 1 for value in samples_per_pixel):
        errors.append(
            "The planar omeify profile requires SamplesPerPixel=1 for every Channel"
        )
    if interleaved is True:
        if len(channels) != 1 or samples_per_pixel != [3] or size_c != 3:
            errors.append(
                "The interleaved RGB omeify profile requires SizeC=3 and one Channel "
                "with SamplesPerPixel=3"
            )

    if len(tiff_data) != 1:
        errors.append(
            "The omeify image header must contain exactly one TiffData element; "
            f"found {len(tiff_data)}"
        )
    elif tiff_data[0].get("ifd") != 0:
        errors.append("The omeify TiffData element must begin at IFD=0")

    if (
        isinstance(size_z, int)
        and isinstance(size_t, int)
        and tiff_data
        and all(isinstance(item.get("plane_count"), int) for item in tiff_data)
    ):
        expected_planes = len(channels) * size_z * size_t
        actual_planes = sum(item["plane_count"] for item in tiff_data)
        if actual_planes != expected_planes:
            errors.append(
                f"TiffData PlaneCount totals {actual_planes}, but the logical Channel "
                f"count times SizeZ*SizeT is {expected_planes}"
            )

    expected_bits = _OME_PIXEL_BITS.get(pixel_type)
    if expected_bits is not None and significant_bits != expected_bits:
        errors.append(
            f"Pixels SignificantBits={significant_bits!r} does not match "
            f"Type={pixel_type!r} storage width ({expected_bits})"
        )

    channel_ids = [channel.get("id") for channel in channels]
    if len(channel_ids) != len(set(channel_ids)):
        errors.append("Channel IDs must be unique within the Image")

    deduplicated_errors = _deduplicate(errors)
    return MITIHeaderValidation(
        is_valid=not deduplicated_errors,
        errors=deduplicated_errors,
        warnings=_deduplicate(warnings),
    )
