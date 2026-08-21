from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Any, Iterable

from jsonschema import Draft202012Validator
from lxml import etree

_OME_NAMESPACE = "http://www.openmicroscopy.org/Schemas/OME/2016-06"
_PYRAMID_NAMESPACE = "openmicroscopy.org/PyramidResolution"
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
    """Assessment of OME metadata against omeify's MITI header profile.

    ``extra_metadata`` is informational. MITI is a minimum-information profile,
    so additional OME metadata does not make an otherwise conforming header
    invalid. The list identifies metadata outside omeify's deliberately small
    output vocabulary so callers can review it before sharing a report or file.
    """

    is_valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    missing_fields: tuple[str, ...] = ()
    extra_metadata: tuple[str, ...] = ()
    profile: str = _PROFILE_NAME
    source: str = _MITI_HEADER_SOURCE
    schema_resource: str = _SCHEMA_RESOURCE

    @property
    def status(self) -> str:
        return "pass" if self.is_valid else "fail"

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "source": self.source,
            "schema_resource": self.schema_resource,
            "status": self.status,
            "is_valid": self.is_valid,
            "errors": list(self.errors),
            "missing_fields": list(self.missing_fields),
            "warnings": list(self.warnings),
            "has_extra_metadata": bool(self.extra_metadata),
            "extra_metadata": list(self.extra_metadata),
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
        return None
    try:
        return int(raw)
    except ValueError:
        errors.append(f"{field} must be an integer, found {raw!r}")
        return None


def _parse_float(raw: str | None, field: str, errors: list[str]) -> float | None:
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError:
        errors.append(f"{field} must be numeric, found {raw!r}")
        return None


def _deduplicate(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _field_label(path: Iterable[Any], image_index: int) -> str:
    parts = list(path)
    prefix = f"Image[{image_index}]"
    direct = {
        "image_id": f"{prefix}/@ID",
        "pixels_id": f"{prefix}/Pixels/@ID",
        "big_endian": f"{prefix}/Pixels/@BigEndian",
        "dimension_order": f"{prefix}/Pixels/@DimensionOrder",
        "interleaved": f"{prefix}/Pixels/@Interleaved",
        "physical_size_x": f"{prefix}/Pixels/@PhysicalSizeX",
        "physical_size_x_unit": f"{prefix}/Pixels/@PhysicalSizeXUnit",
        "physical_size_y": f"{prefix}/Pixels/@PhysicalSizeY",
        "physical_size_y_unit": f"{prefix}/Pixels/@PhysicalSizeYUnit",
        "physical_size_z": f"{prefix}/Pixels/@PhysicalSizeZ",
        "physical_size_z_unit": f"{prefix}/Pixels/@PhysicalSizeZUnit",
        "significant_bits": f"{prefix}/Pixels/@SignificantBits",
        "size_c": f"{prefix}/Pixels/@SizeC",
        "size_t": f"{prefix}/Pixels/@SizeT",
        "size_x": f"{prefix}/Pixels/@SizeX",
        "size_y": f"{prefix}/Pixels/@SizeY",
        "size_z": f"{prefix}/Pixels/@SizeZ",
        "pixel_type": f"{prefix}/Pixels/@Type",
        "channels": f"{prefix}/Pixels/Channel",
        "tiff_data": f"{prefix}/Pixels/TiffData",
    }
    if not parts:
        return prefix
    first = str(parts[0])
    if first == "channels":
        if len(parts) == 1:
            return direct[first]
        channel_index = parts[1]
        base = f"{prefix}/Pixels/Channel[{channel_index}]"
        if len(parts) == 2:
            return base
        attr = {
            "id": "ID",
            "name": "Name",
            "samples_per_pixel": "SamplesPerPixel",
        }.get(str(parts[2]), str(parts[2]))
        return f"{base}/@{attr}"
    if first == "tiff_data":
        if len(parts) == 1:
            return direct[first]
        data_index = parts[1]
        base = f"{prefix}/Pixels/TiffData[{data_index}]"
        if len(parts) == 2:
            return base
        attr = {"ifd": "IFD", "plane_count": "PlaneCount"}.get(
            str(parts[2]), str(parts[2])
        )
        return f"{base}/@{attr}"
    return direct.get(first, f"{prefix}/Pixels/{'/'.join(str(item) for item in parts)}")


def _required_missing_fields(error: Any, image_index: int) -> list[str]:
    missing: list[str] = []
    path = list(error.path)
    if error.validator == "required" and isinstance(error.instance, dict):
        for required in error.validator_value:
            if required not in error.instance:
                missing.append(_field_label([*path, required], image_index))
    elif error.validator == "minLength" and error.instance == "":
        missing.append(_field_label(path, image_index))
    elif error.validator == "minItems" and not error.instance:
        missing.append(_field_label(path, image_index))
    return missing


def _normalized_record(
    root: etree._Element,
    image: etree._Element,
    image_index: int,
    errors: list[str],
) -> tuple[dict[str, Any], bool | None]:
    pixels_nodes = [child for child in image if _local_name(child.tag) == "Pixels"]
    if not pixels_nodes:
        errors.append(f"Image[{image_index}] does not contain a Pixels element")
        return {}, None
    pixels = pixels_nodes[0]

    record: dict[str, Any] = {}
    for xml_name, json_name in {"UUID": "uuid", "Creator": "creator"}.items():
        value = root.get(xml_name)
        if value is not None:
            record[json_name] = value

    record["image_id"] = image.get("ID") or ""
    record["pixels_id"] = pixels.get("ID") or ""

    local_errors: list[str] = []
    big_endian = _parse_bool(pixels.get("BigEndian"), "Pixels BigEndian", local_errors)
    if big_endian is not None:
        record["big_endian"] = big_endian
    interleaved = _parse_bool(pixels.get("Interleaved"), "Pixels Interleaved", local_errors)
    if interleaved is not None:
        record["interleaved"] = interleaved

    record["dimension_order"] = pixels.get("DimensionOrder") or ""
    record["physical_size_x_unit"] = pixels.get("PhysicalSizeXUnit") or ""
    record["physical_size_y_unit"] = pixels.get("PhysicalSizeYUnit") or ""
    record["pixel_type"] = pixels.get("Type") or ""

    for xml_name, json_name in {
        "PhysicalSizeX": "physical_size_x",
        "PhysicalSizeY": "physical_size_y",
    }.items():
        value = _parse_float(pixels.get(xml_name), f"Pixels {xml_name}", local_errors)
        if value is not None:
            record[json_name] = value

    for xml_name, json_name in {
        "SignificantBits": "significant_bits",
        "SizeC": "size_c",
        "SizeT": "size_t",
        "SizeX": "size_x",
        "SizeY": "size_y",
        "SizeZ": "size_z",
    }.items():
        value = _parse_int(pixels.get(xml_name), f"Pixels {xml_name}", local_errors)
        if value is not None:
            record[json_name] = value

    if "PhysicalSizeZ" in pixels.attrib:
        value = _parse_float(pixels.get("PhysicalSizeZ"), "Pixels PhysicalSizeZ", local_errors)
        if value is not None:
            record["physical_size_z"] = value
    if "PhysicalSizeZUnit" in pixels.attrib:
        record["physical_size_z_unit"] = pixels.get("PhysicalSizeZUnit") or ""

    channels: list[dict[str, Any]] = []
    tiff_data: list[dict[str, Any]] = []
    for child_index, element in enumerate(pixels):
        local = _local_name(element.tag)
        if local == "Channel":
            samples = _parse_int(
                element.get("SamplesPerPixel"),
                f"Channel {len(channels)} SamplesPerPixel",
                local_errors,
            )
            channel_record: dict[str, Any] = {
                "id": element.get("ID") or "",
                "name": element.get("Name") or "",
            }
            if samples is not None:
                channel_record["samples_per_pixel"] = samples
            channels.append(channel_record)
        elif local == "TiffData":
            ifd = _parse_int(
                element.get("IFD", "0"),
                f"TiffData {len(tiff_data)} IFD",
                local_errors,
            )
            plane_count = _parse_int(
                element.get("PlaneCount"),
                f"TiffData {len(tiff_data)} PlaneCount",
                local_errors,
            )
            item: dict[str, Any] = {}
            if ifd is not None:
                item["ifd"] = ifd
            if plane_count is not None:
                item["plane_count"] = plane_count
            tiff_data.append(item)
    record["channels"] = channels
    record["tiff_data"] = tiff_data
    errors.extend(f"Image[{image_index}]: {item}" for item in local_errors)
    return record, interleaved


def _validate_record(
    record: dict[str, Any],
    *,
    image_index: int,
    interleaved: bool | None,
    errors: list[str],
    missing_fields: list[str],
) -> None:
    if not record:
        missing_fields.append(f"Image[{image_index}]/Pixels")
        return

    for schema_error in sorted(
        _validator().iter_errors(record),
        key=lambda item: tuple(str(part) for part in item.path),
    ):
        label = _field_label(schema_error.path, image_index)
        errors.append(f"{label}: {schema_error.message}")
        missing_fields.extend(_required_missing_fields(schema_error, image_index))

    channels = record.get("channels", [])
    tiff_data = record.get("tiff_data", [])
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
                f"Image[{image_index}]/Pixels declares SizeC={size_c}, but Channel "
                f"SamplesPerPixel values sum to {total_samples}"
            )

    if interleaved is False and channels and any(value != 1 for value in samples_per_pixel):
        errors.append(
            f"Image[{image_index}]: the planar omeify profile requires "
            "SamplesPerPixel=1 for every Channel"
        )
    if interleaved is True:
        if len(channels) != 1 or samples_per_pixel != [3] or size_c != 3:
            errors.append(
                f"Image[{image_index}]: the interleaved RGB omeify profile requires "
                "SizeC=3 and one Channel with SamplesPerPixel=3"
            )

    if len(tiff_data) != 1:
        errors.append(
            f"Image[{image_index}]: the omeify header profile requires exactly one "
            f"TiffData element; found {len(tiff_data)}"
        )
    elif tiff_data[0].get("ifd") != 0:
        errors.append(f"Image[{image_index}]: omeify TiffData must begin at IFD=0")

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
                f"Image[{image_index}]: TiffData PlaneCount totals {actual_planes}, but "
                f"logical Channel count times SizeZ*SizeT is {expected_planes}"
            )

    expected_bits = _OME_PIXEL_BITS.get(pixel_type)
    if (
        expected_bits is not None
        and significant_bits is not None
        and significant_bits != expected_bits
    ):
        errors.append(
            f"Image[{image_index}]/Pixels SignificantBits={significant_bits!r} does not "
            f"match Type={pixel_type!r} storage width ({expected_bits})"
        )

    channel_ids = [channel.get("id") for channel in channels]
    if len(channel_ids) != len(set(channel_ids)):
        errors.append(f"Image[{image_index}]: Channel IDs must be unique within the Image")


_ALLOWED_ATTRIBUTES: dict[str, set[str]] = {
    "OME": {"Creator", "UUID", "schemaLocation"},
    "Image": {"ID"},
    "Pixels": {
        "ID",
        "BigEndian",
        "DimensionOrder",
        "Interleaved",
        "PhysicalSizeX",
        "PhysicalSizeXUnit",
        "PhysicalSizeY",
        "PhysicalSizeYUnit",
        "PhysicalSizeZ",
        "PhysicalSizeZUnit",
        "SignificantBits",
        "SizeC",
        "SizeT",
        "SizeX",
        "SizeY",
        "SizeZ",
        "Type",
    },
    "Channel": {"ID", "Name", "SamplesPerPixel"},
    "LightPath": set(),
    "TiffData": {"IFD", "PlaneCount", "FirstC", "FirstZ", "FirstT"},
    "AnnotationRef": {"ID"},
    "StructuredAnnotations": set(),
    "MapAnnotation": {"ID", "Namespace"},
    "Value": set(),
    "M": {"K"},
}
_ALLOWED_CHILDREN: dict[str, set[str]] = {
    "OME": {"Image", "StructuredAnnotations"},
    "Image": {"Pixels", "AnnotationRef"},
    "Pixels": {"Channel", "TiffData"},
    "Channel": {"LightPath"},
    "LightPath": set(),
    "TiffData": set(),
    "AnnotationRef": set(),
    "StructuredAnnotations": {"MapAnnotation"},
    "MapAnnotation": {"Value"},
    "Value": {"M"},
    "M": set(),
}


def _indexed_child_path(parent_path: str, child: etree._Element) -> str:
    tag = _local_name(child.tag)
    parent = child.getparent()
    if parent is None:
        return f"{parent_path}/{tag}"
    siblings = [item for item in parent if _local_name(item.tag) == tag]
    index = siblings.index(child)
    return f"{parent_path}/{tag}[{index}]"


def _find_extra_metadata(root: etree._Element) -> tuple[str, ...]:
    extras: list[str] = []

    def visit(element: etree._Element, path: str) -> None:
        tag = _local_name(element.tag)
        allowed_attributes = _ALLOWED_ATTRIBUTES.get(tag, set())
        for raw_name in element.attrib:
            name = _local_name(raw_name)
            if name not in allowed_attributes:
                extras.append(f"{path}/@{name}")

        if tag == "MapAnnotation" and element.get("Namespace") != _PYRAMID_NAMESPACE:
            extras.append(path)
            return

        allowed_children = _ALLOWED_CHILDREN.get(tag, set())
        for child in element:
            child_path = _indexed_child_path(path, child)
            child_tag = _local_name(child.tag)
            if child_tag not in allowed_children:
                extras.append(child_path)
                continue
            visit(child, child_path)

    visit(root, "OME")
    return _deduplicate(sorted(extras))


def validate_miti_ome_tiff_header(xml_string: str) -> MITIHeaderValidation:
    """Validate all OME Images against omeify's MITI-aligned header profile."""

    errors: list[str] = []
    missing_fields: list[str] = []
    warnings: list[str] = []

    parser = etree.XMLParser(resolve_entities=False, no_network=True)
    try:
        root = etree.fromstring(xml_string.encode("utf-8"), parser=parser)
    except etree.XMLSyntaxError as exc:
        return MITIHeaderValidation(
            is_valid=False,
            errors=(f"OME-XML is not well formed: {exc}",),
            warnings=(),
            missing_fields=(),
            extra_metadata=(),
        )

    root_namespace = etree.QName(root).namespace
    if _local_name(root.tag) != "OME":
        errors.append(f"Root element must be OME, found {_local_name(root.tag)!r}")
    if root_namespace != _OME_NAMESPACE:
        errors.append(
            f"OME namespace must be {_OME_NAMESPACE!r}, found {root_namespace!r}"
        )

    images = [child for child in root if _local_name(child.tag) == "Image"]
    if not images:
        return MITIHeaderValidation(
            is_valid=False,
            errors=("OME-XML does not contain an Image element",),
            warnings=(),
            missing_fields=("OME/Image",),
            extra_metadata=_find_extra_metadata(root),
        )
    if len(images) > 1:
        warnings.append(
            f"The omeify output profile normally contains one Image; validating all {len(images)}"
        )

    for image_index, image in enumerate(images):
        record, interleaved = _normalized_record(root, image, image_index, errors)
        _validate_record(
            record,
            image_index=image_index,
            interleaved=interleaved,
            errors=errors,
            missing_fields=missing_fields,
        )

    deduplicated_errors = _deduplicate(errors)
    return MITIHeaderValidation(
        is_valid=not deduplicated_errors,
        errors=deduplicated_errors,
        warnings=_deduplicate(warnings),
        missing_fields=_deduplicate(missing_fields),
        extra_metadata=_find_extra_metadata(root),
    )
