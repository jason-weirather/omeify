from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lxml import etree

_OME_NAMESPACE = "http://www.openmicroscopy.org/Schemas/OME/2016-06"
_MITI_HEADER_PROFILE = (
    "miti-consortium/MITI yaml/05-ome-tiff-header.yaml@"
    "6bffc09c7673500cffcdd237d7fc630a81406da9"
)
_MITI_PIXEL_TYPES = {"uint16", "float"}
_OME_DIMENSION_ORDERS = {
    "XYZCT",
    "XYZTC",
    "XYCZT",
    "XYCTZ",
    "XYTCZ",
    "XYTZC",
}


@dataclass(frozen=True)
class MITIHeaderValidation:
    """Result of checking the current MITI OME-TIFF header profile.

    This is intentionally separate from OME XSD validation.  An OME-TIFF can be
    perfectly valid OME while failing a narrower MITI constraint.  The current
    MITI YAML, for example, lists only ``uint16`` and ``float`` as accepted
    pixel types.  omeify reports that mismatch instead of silently changing a
    native ``uint8`` image into a different dtype.
    """

    strict_is_valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    profile: str = _MITI_HEADER_PROFILE

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "strict_is_valid": self.strict_is_valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def _positive_float(
    element: etree._Element,
    attribute: str,
    errors: list[str],
) -> float | None:
    raw = element.get(attribute)
    if raw is None or not raw.strip():
        errors.append(f"Pixels {attribute} is required")
        return None
    try:
        value = float(raw)
    except ValueError:
        errors.append(f"Pixels {attribute} must be numeric, found {raw!r}")
        return None
    if value <= 0:
        errors.append(f"Pixels {attribute} must be greater than zero, found {value}")
        return None
    return value


def _positive_int(
    element: etree._Element,
    attribute: str,
    errors: list[str],
) -> int | None:
    raw = element.get(attribute)
    if raw is None or not raw.strip():
        errors.append(f"Pixels {attribute} is required")
        return None
    try:
        value = int(raw)
    except ValueError:
        errors.append(f"Pixels {attribute} must be an integer, found {raw!r}")
        return None
    if value <= 0:
        errors.append(f"Pixels {attribute} must be greater than zero, found {value}")
        return None
    return value


def validate_miti_ome_tiff_header(xml_string: str) -> MITIHeaderValidation:
    """Check one OME image against MITI's current header-level requirements.

    The full MITI standard also requires file-level, channel-level,
    biospecimen, acquisition, reagent, and processing metadata.  This function
    validates only ``yaml/05-ome-tiff-header.yaml``.
    """

    errors: list[str] = []
    warnings: list[str] = []

    try:
        root = etree.fromstring(xml_string.encode("utf-8"))
    except etree.XMLSyntaxError as exc:
        return MITIHeaderValidation(
            strict_is_valid=False,
            errors=(f"OME-XML is not well formed: {exc}",),
            warnings=(),
        )

    ns = {"ome": _OME_NAMESPACE}
    images = root.xpath("./ome:Image", namespaces=ns)
    if not images:
        return MITIHeaderValidation(
            strict_is_valid=False,
            errors=("OME-XML does not contain an Image element",),
            warnings=(),
        )
    if len(images) > 1:
        warnings.append(
            "Header validator checked only the first Image element; the mIF writer "
            "normally emits one"
        )
    image = images[0]

    image_id = image.get("ID")
    if image_id is None or not image_id.strip():
        errors.append("Image ID is required")

    pixels_nodes = image.xpath("./ome:Pixels", namespaces=ns)
    if not pixels_nodes:
        return MITIHeaderValidation(
            strict_is_valid=False,
            errors=tuple(errors + ["Image does not contain a Pixels element"]),
            warnings=tuple(warnings),
        )
    pixels = pixels_nodes[0]

    big_endian = (pixels.get("BigEndian") or "").lower()
    if big_endian not in {"true", "false"}:
        errors.append("Pixels BigEndian is required and must be true or false")

    dimension_order = pixels.get("DimensionOrder") or ""
    if dimension_order not in _OME_DIMENSION_ORDERS:
        errors.append(
            "Pixels DimensionOrder is required and must be a valid OME dimension order; "
            f"found {dimension_order!r}"
        )

    _positive_float(pixels, "PhysicalSizeX", errors)
    _positive_float(pixels, "PhysicalSizeY", errors)
    for unit_attribute in ("PhysicalSizeXUnit", "PhysicalSizeYUnit"):
        unit = pixels.get(unit_attribute)
        if unit is None or not unit.strip():
            errors.append(f"Pixels {unit_attribute} is required")

    size_c = _positive_int(pixels, "SizeC", errors)
    size_t = _positive_int(pixels, "SizeT", errors)
    _positive_int(pixels, "SizeX", errors)
    _positive_int(pixels, "SizeY", errors)
    size_z = _positive_int(pixels, "SizeZ", errors)

    if size_z is not None and size_z > 1:
        _positive_float(pixels, "PhysicalSizeZ", errors)
        unit = pixels.get("PhysicalSizeZUnit")
        if unit is None or not unit.strip():
            errors.append("Pixels PhysicalSizeZUnit is required when SizeZ is greater than one")

    pixel_type = pixels.get("Type") or ""
    if pixel_type not in _MITI_PIXEL_TYPES:
        errors.append(
            f"Pixels Type={pixel_type!r} is valid OME only if allowed by the OME schema, but "
            "the current MITI header YAML accepts only 'uint16' or 'float'.  omeify preserves "
            "the native dtype rather than silently upcasting or rescaling it."
        )

    channels = pixels.xpath("./ome:Channel", namespaces=ns)
    if size_c is not None and len(channels) != size_c:
        errors.append(
            f"Pixels declares SizeC={size_c} but contains {len(channels)} Channel elements"
        )
    for index, channel in enumerate(channels):
        name = channel.get("Name")
        if name is None or not name.strip():
            errors.append(f"Channel {index} Name is required")

    tiff_data = pixels.xpath("./ome:TiffData", namespaces=ns)
    if not tiff_data:
        errors.append("At least one TiffData element with PlaneCount is required")
    else:
        plane_counts: list[int] = []
        for index, element in enumerate(tiff_data):
            raw = element.get("PlaneCount")
            if raw is None:
                errors.append(f"TiffData {index} PlaneCount is required by MITI")
                continue
            try:
                value = int(raw)
            except ValueError:
                errors.append(f"TiffData {index} PlaneCount must be an integer, found {raw!r}")
                continue
            if value <= 0:
                errors.append(f"TiffData {index} PlaneCount must be greater than zero")
                continue
            plane_counts.append(value)

        if size_c is not None and size_z is not None and size_t is not None and plane_counts:
            expected = size_c * size_z * size_t
            actual = sum(plane_counts)
            if actual != expected:
                errors.append(
                    f"TiffData PlaneCount totals {actual}, but SizeC*SizeZ*SizeT is {expected}"
                )

    return MITIHeaderValidation(
        strict_is_valid=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )
