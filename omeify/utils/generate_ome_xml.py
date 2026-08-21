from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from lxml import etree
from tifffile import OmeXml

from omeify._version import __version__

_OME_NAMESPACE = "http://www.openmicroscopy.org/Schemas/OME/2016-06"
_PYRAMID_NAMESPACE = "openmicroscopy.org/PyramidResolution"


def _level_yx(shape: Sequence[int], axes: str) -> tuple[int, int]:
    if len(shape) != len(axes):
        raise ValueError(f"Shape {tuple(shape)} does not match axes {axes!r}")
    return int(shape[axes.index("Y")]), int(shape[axes.index("X")])


def generate_ome_xml(
    tiff_features: Any,
    level_shapes: Sequence[Sequence[int]],
    *,
    display_uuid: bool = True,
    rename_channels: Mapping[str, str] | None = None,
    output_byteorder: Literal["<", ">"] = "<",
) -> dict[str, str | None]:
    """Generate minimal OME-XML for one planar mIF or interleaved RGB image."""

    if output_byteorder not in {"<", ">"}:
        raise ValueError("output_byteorder must be '<' or '>'")

    rename_channels = dict(rename_channels or {})
    channel_names = [rename_channels.get(name, name) for name in tiff_features.channel_names]
    file_uuid = str(uuid.uuid4())

    ome = OmeXml(
        Creator=f"omeify {__version__}",
        UUID=file_uuid,
    )

    image_metadata: dict[str, object] = {
        "SignificantBits": tiff_features.significant_bits,
        "PhysicalSizeX": tiff_features.physical_size_x_um,
        "PhysicalSizeXUnit": "µm",
        "PhysicalSizeY": tiff_features.physical_size_y_um,
        "PhysicalSizeYUnit": "µm",
        "Channel": {"Name": channel_names},
    }

    if len(level_shapes) > 1:
        annotation: dict[str, str] = {"Namespace": _PYRAMID_NAMESPACE}
        for index, shape in enumerate(level_shapes[1:], start=1):
            level_y, level_x = _level_yx(shape, tiff_features.output_axes)
            annotation[str(index)] = f"{level_x} {level_y}"
        image_metadata["MapAnnotation"] = annotation

    stored_shape = (
        tiff_features.plane_count,
        1,
        1,
        tiff_features.size_y,
        tiff_features.size_x,
        tiff_features.samples_per_pixel,
    )
    ome.addimage(
        tiff_features.dtype,
        tiff_features.output_shape,
        stored_shape,
        axes=tiff_features.output_axes,
        **image_metadata,
    )

    root = etree.fromstring(ome.tostring(declaration=True).encode("utf-8"))

    # OME Image/@Name is optional. Tifffile supplies a default (for example,
    # "Image0") when no name is provided, so remove it explicitly to keep the
    # generated header minimized and free of unnecessary image labels.
    namespace = {"ome": _OME_NAMESPACE}
    image = root.find("./ome:Image", namespaces=namespace)
    if image is None:
        raise RuntimeError("tifffile generated OME-XML without an Image element")
    image.attrib.pop("Name", None)

    if not display_uuid:
        root.attrib.pop("UUID", None)

    pixels = root.find(".//ome:Pixels", namespaces=namespace)
    if pixels is None:
        raise RuntimeError("tifffile generated OME-XML without a Pixels element")
    # These values describe the file that omeify writes, not the source file.
    pixels.set("BigEndian", "true" if output_byteorder == ">" else "false")
    # Preserve the historical omeify dimension-order convention. With
    # SizeZ=SizeT=1, XYZCT maps planar channel pages in order and also
    # correctly describes one interleaved RGB plane.
    pixels.set("DimensionOrder", "XYZCT")
    pixels.set("Interleaved", "true" if tiff_features.is_rgb else "false")

    xml_string = etree.tostring(
        root,
        encoding="utf-8",
        xml_declaration=True,
        pretty_print=False,
    ).decode("utf-8")
    return {
        "xml_string": xml_string,
        "uuid": file_uuid if display_uuid else None,
    }
