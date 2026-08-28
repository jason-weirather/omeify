from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any, Literal

from lxml import etree
from tifffile import OmeXml

from omeify._version import __version__

_OME_NAMESPACE = "http://www.openmicroscopy.org/Schemas/OME/2016-06"
_PYRAMID_NAMESPACE = "openmicroscopy.org/PyramidResolution"
OMEIFY_PROVENANCE_NAMESPACE = (
    "https://github.com/jason-weirather/omeify/ns/provenance/1.0"
)


def _level_yx(shape: Sequence[int], axes: str) -> tuple[int, int]:
    if len(shape) != len(axes):
        raise ValueError(f"Shape {tuple(shape)} does not match axes {axes!r}")
    return int(shape[axes.index("Y")]), int(shape[axes.index("X")])


def _image_metadata(
    tiff_features: Any,
    level_shapes: Sequence[Sequence[int]],
) -> dict[str, object]:
    pixel_size = tiff_features.pixel_size
    metadata: dict[str, object] = {
        "SignificantBits": tiff_features.significant_bits,
        "PhysicalSizeX": pixel_size.x,
        "PhysicalSizeXUnit": pixel_size.unit,
        "PhysicalSizeY": pixel_size.y,
        "PhysicalSizeYUnit": pixel_size.unit,
        "Channel": {"Name": list(tiff_features.channel_names)},
    }
    if len(level_shapes) > 1:
        annotation: dict[str, str] = {"Namespace": _PYRAMID_NAMESPACE}
        for index, shape in enumerate(level_shapes[1:], start=1):
            level_y, level_x = _level_yx(shape, tiff_features.output_axes)
            annotation[str(index)] = f"{level_x} {level_y}"
        metadata["MapAnnotation"] = annotation
    return metadata


def _add_image(
    ome: OmeXml,
    tiff_features: Any,
    level_shapes: Sequence[Sequence[int]],
    *,
    name: str | None,
) -> None:
    stored_shape = (
        tiff_features.plane_count,
        1,
        1,
        tiff_features.size_y,
        tiff_features.size_x,
        tiff_features.samples_per_pixel,
    )
    options = _image_metadata(tiff_features, level_shapes)
    if name is not None:
        options["Name"] = name
    ome.addimage(
        tiff_features.dtype,
        tiff_features.output_shape,
        stored_shape,
        axes=tiff_features.output_axes,
        **options,
    )


def _normalize_root(
    root: etree._Element,
    *,
    display_uuid: bool,
    output_byteorder: Literal["<", ">"],
    remove_image_names: bool,
) -> list[etree._Element]:
    namespace = {"ome": _OME_NAMESPACE}
    images = root.findall("./ome:Image", namespaces=namespace)
    if not images:
        raise RuntimeError("tifffile generated OME-XML without an Image element")

    if remove_image_names:
        # OME Image/@Name is optional. Tifffile supplies a default when no
        # name is provided, so remove it explicitly for the minimized
        # single-image primary-data writer.
        for image in images:
            image.attrib.pop("Name", None)

    if not display_uuid:
        root.attrib.pop("UUID", None)

    pixels_nodes = root.findall("./ome:Image/ome:Pixels", namespaces=namespace)
    if len(pixels_nodes) != len(images):
        raise RuntimeError("tifffile generated an OME Image without Pixels")
    for pixels in pixels_nodes:
        # These values describe the file that omeify writes, not the source.
        pixels.set("BigEndian", "true" if output_byteorder == ">" else "false")
        # With SizeZ=SizeT=1, XYZCT maps planar channel pages in order and
        # also correctly describes one interleaved RGB plane.
        pixels.set("DimensionOrder", "XYZCT")
        channels = pixels.findall("./ome:Channel", namespaces=namespace)
        interleaved = (
            len(channels) == 1
            and int(channels[0].get("SamplesPerPixel", "1")) == 3
        )
        pixels.set("Interleaved", "true" if interleaved else "false")
    return images


def _attach_json_provenance(
    root: etree._Element,
    images: Sequence[etree._Element],
    provenance_json: str,
) -> None:
    namespace = {"ome": _OME_NAMESPACE}
    structured = root.find("./ome:StructuredAnnotations", namespaces=namespace)
    if structured is None:
        structured = etree.SubElement(
            root,
            etree.QName(_OME_NAMESPACE, "StructuredAnnotations"),
        )

    annotation_id = "Annotation:Provenance:0"
    existing_ids = {
        element.get("ID")
        for element in root.findall(".//*[@ID]")
        if element.get("ID") is not None
    }
    if annotation_id in existing_ids:
        raise RuntimeError(f"Generated OME annotation ID collision: {annotation_id}")

    annotation = etree.SubElement(
        structured,
        etree.QName(_OME_NAMESPACE, "MapAnnotation"),
        ID=annotation_id,
        Namespace=OMEIFY_PROVENANCE_NAMESPACE,
    )
    value = etree.SubElement(annotation, etree.QName(_OME_NAMESPACE, "Value"))
    item = etree.SubElement(value, etree.QName(_OME_NAMESPACE, "M"), K="json")
    item.text = provenance_json

    for image in images:
        etree.SubElement(
            image,
            etree.QName(_OME_NAMESPACE, "AnnotationRef"),
            ID=annotation_id,
        )


def _serialize(root: etree._Element) -> str:
    return etree.tostring(
        root,
        encoding="utf-8",
        xml_declaration=True,
        pretty_print=False,
    ).decode("utf-8")


def generate_ome_xml(
    tiff_features: Any,
    level_shapes: Sequence[Sequence[int]],
    *,
    display_uuid: bool = True,
    output_byteorder: Literal["<", ">"] = "<",
) -> dict[str, str | None]:
    """Generate minimal OME-XML for one planar, RGB, or label image."""

    if output_byteorder not in {"<", ">"}:
        raise ValueError("output_byteorder must be '<' or '>'")

    file_uuid = str(uuid.uuid4())
    ome = OmeXml(Creator=f"omeify {__version__}", UUID=file_uuid)
    _add_image(ome, tiff_features, level_shapes, name=None)
    root = etree.fromstring(ome.tostring(declaration=True).encode("utf-8"))
    _normalize_root(
        root,
        display_uuid=display_uuid,
        output_byteorder=output_byteorder,
        remove_image_names=True,
    )
    return {
        "xml_string": _serialize(root),
        "uuid": file_uuid if display_uuid else None,
    }


def generate_multi_series_ome_xml(
    images: Sequence[tuple[str, Any, Sequence[Sequence[int]]]],
    *,
    provenance_json: str | None = None,
    display_uuid: bool = True,
    output_byteorder: Literal["<", ">"] = "<",
) -> dict[str, str | None]:
    """Generate OME-XML for named heterogeneous image series.

    Every tuple contains ``(series_name, image_spec, pyramid_level_shapes)``.
    Series names are intentionally retained as OME ``Image/@Name`` values so
    viewers and readers can navigate a derived multi-image product without
    guessing from IFD position or dtype.
    """

    if output_byteorder not in {"<", ">"}:
        raise ValueError("output_byteorder must be '<' or '>'")
    normalized = tuple(images)
    if not normalized:
        raise ValueError("multi-series OME-XML requires at least one image")
    names = tuple(name.strip() for name, _, _ in normalized)
    if any(not name for name in names):
        raise ValueError("OME series names must be non-empty")
    if len(names) != len(set(names)):
        raise ValueError("OME series names must be unique")

    file_uuid = str(uuid.uuid4())
    ome = OmeXml(Creator=f"omeify {__version__}", UUID=file_uuid)
    for name, tiff_features, level_shapes in normalized:
        _add_image(ome, tiff_features, level_shapes, name=name.strip())

    root = etree.fromstring(ome.tostring(declaration=True).encode("utf-8"))
    image_elements = _normalize_root(
        root,
        display_uuid=display_uuid,
        output_byteorder=output_byteorder,
        remove_image_names=False,
    )
    if provenance_json is not None:
        if not isinstance(provenance_json, str) or not provenance_json:
            raise ValueError("provenance_json must be a non-empty string or None")
        _attach_json_provenance(root, image_elements, provenance_json)

    return {
        "xml_string": _serialize(root),
        "uuid": file_uuid if display_uuid else None,
        "provenance_namespace": (
            OMEIFY_PROVENANCE_NAMESPACE if provenance_json is not None else None
        ),
    }
