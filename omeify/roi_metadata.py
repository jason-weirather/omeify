"""Explicit, lossless OME-TIFF ROI metadata retention for image operations.

Default writers do not copy source annotations. This module is used ONLY with
preserve_rois=True, and refuses ambiguous source/image association or changed
axes rather than writing spatially misleading annotations. No shape pixels are
rasterized, and OME Mask payloads are never decompressed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tifffile
from lxml import etree

_OME = "http://www.openmicroscopy.org/Schemas/OME/2016-06"


def _q(name: str) -> str:
    return f"{{{_OME}}}{name}"


def _tree(xml: str) -> etree._Element:
    from omeify.inspection import _xml_bytes

    parser = etree.XMLParser(resolve_entities=False, load_dtd=False, no_network=True)
    root = etree.fromstring(_xml_bytes(xml), parser=parser)
    if root.getroottree().docinfo.doctype:
        raise ValueError("Cannot preserve ROI metadata from OME-XML containing a DTD")
    if root.tag != _q("OME"):
        raise ValueError("ROI preservation requires OME 2016-06 metadata")
    return root


def _signature(node: etree._Element):
    return (node.tag, tuple(sorted(node.attrib.items())), node.text,
            tuple(_signature(child) for child in node if isinstance(child.tag, str)))


@dataclass(frozen=True)
class PreservedROIs:
    """A detached collection of source ROIs, bound to one original image."""

    elements: tuple[bytes, ...]
    ids: tuple[str, ...]
    size_x: int
    size_y: int
    size_z: int
    size_t: int
    size_c: int

    @property
    def count(self) -> int:
        return len(self.elements)

    def require_matching_geometry(self, spec) -> None:
        source = (self.size_x, self.size_y, self.size_z, self.size_t, self.size_c)
        actual = (spec.size_x, spec.size_y, spec.size_z, spec.size_t, spec.size_c)
        if source != actual:
            raise ValueError(
                "--preserve-rois requires unchanged X/Y/Z/T/C dimensions and complete "
                "source axes; ROI coordinate remapping for selected subsets is not yet "
                f"supported (source={source}, output={actual}). Omit --preserve-rois "
                "or export the complete series."
            )

    def attach(self, root: etree._Element, image: etree._Element) -> None:
        if not self.elements:
            return
        # Source ROI IDs are intentionally retained. Do not collide with
        # freshly generated image, channel, or linked annotation IDs.
        generated = {n.get("ID") for n in root.iter() if n.get("ID") is not None}
        if any(roi_id in generated for roi_id in self.ids):
            raise ValueError("Source ROI ID collides with generated OME metadata")

        # OME 2016-06 requires StructuredAnnotations BEFORE top-level ROI.
        for data in self.elements:
            root.append(etree.fromstring(data))

        # ROIRef belongs after Pixels but before AnnotationRef in OME Image.
        ref_insertion = next(
            (i for i, child in enumerate(image) if child.tag in
             {_q("AnnotationRef"), _q("MicrobeamManipulationRef")}), len(image),
        )
        for roi_id in self.ids:
            image.insert(ref_insertion, etree.Element(_q("ROIRef"), ID=roi_id))
            ref_insertion += 1


def load_preserved_rois(path: str | Path, *, series: int = 0) -> PreservedROIs:
    """Resolve the selected OME Image via physical IFDs, not image ordinal."""
    from omeify.io._tiff_sources.ome_index import OMEPlaneIndex

    with tifffile.TiffFile(path, _multifile=False) as tiff:
        if not tiff.is_ome or not tiff.ome_metadata:
            raise ValueError("--preserve-rois requires an OME-TIFF input")
        pixels = OMEPlaneIndex(tiff).map_series(series).pixels
        image = pixels.getparent()
        root = pixels.getroottree().getroot()
        # A source with ROIs but missing image links cannot be disambiguated.
        definitions = [n for n in root if n.tag == _q("ROI")]
        by_id = {}
        for node in definitions:
            roi_id = node.get("ID")
            if not roi_id or roi_id in by_id:
                raise ValueError("Missing or duplicate source ROI ID")
            by_id[roi_id] = node
        ids = tuple(child.get("ID") for child in image if child.tag == _q("ROIRef"))
        if len(ids) != len(set(ids)) or any(roi_id not in by_id for roi_id in ids):
            raise ValueError("Duplicate or unresolved source image ROIRef")
        if definitions and not ids and not any(
            child.tag == _q("ROIRef")
            for candidate in root if candidate.tag == _q("Image")
            for child in candidate
        ):
            raise ValueError("Source contains unassociated ROIs without any image ROIRef")
        if any(node.get("ID") is None for node in image if node.tag == _q("ROIRef")):
            raise ValueError("Invalid source ROIRef ID")
        sizes = tuple(int(pixels.get("Size" + axis, "0")) for axis in "XYZTC")
        if any(n <= 0 for n in sizes):
            raise ValueError("Cannot preserve ROI metadata with invalid image dimensions")
        # No geometric transformations are performed. Bad plane selectors should
        # be rejected rather than silently misattached in the new image.
        for roi_id in ids:
            for node in by_id[roi_id].iter():
                if not isinstance(node.tag, str):
                    continue
                for axis, extent in (("Z", sizes[2]), ("T", sizes[3]), ("C", sizes[4])):
                    raw = node.get("The" + axis)
                    if raw is not None and (not raw.isdecimal() or int(raw) >= extent):
                        raise ValueError(f"ROI {roi_id!r} has invalid The{axis}={raw!r}")
        for roi_id in ids:
            if any(n.tag == _q("AnnotationRef") for n in by_id[roi_id].iter()
                   if isinstance(n.tag, str)):
                raise ValueError("ROI-linked external annotations cannot yet be safely preserved")
        data = tuple(etree.tostring(by_id[roi_id], encoding="utf-8") for roi_id in ids)
    return PreservedROIs(data, ids, *sizes)


def verify_preserved_rois(output_path: Path, expected: PreservedROIs) -> None:
    """Verify annotations and links before the staged TIFF is published."""
    with tifffile.TiffFile(output_path, _multifile=False) as tiff:
        root = _tree(tiff.ome_metadata)
        image = next(child for child in root if child.tag == _q("Image"))
        ids = tuple(child.get("ID") for child in image if child.tag == _q("ROIRef"))
        if ids != expected.ids:
            raise ValueError("Written OME ROIRef list differs from requested source")
        actual = {node.get("ID"): _signature(node) for node in root if node.tag == _q("ROI")}
        desired = {roi_id: _signature(etree.fromstring(xml))
                   for roi_id, xml in zip(expected.ids, expected.elements, strict=True)}
        if actual != desired or len(actual) != expected.count:
            raise ValueError("Written OME ROI geometry differs from requested source")
