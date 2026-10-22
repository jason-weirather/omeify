"""Strict Leica SCN brightfield image/IFD interpretation (not a pixel decoder).

SCN's first TIFF ImageDescription maps image pyramids to top-level IFDs.
The slide label and other supplemental IFDs are deliberately not image series.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import tifffile
from lxml import etree

from .pixel_size import PixelSize

_SCN_NAMESPACES = frozenset({
    "http://www.leica-microsystems.com/scn/2010/03/10",
    "http://www.leica-microsystems.com/scn/2010/10/01",
})
_XML_ENCODING = re.compile(r"(<\?xml\b[^>]*?)\s+encoding=(['\"])[^'\"]+\2", re.I)
_MAX_XML_BYTES = 1_048_576


def _child(element: etree._Element, name: str) -> etree._Element | None:
    """Only direct children in the declared SCN namespace count."""
    return element.find(f"{{{etree.QName(element).namespace}}}{name}")


def _integer(element: etree._Element, name: str, *, minimum: int | None = None,
             default: int | None = None) -> int:
    raw = element.get(name)
    if raw is None and default is not None:
        return default
    if raw is None or not re.fullmatch(r"-?[0-9]+", raw):
        raise ValueError(
            f"Leica SCN has missing or invalid {name} on {etree.QName(element).localname}"
        )
    value = int(raw)
    if minimum is not None and value < minimum:
        raise ValueError(f"Leica SCN {name} must be >= {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class SCNDimension:
    ifd: int
    level: int
    z: int
    channel: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class SCNImage:
    name: str | None
    dimensions: tuple[SCNDimension, ...]
    illumination: str | None
    # The SCN view specifies physical pixel-edge bounds in slide-local nanometers.
    view_offset_yx_nm: tuple[int, int] | None
    view_size_yx_nm: tuple[int, int] | None
    width: int
    height: int

    def view_pixel_size(self) -> PixelSize | None:
        if self.view_size_yx_nm is None:
            return None
        h_nm, w_nm = self.view_size_yx_nm
        return PixelSize(w_nm / (1000.0 * self.width), h_nm / (1000.0 * self.height), "µm")

    def center_origin(self, pixel_size: PixelSize | None) -> tuple[float, float] | None:
        if self.view_offset_yx_nm is None or pixel_size is None:
            return None
        y_nm, x_nm = self.view_offset_yx_nm
        size = pixel_size.converted_to("µm")
        # ImageLevel origins are pixel *centers*, whereas SCN view offsets are edges.
        return (y_nm / 1000.0 + size.y / 2.0, x_nm / 1000.0 + size.x / 2.0)


@dataclass(frozen=True, slots=True)
class SCNMetadata:
    images: tuple[SCNImage, ...]


def parse_leica_scn(tiff: tifffile.TiffFile) -> SCNMetadata:
    """Parse the two supported SCN namespaces without resolving external XML."""
    if not tiff.is_bigtiff or not tiff.pages or not tiff.pages.first.is_tiled:
        raise ValueError("Leica SCN H&E requires a tiled BigTIFF with SCN ImageDescription")
    description = tiff.pages.first.description
    if not isinstance(description, str) or not description.strip():
        raise ValueError("Leica SCN is missing the first IFD's XML ImageDescription")
    xml = _XML_ENCODING.sub(r"\1", description, count=1).encode("utf-8")
    if len(xml) > _MAX_XML_BYTES or b"<!DOCTYPE" in xml.upper():
        raise ValueError("Leica SCN has oversized or unsafe XML")
    try:
        parser = etree.XMLParser(resolve_entities=False, load_dtd=False, no_network=True)
        root = etree.fromstring(xml, parser=parser)
    except (etree.XMLSyntaxError, ValueError) as exc:
        raise ValueError("Leica SCN ImageDescription is not valid XML") from exc
    if root.getroottree().docinfo.doctype or etree.QName(root).localname != "scn" or (
        etree.QName(root).namespace not in _SCN_NAMESPACES
    ):
        raise ValueError("Leica SCN H&E requires a recognized Leica <scn> XML namespace")
    namespace = etree.QName(root).namespace
    collections = root.findall(f"{{{namespace}}}collection")
    if len(collections) != 1:
        raise ValueError("Leica SCN requires exactly one <collection> element")
    collection = collections[0]

    images: list[SCNImage] = []
    used_ifds: set[int] = set()
    for image in collection.findall(f"{{{namespace}}}image"):
        pixels = _child(image, "pixels")
        if pixels is None:
            raise ValueError("Leica SCN image has no <pixels> element")
        width, height = _integer(pixels, "sizeX", minimum=1), _integer(pixels, "sizeY", minimum=1)
        dims: list[SCNDimension] = []
        for dim in pixels.findall(f"{{{namespace}}}dimension"):
            ifd = _integer(dim, "ifd", minimum=0)
            if ifd in used_ifds:
                raise ValueError(f"Leica SCN IFD {ifd} is declared more than once")
            used_ifds.add(ifd)
            dims.append(SCNDimension(
                ifd=ifd, level=_integer(dim, "r", minimum=0, default=0),
                z=_integer(dim, "z", minimum=0, default=0),
                channel=_integer(dim, "c", minimum=0, default=0),
                width=_integer(dim, "sizeX", minimum=1),
                height=_integer(dim, "sizeY", minimum=1),
            ))
        if not dims:
            raise ValueError("Leica SCN image contains no <dimension> IFD mappings")
        view = _child(image, "view")
        offset = size = None
        if view is not None:
            offset = (_integer(view, "offsetY"), _integer(view, "offsetX"))
            size = (_integer(view, "sizeY", minimum=1), _integer(view, "sizeX", minimum=1))
        scan = _child(image, "scanSettings")
        illumination = _child(scan, "illuminationSettings") if scan is not None else None
        source = _child(illumination, "illuminationSource") if illumination is not None else None
        images.append(SCNImage(
            name=image.get("name") or None,
            dimensions=tuple(dims),
            illumination=(source.text or "").strip().lower() if source is not None else None,
            view_offset_yx_nm=offset, view_size_yx_nm=size,
            width=width, height=height,
        ))
    if not images:
        raise ValueError("Leica SCN collection has no <image> entries")
    return SCNMetadata(tuple(images))


def build_leica_scn_he_series(
    tiff: tifffile.TiffFile, image: SCNImage,
) -> tifffile.TiffPageSeries:
    """Construct a YXS pyramid only from this image's explicitly mapped IFDs."""
    if image.illumination != "brightfield":
        raise ValueError("scn_he requires a brightfield Leica SCN image, not fluorescence/mIF")
    dims = sorted(image.dimensions, key=lambda d: d.level)
    if [dim.level for dim in dims] != list(range(len(dims))):
        raise ValueError("scn_he supports one RGB plane per contiguous pyramid level, from r=0")
    if any(dim.z != 0 or dim.channel != 0 for dim in dims):
        raise ValueError("scn_he does not support SCN Z stacks or multiplex channels")
    if (dims[0].height, dims[0].width) != (image.height, image.width):
        raise ValueError("Leica SCN level zero disagrees with <pixels> geometry")
    levels: list[tifffile.TiffPageSeries] = []
    prior_size = None
    for dim in dims:
        if dim.ifd >= len(tiff.pages):
            raise ValueError(f"Leica SCN refers to absent top-level IFD {dim.ifd}")
        page = tiff.pages[dim.ifd].aspage()
        shape = (dim.height, dim.width)
        if shape != (int(page.imagelength), int(page.imagewidth)):
            raise ValueError(f"Leica SCN IFD {dim.ifd} dimensions disagree with XML")
        if prior_size is not None and any(a > b for a, b in zip(shape, prior_size)):
            raise ValueError("Leica SCN pyramid size increases with resolution level")
        prior_size = shape
        levels.append(tifffile.TiffPageSeries(
            [page], (dim.height, dim.width, 3), page.dtype, "YXS",
            name=image.name, kind="scn",
        ))
    base = levels[0]
    base.levels = levels
    return base


def scn_he_pixel_size(image: SCNImage, page: tifffile.TiffPage) -> PixelSize | None:
    """Prefer standard TIFF X/Y calibration; validate independent SCN view extent."""
    from .pixel_size import pixel_size_from_tiff_resolution

    tiff_size = pixel_size_from_tiff_resolution(page)
    view_size = image.view_pixel_size()
    if tiff_size is not None and view_size is not None:
        if not (math.isclose(tiff_size.x, view_size.x, rel_tol=0.01)
                and math.isclose(tiff_size.y, view_size.y, rel_tol=0.01)):
            raise ValueError("Leica SCN view extent and TIFF physical calibration disagree")
    return tiff_size or view_size
