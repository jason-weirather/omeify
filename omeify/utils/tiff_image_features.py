from __future__ import annotations

import logging
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence
from xml.etree import ElementTree

import numpy as np
import tifffile

LOGGER = logging.getLogger(__name__)

InputProfile = Literal[
    "akoya_mif_qptiff",
    "halo_mif",
    "component",
]

_OME_DTYPES = {
    "int8",
    "int16",
    "int32",
    "uint8",
    "uint16",
    "uint32",
    "float32",
    "float64",
    "complex64",
    "complex128",
    "bool",
}


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _first_child_text(root: ElementTree.Element, local_name: str) -> str | None:
    for child in root:
        if _local_name(child.tag) == local_name and child.text:
            value = child.text.strip()
            if value:
                return value
    return None


def _first_descendant_text(root: ElementTree.Element, local_name: str) -> str | None:
    for element in root.iter():
        if _local_name(element.tag) == local_name and element.text:
            value = element.text.strip()
            if value:
                return value
    return None


def _parse_xml(xml: str | bytes | None) -> ElementTree.Element | None:
    if not xml:
        return None
    try:
        return ElementTree.fromstring(xml)
    except (ElementTree.ParseError, ValueError, TypeError):
        return None


def _ome_images(root: ElementTree.Element) -> list[ElementTree.Element]:
    return [element for element in root.iter() if _local_name(element.tag) == "Image"]


def _ome_pixels(image: ElementTree.Element) -> ElementTree.Element | None:
    for element in image:
        if _local_name(element.tag) == "Pixels":
            return element
    return None


def _unit_to_micrometers(value: float, unit: str | None) -> float:
    normalized = (unit or "µm").strip().lower().replace("μ", "µ")
    factors = {
        "µm": 1.0,
        "um": 1.0,
        "micrometer": 1.0,
        "micrometre": 1.0,
        "nm": 1e-3,
        "nanometer": 1e-3,
        "nanometre": 1e-3,
        "mm": 1e3,
        "millimeter": 1e3,
        "millimetre": 1e3,
        "cm": 1e4,
        "centimeter": 1e4,
        "centimetre": 1e4,
        "m": 1e6,
        "meter": 1e6,
        "metre": 1e6,
    }
    if normalized not in factors:
        raise ValueError(f"Unsupported physical-size unit {unit!r}")
    return float(value) * factors[normalized]


def _resolution_tag_to_um(page: tifffile.TiffPage) -> tuple[float, float] | None:
    """Return pixel size from standard TIFF resolution tags, when trustworthy."""

    try:
        x_num, x_den = page.tags["XResolution"].value
        y_num, y_den = page.tags["YResolution"].value
        x_ppu = float(x_num) / float(x_den)
        y_ppu = float(y_num) / float(y_den)
        unit_value = int(page.tags["ResolutionUnit"].value)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None

    if x_ppu <= 0 or y_ppu <= 0:
        return None
    if unit_value == 3:  # centimeter
        return 1e4 / x_ppu, 1e4 / y_ppu
    if unit_value == 2:  # inch
        return 25400.0 / x_ppu, 25400.0 / y_ppu
    return None


@dataclass(frozen=True)
class ImageMetadata:
    input_path: Path
    series_index: int
    input_type: str
    image_name: str
    size_c: int
    size_y: int
    size_x: int
    dtype: np.dtype
    significant_bits: int
    channel_names: tuple[str, ...]
    physical_size_x_um: float
    physical_size_y_um: float
    source_axes: str
    source_byte_order: str

    @property
    def shape_cyx(self) -> tuple[int, int, int]:
        return self.size_c, self.size_y, self.size_x


class TiffPlaneReader:
    """Random-access reader for one grayscale TIFF page.

    Only intersecting tiles or strips are decoded.  A small LRU cache prevents
    repeated decoding when the requested output tile grid does not exactly match
    the source grid.
    """

    def __init__(
        self,
        page: tifffile.TiffPage,
        *,
        lock: threading.RLock | None = None,
        cache_mib: int = 64,
    ) -> None:
        if int(page.samplesperpixel) != 1:
            raise ValueError(
                "The mIF writer requires one grayscale sample per TIFF page; "
                f"found SamplesPerPixel={page.samplesperpixel}."
            )

        self.page = page
        self.filehandle = page.parent.filehandle
        self.lock = lock or threading.RLock()
        self.height = int(page.imagelength)
        self.width = int(page.imagewidth)
        self.dtype = np.dtype(page.dtype).newbyteorder("=")
        self._cache: OrderedDict[int, tuple[np.ndarray, int, int]] = OrderedDict()

        if page.is_tiled:
            self.segment_height = int(page.tilelength)
            self.segment_width = int(page.tilewidth)
            self.segments_across = math.ceil(self.width / self.segment_width)
        else:
            self.segment_height = int(page.rowsperstrip or self.height)
            self.segment_width = self.width
            self.segments_across = 1

        nominal_segment_bytes = max(
            1,
            self.segment_height * self.segment_width * max(1, self.dtype.itemsize),
        )
        self.cache_capacity = max(
            1,
            min(128, (max(1, cache_mib) * 1024 * 1024) // nominal_segment_bytes),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    def clear_cache(self) -> None:
        self._cache.clear()

    def _segment_index(self, segment_y: int, segment_x: int) -> int:
        return segment_y * self.segments_across + segment_x

    @staticmethod
    def _as_2d(decoded: np.ndarray) -> np.ndarray:
        array = np.asarray(decoded)
        if array.ndim == 4:
            if array.shape[0] != 1 or array.shape[-1] != 1:
                raise ValueError(f"Unsupported decoded TIFF segment shape {array.shape}")
            array = array[0, :, :, 0]
        elif array.ndim == 3:
            if array.shape[0] == 1:
                array = array[0]
            elif array.shape[-1] == 1:
                array = array[..., 0]
        if array.ndim != 2:
            raise ValueError(f"Unsupported decoded TIFF segment shape {array.shape}")
        return array

    def _decode_segment(self, index: int) -> tuple[np.ndarray, int, int]:
        cached = self._cache.get(index)
        if cached is not None:
            self._cache.move_to_end(index)
            return cached

        try:
            offset = int(self.page.dataoffsets[index])
            bytecount = int(self.page.databytecounts[index])
        except IndexError as exc:
            raise IndexError(
                f"TIFF segment index {index} is outside page segment table "
                f"({len(self.page.dataoffsets)} segments)"
            ) from exc

        encoded: bytes | None
        if offset == 0 or bytecount == 0:
            encoded = None
        else:
            with self.lock:
                self.filehandle.seek(offset)
                encoded = self.filehandle.read(bytecount)
            if len(encoded) != bytecount:
                raise OSError(
                    f"Short read for TIFF segment {index}: expected {bytecount} bytes, "
                    f"received {len(encoded)}"
                )

        decoded, indices, _shape = self.page.decode(
            encoded,
            index,
            jpegtables=self.page.jpegtables,
        )

        if indices is not None and len(indices) >= 4:
            origin_y = int(indices[2])
            origin_x = int(indices[3])
        else:
            segment_y, segment_x = divmod(index, self.segments_across)
            origin_y = segment_y * self.segment_height
            origin_x = segment_x * self.segment_width

        if decoded is None:
            valid_h = min(self.segment_height, self.height - origin_y)
            valid_w = min(self.segment_width, self.width - origin_x)
            array = np.zeros((max(0, valid_h), max(0, valid_w)), dtype=self.dtype)
        else:
            array = self._as_2d(decoded)
            if array.dtype != self.dtype:
                array = array.astype(self.dtype, copy=False)
            valid_h = min(array.shape[0], self.height - origin_y)
            valid_w = min(array.shape[1], self.width - origin_x)
            array = np.ascontiguousarray(array[:valid_h, :valid_w])

        value = (array, origin_y, origin_x)
        self._cache[index] = value
        self._cache.move_to_end(index)
        while len(self._cache) > self.cache_capacity:
            self._cache.popitem(last=False)
        return value

    def read_region(self, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        if not (0 <= y0 <= y1 <= self.height and 0 <= x0 <= x1 <= self.width):
            raise ValueError(
                f"Requested region {(y0, y1, x0, x1)} is outside page shape {self.shape}"
            )
        output = np.zeros((y1 - y0, x1 - x0), dtype=self.dtype)
        if output.size == 0:
            return output

        first_segment_y = y0 // self.segment_height
        last_segment_y = (y1 - 1) // self.segment_height
        first_segment_x = x0 // self.segment_width
        last_segment_x = (x1 - 1) // self.segment_width

        for segment_y in range(first_segment_y, last_segment_y + 1):
            for segment_x in range(first_segment_x, last_segment_x + 1):
                index = self._segment_index(segment_y, segment_x)
                segment, seg_y0, seg_x0 = self._decode_segment(index)
                seg_y1 = seg_y0 + segment.shape[0]
                seg_x1 = seg_x0 + segment.shape[1]

                copy_y0 = max(y0, seg_y0)
                copy_y1 = min(y1, seg_y1)
                copy_x0 = max(x0, seg_x0)
                copy_x1 = min(x1, seg_x1)
                if copy_y0 >= copy_y1 or copy_x0 >= copy_x1:
                    continue

                output[
                    copy_y0 - y0 : copy_y1 - y0,
                    copy_x0 - x0 : copy_x1 - x0,
                ] = segment[
                    copy_y0 - seg_y0 : copy_y1 - seg_y0,
                    copy_x0 - seg_x0 : copy_x1 - seg_x0,
                ]
        return output


class TiffMIFSource:
    """Open and normalize a TIFF-like mIF source into planar C/Y/X pages."""

    def __init__(
        self,
        input_path: str | Path,
        *,
        series: int = 0,
        profile: InputProfile,
        input_type: str,
        image_name: str = "WholeSlideMIF",
        physical_size_x_um: float | None = None,
        physical_size_y_um: float | None = None,
    ) -> None:
        self.input_path = Path(input_path)
        self.series_index = int(series)
        self.profile = profile
        self.input_type = input_type
        self.image_name = image_name
        self.physical_size_x_override = physical_size_x_um
        self.physical_size_y_override = physical_size_y_um
        self.tiff: tifffile.TiffFile | None = None
        self.series = None
        self.level0 = None
        self.pages: list[tifffile.TiffPage] = []
        self.metadata: ImageMetadata | None = None
        self._read_lock = threading.RLock()

    def __enter__(self) -> "TiffMIFSource":
        self.tiff = tifffile.TiffFile(self.input_path)
        try:
            series_collection = self.tiff.series
            if self.series_index < 0 or self.series_index >= len(series_collection):
                raise IndexError(
                    f"Series {self.series_index} does not exist; file has "
                    f"{len(series_collection)} series."
                )
            self.series = series_collection[self.series_index]
            levels = getattr(self.series, "levels", None)
            self.level0 = levels[0] if levels else self.series
            # TiffPageSeries.pages may contain lightweight TiffFrame objects.
            # Materialize their IFD metadata so per-page tags (including Akoya
            # channel descriptions) and structural fields are available.  This
            # reads only TIFF directory metadata, not image pixels.
            self.pages = [page.aspage() for page in self.level0.pages]
            self.metadata = self._inspect()
            return self
        except Exception:
            self.tiff.close()
            self.tiff = None
            raise

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.tiff is not None:
            self.tiff.close()
        self.tiff = None
        self.pages = []

    @property
    def features(self) -> ImageMetadata:
        if self.metadata is None:
            raise RuntimeError("TiffMIFSource must be opened as a context manager")
        return self.metadata

    def plane_readers(self, *, cache_mib: int = 64) -> list[TiffPlaneReader]:
        if not self.pages:
            raise RuntimeError("TiffMIFSource must be opened as a context manager")
        return [
            TiffPlaneReader(page, lock=self._read_lock, cache_mib=cache_mib)
            for page in self.pages
        ]

    def _inspect(self) -> ImageMetadata:
        if self.tiff is None or self.level0 is None or not self.pages:
            raise ValueError("Selected series contains no readable TIFF pages")

        page0 = self.pages[0]
        size_y = int(page0.imagelength)
        size_x = int(page0.imagewidth)
        dtype = np.dtype(self.level0.dtype).newbyteorder("=")
        if dtype.name not in _OME_DTYPES:
            raise TypeError(f"OME-TIFF does not support source dtype {dtype}")

        for index, page in enumerate(self.pages):
            if int(page.imagelength) != size_y or int(page.imagewidth) != size_x:
                raise ValueError(
                    "All mIF channel pages must have the same full-resolution shape; "
                    f"page 0 is {(size_y, size_x)}, page {index} is "
                    f"{(page.imagelength, page.imagewidth)}."
                )
            if np.dtype(page.dtype).newbyteorder("=") != dtype:
                raise TypeError(
                    "All mIF channel pages must have the same dtype; "
                    f"page 0 is {dtype}, page {index} is {page.dtype}."
                )
            if int(page.samplesperpixel) != 1:
                raise ValueError(
                    "RGB/contiguous-sample TIFF input is intentionally deferred to the H&E "
                    "iteration.  The mIF path requires SamplesPerPixel=1 for every page."
                )

        axes = str(getattr(self.level0, "axes", ""))
        shape = tuple(int(value) for value in getattr(self.level0, "shape", ()))
        axis_sizes = dict(zip(axes, shape))
        for axis, axis_size in axis_sizes.items():
            if axis not in {"X", "Y", "C", "I", "Q"} and axis_size != 1:
                raise ValueError(
                    f"Unsupported non-singleton axis {axis}={axis_size} in mIF input "
                    f"with axes {axes!r}."
                )

        candidate_channel_size = None
        for axis in ("C", "I", "Q"):
            if axis in axis_sizes:
                candidate_channel_size = axis_sizes[axis]
                break
        size_c = candidate_channel_size or len(self.pages)
        if len(self.pages) != size_c:
            # For planar mIF data, the physical page count is the strongest signal.
            if len(self.pages) > 1:
                LOGGER.warning(
                    "Series axes %s imply %s channels but the level contains %s pages; "
                    "using the page count.",
                    axes,
                    size_c,
                    len(self.pages),
                )
                size_c = len(self.pages)
            else:
                raise ValueError(
                    f"Cannot map series axes {axes!r} and shape {shape} onto planar channels."
                )

        channel_names = self._channel_names(size_c)
        physical_x, physical_y = self._physical_sizes(page0)
        significant_bits = self._significant_bits(dtype)

        return ImageMetadata(
            input_path=self.input_path,
            series_index=self.series_index,
            input_type=self.input_type,
            image_name=self.image_name,
            size_c=size_c,
            size_y=size_y,
            size_x=size_x,
            dtype=dtype,
            significant_bits=significant_bits,
            channel_names=tuple(channel_names),
            physical_size_x_um=physical_x,
            physical_size_y_um=physical_y,
            source_axes=axes,
            source_byte_order="big" if self.tiff.byteorder == ">" else "little",
        )

    def _channel_names(self, size_c: int) -> list[str]:
        if self.profile in {"akoya_mif_qptiff", "component"}:
            names = [self._akoya_page_name(page) for page in self.pages]
        elif self.profile == "halo_mif":
            names = self._ome_channel_names()
        else:
            names = []

        result: list[str] = []
        for index in range(size_c):
            name = names[index] if index < len(names) else None
            result.append(name or f"Channel {index + 1}")
        return result

    @staticmethod
    def _akoya_page_name(page: tifffile.TiffPage) -> str | None:
        root = _parse_xml(page.description)
        if root is None:
            return None
        return _first_child_text(root, "Name") or _first_descendant_text(root, "Name")

    def _selected_ome_pixels(self) -> ElementTree.Element | None:
        if self.tiff is None:
            return None
        root = _parse_xml(self.tiff.ome_metadata)
        if root is None:
            root = _parse_xml(self.pages[0].description if self.pages else None)
        if root is None:
            return None
        images = _ome_images(root)
        if not images:
            return None
        image_index = min(self.series_index, len(images) - 1)
        return _ome_pixels(images[image_index])

    def _ome_channel_names(self) -> list[str]:
        pixels = self._selected_ome_pixels()
        if pixels is None:
            return []
        names: list[str] = []
        for element in pixels:
            if _local_name(element.tag) == "Channel":
                names.append(element.attrib.get("Name") or "")
        return names

    def _physical_sizes(self, page0: tifffile.TiffPage) -> tuple[float, float]:
        if self.physical_size_x_override is not None or self.physical_size_y_override is not None:
            if self.physical_size_x_override is None or self.physical_size_y_override is None:
                raise ValueError("Both physical_size_x_um and physical_size_y_um are required")
            x_um = float(self.physical_size_x_override)
            y_um = float(self.physical_size_y_override)
        elif self.profile in {"akoya_mif_qptiff", "component"}:
            root = _parse_xml(page0.description)
            pixel_size = (
                _first_descendant_text(root, "PixelSizeMicrons")
                if root is not None
                else None
            )
            if pixel_size is not None:
                x_um = y_um = float(pixel_size)
            else:
                fallback = _resolution_tag_to_um(page0)
                if fallback is None:
                    raise ValueError(
                        "Akoya metadata does not contain PixelSizeMicrons and TIFF resolution "
                        "tags do not define a physical unit."
                    )
                x_um, y_um = fallback
        elif self.profile == "halo_mif":
            pixels = self._selected_ome_pixels()
            if pixels is None:
                raise ValueError("HALO mIF TIFF does not contain readable OME-XML Pixels metadata")
            try:
                x_um = _unit_to_micrometers(
                    float(pixels.attrib["PhysicalSizeX"]),
                    pixels.attrib.get("PhysicalSizeXUnit"),
                )
                y_um = _unit_to_micrometers(
                    float(pixels.attrib["PhysicalSizeY"]),
                    pixels.attrib.get("PhysicalSizeYUnit"),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "HALO OME metadata must contain valid PhysicalSizeX and PhysicalSizeY."
                ) from exc
        else:
            raise AssertionError(f"Unhandled input profile {self.profile}")

        if not math.isfinite(x_um) or not math.isfinite(y_um) or x_um <= 0 or y_um <= 0:
            raise ValueError(
                f"Physical pixel sizes must be positive finite values, got {x_um}, {y_um}"
            )
        return x_um, y_um

    def _significant_bits(self, dtype: np.dtype) -> int:
        pixels = self._selected_ome_pixels()
        if pixels is not None and "SignificantBits" in pixels.attrib:
            try:
                value = int(pixels.attrib["SignificantBits"])
                if 0 < value <= dtype.itemsize * 8:
                    return value
            except ValueError:
                pass

        try:
            bits_per_sample = self.pages[0].bitspersample
            if isinstance(bits_per_sample, Sequence) and not isinstance(
                bits_per_sample, (str, bytes)
            ):
                bits_per_sample = bits_per_sample[0]
            value = int(bits_per_sample)
            if 0 < value <= dtype.itemsize * 8:
                return value
        except (AttributeError, TypeError, ValueError, IndexError):
            pass
        return dtype.itemsize * 8


class TiffImageFeatures:
    """Compatibility metadata facade for code that imported the old class.

    The class now uses tifffile directly and does not depend on tiff-inspector.
    For conversion, use the input classes in :mod:`omeify.inputs`.
    """

    def __init__(
        self,
        tiff_file_path: str | Path,
        series: int = 0,
        *,
        profile: InputProfile = "akoya_mif_qptiff",
        input_type: str = "TIFF mIF",
        image_name: str = "WholeSlideMIF",
        physical_size_x_um: float | None = None,
        physical_size_y_um: float | None = None,
    ) -> None:
        with TiffMIFSource(
            tiff_file_path,
            series=series,
            profile=profile,
            input_type=input_type,
            image_name=image_name,
            physical_size_x_um=physical_size_x_um,
            physical_size_y_um=physical_size_y_um,
        ) as source:
            self._metadata = source.features

    @property
    def metadata(self) -> ImageMetadata:
        return self._metadata

    @property
    def image_id(self) -> str:
        return "Image:0"

    @property
    def pixel_id(self) -> str:
        return "Pixels:0"

    @property
    def name(self) -> str:
        return self._metadata.image_name

    @property
    def big_endian(self) -> str:
        return "true" if self._metadata.source_byte_order == "big" else "false"

    @property
    def dimension_order(self) -> str:
        return "XYZCT"

    @property
    def interleaved(self) -> str:
        return "false"

    @property
    def physical_size_x(self) -> float:
        return self._metadata.physical_size_x_um

    @property
    def physical_size_y(self) -> float:
        return self._metadata.physical_size_y_um

    @property
    def physical_size_x_unit(self) -> str:
        return "µm"

    @property
    def physical_size_y_unit(self) -> str:
        return "µm"

    @property
    def size_c(self) -> int:
        return self._metadata.size_c

    @property
    def size_y(self) -> int:
        return self._metadata.size_y

    @property
    def size_x(self) -> int:
        return self._metadata.size_x

    @property
    def size_z(self) -> int:
        return 1

    @property
    def size_t(self) -> int:
        return 1

    @property
    def type(self) -> str:
        return {
            "float32": "float",
            "float64": "double",
            "complex64": "complex",
            "complex128": "double-complex",
            "bool": "bit",
        }.get(self._metadata.dtype.name, self._metadata.dtype.name)

    @property
    def significant_bits(self) -> int:
        return self._metadata.significant_bits

    @property
    def plane_count(self) -> int:
        return self._metadata.size_c

    @property
    def channels(self) -> list[dict[str, object]]:
        return [
            {
                "ID": f"Channel:0:{index}",
                "Name": name,
                "SamplesPerPixel": 1,
            }
            for index, name in enumerate(self._metadata.channel_names)
        ]
