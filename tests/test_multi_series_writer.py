from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
from lxml import etree

from omeify import (
    OMEImageSeries,
    OMEMultiSeriesWriter,
    OMETiffReader,
    PixelSize,
)
from omeify.utils.generate_ome_xml import OMEIFY_PROVENANCE_NAMESPACE

_OME_NAMESPACE = "http://www.openmicroscopy.org/Schemas/OME/2016-06"


def _mean2(values: np.ndarray) -> np.ndarray:
    return values.reshape(
        values.shape[0],
        values.shape[1] // 2,
        2,
        values.shape[2] // 2,
        2,
    ).mean(axis=(2, 4), dtype=np.float64).astype(values.dtype)


def test_multi_series_writer_preserves_heterogeneous_series_and_provenance(
    tmp_path: Path,
) -> None:
    output = tmp_path / "derived.ome.tif"
    pixel_size = PixelSize(0.5, 0.6, "µm")
    intensity = np.arange(2 * 32 * 48, dtype=np.float32).reshape(2, 32, 48)
    labels = np.arange(32 * 48, dtype=np.uint16).reshape(32, 48) % 17
    provenance = {
        "schema": "example.provenance/1",
        "software": {"name": "example", "version": "2.0"},
        "parameters": {"threshold": 0.25, "seed": 42},
    }
    series = (
        OMEImageSeries.from_array(
            "Normalized signal",
            intensity,
            image_type="multichannel",
            channel_names=("A", "B"),
            pixel_size=pixel_size,
        ),
        OMEImageSeries.from_array(
            "Object labels",
            labels,
            image_type="label",
            channel_names=("Object labels",),
            pixel_size=pixel_size,
        ),
    )

    report = OMEMultiSeriesWriter(
        output,
        compression="Uncompressed",
        tile_size=16,
        pyramid_levels=1,
    ).write(series, provenance=provenance)

    assert report["miti_header"]["is_valid"] is True
    assert report["provenance"] == {
        "namespace": OMEIFY_PROVENANCE_NAMESPACE,
        "value": provenance,
    }
    assert report["verification"]["provenance_annotation_linked"] is True
    assert report["verification"]["base_pixel_values_match"] is True
    assert [item["name"] for item in report["series"]] == [
        "Normalized signal",
        "Object labels",
    ]
    assert [item["dtype"] for item in report["series"]] == ["float32", "uint16"]
    assert [item["downsample_method"] for item in report["series"]] == [
        "mean",
        "nearest",
    ]

    with tifffile.TiffFile(output) as tiff:
        assert tiff.is_ome
        assert len(tiff.series) == 2
        assert [item.name for item in tiff.series] == [
            "Normalized signal",
            "Object labels",
        ]
        np.testing.assert_array_equal(tiff.series[0].levels[0].asarray(), intensity)
        np.testing.assert_array_equal(tiff.series[0].levels[1].asarray(), _mean2(intensity))
        np.testing.assert_array_equal(tiff.series[1].levels[0].asarray(), labels)
        np.testing.assert_array_equal(tiff.series[1].levels[1].asarray(), labels[::2, ::2])

        assert tiff.ome_metadata is not None
        root = etree.fromstring(tiff.ome_metadata.encode("utf-8"))
        namespace = {"ome": _OME_NAMESPACE}
        images = root.findall("./ome:Image", namespaces=namespace)
        mappings = [
            image.find("./ome:Pixels/ome:TiffData", namespaces=namespace)
            for image in images
        ]
        assert [int(item.get("IFD")) for item in mappings if item is not None] == [0, 2]
        assert [
            int(item.get("PlaneCount")) for item in mappings if item is not None
        ] == [2, 1]

        annotations = root.findall(
            "./ome:StructuredAnnotations/ome:MapAnnotation",
            namespaces=namespace,
        )
        provenance_annotations = [
            item
            for item in annotations
            if item.get("Namespace") == OMEIFY_PROVENANCE_NAMESPACE
        ]
        assert len(provenance_annotations) == 1
        payload = provenance_annotations[0].find(
            "./ome:Value/ome:M[@K='json']",
            namespaces=namespace,
        )
        assert payload is not None
        assert json.loads(payload.text or "") == provenance
        provenance_id = provenance_annotations[0].get("ID")
        assert all(
            provenance_id
            in {
                ref.get("ID")
                for ref in image.findall("./ome:AnnotationRef", namespaces=namespace)
            }
            for image in images
        )

    with OMETiffReader(output, series=1) as reader:
        assert reader.series_count == 2
        assert reader.series_name == "Object labels"
        assert reader.series_names == ("Normalized signal", "Object labels")
        np.testing.assert_array_equal(reader.asarray(), labels)


def test_multi_series_writer_rejects_ambiguous_names_and_non_json_provenance(
    tmp_path: Path,
) -> None:
    pixel_size = PixelSize(0.5, 0.5, "µm")
    image = OMEImageSeries.from_array(
        "Duplicate",
        np.zeros((16, 16), dtype=np.uint16),
        image_type="label",
        channel_names=("Labels",),
        pixel_size=pixel_size,
    )

    duplicate_output = tmp_path / "duplicate.ome.tif"
    with pytest.raises(ValueError, match="series names must be unique"):
        OMEMultiSeriesWriter(
            duplicate_output,
            compression="Uncompressed",
            tile_size=16,
            pyramid_levels=0,
        ).write((image, image))
    assert not duplicate_output.exists()

    provenance_output = tmp_path / "bad-provenance.ome.tif"
    with pytest.raises(TypeError, match="finite JSON-serializable"):
        OMEMultiSeriesWriter(
            provenance_output,
            compression="Uncompressed",
            tile_size=16,
            pyramid_levels=0,
        ).write((image,), provenance={"path": tmp_path})
    assert not provenance_output.exists()


def test_miti_validator_requires_contiguous_ifd_mapping_across_images() -> None:
    from omeify.io.spec import OMEImageSpec
    from omeify.utils.generate_ome_xml import generate_multi_series_ome_xml
    from omeify.utils.miti_header_validator import validate_miti_ome_tiff_header

    pixel_size = PixelSize(0.5, 0.5, "µm")
    first = OMEImageSpec.from_shape(
        image_type="multichannel",
        axes="CYX",
        shape=(2, 16, 16),
        dtype=np.uint16,
        channel_names=("A", "B"),
        pixel_size=pixel_size,
    )
    second = OMEImageSpec.from_shape(
        image_type="label",
        axes="YX",
        shape=(16, 16),
        dtype=np.uint16,
        channel_names=("Labels",),
        pixel_size=pixel_size,
    )
    xml = generate_multi_series_ome_xml(
        (
            ("Signal", first, (first.output_shape,)),
            ("Labels", second, (second.output_shape,)),
        )
    )["xml_string"]
    assert isinstance(xml, str)
    root = etree.fromstring(xml.encode("utf-8"))
    namespace = {"ome": _OME_NAMESPACE}
    mappings = root.findall("./ome:Image/ome:Pixels/ome:TiffData", namespaces=namespace)
    assert [int(item.get("IFD")) for item in mappings] == [0, 2]

    mappings[1].set("IFD", "3")
    broken = etree.tostring(root, encoding="unicode")
    assessment = validate_miti_ome_tiff_header(broken)

    assert assessment.is_valid is False
    assert any(
        "contiguous multi-image mapping requires IFD=2" in error
        for error in assessment.errors
    )


def test_multi_series_writer_allows_per_series_grayscale_jpeg(tmp_path: Path) -> None:
    pytest.importorskip("imagecodecs")
    output = tmp_path / "mixed-compression.ome.tif"
    pixel_size = PixelSize(0.5, 0.5, "µm")
    visualization = np.arange(32 * 48, dtype=np.uint8).reshape(32, 48)
    labels = (np.arange(32 * 48, dtype=np.uint16).reshape(32, 48) % 17)
    series = (
        OMEImageSeries.from_array(
            "Visualization",
            visualization,
            image_type="multichannel",
            channel_names=("Visualization",),
            pixel_size=pixel_size,
            compression="JPEG",
        ),
        OMEImageSeries.from_array(
            "Labels",
            labels,
            image_type="label",
            channel_names=("Labels",),
            pixel_size=pixel_size,
        ),
    )

    report = OMEMultiSeriesWriter(
        output,
        compression="LZW",
        jpeg_quality=90,
        tile_size=16,
        pyramid_levels=0,
    ).write(series)

    assert [item["compression"] for item in report["series"]] == ["JPEG", "LZW"]
    assert report["output_file"]["lossless_compression"] is False
    with tifffile.TiffFile(output) as tiff:
        assert int(tiff.series[0].pages[0].compression) == 7
        assert int(tiff.series[1].pages[0].compression) == 5
        decoded = tiff.series[0].asarray()
        assert decoded.dtype == np.uint8
        assert decoded.shape == visualization.shape
        np.testing.assert_array_equal(tiff.series[1].asarray(), labels)


def test_multi_series_writer_rejects_jpeg_label_series(tmp_path: Path) -> None:
    pixel_size = PixelSize(0.5, 0.5, "µm")
    labels = OMEImageSeries.from_array(
        "Labels",
        np.zeros((16, 16), dtype=np.uint8),
        image_type="label",
        channel_names=("Labels",),
        pixel_size=pixel_size,
        compression="JPEG",
    )

    with pytest.raises(ValueError, match="require lossless compression"):
        OMEMultiSeriesWriter(
            tmp_path / "bad-label.ome.tif",
            compression="LZW",
            tile_size=16,
            pyramid_levels=0,
        ).write((labels,))
