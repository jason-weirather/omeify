from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from omeify.converters.tifffile_converter import _mean_downsample_2x
from omeify.inputs import AkoyaMIFQptiff, HaloMIFTiff


def _write_source(
    path: Path,
    data: np.ndarray,
    *,
    byteorder: str | None = None,
    tile: tuple[int, int] | None = (16, 16),
    rowsperstrip: int | None = None,
    significant_bits: int | None = None,
) -> None:
    metadata = {
        "axes": "CYX",
        "PhysicalSizeX": 0.5,
        "PhysicalSizeXUnit": "µm",
        "PhysicalSizeY": 0.6,
        "PhysicalSizeYUnit": "µm",
        "Channel": {"Name": [f"Marker {index + 1}" for index in range(data.shape[0])]},
    }
    if significant_bits is not None:
        metadata["SignificantBits"] = significant_bits
    with tifffile.TiffWriter(path, bigtiff=True, ome=True, byteorder=byteorder) as writer:
        writer.write(
            data,
            tile=tile,
            rowsperstrip=rowsperstrip,
            photometric="minisblack",
            compression=None,
            metadata=metadata,
        )


def _mean2(array: np.ndarray) -> np.ndarray:
    channels, height, width = array.shape
    output = np.zeros((channels, (height + 1) // 2, (width + 1) // 2), dtype=array.dtype)
    for c in range(channels):
        for y in range(output.shape[1]):
            for x in range(output.shape[2]):
                block = array[c, y * 2 : y * 2 + 2, x * 2 : x * 2 + 2]
                output[c, y, x] = np.rint(block.astype(np.float64).mean()).astype(array.dtype)
    return output


def _akoya_description(channel_name: str, pixel_size_um: float = 0.5) -> str:
    return (
        "<PerkinElmer-QPI-ImageDescription>"
        f"<Name>{channel_name}</Name>"
        "<ScanProfile><root><ScanResolution>"
        f"<PixelSizeMicrons>{pixel_size_um}</PixelSizeMicrons>"
        "</ScanResolution></root></ScanProfile>"
        "</PerkinElmer-QPI-ImageDescription>"
    )


def _write_synthetic_qptiff(path: Path, base: np.ndarray) -> None:
    channel_names = [f"Akoya {index + 1}" for index in range(base.shape[0])]
    source_pyramid = np.full(
        (base.shape[0], base.shape[1] // 2, base.shape[2] // 2),
        np.iinfo(base.dtype).max,
        dtype=base.dtype,
    )
    thumbnail = np.zeros((8, 12), dtype=np.uint8)

    # QPTIFF lays out all full-resolution channel IFDs, an associated
    # thumbnail, and then channel IFDs for each reduced-resolution level.
    with tifffile.TiffWriter(path, bigtiff=True) as writer:
        for index, plane in enumerate(base):
            writer.write(
                plane,
                tile=(16, 16),
                description=_akoya_description(channel_names[index]),
                software="PerkinElmer-QPI synthetic" if index == 0 else False,
                metadata=None,
            )
        writer.write(
            thumbnail,
            description=_akoya_description("Thumbnail"),
            software=False,
            metadata=None,
        )
        for index, plane in enumerate(source_pyramid):
            writer.write(
                plane,
                tile=(16, 16),
                description=_akoya_description(channel_names[index]),
                software=False,
                subfiletype=1,
                metadata=None,
            )


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_conversion_preserves_dtype_and_rebuilds_pyramid(tmp_path: Path, dtype) -> None:
    base = np.arange(3 * 35 * 49, dtype=np.uint32).reshape(3, 35, 49)
    data = (base % np.iinfo(dtype).max).astype(dtype)
    source = tmp_path / "source.ome.tif"
    output = tmp_path / "output.ome.tif"
    _write_source(source, data)

    processor = HaloMIFTiff(source)
    report = processor.convert(
        output,
        compression="Uncompressed",
        tile_size=16,
        pyramid_levels=2,
        downsample="mean",
        calculate_checksums=False,
    )

    assert report["output_file"]["dtype"] == np.dtype(dtype).name
    assert report["input_file"]["sha256_checksum"] is None
    assert report["output_file"]["sha256_checksum"] is None
    assert report["verification"]["ome_tiff_recognized"] is True
    assert report["verification"]["bigtiff"] is True
    assert report["verification"]["output_byte_order"] == "little"
    assert report["verification"]["byte_order_metadata_matches_tiff"] is True
    assert report["verification"]["significant_bits_matches_dtype"] is True
    assert report["verification"]["tiff_data_mapping_matches"] is True
    assert report["verification"]["pyramid_annotation_linked"] is True
    assert report["verification"]["dtype_matches_source"] is True
    assert report["verification"]["pyramid_level_shapes_match"] is True
    assert report["verification"]["top_level_ifd_count_matches"] is True
    assert report["verification"]["subifd_layout_matches"] is True
    assert report["verification"]["all_levels_tiled"] is True
    assert report["verification"]["base_pixel_values_checked"] is True
    assert report["verification"]["base_pixel_values_match"] is True
    assert report["verification"]["channels_checked"] == data.shape[0]
    assert report["verification"]["points_per_channel"] == 3
    assert report["miti_header"]["is_valid"] is True
    assert report["miti_header"]["errors"] == []
    with tifffile.TiffFile(output) as tif:
        assert tif.is_ome
        series = tif.series[0]
        assert series.axes == "CYX"
        assert series.dtype == np.dtype(dtype)
        assert len(series.levels) == 3
        np.testing.assert_array_equal(series.levels[0].asarray(), data)
        level1 = _mean2(data)
        np.testing.assert_array_equal(series.levels[1].asarray(), level1)
        np.testing.assert_array_equal(series.levels[2].asarray(), _mean2(level1))
        assert 'Name="Marker 1"' in tif.ome_metadata
        assert 'DimensionOrder="XYZCT"' in tif.ome_metadata
        assert 'BigEndian="false"' in tif.ome_metadata
        assert '<AnnotationRef ID="Annotation:0"/>' in tif.ome_metadata
        assert 'Namespace="openmicroscopy.org/PyramidResolution"' in tif.ome_metadata


def test_channel_rename_and_omit_uuid(tmp_path: Path) -> None:
    data = np.arange(2 * 32 * 32, dtype=np.uint8).reshape(2, 32, 32)
    source = tmp_path / "source.ome.tif"
    output = tmp_path / "output.ome.tif"
    _write_source(source, data)

    processor = HaloMIFTiff(source, rename_channels={"Marker 1": "DAPI"})
    report = processor.convert(
        output,
        display_uuid=False,
        compression="Uncompressed",
        tile_size=16,
        pyramid_levels=1,
        calculate_checksums=False,
    )
    assert report["ome"]["uuid"] is None
    with tifffile.TiffFile(output) as tif:
        assert 'Name="DAPI"' in tif.ome_metadata
        assert " UUID=" not in tif.ome_metadata


def test_akoya_qptiff_profile_ignores_source_pyramid(tmp_path: Path) -> None:
    data = np.arange(2 * 32 * 48, dtype=np.uint16).reshape(2, 32, 48)
    source = tmp_path / "source.qptiff"
    output = tmp_path / "output.ome.tif"
    _write_synthetic_qptiff(source, data)

    report = AkoyaMIFQptiff(source).convert(
        output,
        compression="Uncompressed",
        tile_size=16,
        pyramid_levels=1,
        calculate_checksums=False,
    )

    assert report["input_file"]["source_axes"] == "CYX"
    assert report["image"]["channel_names"] == ["Akoya 1", "Akoya 2"]
    assert report["image"]["physical_size_x_um"] == pytest.approx(0.5)
    with tifffile.TiffFile(output) as tif:
        np.testing.assert_array_equal(tif.series[0].levels[0].asarray(), data)
        np.testing.assert_array_equal(tif.series[0].levels[1].asarray(), _mean2(data))
        # The synthetic source pyramid is all max-valued pixels.  This proves
        # the output level was rebuilt from full resolution instead of copied.
        assert not np.all(
            tif.series[0].levels[1].asarray() == np.iinfo(data.dtype).max
        )


def test_zero_pyramid_levels_writes_base_only(tmp_path: Path) -> None:
    data = np.arange(2 * 32 * 48, dtype=np.uint16).reshape(2, 32, 48)
    source = tmp_path / "source.ome.tif"
    output = tmp_path / "output.ome.tif"
    _write_source(source, data)

    report = HaloMIFTiff(source).convert(
        output,
        compression="Uncompressed",
        tile_size=16,
        pyramid_levels=0,
        calculate_checksums=False,
    )

    assert report["pyramid"]["subresolution_count"] == 0
    assert report["miti_header"]["is_valid"] is True
    with tifffile.TiffFile(output) as tif:
        assert len(tif.series[0].levels) == 1
        np.testing.assert_array_equal(tif.series[0].asarray(), data)


def test_strip_source_and_mismatched_output_grid(tmp_path: Path) -> None:
    data = np.arange(3 * 37 * 53, dtype=np.uint16).reshape(3, 37, 53)
    source = tmp_path / "source-strips.ome.tif"
    output = tmp_path / "output.ome.tif"
    _write_source(source, data, tile=None, rowsperstrip=7)

    report = HaloMIFTiff(source).convert(
        output,
        compression="Uncompressed",
        tile_size=16,
        pyramid_levels=2,
        calculate_checksums=False,
    )

    assert report["verification"]["all_levels_tiled"] is True
    with tifffile.TiffFile(output) as tif:
        np.testing.assert_array_equal(tif.series[0].levels[0].asarray(), data)
        np.testing.assert_array_equal(tif.series[0].levels[1].asarray(), _mean2(data))


def test_float32_dtype_and_miti_type_are_preserved(tmp_path: Path) -> None:
    data = np.linspace(0.0, 1.0, 2 * 33 * 47, dtype=np.float32).reshape(2, 33, 47)
    source = tmp_path / "source-float.ome.tif"
    output = tmp_path / "output.ome.tif"
    _write_source(source, data)

    report = HaloMIFTiff(source).convert(
        output,
        compression="Uncompressed",
        tile_size=16,
        pyramid_levels=1,
        calculate_checksums=False,
    )

    assert report["output_file"]["dtype"] == "float32"
    assert report["miti_header"]["is_valid"] is True
    assert 'Type="float"' in report["ome"]["xml_string"]
    with tifffile.TiffFile(output) as tif:
        np.testing.assert_array_equal(tif.series[0].levels[0].asarray(), data)


def test_significant_bits_matches_dtype_width(tmp_path: Path) -> None:
    data = np.arange(2 * 32 * 48, dtype=np.uint16).reshape(2, 32, 48)
    source = tmp_path / "source-12bit.ome.tif"
    output = tmp_path / "output.ome.tif"
    _write_source(source, data, significant_bits=12)

    report = HaloMIFTiff(source).convert(
        output,
        compression="Uncompressed",
        tile_size=16,
        pyramid_levels=0,
        calculate_checksums=False,
    )

    assert report["image"]["significant_bits"] == 16
    assert report["verification"]["significant_bits_matches_dtype"] is True
    with tifffile.TiffFile(output) as tif:
        assert 'SignificantBits="16"' in tif.ome_metadata


def test_deflate_is_lossless_and_uses_the_same_streaming_path(tmp_path: Path) -> None:
    data = np.arange(2 * 35 * 49, dtype=np.uint16).reshape(2, 35, 49)
    source = tmp_path / "source.ome.tif"
    output = tmp_path / "output.ome.tif"
    _write_source(source, data)

    report = HaloMIFTiff(source).convert(
        output,
        compression="Deflate",
        tile_size=16,
        pyramid_levels=1,
        calculate_checksums=False,
    )

    assert report["output_file"]["lossless_compression"] is True
    assert report["verification"]["base_pixel_values_match"] is True
    with tifffile.TiffFile(output) as tif:
        np.testing.assert_array_equal(tif.series[0].levels[0].asarray(), data)


def test_unsupported_dtype_fails_without_casting(tmp_path: Path) -> None:
    data = np.arange(2 * 16 * 16, dtype=np.uint64).reshape(2, 16, 16)
    source = tmp_path / "source-uint64.tif"
    output = tmp_path / "output.ome.tif"
    tifffile.imwrite(source, data, photometric="minisblack", metadata={"axes": "CYX"})

    with pytest.raises(TypeError, match="Unsupported mIF source dtype uint64"):
        HaloMIFTiff(source).convert(
            output,
            compression="Uncompressed",
            tile_size=16,
            pyramid_levels=0,
            calculate_checksums=False,
        )
    assert not output.exists()


def test_input_and_output_must_differ(tmp_path: Path) -> None:
    data = np.arange(2 * 32 * 32, dtype=np.uint16).reshape(2, 32, 32)
    source = tmp_path / "source.ome.tif"
    _write_source(source, data)

    with pytest.raises(ValueError, match="must be different"):
        HaloMIFTiff(source).convert(
            source,
            compression="Uncompressed",
            tile_size=16,
            pyramid_levels=0,
            calculate_checksums=False,
        )


def test_big_endian_input_is_written_little_endian_without_value_change(
    tmp_path: Path,
) -> None:
    data = np.arange(2 * 32 * 48, dtype=np.uint16).reshape(2, 32, 48)
    source = tmp_path / "source-big-endian.ome.tif"
    output = tmp_path / "output.ome.tif"
    _write_source(source, data, byteorder=">")

    report = HaloMIFTiff(source).convert(
        output,
        compression="Uncompressed",
        tile_size=16,
        pyramid_levels=0,
        calculate_checksums=False,
    )

    assert report["input_file"]["byte_order"] == "big"
    assert report["output_file"]["byte_order"] == "little"
    assert report["verification"]["byte_order_metadata_matches_tiff"] is True
    with tifffile.TiffFile(output) as tif:
        assert tif.byteorder == "<"
        assert 'BigEndian="false"' in tif.ome_metadata
        np.testing.assert_array_equal(tif.series[0].asarray(), data)


def test_explicit_pyramid_level_count_must_be_possible(tmp_path: Path) -> None:
    data = np.arange(2 * 16 * 16, dtype=np.uint16).reshape(2, 16, 16)
    source = tmp_path / "source.ome.tif"
    output = tmp_path / "output.ome.tif"
    _write_source(source, data)

    with pytest.raises(ValueError, match="already reached 1x1"):
        HaloMIFTiff(source).convert(
            output,
            compression="Uncompressed",
            tile_size=16,
            pyramid_levels=10,
            calculate_checksums=False,
        )
    assert not output.exists()


def test_mean_downsample_preserves_signed_integer_dtype() -> None:
    data = np.array([[-3, -2], [2, 3]], dtype=np.int16)
    result = _mean_downsample_2x(data, (1, 1))
    assert result.dtype == np.dtype("int16")
    assert result[0, 0] == 0


def test_compatibility_features_expose_old_metadata_properties(tmp_path: Path) -> None:
    from omeify.utils import TiffImageFeatures

    data = np.arange(2 * 32 * 32, dtype=np.uint16).reshape(2, 32, 32)
    source = tmp_path / "source.ome.tif"
    _write_source(source, data, byteorder=">")

    features = TiffImageFeatures(
        source,
        profile="halo_mif",
        input_type="HALO mIF TIFF",
    )

    assert features.image_id == "Image:0"
    assert features.pixel_id == "Pixels:0"
    assert features.big_endian == "true"
    assert features.dimension_order == "XYZCT"
    assert features.interleaved == "false"
    assert features.physical_size_x_unit == "µm"
    assert features.physical_size_y_unit == "µm"
    assert features.size_z == 1
    assert features.size_t == 1
    assert features.plane_count == 2
    assert features.type == "uint16"


def test_click_version_is_eager_and_json() -> None:
    import json

    from click.testing import CliRunner

    from omeify.cli import main

    result = CliRunner().invoke(main, ["--version"])
    assert result.exit_code == 0
    version_info = json.loads(result.output)
    assert version_info["omeify"] == "0.4.1"
    assert "tifffile" in version_info

    help_result = CliRunner().invoke(main, ["--help"])
    assert help_result.exit_code == 0
    assert "--strict-miti" not in help_result.output

def test_pyproject_is_the_version_authority() -> None:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    from omeify import __version__

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as handle:
        project_version = tomllib.load(handle)["project"]["version"]
    assert __version__ == project_version == "0.4.1"


def test_bundled_miti_json_schema_is_available() -> None:
    import json
    from importlib.resources import files

    resource = files("omeify.schemas").joinpath("miti_ome_tiff_header.schema.json")
    schema = json.loads(resource.read_text(encoding="utf-8"))
    assert schema["$schema"].endswith("draft/2020-12/schema")
    assert "uint8" in schema["properties"]["pixel_type"]["enum"]
