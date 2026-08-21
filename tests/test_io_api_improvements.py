from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
import tifffile
from click.testing import CliRunner

from omeify import (
    AkoyaFusionQPTiffReader,
    Channel,
    OMETiffLabelReader,
    OMETiffReader,
    PixelSize,
    TemporaryOMETiffWriter,
    write_ometiff,
)
from omeify.cli import main
from omeify.inputs import AkoyaFusionQPTiff, HaloMIFTiff


def _fusion_description(
    *,
    name: str | None,
    biomarker: str | None,
    pixel_size_microns: float | None,
    stale_encoding: bool = False,
) -> str:
    declaration = '<?xml version="1.0" encoding="utf-16"?>' if stale_encoding else ""
    name_xml = f"<Name>{name}</Name>" if name is not None else ""
    biomarker_xml = f"<Biomarker>{biomarker}</Biomarker>" if biomarker is not None else ""
    pixel_xml = (
        f"<PixelSizeMicrons>{pixel_size_microns}</PixelSizeMicrons>"
        if pixel_size_microns is not None
        else ""
    )
    return (
        declaration
        + "<PerkinElmer-QPI-ImageDescription>"
        + name_xml
        + biomarker_xml
        + "<ScanProfile><root><ScanResolution>"
        + pixel_xml
        + "</ScanResolution></root></ScanProfile>"
        + "</PerkinElmer-QPI-ImageDescription>"
    )


def _write_fusion_qptiff(
    path: Path,
    data: np.ndarray,
    *,
    names: list[str | None] | None = None,
    biomarkers: list[str | None] | None = None,
    pixel_sizes: list[float | None] | None = None,
) -> None:
    channel_count = int(data.shape[0])
    names = names or [f"Opal {index + 1}" for index in range(channel_count)]
    biomarkers = biomarkers or [f"Marker {index + 1}" for index in range(channel_count)]
    pixel_sizes = pixel_sizes or [0.5068] * channel_count
    with tifffile.TiffWriter(path, bigtiff=True) as writer:
        for index, plane in enumerate(data):
            writer.write(
                plane,
                tile=(16, 16),
                description=_fusion_description(
                    name=names[index],
                    biomarker=biomarkers[index],
                    pixel_size_microns=pixel_sizes[index],
                    stale_encoding=index == 0,
                ),
                software="PerkinElmer-QPI synthetic" if index == 0 else False,
                metadata=None,
            )


def _write_planar_ome(
    path: Path,
    data: np.ndarray,
    *,
    names: list[str] | None = None,
    pixel_size: PixelSize = PixelSize(0.5, 0.6, "µm"),
) -> None:
    names = names or [f"Marker {index + 1}" for index in range(data.shape[0])]
    tifffile.imwrite(
        path,
        data,
        ome=True,
        tile=(16, 16),
        photometric="minisblack",
        metadata={
            "axes": "CYX",
            "PhysicalSizeX": pixel_size.x,
            "PhysicalSizeXUnit": pixel_size.unit,
            "PhysicalSizeY": pixel_size.y,
            "PhysicalSizeYUnit": pixel_size.unit,
            "Channel": {"Name": names},
        },
    )


def _read_ome_channel_names(path: Path) -> list[str]:
    with tifffile.TiffFile(path) as tiff:
        assert tiff.ome_metadata is not None
        import xml.etree.ElementTree as ET

        root = ET.fromstring(tiff.ome_metadata)
        return [
            element.attrib["Name"]
            for element in root.iter()
            if element.tag.rsplit("}", 1)[-1] == "Channel"
        ]


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("name", ("Opal 1", "Opal 2", "Opal 3")),
        ("biomarker", ("CD3", "PanCK", "DAPI")),
        ("auto", ("CD3", "PanCK", "DAPI")),
    ],
)
def test_fusion_reader_preserves_name_biomarker_and_selects_normalized_name(
    tmp_path: Path,
    field: str,
    expected: tuple[str, ...],
) -> None:
    source = tmp_path / "fusion.qptiff"
    data = np.arange(3 * 32 * 48, dtype=np.uint16).reshape(3, 32, 48)
    _write_fusion_qptiff(
        source,
        data,
        names=["Opal 1", "Opal 2", "Opal 3"],
        biomarkers=["CD3", "PanCK", "DAPI"],
    )

    with AkoyaFusionQPTiffReader(source, channel_name_field=field) as reader:
        assert reader.channel_names == expected
        assert reader.pixel_size == PixelSize(0.5068, 0.5068, "µm")
        assert reader[0].source_metadata["name"] == "Opal 1"
        assert reader[0].source_metadata["biomarker"] == "CD3"
        assert reader[0].id == "Channel:0:0"
        assert reader[0].source_id is None
        assert reader[0].id_is_generated is True
        np.testing.assert_array_equal(reader[1].read_region(2, 8, 3, 11), data[1, 2:8, 3:11])
        assert reader[2].array.dtype == np.dtype("uint16")
        assert reader[2].array.shape == (32, 48)
        np.testing.assert_array_equal(reader[2].array, data[2])


def test_fusion_auto_falls_back_to_name_when_biomarker_is_missing(tmp_path: Path) -> None:
    source = tmp_path / "fusion.qptiff"
    data = np.zeros((2, 32, 48), dtype=np.uint16)
    _write_fusion_qptiff(
        source,
        data,
        names=["Opal 520", "Opal 570"],
        biomarkers=["CD3", None],
    )

    with AkoyaFusionQPTiffReader(source, channel_name_field="auto") as reader:
        assert reader.channel_names == ("CD3", "Opal 570")
        assert reader[1].source_metadata["selected_name_field"] == "name"

    with pytest.raises(ValueError, match="missing Biomarker"):
        with AkoyaFusionQPTiffReader(source, channel_name_field="biomarker"):
            pass


def test_fusion_reader_rejects_inconsistent_pixel_sizes(tmp_path: Path) -> None:
    source = tmp_path / "fusion.qptiff"
    data = np.zeros((3, 32, 48), dtype=np.uint16)
    _write_fusion_qptiff(source, data, pixel_sizes=[0.5068, 0.5068, 0.5])

    with pytest.raises(ValueError, match="inconsistent PixelSizeMicrons"):
        with AkoyaFusionQPTiffReader(source):
            pass

    output = tmp_path / "output.ome.tif"
    with pytest.raises(ValueError, match="inconsistent PixelSizeMicrons"):
        AkoyaFusionQPTiff(source).convert(
            output,
            compression="Uncompressed",
            tile_size=16,
            pyramid_levels=0,
            calculate_checksums=False,
        )
    assert not output.exists()


def test_fusion_conversion_profile_preserves_source_fields_but_writes_selected_name(
    tmp_path: Path,
) -> None:
    source = tmp_path / "fusion.qptiff"
    output = tmp_path / "fusion.ome.tif"
    data = np.arange(2 * 32 * 48, dtype=np.uint16).reshape(2, 32, 48)
    _write_fusion_qptiff(
        source,
        data,
        names=["Opal 520", "Opal 570"],
        biomarkers=["CD3", "PanCK"],
    )

    report = AkoyaFusionQPTiff(
        source,
        channel_name_field="biomarker",
    ).convert(
        output,
        compression="Uncompressed",
        tile_size=16,
        pyramid_levels=0,
        calculate_checksums=False,
    )

    assert report["image"]["channel_names"] == ["CD3", "PanCK"]
    assert report["input_file"]["pixel_size"] == [0.5068, 0.5068, "µm"]
    assert report["image"]["source_channel_metadata"][0]["name"] == "Opal 520"
    assert report["image"]["source_channel_metadata"][0]["biomarker"] == "CD3"
    assert _read_ome_channel_names(output) == ["CD3", "PanCK"]


def test_python_channel_renames_require_explicit_mode_and_support_indices(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.ome.tif"
    output = tmp_path / "output.ome.tif"
    data = np.arange(3 * 32 * 48, dtype=np.uint16).reshape(3, 32, 48)
    _write_planar_ome(source, data, names=["A", "B", "C"])

    with pytest.raises(ValueError, match="explicitly set"):
        HaloMIFTiff(source, rename_channels={"A": "Alpha"})

    report = HaloMIFTiff(
        source,
        rename_channels={0: "Alpha", 2: "Gamma"},
        rename_channels_by="index",
    ).convert(
        output,
        compression="Uncompressed",
        tile_size=16,
        pyramid_levels=0,
        calculate_checksums=False,
    )
    assert report["image"]["channel_names"] == ["Alpha", "B", "Gamma"]

    with pytest.raises(TypeError, match="must be integers"):
        HaloMIFTiff(
            source,
            rename_channels={0: "Alpha", "B": "Beta"},  # type: ignore[dict-item]
            rename_channels_by="index",
        )

    duplicate_source = tmp_path / "duplicate-source.ome.tif"
    _write_planar_ome(duplicate_source, data, names=["A", "A", "C"])
    with pytest.raises(ValueError, match="ambiguous for duplicate"):
        HaloMIFTiff(
            duplicate_source,
            rename_channels={"A": "Alpha"},
            rename_channels_by="name",
        ).convert(
            tmp_path / "duplicate-output.ome.tif",
            compression="Uncompressed",
            tile_size=16,
            pyramid_levels=0,
            calculate_checksums=False,
        )


def test_cli_channel_rename_json_by_name_and_index(tmp_path: Path) -> None:
    source = tmp_path / "source.ome.tif"
    data = np.arange(3 * 32 * 48, dtype=np.uint16).reshape(3, 32, 48)
    _write_planar_ome(source, data, names=["DAPI", "PanCK", "CD3"])
    runner = CliRunner()

    name_map = tmp_path / "name.json"
    name_map.write_text(json.dumps({"DAPI": "DNA", "CD3": "CD3e"}), encoding="utf-8")
    name_output = tmp_path / "name.ome.tif"
    result = runner.invoke(
        main,
        [
            "convert",
            str(source),
            str(name_output),
            "--type",
            "halo_mif",
            "--rename-channels-json",
            str(name_map),
            "--rename-channels-by",
            "name",
            "--compression",
            "Uncompressed",
            "--tile-size",
            "16",
            "--pyramid-levels",
            "0",
            "--no-checksums",
        ],
    )
    assert result.exit_code == 0, result.output
    assert _read_ome_channel_names(name_output) == ["DNA", "PanCK", "CD3e"]

    index_map = tmp_path / "index.json"
    index_map.write_text(json.dumps({"0": "DNA", "2": "CD3e"}), encoding="utf-8")
    index_output = tmp_path / "index.ome.tif"
    result = runner.invoke(
        main,
        [
            "convert",
            str(source),
            str(index_output),
            "--type",
            "halo_mif",
            "--rename-channels-json",
            str(index_map),
            "--rename-channels-by",
            "index",
            "--compression",
            "Uncompressed",
            "--tile-size",
            "16",
            "--pyramid-levels",
            "0",
            "--no-checksums",
        ],
    )
    assert result.exit_code == 0, result.output
    assert _read_ome_channel_names(index_output) == ["DNA", "PanCK", "CD3e"]


def test_cli_rejects_missing_malformed_and_mixed_channel_rename_modes(tmp_path: Path) -> None:
    source = tmp_path / "source.ome.tif"
    data = np.zeros((2, 32, 48), dtype=np.uint16)
    _write_planar_ome(source, data, names=["DAPI", "PanCK"])
    runner = CliRunner()

    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps({"0": "DNA", "PanCK": "Epithelial"}), encoding="utf-8")

    missing_mode = runner.invoke(
        main,
        [
            "convert",
            str(source),
            str(tmp_path / "missing.ome.tif"),
            "--type",
            "halo_mif",
            "--rename-channels-json",
            str(mapping),
        ],
    )
    assert missing_mode.exit_code != 0
    assert "--rename-channels-by" in missing_mode.output

    mixed_index = runner.invoke(
        main,
        [
            "convert",
            str(source),
            str(tmp_path / "mixed-index.ome.tif"),
            "--type",
            "halo_mif",
            "--rename-channels-json",
            str(mapping),
            "--rename-channels-by",
            "index",
        ],
    )
    assert mixed_index.exit_code != 0
    assert "integer strings" in mixed_index.output

    numeric_name = tmp_path / "numeric-name.json"
    numeric_name.write_text(json.dumps({"0": "DNA"}), encoding="utf-8")
    mixed_name = runner.invoke(
        main,
        [
            "convert",
            str(source),
            str(tmp_path / "mixed-name.ome.tif"),
            "--type",
            "halo_mif",
            "--rename-channels-json",
            str(numeric_name),
            "--rename-channels-by",
            "name",
        ],
    )
    assert mixed_name.exit_code != 0
    assert "integer-like keys" in mixed_name.output


def test_channel_metadata_lookup_region_and_array_are_lazy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.ome.tif"
    data = np.arange(3 * 32 * 48, dtype=np.uint16).reshape(3, 32, 48)
    _write_planar_ome(source, data, names=["DAPI", "PanCK", "CD3"])

    original_asarray = tifffile.TiffPage.asarray
    materializations = {"count": 0}

    def forbidden_asarray(self, *args, **kwargs):
        materializations["count"] += 1
        raise AssertionError("channel metadata access must not materialize pixels")

    monkeypatch.setattr(tifffile.TiffPage, "asarray", forbidden_asarray)
    with OMETiffReader(source) as reader:
        assert reader.pixel_size == PixelSize(0.5, 0.6, "µm")
        assert tuple(channel.index for channel in reader.channels) == (0, 1, 2)
        assert tuple(channel.id for channel in reader.channels) == (
            "Channel:0:0",
            "Channel:0:1",
            "Channel:0:2",
        )
        assert reader[1].name == "PanCK"
        assert reader.get_by_name("CD3").index == 2
        assert reader.get_by_id("Channel:0:0").name == "DAPI"
        assert reader[0].dtype == np.dtype("uint16")
        assert reader[0].shape == (32, 48)
        np.testing.assert_array_equal(
            reader[2].read_region(2, 9, 3, 12),
            data[2, 2:9, 3:12],
        )
        assert materializations["count"] == 0

    monkeypatch.setattr(tifffile.TiffPage, "asarray", original_asarray)
    with OMETiffReader(source) as reader:
        channel_array = reader[1].array
        assert channel_array.dtype == np.dtype("uint16")
        assert channel_array.shape == (32, 48)
        np.testing.assert_array_equal(channel_array, data[1])


def test_duplicate_channel_name_lookup_is_explicitly_ambiguous(tmp_path: Path) -> None:
    source = tmp_path / "duplicates.ome.tif"
    data = np.zeros((3, 32, 48), dtype=np.uint16)
    _write_planar_ome(source, data, names=["DAPI", "DAPI", "CD3"])

    with OMETiffReader(source) as reader:
        with pytest.raises(ValueError, match="ambiguous"):
            reader.get_by_name("DAPI")
        with pytest.raises(KeyError, match="No channel"):
            reader.get_by_name("missing")
        with pytest.raises(IndexError, match="outside"):
            _ = reader[-1]


def test_pixel_size_is_immutable_serializable_and_scalable() -> None:
    original = PixelSize(0.5, 0.6, "µm")
    assert original.to_tuple() == (0.5, 0.6, "µm")
    assert PixelSize.from_tuple(original.to_tuple()) == original
    scaled = original.scaled(2)
    assert scaled == PixelSize(1.0, 1.2, "µm")
    assert original == PixelSize(0.5, 0.6, "µm")
    converted = original.converted_to("nm")
    assert converted.unit == "nm"
    assert converted.x == pytest.approx(500.0)
    assert converted.y == pytest.approx(600.0)
    with pytest.raises(FrozenInstanceError):
        original.x = 1.0  # type: ignore[misc]


def test_reader_reports_adjusted_pixel_size_for_pyramid_level(tmp_path: Path) -> None:
    output = tmp_path / "pyramid.ome.tif"
    data = np.arange(2 * 32 * 48, dtype=np.uint16).reshape(2, 32, 48)
    write_ometiff(
        output,
        channels=data,
        channel_names=["A", "B"],
        pixel_size=PixelSize(0.5, 0.6, "µm"),
        compression="Uncompressed",
        tile_size=16,
        pyramid_levels=1,
    )

    with OMETiffReader(output) as reader:
        assert reader.pixel_size == PixelSize(0.5, 0.6, "µm")
        assert reader.pixel_size_at_level(1) == PixelSize(1.0, 1.2, "µm")


def test_temporary_ome_tiff_writer_creates_and_cleans_up(tmp_path: Path) -> None:
    data = np.zeros((1, 16, 16), dtype=np.uint8)
    with TemporaryOMETiffWriter(
        directory=tmp_path,
        channel_names=["DAPI"],
        pixel_size=PixelSize(0.5, 0.5, "µm"),
        compression="Uncompressed",
        tile_size=16,
        pyramid_levels=0,
    ) as writer:
        writer.write(data)
        temporary_path = writer.path
        assert temporary_path.exists()
        with OMETiffReader(temporary_path) as reader:
            assert reader.channel_names == ("DAPI",)
    assert not temporary_path.exists()


def test_temporary_ome_tiff_writer_cleans_up_after_exception(tmp_path: Path) -> None:
    temporary_path: Path | None = None
    with pytest.raises(RuntimeError, match="boom"):
        with TemporaryOMETiffWriter(
            directory=tmp_path,
            channel_names=["DAPI"],
            pixel_size=PixelSize(0.5, 0.5, "µm"),
            compression="Uncompressed",
            tile_size=16,
            pyramid_levels=0,
        ) as writer:
            writer.write(np.zeros((1, 16, 16), dtype=np.uint8))
            temporary_path = writer.path
            assert temporary_path.exists()
            raise RuntimeError("boom")
    assert temporary_path is not None
    assert not temporary_path.exists()


def test_write_ometiff_accepts_cyx_array_and_list_of_yx_arrays(tmp_path: Path) -> None:
    data = np.arange(3 * 32 * 48, dtype=np.uint16).reshape(3, 32, 48)
    pixel_size = PixelSize(0.5, 0.6, "µm")

    array_output = tmp_path / "array.ome.tif"
    write_ometiff(
        array_output,
        channels=data,
        channel_names=["DAPI", "PanCK", "CD3"],
        pixel_size=pixel_size,
        compression="Uncompressed",
        tile_size=16,
        pyramid_levels=0,
    )
    with OMETiffReader(array_output) as reader:
        np.testing.assert_array_equal(reader.asarray(), data)
        assert reader.channel_names == ("DAPI", "PanCK", "CD3")

    list_output = tmp_path / "list.ome.tif"
    write_ometiff(
        list_output,
        channels=[data[0], data[1], data[2]],
        channel_names=["DAPI", "PanCK", "CD3"],
        pixel_size=pixel_size,
        compression="Uncompressed",
        tile_size=16,
        pyramid_levels=0,
    )
    with OMETiffReader(list_output) as reader:
        np.testing.assert_array_equal(reader.asarray(), data)


def test_write_ometiff_streams_lazy_channel_objects_without_array_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.ome.tif"
    output = tmp_path / "selected.ome.tif"
    data = np.arange(3 * 32 * 48, dtype=np.uint16).reshape(3, 32, 48)
    _write_planar_ome(source, data, names=["DAPI", "PanCK", "CD3"])

    def forbidden_asarray(self, *, level: int = 0):
        raise AssertionError("write_ometiff must stream Channel objects instead of using .array")

    monkeypatch.setattr(Channel, "asarray", forbidden_asarray)
    with OMETiffReader(source) as reader:
        assert reader.pixel_size is not None
        report = write_ometiff(
            output,
            channels=[reader.get_by_name("DAPI"), reader.get_by_name("CD3")],
            pixel_size=reader.pixel_size,
            compression="Uncompressed",
            tile_size=16,
            pyramid_levels=0,
        )
    assert report["image"]["channel_names"] == ["DAPI", "CD3"]
    with tifffile.TiffFile(output) as tiff:
        np.testing.assert_array_equal(tiff.series[0].asarray(), data[[0, 2]])


def test_fusion_cli_profile_and_subcommand_only_policy(tmp_path: Path) -> None:
    source = tmp_path / "fusion.qptiff"
    output = tmp_path / "fusion.ome.tif"
    data = np.arange(2 * 32 * 48, dtype=np.uint16).reshape(2, 32, 48)
    _write_fusion_qptiff(
        source,
        data,
        names=["Opal 1", "Opal 2"],
        biomarkers=["CD3", "PanCK"],
    )
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "convert",
            str(source),
            str(output),
            "--type",
            "qptiff_fusion",
            "--channel-name-field",
            "name",
            "--compression",
            "Uncompressed",
            "--tile-size",
            "16",
            "--pyramid-levels",
            "0",
            "--no-checksums",
        ],
    )
    assert result.exit_code == 0, result.output
    assert _read_ome_channel_names(output) == ["Opal 1", "Opal 2"]

    old_form = runner.invoke(
        main,
        [
            str(source),
            str(tmp_path / "old.ome.tif"),
            "--type",
            "qptiff_fusion",
        ],
    )
    assert old_form.exit_code != 0
    assert "No such command" in old_form.output


def test_label_reader_exposes_pixel_size(tmp_path: Path) -> None:
    output = tmp_path / "labels.ome.tif"
    labels = np.arange(32 * 48, dtype=np.uint16).reshape(32, 48)
    write_ometiff(
        output,
        channels=[labels],
        channel_names=["Labels"],
        pixel_size=PixelSize(0.7, 0.8, "µm"),
        image_type="label",
        compression="Uncompressed",
        tile_size=16,
        pyramid_levels=0,
    )
    with OMETiffLabelReader(output) as reader:
        assert reader.pixel_size == PixelSize(0.7, 0.8, "µm")


def test_writer_serializes_pixel_size_unit_without_forcing_micrometers(
    tmp_path: Path,
) -> None:
    output = tmp_path / "nanometers.ome.tif"
    write_ometiff(
        output,
        channels=np.zeros((1, 16, 16), dtype=np.uint8),
        channel_names=["DAPI"],
        pixel_size=PixelSize(500.0, 600.0, "nm"),
        compression="Uncompressed",
        tile_size=16,
        pyramid_levels=0,
    )

    with OMETiffReader(output) as reader:
        assert reader.pixel_size == PixelSize(500.0, 600.0, "nm")
    with tifffile.TiffFile(output) as tiff:
        assert tiff.ome_metadata is not None
        assert 'PhysicalSizeX="500.0"' in tiff.ome_metadata
        assert 'PhysicalSizeXUnit="nm"' in tiff.ome_metadata
        assert 'PhysicalSizeY="600.0"' in tiff.ome_metadata
        assert 'PhysicalSizeYUnit="nm"' in tiff.ome_metadata
