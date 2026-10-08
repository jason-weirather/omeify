"""Input color meaning is checked before decoding or relabeling sample values."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from omeify import OMETiffReader, convert
from omeify.io.source_reader import source_reader
from omeify.io.tiff import TiffPlaneReader


def color_source(path: Path, profile: str, *, levels: bool, compression: str | None = None):
    pixels = np.full((64, 96, 3), (230, 91, 170), dtype=np.uint8)
    metadata = {"axes": "YXS", "PhysicalSizeX": .5, "PhysicalSizeY": .5,
                "PhysicalSizeXUnit": "µm", "PhysicalSizeYUnit": "µm"}
    with tifffile.TiffWriter(path, ome=profile == "ome_tiff") as writer:
        writer.write(
            pixels, photometric="rgb", tile=(32, 32), compression=compression,
            subifds=1 if levels and profile == "ome_tiff" else None,
            metadata=metadata if profile == "ome_tiff" else None,
            description=None if profile == "ome_tiff" else "Aperio Image Library v10|MPP = 0.5",
        )
        if levels:
            if profile == "svs":
                writer.write(pixels[::4, ::4], photometric="rgb", metadata=None)
            writer.write(pixels[::2, ::2], photometric="rgb", tile=(32, 32),
                         compression=compression, subfiletype=1 if profile == "ome_tiff" else 0,
                         metadata=None)
    return pixels


@pytest.mark.parametrize("photometric", [6, 8], ids=["raw-ycbcr", "cielab"])
@pytest.mark.parametrize("compression", [None, "deflate"])
@pytest.mark.parametrize("profile", ["ome_tiff", "svs"])
@pytest.mark.parametrize("bad_level", [0, 1], ids=["base", "reduced"])
def test_unsupported_colors_fail_before_decode_and_output(
    tmp_path, monkeypatch, photometric, compression, profile, bad_level,
):
    source = tmp_path / "source.tif"
    output = tmp_path / "keep.ome.tif"
    color_source(source, profile, levels=True, compression=compression)
    # Change the interpretation of uncompressed/Deflate triples. No JPEG codec
    # or guessed color transform is involved in these negative input fixtures.
    with tifffile.TiffFile(source, mode="r+") as tiff:
        page = (tiff.pages[0] if bad_level == 0 else
                tiff.pages[0].pages[0] if profile == "ome_tiff" else tiff.pages[2])
        page.tags["PhotometricInterpretation"].overwrite(photometric)
    original = source.read_bytes()
    output.write_bytes(b"KEEP EXISTING OUTPUT")
    monkeypatch.setattr(TiffPlaneReader, "_decode_segment",
                        lambda *a, **k: pytest.fail("Input preflight must be metadata-only"))
    with pytest.raises(ValueError, match="Unsupported RGB photometric/codec") as error:
        with source_reader(source, input_type=profile, series=0, channel_name_field=None):
            pass
    assert f"level {bad_level}" in str(error.value)
    with pytest.raises(ValueError, match="Unsupported RGB photometric/codec"):
        convert(source, output, input_type=profile, compression="Uncompressed")
    assert output.read_bytes() == b"KEEP EXISTING OUTPUT"
    assert source.read_bytes() == original
    assert not list(tmp_path.glob(".omeify-*.partial"))


@pytest.mark.parametrize("photometric", [6, 8])
def test_direct_plane_reader_does_not_admit_unsupported_color_triples(tmp_path, photometric):
    path = tmp_path / "raw.tif"
    color_source(path, "ome_tiff", levels=False)
    with tifffile.TiffFile(path, mode="r+") as tiff:
        tiff.pages[0].tags["PhotometricInterpretation"].overwrite(photometric)
    with tifffile.TiffFile(path) as tiff, pytest.raises(ValueError, match="not relabeled as RGB"):
        TiffPlaneReader(tiff.pages[0])


@pytest.mark.parametrize("profile", ["ome_tiff", "svs"])
@pytest.mark.parametrize("compression", [None, "deflate"])
def test_supported_rgb_keeps_complete_small_rasters_at_every_level(tmp_path, profile, compression):
    path = tmp_path / "rgb.tif"
    pixels = color_source(path, profile, levels=True, compression=compression)
    with source_reader(path, input_type=profile, series=0, channel_name_field=None) as image:
        assert image.image_type == "rgb" and image.channel_count == 1
        np.testing.assert_array_equal(image.asarray(), pixels)
        np.testing.assert_array_equal(image.asarray(level=1), pixels[::2, ::2])


@pytest.mark.parametrize("sampling", ["444", "422", "420", "411"])
def test_jpeg_rgb_and_ycbcr_inputs_remain_supported(tmp_path, sampling):
    pytest.importorskip("imagecodecs", reason="Real JPEG encoding and decoding required")
    pixels = np.full((64, 96, 3), (230, 91, 170), dtype=np.uint8)
    path = tmp_path / "jpeg.ome.tif"
    subsampling = {"444": (1, 1), "422": (2, 1), "420": (2, 2), "411": (4, 1)}[sampling]
    with tifffile.TiffWriter(path, ome=True) as writer:
        for level in range(2):
            writer.write(
                pixels[::2**level, ::2**level], photometric="rgb", compression="jpeg",
                compressionargs={"level": 95, "outcolorspace": "RGB" if sampling == "444" else "YCBCR"},
                subsampling=subsampling, tile=(32, 32), subifds=1 if level == 0 else None,
                subfiletype=0 if level == 0 else 1, metadata={"axes": "YXS"},
            )
    with OMETiffReader(path) as image:
        for level in range(2):
            np.testing.assert_allclose(image.asarray(level=level)[8, 8], pixels[8, 8], atol=4, rtol=0)
