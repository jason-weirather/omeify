"""File-free sources and shared writers. Import run_example from a notebook.

The coordinate pattern is a plumbing fixture, not simulated biological tissue.
Only requested regions are calculated. No Mocktome dependency is introduced.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from omeify import (
    ImageMetadata,
    ImageSource,
    LabelImage,
    MultichannelImage,
    OMEImageSeries,
    OMEMultiSeriesWriter,
    OMETiffReader,
    OMETiffWriter,
    PixelSize,
    RGBImage,
)


class CoordinateSource(ImageSource):
    """A deterministic procedural multichannel image with no backing array/file."""

    def __init__(self, *, height: int = 2048, width: int = 3072) -> None:
        super().__init__(ImageMetadata(
            axes="CYX",
            shape=(3, height, width),
            dtype=np.dtype("uint16"),
            image_type="multichannel",
            channel_names=("X pattern", "Y pattern", "Combined pattern"),
            pixel_size=PixelSize(0.5, 0.5, "µm"),
        ))
        self.regions_evaluated = 0

    def _read_region(
        self, y0: int, y1: int, x0: int, x1: int, *,
        level: int, channels: tuple[int, ...],
    ) -> np.ndarray:
        # ImageSource has already validated bounds, level, and selected channels.
        # Absolute coordinates keep overlapping reads and repeat reads identical.
        y = np.arange(y0, y1, dtype=np.uint64)[:, None]
        x = np.arange(x0, x1, dtype=np.uint64)[None, :]
        self.regions_evaluated += 1
        values = []
        for channel in channels:
            if channel == 0:
                plane = np.broadcast_to(x % 4096, (y1 - y0, x1 - x0))
            elif channel == 1:
                plane = np.broadcast_to(y % 4096, (y1 - y0, x1 - x0))
            else:
                plane = (3 * x + 7 * y) % 4096
            values.append(plane.astype(np.uint16))
        return np.stack(values)


def run_example(
    output_dir: str | Path = "Scratch/omeify_sources", *, overwrite: bool = False,
) -> dict[str, object]:
    """Write an on-demand image and an array-backed heterogeneous output file."""
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(f"Destination is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    signal_path = out / "procedural.ome.tif"
    mixed_path = out / "mixed_products.ome.tif"
    size = PixelSize(0.5, 0.5, "µm")

    with MultichannelImage(CoordinateSource()) as image:
        print(image)
        print("Regions before reading:", image.source.regions_evaluated)
        patch = image.read_region(100, 164, 200, 264, channels=[2, 0])
        # Image metadata supplies channel names and calibration to the writer.
        report = OMETiffWriter(
            signal_path, tile_size=256, compression="Deflate", overwrite=overwrite,
        ).write(image)
        print("Regions after writing:", image.source.regions_evaluated)

    with OMETiffReader(signal_path) as real:
        np.testing.assert_array_equal(
            real.read_region(100, 164, 200, 264, channels=[2, 0]), patch,
        )
        print("Readback:", real.channel_names, real.pixel_size)

    # Output arrays from any other program use the same image and writer contracts.
    rgb = np.zeros((129, 193, 3), dtype=np.uint8)
    rgb[..., 0] = np.arange(193, dtype=np.uint8)[None, :]
    labels = np.zeros((129, 193), dtype=np.uint32)
    labels[20:75, 30:90] = 1001
    labels[60:110, 110:180] = 9007
    with (
        RGBImage.from_array(rgb, pixel_size=size) as preview,
        LabelImage.from_array(labels, pixel_size=size) as objects,
    ):
        multi_report = OMEMultiSeriesWriter(
            mixed_path, tile_size=64, compression="Deflate", overwrite=overwrite,
        ).write((
            OMEImageSeries.from_image("RGB preview", preview),
            OMEImageSeries.from_image("Object labels", objects),
        ))

    print("Wrote:", signal_path.resolve())
    print("Wrote:", mixed_path.resolve())
    return {"procedural_path": signal_path, "products_path": mixed_path,
            "procedural_report": report, "products_report": multi_report}


if __name__ == "__main__":
    run_example()
