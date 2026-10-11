"""Create and inspect a small calibrated OME-Zarr without a TIFF intermediate."""
from pathlib import Path

import numpy as np

from omeify import MultichannelImage, OMEZarrReader, OMEZarrWriter, PixelSize


def run_example(output_path: str | Path = "Scratch/example.ome.zarr") -> dict:
    """Uses the ordinary Omeify installation. Refuses an existing output directory."""
    y, x = np.indices((257, 385), dtype=np.uint16)
    data = np.stack((x, y))
    with MultichannelImage.from_array(
        data, axes="CYX", channel_names=("X ramp", "Y ramp"),
        pixel_size=PixelSize(0.5, 0.5, "µm"),
    ) as image:
        report = OMEZarrWriter(output_path, tile_size=128).write(image)
    with OMEZarrReader(output_path) as image:
        np.testing.assert_array_equal(image.asarray(), data)
        print(image.inspect())
    return report


if __name__ == "__main__":
    run_example()
