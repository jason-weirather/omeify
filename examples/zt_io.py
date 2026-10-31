"""Create small deterministic Z/T products through real public writers.

Run: python -m examples.zt_io ~/Scratch/omeify-zt-demo
Requires ordinary declared dependencies, including Zarr, imagecodecs, and ome-schema.
The destination products must not exist. This is a test/demo, not a download tool.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from omeify import AxisInfo, ImageDataset, OMETiffWriter, OMEZarrWriter, PixelSize, open_dataset


def run_example(directory: str | Path) -> dict[str, Path]:
    directory = Path(directory).expanduser()
    products = {"tiff": directory / "synthetic.ome.tif",
                "zarr_xy": directory / "synthetic-xy.zarr",
                "zarr_xyz": directory / "synthetic-xyz.zarr"}
    for path in products.values():
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to replace demo product: {path}")
    directory.mkdir(parents=True, exist_ok=True)
    t, c, z, y, x = np.indices((3, 2, 7, 65, 81))
    values = (t * 10000 + c * 3000 + z * 300 + y * 3 + x).astype(np.uint16)
    with ImageDataset.from_array(
        values, axes="TCZYX", channel_names=("Signal A", "Signal B"),
        pixel_size=PixelSize(0.5, 0.5, "µm"),
        axis_info=(AxisInfo("Z", 7, "space", "µm", 2.0, 0.0),
                   AxisInfo("T", 3, "time", "s", 30.0, 0.0)),
    ) as dataset:
        OMETiffWriter(products["tiff"], compression="Deflate", tile_size=32,
                      pyramid_levels=2, overwrite=False).write_dataset(dataset)
        for key, axes in (("zarr_xy", "xy"), ("zarr_xyz", "xyz")):
            OMEZarrWriter(products[key], tile_size=32, pyramid_levels=2).write_dataset(
                dataset, pyramid_axes=axes)
    for key, path in products.items():
        with open_dataset(path, input_type="ome_tiff" if key == "tiff" else "ome_zarr") as dataset:
            for ti in range(3):
                for zi in range(7):
                    with dataset.plane(t=ti, z=zi) as image:
                        np.testing.assert_array_equal(image.asarray(), values[ti, :, zi])
            print(f"{key}: {path} | {dataset.axes} {dataset.shape}")
    return products


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    run_example(parser.parse_args().directory)
