"""Use existing Mocktome 0.4 observations through Omeify's ordinary image API.

Mocktome is an optional caller-owned installation, not an Omeify dependency.
The simulation is eager; array wrapping itself makes no full pixel copy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from omeify import (
    LabelImage,
    MultichannelImage,
    OMEImageSeries,
    OMEMultiSeriesWriter,
    OMETiffLabelReader,
    OMETiffReader,
    OMETiffWriter,
    PixelSize,
    RGBImage,
)


def run_example(
    output_dir: str | Path = "Scratch/mocktome_omeify", *,
    width_um: float = 128, height_um: float = 128, overwrite: bool = False,
) -> dict[str, object]:
    """Render one small section, write both image kinds and labels, and verify pixels."""
    from mocktome import Mocktome, SpectralAcquisitionSpec, tonsil_like_recipe

    out = Path(output_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())) and not overwrite:
        raise FileExistsError(f"Destination is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    recipe = tonsil_like_recipe(
        width_um=width_um, height_um=height_um, depth_um=32,
        microns_per_pixel=0.5, seed=7,
    )
    engine = Mocktome(recipe)
    volume = engine.build_volume()
    section = engine.section(volume, z_um=14, thickness_um=4)
    mif = engine.render_mif(
        section, acquisition=SpectralAcquisitionSpec(seed=101, retain_spectral=False),
    )
    he = engine.render_he(section)
    size = PixelSize(recipe.microns_per_pixel, recipe.microns_per_pixel, "µm")
    mif_path = out / "mif.ome.tif"
    products_path = out / "paired_products.ome.tif"
    with (
        MultichannelImage.from_array(
            mif.data, axes="CYX", channel_names=mif.channel_names, pixel_size=size,
        ) as signals,
        RGBImage.from_array(he.rgb, pixel_size=size) as brightfield,
        LabelImage.from_array(section.nucleus_labels, pixel_size=size) as nuclei,
    ):
        mif_report = OMETiffWriter(
            mif_path, compression="Deflate", tile_size=128, overwrite=overwrite,
        ).write(signals)
        products_report = OMEMultiSeriesWriter(
            products_path, compression="Deflate", tile_size=128, overwrite=overwrite,
        ).write((
            OMEImageSeries("H&E", brightfield),
            OMEImageSeries("Nuclear truth", nuclei),
        ))

    with OMETiffReader(mif_path) as actual:
        np.testing.assert_array_equal(actual.asarray(), mif.data)
        assert actual.channel_names == tuple(mif.channel_names)
    with OMETiffReader(products_path, series=0) as actual:
        np.testing.assert_array_equal(actual.asarray(), he.rgb)
    with OMETiffLabelReader(products_path, series=1) as actual:
        np.testing.assert_array_equal(actual.asarray(), section.nucleus_labels)
    print(f"Wrote {mif_path.resolve()} and {products_path.resolve()}")
    return {
        "mif_path": mif_path, "products_path": products_path,
        "mif_report": mif_report, "products_report": products_report,
    }


if __name__ == "__main__":
    run_example()
