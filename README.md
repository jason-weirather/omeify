# omeify

`omeify` converts multiplexed tissue images into a standardized, deidentified, tiled and pyramidal OME-TIFF. Version 0.4 replaces the Java `bioformats2raw` → Zarr → `raw2ometiff` chain with a pure-Python `tifffile` implementation.

## What this iteration supports

- Akoya mIF QPTIFF
- HALO planar mIF TIFF / OME-TIFF
- Akoya component TIFF
- `uint8`, `uint16`, and the other scalar dtypes supported by OME-TIFF, without intensity rescaling
- Fresh pyramid construction from the full-resolution source pixels
- Lossless LZW, Deflate, ZSTD, or uncompressed output; JPEG remains available for `uint8` only
- Click-based CLI, `pyproject.toml` packaging, and no Java runtime

The H&E class remains importable but intentionally raises a clear `NotImplementedError`. The next iteration should write H&E as one interleaved RGB image (`SamplesPerPixel=3`), not as three grayscale pages.

## MITI scope

The output covers the standardized OME-TIFF image-file portion of MITI. Whether an image is Level 2 or Level 3 depends on the processing and QC performed upstream. A complete MITI submission also requires linked file, biospecimen, reagent, acquisition, channel, instrument, processing, and related manifest metadata. The OME root UUID is valid OME-XML and is retained by default.

The current MITI `yaml/05-ome-tiff-header.yaml` profile lists only `uint16` and `float` as accepted pixel types. Therefore, a native `uint8` image can be valid OME-TIFF while failing that narrower MITI header profile. `omeify` does not mutate the scientific data to make the checkbox turn green: it preserves the native dtype, records the header result in the conversion report, and offers `--strict-miti` when a hard failure is desired.

## Install

```bash
python -m pip install .
```

For development:

```bash
python -m pip install -e '.[dev]'
pytest
```

## CLI

```bash
omeify input.qptiff output.ome.tif \
  --type qptiff_mif \
  --compression LZW \
  --tile-size 1024 \
  --downsample mean \
  --cache-directory /fast/scratch
```

Useful options:

```text
--pyramid-levels N       Explicit subresolution count; auto by default
--rename-channels-json   JSON map from source names to output names
--omit-uuid              Omit the optional OME root UUID
--workers N              TIFF compression workers
--no-checksums           Skip the final whole-file checksum pass
--strict-miti            Fail on the current MITI header-profile check
```

The existing underscore spellings such as `--rename_channels_json` remain accepted as aliases.

The conversion report includes separate `ome`, `miti_header`, and `verification` sections. This keeps schema validity, MITI profile conformance, and binary-image checks from being blended into one suspiciously cheerful boolean.

## Python API

```python
from omeify.inputs import AkoyaMIFQptiff

converter = AkoyaMIFQptiff("input.qptiff", series=0)
converter.rename_channels = {"FITC": "PanCK"}
converter.cache_directory = "/fast/scratch"

report = converter.convert(
    "output.ome.tif",
    compression="LZW",
    tile_size=1024,
    downsample="mean",
    strict_miti=False,
)
```

## I/O strategy

The full-resolution source is never materialized as one giant NumPy array. `omeify` decodes only the source strips or tiles needed for each output tile. Newly generated lower-resolution levels are staged as temporary uncompressed tiled BigTIFF files, one level at a time. Peak RAM is therefore governed primarily by a few output tiles plus the largest source strip or tile, while temporary disk use is approximately one third of the uncompressed base image for a complete 2× pyramid. The final file is first written and verified beside the requested destination and then atomically moved into place, so a cache directory on another filesystem does not break finalization.
