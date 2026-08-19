# omeify

`omeify` converts supported multiplexed tissue images into standardized, deidentified, tiled, pyramidal OME-TIFF. Version 0.4 replaces the Java `bioformats2raw` → Zarr → `raw2ometiff` chain with a pure-Python `tifffile` implementation.

## What this iteration supports

- Akoya mIF QPTIFF
- HALO planar mIF TIFF / OME-TIFF
- Akoya component TIFF
- Native `uint8`, `uint16`, `uint32`, `int8`, `int16`, `int32`, `float32`, and `float64` pixels
- Exact dtype preservation with no automatic casting or intensity rescaling
- Fresh pyramid construction from the full-resolution source pixels
- Lossless LZW, Deflate, ZSTD, or uncompressed output; JPEG remains available for `uint8` only
- Click-based CLI, `pyproject.toml` packaging, and no Java runtime

Unsupported dtypes fail before conversion. The H&E class remains importable but intentionally raises `NotImplementedError`; the H&E iteration should write one interleaved RGB image with `SamplesPerPixel=3`, not three grayscale pages.

## Output conventions

The current mIF writer deliberately emits one little-endian BigTIFF series. The TIFF byte order is `<`, and the generated OME-XML therefore declares `BigEndian="false"`. Converting a big-endian source to this output representation changes byte encoding, not the numeric dtype or pixel values.

`SignificantBits` is the full storage width of the output pixel type: 8 for `uint8`, 16 for `uint16`, 32 for `float32`, and so on. The OME root UUID remains enabled by default and may be omitted explicitly with `--omit-uuid`.

Each full-resolution channel is a top-level IFD. Rebuilt lower resolutions are tiled SubIFDs marked as reduced-resolution images. `TiffData IFD="0"` maps only the full-resolution channel planes; pyramid SubIFDs are not counted as additional OME planes.

## MITI header scope

`omeify` validates the OME-TIFF header fields relevant to the MITI image-header minimums. The normalized validation record is checked against the bundled JSON Schema at:

```text
omeify/schemas/miti_ome_tiff_header.schema.json
```

The published MITI header YAML lists only `uint16` and `float` in its pixel-type enumeration. `omeify` treats the absence of valid OME types such as `uint8` as an incomplete enumeration, accepts the scalar OME types listed above, and never changes scientific data to satisfy that list. Genuine header-validation failures stop conversion; there is no separate strict mode.

MITI also defines companion biospecimen, reagent, acquisition, channel, instrument, processing, and analysis metadata. This package is concerned with the OME-TIFF image header and file structure, not with replacing those companion records.

## Install

```bash
python -m pip install .
```

For development:

```bash
python -m pip install -e '.[dev]'
pytest
```

The authoritative package version is the static `version` field in `pyproject.toml`. Installed code reads that value from distribution metadata through `importlib.metadata`; direct source-tree imports fall back to the neighboring `pyproject.toml`.

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
```

The existing underscore spellings such as `--rename_channels_json` remain accepted as aliases.

The conversion report includes separate `ome`, `miti_header`, and `verification` sections. Structural verification checks BigTIFF, TIFF/OME byte-order agreement, dtype, SignificantBits, TiffData mapping, pyramid geometry, tiled storage, SubIFDs, reduced-resolution flags, and the linked pyramid annotation. Lossless output also spot-checks base pixels against the source in every channel.

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
)
```

## I/O strategy

The full-resolution source is never materialized as one giant NumPy array. `omeify` decodes only the source strips or tiles required for the current output tile. Rebuilt lower-resolution levels are staged as temporary uncompressed tiled BigTIFF files, one level at a time. Peak RAM is therefore governed primarily by a few image tiles plus the largest decoded source segment, while temporary disk use is approximately one third of the uncompressed base image for a complete 2× pyramid.

The final file is written and verified beside the requested destination and then atomically moved into place. A failed write or verification does not destroy an existing valid output.
