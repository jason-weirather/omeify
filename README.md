# omeify

`omeify` converts supported tissue-image formats into standardized, metadata-deidentified, tiled,
pyramidal OME-TIFF images. Metadata deidentification follows MITI (Minimum Information about Highly Multiplexed Tissue Imaging)
guidelines (Schapiro et. al. Nat Methods. 2022). 

> ⚠ ️ **Deidentification note:** rebuilding a minimal header avoids carrying arbitrary source metadata into the output. It does not inspect pixels for burned-in labels or other identifying content.


## Design goals

- Produce a predictable OME-TIFF representation from other image format inputs
- Preserve native pixel dtype and full-resolution image quality
- Minimize carried-forward metadata while retaining image geometry, physical scale, and color
- Rebuild pyramid levels from full resolution rather than trusting source pyramids
- Keep peak memory bounded by processing strips and tiles instead of materializing a whole slide
- Fail on unsupported layouts or dtypes rather than silently coercing data
- Inspect the hierarchy and metadata of any TIFF without decoding the image raster
- Provide context-managed Python readers for multichannel and label OME-TIFF images
- Provide one standards-enforcing OME-TIFF writer used by both the Python API and `convert`
- Keep image analysis, segmentation, patch generation, stitching, and region measurements outside omeify

## Supported inputs

### Planar multiplex images

- Akoya mIF QPTIFF
- HALO planar mIF TIFF / OME-TIFF
- Akoya component TIFF
- Native `uint8`, `uint16`, `uint32`, `int8`, `int16`, `int32`, `float32`, and
  `float64` pixels

The planar profile writes `CYX`: one grayscale top-level IFD per logical channel,
`SamplesPerPixel=1`, and `Interleaved=false`.

### Brightfield RGB images

- Akoya H&E QPTIFF from PhenoImager HT or PhenoCycler Fusion
- Aperio SVS
- Interleaved `uint8` RGB with source axes `YXS`, `SamplesPerPixel=3`, and contiguous samples

The RGB profile writes one top-level interleaved RGB IFD, not three grayscale pages. Its OME
model uses `SizeC=3`, one `Channel` with `SamplesPerPixel=3`, `Interleaved=true`, and one
physical TIFF plane in `TiffData`.

Akoya physical pixel size is read from `PixelSizeMicrons` in the QPI description. Aperio pixel
size is read from `MPP`, with standard TIFF resolution tags used as a fallback. A source ICC
profile is preserved when present. Arbitrary vendor descriptions, filenames, user names,
scanner identifiers, dates, and other free text are not copied into the output OME-XML.

Unsupported dtypes and image layouts fail before conversion. `omeify` does not automatically
cast or rescale unsupported pixel data.

## Compression policy

Planar mIF and component inputs default to lossless LZW. Brightfield `qptiff_he` and `svs`
inputs default to JPEG because whole-slide RGB pathology images become extraordinarily large
under lossless compression.

The omeify brightfield default is:

- JPEG quality `90`
- 4:4:4 chroma sampling, represented to tifffile as subsampling `(1, 1)`

This is a conservative project default, not a universal pathology standard. It avoids chroma
subsampling while still providing substantial compression. Configure it with
`--jpeg-quality` and `--jpeg-subsampling`. Available sampling choices are `444`, `422`, `420`,
and `411`. Tile dimensions must satisfy the JPEG sampling alignment; omeify reports a clear
error when they do not.

Lossless LZW, Deflate, ZSTD, and uncompressed output remain available for RGB images. JPEG is
restricted to `uint8` input.

## Output conventions

All profiles emit one little-endian BigTIFF series. The generated OME-XML therefore declares
`BigEndian="false"`. Converting a big-endian source changes byte encoding, not its numeric dtype
or values.

`SignificantBits` is the full storage width of the output pixel type: 8 for `uint8`, 16 for
`uint16`, 32 for `float32`, and so on.

Every physical full-resolution plane is a top-level IFD. Rebuilt lower resolutions are tiled
SubIFDs marked as reduced-resolution images. For planar mIF, the physical plane count equals
the logical channel count. For interleaved RGB, three color samples occupy one physical plane.
Pyramid SubIFDs are not additional OME planes.

The source pyramid is ignored for all profiles. Each reduced level is rebuilt from the preceding
level using deterministic local 2× mean by default, or nearest-neighbor when requested. For RGB,
the operation is applied independently to all three samples while preserving the `YXS` layout.

The OME root UUID is generated and retained by default. It can be omitted with `--omit-uuid`.

## MITI-aligned metadata minimization

`omeify` constructs a new OME-XML header instead of copying arbitrary source metadata. The goal
is to retain the minimum information needed to unambiguously interpret the pixels while avoiding
unnecessary vendor, acquisition, or potentially identifying metadata.

The guiding resource for MITI (Minimum Information about highly multiplexed Tissue Imaging) is:

> **Schapiro D, Yapp C, Sokolov A, et al.** MITI minimum information guidelines for highly multiplexed tissue images. *Nat Methods.* 2022;19:262–267. doi:10.1038/s41592-022-01415-4.

The normalized header record is validated against the bundled JSON Schema:

```text
omeify/schemas/miti_ome_tiff_header.schema.json
```

This schema implements the intent of the MITI OME-TIFF header minimums for the image
representations written by `omeify`. It is intentionally stronger than a literal transcription
of the MITI table: fields such as `Pixels ID`, `Interleaved`, `SignificantBits`, channel IDs, and
TIFF IFD mapping are also checked because they make the generated OME-TIFF mechanically
interpretable.

The published MITI header definition does not enumerate every valid OME scalar pixel type.
`omeify` accepts the scalar numeric types its writers can preserve and never changes scientific
data merely to satisfy an incomplete enumeration.

### Header fields

| Field | Requirement | Why it is retained |
|---|---|---|
| `Image ID` | Required | Provides the internal OME identifier for the image object. |
| `Pixels ID` | Required | Provides the internal OME identifier for the pixel object. |
| `BigEndian` | Required | Declares the byte order of the written TIFF and must match the file. |
| `DimensionOrder` | Required | Defines how Z, C, and T map onto the TIFF plane sequence. |
| `Interleaved` | Required by the omeify profile | `false` for planar mIF and `true` for interleaved RGB. |
| `PhysicalSizeX`, `PhysicalSizeY`, and units | Required | Converts pixel coordinates into physical distance. |
| `PhysicalSizeZ` and unit | Required when `SizeZ > 1` | Provides physical calibration for a Z stack. Current profiles use `SizeZ=1`. |
| `SizeX`, `SizeY` | Required | Defines the full-resolution raster dimensions. |
| `SizeC`, `SizeZ`, `SizeT` | Required | Defines the logical dimensional shape. RGB has `SizeC=3`. |
| `Type` | Required | Defines the numeric representation of each sample. |
| `SignificantBits` | Required by the omeify profile | Records and validates the output sample storage width. |
| Channel `ID` | Required by the omeify profile | Gives every logical OME channel element a unique identifier. |
| Channel `Name` | Required | Retains a human-readable channel identity. RGB uses the neutral name `RGB`. |
| Channel `SamplesPerPixel` | Required by the omeify profile | `1` for planar channels and `3` for the single interleaved RGB channel. |
| `TiffData IFD` | Required by the omeify profile | Maps the OME plane model to top-level TIFF IFDs beginning at IFD 0. |
| `TiffData PlaneCount` | Required | Counts physical full-resolution planes, which is `1` for one RGB image. |

These fields reconstruct what is stored in the file: raster dimensions, sample type and byte
encoding, physical scale, channel/sample organization, dimension ordering, and TIFF IFD mapping.
Experimental context such as biospecimen identifiers, antibody clone and lot, staining protocol,
instrument configuration, acquisition date, and analysis history remains valuable but belongs in
associated experimental records.

The MITI header table also recommends a free-text `Comment`. `omeify` does not synthesize one or
copy arbitrary source comments. Conversion provenance is captured by the generated `Creator`
value and the conversion report.

### Additional OME metadata written by omeify

`Image/@Name` is intentionally omitted. It is not needed to interpret the raster and would create
another place for unnecessary source identity to enter the standardized file.

| Metadata | Policy | Purpose |
|---|---|---|
| Root `UUID` | Generated by default; optional with `--omit-uuid` | Gives the OME document a new globally unique identifier. |
| Root `Creator` | Generated | Records the `omeify` version that created the OME-XML. |
| Empty `LightPath` | Generated for each OME channel | Preserves schema structure without inventing acquisition metadata. |
| Pyramid `MapAnnotation` + `AnnotationRef` | Generated when a pyramid is present | Records rebuilt level dimensions and links them to the image. |
| TIFF ICC profile | Preserved for RGB when present | Retains the source color-space characterization without copying free-text metadata. |

This is metadata minimization, not metadata invention. MITI also defines biospecimen, reagent,
acquisition, instrument, processing, analysis, and other companion metadata. Those records remain
important to a complete MITI dataset but are outside the OME-TIFF header generated here.

## Validation and verification

Before an output replaces the destination, `omeify` checks:

- OME 2016-06 XML schema validity
- The bundled MITI-aligned header profile
- BigTIFF and TIFF/OME byte-order agreement
- Native dtype and `SignificantBits`
- Planar versus interleaved `Channel` and `SamplesPerPixel` organization
- `TiffData` physical-plane mapping
- Output axes, full-resolution shape, top-level IFD count, and photometric mode
- Pyramid dimensions, tiled storage, SubIFDs, and reduced-resolution flags
- The linked pyramid annotation
- ICC profile preservation when one was supplied by an RGB source
- Exact full-resolution spot checks for lossless output
- Successful decoding of representative pixels for lossy JPEG output

## Install

```bash
python -m pip install .
```

For development:

```bash
python -m pip install -e '.[dev]'
pytest
```

The authoritative package version is the static `version` field in `pyproject.toml`. Installed
code reads it from distribution metadata; direct source-tree imports fall back to the neighboring
`pyproject.toml`.

## CLI

`omeify` is a command group with three public subcommands:

```text
omeify convert   Convert a supported source into standardized OME-TIFF
omeify inspect   Summarize any TIFF as a text tree or schema-backed JSON
omeify version   Show the omeify version
```

### Convert

Conversion behavior and options are unchanged; they now live under the explicit `convert`
subcommand.

Planar Akoya mIF:

```bash
omeify convert input.qptiff output.ome.tif \
  --type qptiff_mif \
  --compression LZW \
  --tile-size 1024 \
  --downsample mean \
  --cache-directory /fast/scratch
```

Akoya H&E QPTIFF using the brightfield JPEG defaults:

```bash
omeify convert H32_HE.qptiff H32_HE.ome.tif \
  --type qptiff_he \
  --tile-size 1024
```

Aperio SVS with an explicit quality setting:

```bash
omeify convert CMU-1.svs CMU-1.ome.tif \
  --type svs \
  --jpeg-quality 92 \
  --jpeg-subsampling 444
```

Useful conversion options:

```text
--compression NAME        LZW, Deflate, ZSTD, JPEG, or Uncompressed
--jpeg-quality N          JPEG quality from 1 through 100; default 90
--jpeg-subsampling MODE   444, 422, 420, or 411; default 444
--pyramid-levels N        Explicit subresolution count; auto by default
--rename-channels-json    JSON map from source names to output names
--omit-uuid               Omit the optional OME root UUID
--workers N               TIFF compression workers
--no-checksums            Skip the final whole-file checksum pass
```

Only the selected TIFF series is converted. The default is series 0, which is the baseline
whole-slide series for the supported QPTIFF and SVS examples. Use `--series` only when inspection
shows that the desired full-resolution image is elsewhere.

The conversion report keeps separate `ome`, `miti_header`, and `verification` sections so XML
validity, header-profile validity, and binary-image verification remain distinct.

### Inspect

The default report is a compact tree of the file, OME header when present, series, and pyramid
levels. Inspection reads TIFF directories and metadata but does not materialize the image raster.
It works for any TIFF layout understood by `tifffile`, not only formats accepted by `convert`.

```bash
omeify inspect image.tif
omeify inspect image.tif --detail 2
omeify inspect image.tif --detail 3 --max-text-length 500
omeify inspect image.tif --json --output inspection.json
```

Detail levels are cumulative:

```text
0  file and series summaries
1  pyramid levels plus OME image, dimension, channel, and physical-size metadata
2  TIFF pages and lightweight frames
3  TIFF tags and parsed XML or plain-text ImageDescription values
```

The JSON representation is defined by:

```text
omeify/schemas/tiff_inspection.schema.json
```

For an OME-TIFF, `inspect` parses the OME header rather than guessing channel names, dimension
sizes, dimension order, or physical pixel sizes from TIFF pages alone. It also reports whether
each OME header satisfies the bundled omeify MITI header profile, lists missing or invalid fields,
and identifies additional OME metadata outside omeify's minimized output vocabulary. Additional
metadata is informational: MITI is a minimum-information profile, so extra fields do not by
themselves make a header invalid.

> ⚠️ **Inspection is not deidentification.** Text and JSON inspection reports may expose source
> paths, filenames, TIFF tags, vendor XML, scanner fields, or other identifying metadata. Review
> inspection output before sharing it.

### Version

```bash
omeify version
omeify version --json
```

The plain command prints only the omeify version. `--json` includes the Python, tifffile,
imagecodecs, NumPy, Click, XML, and schema-library versions used by the installation.

## Python API

Planar mIF:

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

Brightfield RGB:

```python
from omeify.inputs import AkoyaHEQptiff, AperioSVS

qptiff_report = AkoyaHEQptiff("H32_HE.qptiff").convert(
    "H32_HE.ome.tif",
    jpeg_quality=90,
    jpeg_subsampling="444",
)

svs_report = AperioSVS("CMU-1.svs").convert(
    "CMU-1.ome.tif",
    compression="JPEG",
)
```

### OME-TIFF reader and image interfaces

The generic interfaces `MultichannelImage`, `RGBImage`, and `LabelImage` define the initial
contracts for image readers. `OMETiffReader` is the first concrete multichannel reader and keeps
the underlying TIFF open for efficient metadata and regional access:

```python
from omeify import OMETiffReader

with OMETiffReader("output.ome.tif") as ome:
    print(ome)  # same default tree as: omeify inspect output.ome.tif
    print(ome.axes, ome.shape, ome.dtype)
    print(ome.channel_names)

    patch = ome.read_region(
        y0=10_000,
        y1=11_024,
        x0=20_000,
        x1=21_024,
        channels=[0, 3, 7],
    )
```

`read_region` supports the common planar `CYX`, grayscale `YX`, and interleaved `YXS` OME-TIFF
layouts without materializing the whole slide. `asarray(level=N)` remains available when loading
an entire pyramid level is intentional.

Logical OME channels are distinct from stored samples. A planar multiplex image normally has one
sample per logical channel. RGB has one logical channel named `RGB`, three samples per pixel, and
sample names `Red`, `Green`, and `Blue`.

A label raster uses the same virtual-access boundary without inventing biological semantics:

```python
from omeify import OMETiffLabelReader

with OMETiffLabelReader("cells.ome.tif") as labels:
    block = labels.read_region(10_000, 11_024, 20_000, 21_024)
```

There are no separate `CellLabelImage` or `TissueLabelImage` types. `LabelImage` validates integer
storage and provides image access only. It intentionally does not expose region properties or a
`label_count`: an exact generic label count requires scanning the raster, and `max(label)` is not
a reliable count when IDs are sparse. `scikit-image` is deliberately not an omeify dependency.

### OME-TIFF writer

`OMETiffWriter` writes arrays using the same tiled, pyramidal, metadata-minimized implementation
used by `omeify convert`:

```python
import numpy as np
from omeify import OMETiffWriter

image = np.zeros((3, 4096, 4096), dtype=np.uint16)
report = OMETiffWriter(
    "image.ome.tif",
    image_type="multichannel",
    channel_names=["DAPI", "PanCK", "CD3"],
    physical_size_x_um=0.5,
    physical_size_y_um=0.5,
).write(image)
```

Use `image_type="rgb"` for `YXS` `uint8` RGB data. RGB is written as one logical OME channel
with `SamplesPerPixel=3`. Use `image_type="label"` for one integer `YX` label raster; label
pyramids use nearest-neighbor downsampling and lossless compression. The image-type argument is
writer policy, not a private TIFF tag. OME-TIFF itself does not intrinsically distinguish label
values from intensity values.

Advanced streaming sources can implement the small `PlaneReaderSource` contract and call
`write_source`. The converter uses that path directly, so there is no second private writer
quietly drifting away from the public API.

## Architectural boundary

`omeify` owns the TIFF/OME-TIFF file boundary: inspection, normalization, metadata validation,
virtual region access, and standardized writing. It does not own tile scheduling, overlapping
patch generation, stitching, segmentation, object reconciliation, or morphological analysis.
Those operations belong in downstream computation packages such as OcelliKit.

## I/O strategy

The full-resolution source is never materialized as one giant NumPy array. `omeify` decodes only
the source strips or tiles required for the current output tile. Rebuilt lower-resolution levels
are staged as temporary uncompressed tiled BigTIFF files, one level at a time. Peak RAM is
therefore governed primarily by a few image tiles plus the largest decoded source segment.
Temporary disk use is approximately one third of the uncompressed base image for a complete 2×
pyramid.

The final file is written and verified beside the requested destination and atomically moved into
place. A failed write or verification does not destroy an existing valid output.
