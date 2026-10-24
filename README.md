# omeify

**Predictable OME-TIFF and OME-Zarr image I/O for microscopy.**

Omeify is a command-line tool and Python library for **inspecting, converting,
cropping, concatenating, and deliberately modifying** microscopy images. It reads
supported vendor TIFFs, OME-TIFF, and OME-Zarr, and writes calibrated, pyramidal
**OME-TIFF** or **OME-Zarr** with consistent pixel interpretation, controlled
metadata, and verified construction. OME-TIFF is the default output.

Microscopy metadata can carry loose text in vendor descriptions, scanner fields,
file paths, and annotations, sometimes including patient or specimen identifiers.
Copying it uncritically can propagate that information into later derivatives.
Omeify **rebuilds output metadata from a documented, constrained set of fields**
instead of forwarding arbitrary source descriptions alongside the pixels.
**Controlling this propagation is an essential step in deidentification pipelines**,
but not deidentification on its own: retained channel and ROI names, deliberate
provenance, opaque ICC profiles, and image pixels still require independent review.

The goal is a dependable home base for images from different instruments and
workflows. Omeify produces documented profiles of open formats, not private
Omeify files: an OME-aware application such as **QuPath** can use a compatible
OME-TIFF without installing Omeify.

[Installation](#installation) · [Quick start](#quick-start) ·
[Commands](#command-line) ·
[Supported inputs](#supported-file-conversions) ·
[OME-TIFF contract](#the-ome-tiff-home-base) ·
[OME-Zarr contract](#the-ome-zarr-home-base) ·
[Metadata and MITI](#metadata-minimization-and-miti-alignment) ·
[Python library](#python-library) · [Reference][reference]

## Installation

Omeify requires **Python 3.11 or newer**. A normal installation includes both
OME-TIFF and OME-Zarr support, including reading Zarr over HTTP(S).
From this repository checkout:

```bash
python -m pip install .
omeify version
```

The package-index release may lag this checkout; use the revision matching the
API documented here. For scientific environments that must remain on Python
3.10, **0.21.0** is the compatibility checkpoint.

For development, install `python -m pip install -e '.[dev]'` and run
`python -m pytest -q`. [Development checks](docs/development.md) documents the
supported environments and independent interoperability tests. Ordinary image
I/O does not require an AI model or optional intelligence dependencies.

## Command line

Each command has a distinct job:

| Command | Purpose |
|---|---|
| `omeify inspect` | Examine TIFF or OME-Zarr structure, channels, calibration, and metadata without decoding image pixels. |
| `omeify convert` | Normalize a supported image into OME-TIFF or OME-Zarr, preserving dtype and intensity scale. |
| `omeify crop` | Export pixel-coordinate rectangles or GeoJSON regions, together or as separate files. |
| `omeify concat` | Combine selected, optionally renamed **OME-TIFF image series** in a chosen order. |
| `omeify mutate` | Intentionally change floating-point image values or float32 precision, with a report. |
| `omeify version` | Show the package version; `--json` also reports image-I/O dependency versions. |

`convert`, `crop`, and `mutate` can write either output format with
`--output-format ome_tiff|ome_zarr`. `concat` writes OME-TIFF only.

### Quick start

Inspect a slide, turn it into a standardized OME-TIFF, then inspect the output:

```bash
omeify inspect slide.svs
omeify convert slide.svs -o slide.ome.tif --type svs --no-overwrite
omeify inspect slide.ome.tif
```

To write OME-Zarr instead, choose the output format explicitly:

```bash
omeify convert slide.ome.tif -o slide.ome.zarr \
  --type ome_tiff --output-format ome_zarr
```

The selected input profile (`--type`) determines how pixels and calibration are
interpreted; a filename suffix alone is not enough. `convert` preserves the
source's numerical type and intensity scale, although lossy output compression
can change decoded RGB values. Existing OME-TIFF and supported OME-Zarr images
can also be converted into a fresh, normalized output.

### Supported file conversions

| Input | `--type` | Image representation |
|---|---|---|
| Akoya multiplex QPTIFF | `qptiff_mif` | Planar multiplex |
| Akoya Fusion multiplex QPTIFF | `qptiff_fusion` | Planar multiplex |
| Indica Labs/HALO mIF TIFF | `indica_mif` | Planar multiplex |
| Akoya H&E QPTIFF | `qptiff_he` | RGB |
| Aperio SVS | `svs` | RGB |
| Leica SCN brightfield/H&E | `scn_he` | RGB |
| OME-TIFF | `ome_tiff` | Scalar, multiplex, or RGB |
| OME-Zarr | `ome_zarr` | Selected scalar, multiplex, RGB, or label image |

Source support is intentionally specific, not a promise to convert every TIFF or
all variations of a vendor format. `--series` selects an image where supported;
OME-Zarr also provides `--group`, `--z-index`, and `--t-index` selectors for its
supported image and plane model. See [supported inputs and calibration][supported-inputs]
for the exact rules. Ordinary TIFF `inspect` can describe metadata from layouts
that `convert` does not support.

### Crop an image or a set of regions

```bash
# CLI bounds are X0 Y0 X1 Y1, in full-resolution pixels.
omeify crop slide.ome.tif -o one-region.ome.tif --bounds 10000 5000 12048 7048

# QuPath-style pixel-coordinate GeoJSON can describe several regions.
omeify crop slide.ome.tif -o regions.ome.tif --geojson regions.geojson
```

By default, every region becomes a separately named **image series in one
output**; `-o` names that exact output, not a prefix. Use `--shatter by_index` to
write one file per region, or `--shatter by_name` to group regions into files by
name. Polygons select their bounding rectangles: cropping does **not** mask
polygon interiors. OME-Zarr collections use the same ordered-region model.

With the optional intelligence integration, a vision-capable model can propose
GeoJSON rectangles from a bounded preview. This explicitly sends image pixels
to the selected model source and needs human review:

```bash
omeify inspect slide.ome.tif -i --geojson \
  -q "Find the right tissue and name it right." > right.geojson
omeify crop slide.ome.tif -o right.ome.tif --geojson right.geojson
```

[Cropping guide](docs/regions.md) covers geometry, naming, preview setup,
and output options.

### Concatenate OME-TIFF series

```bash
omeify concat \
  --series slide.cells.ome.tif 0 \
  --named-series "Nuclei" slide.cells.ome.tif 1 \
  --named-series "Appearance" slide.regions.ome.tif 0 \
  -o slide.combined.ome.tif --no-overwrite
```

`concat` combines **complete OME Images**, not channels or mosaics. The order of
`--series` and `--named-series` arguments is the output order; the indices are
zero-based and output names must be unique. Different dimensions, pixel types,
and physical resolutions are allowed, with each image kept as its own series.
Full-resolution decoded pixels are rewritten losslessly by default, including
RGB; pyramids are rebuilt. The operation does not register or align images.
`concat` currently writes OME-TIFF only.

### Make an intentional numeric change

```bash
# Input must contain supported floating-point pixels.
omeify mutate float-image.ome.tif -o compact.ome.tif \
  --type ome_tiff --dtype uint16 --range-mode auto \
  --no-overwrite --output-json compact.mutation.json
```

`mutate` also supports deliberate float32 mantissa trimming using
`--float32-mantissa-bits N`. Its integer `auto` mapping may rescale each channel
independently, so it is **not** a neutral conversion or cohort normalization.
Keep these numerical changes separate from ordinary `convert`.
[Mutation options][commands] documents the mappings and precision policy.

### Reports and command help

Writing commands use **`-o` / `--output`** for their destination and print a
short completion message by default. Use `-v` for progress, `-vv` for diagnostics,
or `-vvv` for the full report. `--output-json FILE` saves a full structured
report, while `--output-json -` writes JSON-only stdout. `inspect` prints its
report by default; `--json` and `-o` control its representation and destination.

Use `omeify COMMAND --help` for installed options and the [report guide](docs/reports.md)
for the report schemas. TIFF destinations can be replaced after verification by
default; `--no-overwrite` refuses replacement. **OME-Zarr output always requires
a new local directory**, regardless of overwrite flags. Reports can contain
source paths and identifying metadata, so review them before sharing.

## Why use omeify instead of tifffile directly?

Omeify uses [tifffile] for TIFF I/O. Use tifffile directly when you need to
control a TIFF's low-level structure and metadata yourself. Use Omeify when you
want a supported source to follow the same rules for pixel layout, calibration,
channel identity, metadata minimization, pyramids, and safe writing every time.
You need not recreate vendor-specific plane mappings or OME-XML generation in
each analysis script.

Its two output formats remain open standards. The **home base** is a clear,
narrower construction contract, rather than a new format or a requirement to
use an Omeify reader downstream.

## The OME-TIFF home base

**Every OME-TIFF written by Omeify follows these rules.** Pixel values,
compression, and content can differ, but the construction is predictable.

| Aspect | Omeify's output contract |
|---|---|
| Container | Self-contained, little-endian BigTIFF with OME 2016-06 XML. Explicit image origins use a linked, controlled geometry annotation. |
| Image layout | Grayscale (`YX`), multiplex channel stacks (`CYX`), or interleaved RGB (`YXS`). RGB is three samples within one logical channel, not three fluorescence channels. |
| Pixel types | Scalar images support 8-, 16-, and 32-bit signed or unsigned integers and 32- or 64-bit floats. RGB requires `uint8`. |
| Pixel values | `convert` preserves dtype and intensity scale. Lossless encoding retains full-resolution numeric values; JPEG is lossy. `mutate` makes requested numerical changes. |
| Calibration | X and Y physical pixel sizes are required, stored in **µm**, and checked against TIFF resolution tags. Supply them explicitly if missing. |
| Pyramids | Tiled SubIFDs are rebuilt, not copied from the source. Intensity images default to 2× mean reduction; declared label images use nearest-neighbor reduction. |
| Metadata | Fresh, controlled OME-XML instead of arbitrary vendor descriptions; crop outputs also retain region names and source-image coordinates. |
| Verification | Generated metadata, TIFF organization, encoding, and sampled pixels are checked before the temporary output replaces the destination. |

By default, reduced-resolution levels are built until the image fits in one
tile. **RGB OME-TIFF** defaults to JPEG quality 90, 4:2:2 chroma sampling, and
512 × 512 tiles. **Grayscale, multiplex, and label images** default to lossless
LZW and 1024 × 1024 tiles. These defaults apply to CLI and Python writers;
explicit settings take precedence. Choose a lossless codec such as
`--compression Deflate` when exact RGB values matter.

The contract governs construction, **not scientific accuracy**. Calibration can
be faithfully recorded without having been correctly measured at acquisition.
See [storage and compression][jpeg-policy] and [verification][verification].

## The OME-Zarr home base

Omeify's canonical OME-Zarr output is **NGFF 0.5 on Zarr v3**. It uses fresh,
allowlisted JSON metadata, calibrated spatial coordinates, 512-pixel spatial
chunks, and lossless Zstd level 3 by default. Pyramids are rebuilt through the
same image model as TIFF. Zarr output is lossless by default even for RGB,
unlike the TIFF JPEG default.

The reader supports selected **OME-Zarr 0.4, 0.5, and 0.6** images, including
local stores and HTTP(S) roots. S3 reads are available with the optional
`omeify[s3]` installation; neither HTTP(S) nor S3 writes are supported. A reader
can select a 2D image or an explicit Z/T plane, but this is not arbitrary scene
registration or a complete implementation of every NGFF spatial relationship.
For 0.6 images with additional spatial context, export requires deliberate
`--intrinsic-only` selection rather than silently dropping that context.

OME-Zarr output is staged, validated, and published as a **new local directory**;
an existing store is never replaced. The writer currently produces unsharded
chunks, although supported sharded input can be read. The
[OME-Zarr guide](docs/ome_zarr.md) has remote examples, coordinate rules,
compatibility boundaries, and the output-version rationale.

## Metadata minimization and MITI alignment

Microscopy files may contain names, paths, specimen identifiers, vendor XML,
and other acquisition metadata that do not belong in every derived image.
Omeify builds **new OME-XML or NGFF JSON** from a controlled set of fields instead
of copying those descriptions wholesale. This limits the paths through which
unreviewed free text can spread into downstream image products, while keeping
metadata needed to interpret the pixels.

Its OME-TIFF minimum-information policy draws on:

[Schapiro D, Yapp C, Sokolov A, et al. **MITI Minimum Information guidelines for highly
multiplexed tissue images.** *Nature Methods.* 2022;19:262-267.
doi:10.1038/s41592-022-01415-4.](https://doi.org/10.1038/s41592-022-01415-4)

MITI covers the broader imaging study, including biospecimen, reagent,
instrument, acquisition, processing, and analysis records. Omeify addresses a
**narrower image-header contract**: retain what is necessary to interpret the
pixels, and leave other experimental records alongside the image rather than
inventing them in the header. This is MITI-aligned construction, **not a claim of
study-wide MITI compliance**.

### What can cross from the source?

| Source information | Omeify policy | Why |
|---|---|---|
| Pixel dtype and image geometry | Preserved by `convert`; numerical changes require explicit mutation | Defines the numeric image being interpreted. |
| Channel names | Read from recognized source fields and optionally renamed | Retains logical channel identities without guessing biomarkers. |
| Physical pixel size | Taken from supported calibration declarations or an explicit override | Required for quantitative spatial interpretation. |
| RGB ICC profile | Preserved opaquely when present on supported RGB input | May affect color interpretation; internal text is not sanitized. |
| Vendor descriptions, filenames, scanner fields, dates, user names, unrelated free text | Not copied into generated OME-XML or NGFF JSON | Avoids an uncontrolled metadata passthrough. |
| Biospecimen, reagent, acquisition, processing, and analysis metadata | Not inferred or manufactured | Belongs in deliberate accompanying study records. |

### What does Omeify put in OME-TIFF?

The generated header is checked against
`omeify/schemas/miti_ome_tiff_header.schema.json`. The minimum includes:

| Header value | Omeify policy | Purpose |
|---|---|---|
| `Image ID`, `Pixels ID` | Generated | Unambiguous internal OME references without copying source IDs. |
| `BigEndian`, `DimensionOrder`, `Interleaved` | Required and storage-checked | Describes the pixel layout. |
| `PhysicalSizeX`, `PhysicalSizeY`, units | Required | Carries physical calibration. |
| `SizeX`, `SizeY`, `SizeC`, `SizeZ`, `SizeT` | Required | Declares image dimensions. |
| `Type`, `SignificantBits` | Required and sample-checked | Declares pixel representation and precision. |
| Channel `ID`, `Name`, `SamplesPerPixel` | Required | Identifies logical channels and RGB sample grouping. |
| `TiffData IFD`, `PlaneCount` | Required and checked | Maps logical OME planes to stored TIFF directories. |
| `Image/@Name` | Omitted normally; named in multi-series output | Provides intentional series identity without arbitrary source names. |
| Root `UUID` | Generated by default; optional | Supplies a file-level identifier. |
| Root `Creator` | Generated from the Omeify version | Identifies the software constructing the header. |
| Empty `LightPath` | Generated without acquisition claims | Meets the supported OME channel structure without inventing instrument details. |
| Pyramid `MapAnnotation` | Generated when reduced levels exist | Describes the rebuilt resolution hierarchy. |
| ICC profile | Preserved when supplied for RGB | Retains available color-management information. |
| Multi-image annotations | Added for explicit provenance or crop records | Describes intentional output context. |

Passing the header check establishes these minimum fields, not a privacy
certificate. In particular, **metadata minimization is not deidentification**:
retained channel/ROI names, caller provenance, ICC profiles, source coordinates,
and pixels (including burned-in labels) still need independent review before
sharing. `inspect` exposes original metadata; it does not minimize or rewrite
it. The [metadata retention and sharing policy](docs/reports.md#metadata-retention-and-sharing)
describes these boundaries for both output formats.

## Python library

The same **Image API** can open files, wrap arrays, or read from external
regional image providers. Images describe their channels, shape, dtype, pixel
size, and pyramid levels; readers return only requested regions rather than
implicitly materializing whole slides.

### Write an image, then read a region

This example creates a synthetic multiplex image, saves a calibrated OME-TIFF,
and reads a DAPI patch back. Use a new filename when rerunning because
`overwrite=False` refuses to replace the example output.

```python
from pathlib import Path

import numpy as np
from omeify import MultichannelImage, OMETiffReader, OMETiffWriter, PixelSize

path = Path("omeify-demo.ome.tif")
values = np.arange(2 * 256 * 384, dtype=np.uint32).reshape(2, 256, 384)
pixels = (values % 4096).astype(np.uint16)

with MultichannelImage.from_array(
    pixels,
    axes="CYX",
    channel_names=("DAPI", "CD3"),
    pixel_size=PixelSize(0.5, 0.5, "µm"),
) as image:
    report = OMETiffWriter(
        path, compression="Deflate", tile_size=128, overwrite=False,
    ).write(image)

with OMETiffReader(path) as image:
    print(image.axes, image.shape, image.channel_names, image.pixel_size)
    dapi_patch = image.get_by_name("DAPI").read_region(0, 128, 0, 128)

np.testing.assert_array_equal(dapi_patch, pixels[0, :128, :128])
```

Image metadata belongs to the **Image**; compression, tiles, and publication
settings belong to the **writer**. Constructors such as `.from_array()` return
unopened images, so use a context manager or explicit `open()` / `close()`.
Regional patches own their returned pixels; lazy channel and crop views still
need the original image open. `.asarray()` deliberately requests a whole image.

The same regional surface works with OME-Zarr:

```python
from omeify import OMEZarrReader

with OMEZarrReader("slide.ome.zarr") as image:
    print(image.channel_names, image.pixel_size, image.levels)
    patch = image.read_region(0, 256, 0, 256)
```

`RGBImage.from_array(...)` and `LabelImage.from_array(...)` represent explicit
RGB and categorical images; three scalar channels are never guessed to be RGB.
`MultichannelImage.from_channels(...)` assembles lazy channel views, and
`image.with_metadata(...)` supplies deliberate calibration or naming changes.
`OMEMultiSeriesWriter` and `OMEZarrCollectionWriter` write named heterogeneous
image collections; callers can implement `ImageSource` to generate only requested
rectangles without building an intermediate TIFF.

[Images and sources][image-sources] describes the shared API, ownership,
coordinates, and external providers. The [Python reference][python-api] and
[Zarr examples](docs/ome_zarr.md#python-access) cover the individual writers.

## Interoperability with QuPath and other applications

OME-TIFF output targets ordinary OME-aware consumers, including **QuPath**,
**Bio-Formats**, and **tifffile**. Omeify's explicit channel mapping, calibrated
tiled planes, and SubIFD pyramids are meant to make images usable outside the
Python library. The receiving application still needs to support the selected
dtype, compression, and image/series layout; no universal viewer certification
is implied. See [QuPath's TIFF guidance][qupath-formats] and the
[OME-TIFF specification][ome-tiff-spec].

**RGB is not an H&E declaration.** QuPath can recognize RGB pixels while
classifying the image as Fluorescence. For known H&E images, choose Brightfield
(H&E) when needed. Omeify does not invent stain identities or acquisition
metadata, and the receiving application's display choice does not alter stored
pixels. Use a lossless codec for exact RGB values; the [JPEG policy][jpeg-policy]
documents encoding choices and older-file considerations.

## Reproducibility and verification

Omeify promises **documented construction, explicit transformations, and
recorded checks**, not byte-identical output across runs or versions. TIFF
writers validate generated OME-XML against a local OME XSD and the minimum
header policy, then verify structure, calibration, encoding, and representative
pixels. Zarr writers validate metadata and arrays, check their planned dense
chunk inventory, and decode representative pixels before publication.

Pixel checks are **sampled**, including checks at reduced levels. They are not
an exhaustive pixel audit or proof of correct input interpretation, specimen
quality, or physical microscope calibration. Keep original acquisition data,
processing parameters, reports, and `omeify version --json` with the study.

Both backends work in bounded image regions instead of loading the whole slide,
but large TIFF pyramids need temporary scratch disk space. TIFF writes are
staged and checked before atomic replacement, and Zarr creates a new verified
store without replacing an existing directory. See [verification][verification],
[TIFF I/O and failure behavior][io-behavior], and
[Zarr safe writing](docs/ome_zarr.md#safe-writing-and-verification).

## Scope and further documentation

Omeify currently writes two-dimensional scalar (`YX`), multiplex (`CYX`), and
RGB (`YXS`) images; RGB requires three `uint8` samples. Zarr readers can select
supported Z/T planes but do not provide general multidimensional or registered
scene output. Omeify does **not** perform segmentation, registration, stain
normalization, object measurements, or scientific quality assessment.

Optional `inspect -i` can request advisory metadata answers using a configured
Sheetbend source and the `omeify[intelligence]` extra (Python 3.13+).
`inspect -i --visual` or `--geojson` explicitly transmits a bounded image
preview to the selected model source, while ordinary inspection and metadata
questions do not send pixels. The visual GeoJSON proposal can feed a subsequent
`crop`; it is not an automatic crop or a tissue-segmentation result. Read the
[intelligence reference][intelligence] and
[visual region guide](docs/regions.md#ask-for-visual-geojson) for setup and
privacy considerations.

For deeper details:

- [Command-line and Python reference][reference]: source profiles, mutations,
  storage options, and complete parameter descriptions.
- [OME-Zarr guide](docs/ome_zarr.md): versions, selected planes, remote reads,
  geometry, and compatibility limits.
- [Images and sources][image-sources]: channels, arrays, image providers, and
  API migration.
- [Reports and metadata retention](docs/reports.md): report schemas, output
  behavior, privacy boundaries, and report migrations.
- [Cropping and visual regions](docs/regions.md): coordinate conventions,
  GeoJSON, splitting, and optional visual localization.
- [Development and release checks](docs/development.md): test environments and
  packaged-wheel verification.

This README describes the current pre-1.0 contracts rather than a promise of
backward compatibility with every earlier release.

## License

Apache License 2.0. See [LICENSE][license].

[tifffile]: https://github.com/cgohlke/tifffile
[qupath-formats]: https://qupath.readthedocs.io/en/stable/docs/intro/formats.html#tiff
[ome-tiff-spec]: https://docs.openmicroscopy.org/ome-model/6.2.2/ome-tiff/specification.html
[reference]: https://github.com/jason-weirather/omeify/blob/main/docs/reference.md
[supported-inputs]: https://github.com/jason-weirather/omeify/blob/main/docs/reference.md#supported-inputs
[commands]: https://github.com/jason-weirather/omeify/blob/main/docs/reference.md#command-line-reference
[python-api]: https://github.com/jason-weirather/omeify/blob/main/docs/reference.md#python-api
[jpeg-policy]: https://github.com/jason-weirather/omeify/blob/main/docs/reference.md#compression-policy
[verification]: https://github.com/jason-weirather/omeify/blob/main/docs/reference.md#validation-and-verification
[io-behavior]: https://github.com/jason-weirather/omeify/blob/main/docs/reference.md#io-scratch-space-and-failure-behavior
[image-sources]: https://github.com/jason-weirather/omeify/blob/main/docs/image_sources.md
[intelligence]: https://github.com/jason-weirather/omeify/blob/main/docs/reference.md#optional-metadata-intelligence
[license]: https://github.com/jason-weirather/omeify/blob/main/LICENSE
