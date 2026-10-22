# omeify

**Predictable OME-TIFF and OME-Zarr home bases for microscopy images.**

Omeify converts supported tissue-imaging formats into calibrated, pyramidal
**OME-TIFF** files or **OME-Zarr 0.5 / Zarr v3** stores with explicit pixel
interpretation and deliberately controlled metadata. The same regional Image API
also reads OME-Zarr **0.4, 0.5 and 0.6**, locally or over HTTP(S).

The Zarr reader supports selected 2D images and explicit Z/T plane selection,
not arbitrary registered scenes. OME-TIFF remains the default output.
See the [OME-Zarr contract and compatibility boundaries](docs/ome_zarr.md).

Microscopy TIFFs can arrive with different channel layouts, calibration
conventions, pyramid structures, and vendor descriptions. Those descriptions
may also contain information that should remain private, including patient
identifiers, specimen labels, user names, and file paths. Omeify resolves
these differences at the image boundary and **minimizes metadata** by building
controlled OME-XML or NGFF JSON rather than copying arbitrary source descriptions.
The results are documented, narrower profiles of open formats, not new private
file formats.

Metadata minimization is an important step when preparing an image for
deidentification review, but **it does not deidentify or certify the image**.
Retained channel/ROI names, explicit provenance, opaque RGB ICC profiles, and
image pixels (including burned-in labels) still require independent review before
sharing. [Retention boundaries](docs/reports.md#metadata-retention-and-sharing)
distinguishes construction guarantees from privacy review.

The file is the common ground. Convert once and use an OME-aware application such
as **QuPath**, read the result with **tifffile**, or build on Omeify's own regional
image API. Consumers do not need Omeify installed to interpret the OME-TIFF.
[Interoperability](#interoperability-with-qupath-and-other-applications) depends on
the consumer supporting the selected image type and encoding.

[Output contract](#the-ome-tiff-home-base) ·
[Metadata and MITI tables](#metadata-minimization-and-miti-alignment) ·
[Installation](#installation) · [Command line](#command-line) ·
[Python library](#python-library) · [Detailed reference][reference]

## Why use omeify instead of tifffile directly?

Omeify uses [tifffile] under the hood. Tifffile already reads and writes
OME-TIFF, supports tiled images and pyramids, and provides access to metadata
and image regions.

Use **tifffile directly** when you want full control over a TIFF's structure
and metadata. Use **Omeify** when you want supported microscopy images to follow
consistent rules for channels, calibration, metadata minimization, pyramids,
and **safe writing**. Omeify verifies the output before replacing an existing file.
This reduces unwanted source metadata, but does not replace deidentification review.

The same rules apply to supported vendor files, NumPy arrays, and calibrated
image providers, so each application doesn't have to build its own OME-XML or
TIFF-plane mapping. The result remains a standard OME-TIFF, even when the input
was written by another tool.

## The OME-TIFF home base

**Every OME-TIFF that Omeify writes follows these rules.** Pixel types,
compression, and content can vary, but the file structure and metadata remain
predictable.

| Aspect | Omeify's output contract |
|---|---|
| Container | Self-contained, little-endian BigTIFF with OME 2016-06 XML. No separate OME metadata file is needed. Explicit image origins use a linked, controlled geometry annotation. |
| Image layout | Grayscale (`YX`), multiplex channel stacks (`CYX`), or interleaved RGB (`YXS`). RGB has three color samples per pixel, not three fluorescence channels. |
| Pixel types | Grayscale and multiplex images support 8-, 16-, and 32-bit signed or unsigned integers (including `uint32`), plus 32- and 64-bit floats. RGB uses `uint8` only. |
| Pixel values | `convert` preserves the original dtype and intensity scale. Lossless compression preserves full-resolution pixel values; JPEG is lossy. Use `mutate` for intentional changes to pixel type or precision. |
| Calibration | Each output requires X and Y pixel sizes, stored in **µm** and checked against TIFF resolution tags. Supply the pixel size explicitly if the source lacks it. |
| Pyramids | Reduced-resolution images are rebuilt as tiled SubIFDs rather than copied from the source. Intensity images use a defined 2× mean reduction by default; declared label images use lossless, nearest-neighbor reduction. |
| Metadata | Writes a new OME-XML header without copying arbitrary vendor descriptions. `crop` also records ROI names and source-image coordinates. |
| Safe writing | Writes and checks a temporary file before replacing the destination. OME metadata, TIFF structure, and representative pixels are verified; a failed write or check leaves any existing destination untouched. |

By default, Omeify builds reduced-resolution levels until the image fits in one
tile. You can request an exact number of levels, including zero; small images may
need none. Pyramid levels are not extra channels. See the [detailed reference][reference]
for supported pixel types, storage options, calibration, and temporary-disk needs.

These are **file-construction guarantees**, not a judgment of scientific quality.
Pixel checks sample representative locations rather than every pixel. Correctly
writing a pixel size cannot prove that the microscope was calibrated correctly.
See [reproducibility and verification](#reproducibility-and-verification).

## The OME-Zarr home base

OME-Zarr is included in the normal installation, alongside OME-TIFF. Both require
**Python 3.11 or newer**. No extra is needed for local or HTTP(S) image reads.
Ordinary image I/O does not require a viewer, segmentation framework, Dask, or an
AI client.

```bash
omeify convert input.ome.tif -o normalized.ome.zarr \
  --type ome_tiff --output-format ome_zarr --no-overwrite
omeify inspect normalized.ome.zarr
omeify mutate normalized.ome.zarr -o integers.ome.zarr \
  --type ome_zarr --output-format ome_zarr --dtype uint16 --range-mode preserve
```

The mutation example requires floating-point input. Conversion preserves the
source dtype and intensity scale; mutation is the deliberate numerical operation.

The canonical writer emits **0.5 metadata on Zarr v3**, fresh allowlisted JSON,
explicit physical coordinates in `micrometer` units, lossless Zstd level 3,
unsharded 512-pixel spatial chunks, and one stored component per chunk. Gzip and
uncompressed encoding are explicit alternatives. RGB components remain explicit;
three scalar channels are never guessed to be RGB. Ordinary scalar output needs
no private metadata. Supplemental declarations retain RGB/ICC, categorical-label,
or explicit float-precision meaning where NGFF's raster model does not encode it.

A Zarr destination must be a **new local directory**, even when a common workflow
permits TIFF replacement. Construction is staged, the dense chunk inventory and
sampled pixels are checked, and a no-replace rename publishes the completed store.
Remote writes, ZIP stores, output sharding, arbitrary scenes, and whole-experiment
metadata conversion are outside this profile. Input sharding is delegated to Zarr;
remote sharded input additionally needs server byte-range support.

[Library examples, coordinate conventions, version decision, tests, and limits](docs/ome_zarr.md)

## Metadata minimization and MITI alignment

Omeify constructs new OME-XML or NGFF JSON instead of copying arbitrary source
metadata. The OME-TIFF minimum-information policy is guided by:

[Schapiro D, Yapp C, Sokolov A, et al. **MITI Minimum Information guidelines for highly
multiplexed tissue images.** *Nature Methods.* 2022;19:262-267.
doi:10.1038/s41592-022-01415-4.](https://doi.org/10.1038/s41592-022-01415-4)

MITI covers metadata for the wider imaging study and specifies OME-TIFF for the
image data. Omeify applies its minimum-information approach to the file itself:
keep what is needed to interpret the pixels, make other fields deliberate, and
leave biospecimen, reagent, instrument, acquisition, processing, and analysis records
in the study's accompanying metadata rather than inventing them in the image header.

Only a small set of source information is eligible to cross that boundary automatically:

| Source information | Omeify policy | Why it is retained or rejected |
|---|---|---|
| Pixel dtype and image geometry | Preserved by `convert`; dtype changes occur only through explicit mutation | The numeric representation and dimensions are intrinsic to interpreting the raster, and conversion should not silently change them. |
| Channel names | Read from exact fields in supported source metadata, normalized according to the selected profile, and optionally renamed explicitly | Logical channel identity is needed to interpret planar multiplex data. |
| Physical pixel size | Read from supported source metadata or standard TIFF calibration, or supplied as an explicit override | Spatial calibration is required by omeify's normalized output contract and is important for quantitative interpretation. |
| RGB ICC profile | Preserved opaquely when present on supported RGB input | Color interpretation can depend on the profile. Its embedded text is not stripped or privacy-reviewed. |
| Vendor descriptions, filenames, scanner fields, dates, user names, and unrelated TIFF text | Not copied into generated OME-XML | Arbitrary free text is not required to interpret the normalized raster and defeats strict metadata control. |
| Biospecimen, reagent, instrument, acquisition, processing, and analysis metadata | Not inferred or invented | These are important MITI companion records, but omeify does not manufacture experimental facts that were not supplied through an explicit contract. |

The normalized header is validated against:

```text
omeify/schemas/miti_ome_tiff_header.schema.json
```

The schema implements omeify's MITI-aligned OME-TIFF header minimums plus structural requirements
needed to interpret the generated file mechanically.

| Header value | Omeify policy | Why it is present |
|---|---|---|
| `Image ID`, `Pixels ID` | Required generated identifiers | Provide unambiguous internal references for the OME object graph without copying source identifiers. |
| `BigEndian`, `DimensionOrder`, `Interleaved` | Required and checked against TIFF storage | Describe how logical pixels map onto the physical TIFF representation. |
| `PhysicalSizeX`, `PhysicalSizeY`, units | Required | Preserve spatial calibration. |
| `SizeX`, `SizeY`, `SizeC`, `SizeZ`, `SizeT` | Required | Define the dimensional extent of the OME image. |
| `Type`, `SignificantBits` | Required and checked against stored samples | Make the numeric sample representation explicit and verifiable. |
| Channel `ID`, `Name`, `SamplesPerPixel` | Required | Identify logical channels and distinguish planar data from interleaved RGB samples. |
| `TiffData IFD`, `PlaneCount` | Required and checked against physical top-level IFDs | Map OME planes onto the TIFF container unambiguously. |
| `Image/@Name` | Omitted for ordinary output; explicitly named for multi-series output | Avoids carrying arbitrary source image names while still giving multi-series files an explicit navigation identity. |
| Root `UUID` | Generated by default; optional | Gives the generated OME document a file-level identifier when desired. |
| Root `Creator` | Generated from the omeify version | Records the software that constructed the OME metadata rather than asserting source acquisition provenance. |
| Empty `LightPath` | Generated without inventing acquisition metadata | Satisfies the supported OME channel structure without fabricating instrument details. |
| Pyramid `MapAnnotation` | Generated and linked when reduced levels exist | Describes the reconstructed pyramid hierarchy to OME-aware consumers. |
| ICC profile | Preserved for RGB when supplied | Retains source color-management information when it is relevant to pixel interpretation. |
| Multi-image annotations | Added only when a Python application explicitly requests them, or when `crop` records its regions | Describe intentional output context without copying arbitrary scanner metadata. |

Passing the MITI header check means the **minimum fields** are present and valid;
extra metadata is reported separately and does not itself fail that minimum check.
It is not a minimization certificate or MITI compliance for the whole study. Keep the original
acquisition records and other experimental metadata with the study; that
information remains important even when it is not embedded in the image.

**Before sharing:** `convert` and `mutate` construct the selected minimized image
product. `crop` additionally stores ROI names and source-image coordinates in
either format ([crop privacy details](docs/regions.md#privacy-when-sharing-crops)).
`inspect` reports original metadata; it neither minimizes nor rewrites it, and
Zarr inspection is not a recursive audit of every object in a store. Review
retained names, explicit provenance, ICC contents, pixels, and separate reports.
`metadata_minimization=true` records the construction policy, not sharing clearance.

## Installation

Omeify 0.22 requires **Python 3.11 or newer**. A normal installation includes both
OME-TIFF and OME-Zarr, including HTTP(S) Zarr reads. The earlier **0.21.0** release
is the Python 3.10 compatibility checkpoint; keep legacy scientific environments
pinned there rather than upgrading their dependencies in place.

From this checkout:

```bash
python -m pip install .
omeify version
```

A package-index release may lag the repository; use the checkout matching the
API documented here. Ordinary reading, writing, conversion, and inspection do not
require a model endpoint or an intelligence extra.

For development:

```bash
python -m pip install -e '.[dev]'
python -m pip check
python -m pytest -q
```

[Development and release checks](docs/development.md) covers the Python 3.11
minimum-dependency environment, newer-Python independent interoperability tests,
and testing an installed wheel outside the checkout. No CI service is required.

The required `ome-schema` distribution supplies a local OME XSD. Writing requires
that validation to run; a missing validator is not treated as a successful check.

## Command line

Omeify separates inspecting, writing, and changing pixel values:

| Command | What it does |
|---|---|
| `omeify inspect` | Show TIFF or OME-Zarr structure and metadata without reading image pixels. |
| `omeify convert` | Write normalized OME-TIFF or OME-Zarr, preserving pixel type and scale and rebuilding pyramids. |
| `omeify crop` | Export rectangles into an OME-TIFF or ordered OME-Zarr image collection, optionally split with `--shatter`. |
| `omeify mutate` | Intentionally change floating-point pixels or float32 precision, to either output format, with a report. |
| `omeify version` | Show the version; `--json` also shows image-I/O dependency versions. |

For `convert` and `mutate`, set the output file with **`--output` / `-o`**;
a second positional filename is not accepted. Writing commands print a short
completion message by default. Add `-v` for progress, `-vv` for diagnostic logs,
or `-vvv` for the full report unless `--output-json` is set. Use
`--output-json FILE` to save that report, or `--output-json -` for JSON-only
stdout. `inspect` prints its results by default, or saves them with `-o`.
See [report options and migration](docs/reports.md) for details.

### Inspect, convert, and retain the report

This example uses an Akoya multiplex QPTIFF. Replace the paths with your own:

```bash
omeify inspect input.qptiff

omeify convert input.qptiff --output normalized.ome.tif \
  --type qptiff_mif \
  --compression LZW \
  --no-overwrite \
  --output-json normalized.conversion.json \
  -v

omeify inspect normalized.ome.tif
omeify version --json > omeify-versions.json
```

The lossless example keeps the full-resolution numeric values, not the original
file bytes or arbitrary vendor metadata. `--type` selects a supported source
profile; a `.tif` suffix alone is not enough. For large slides, use
`--cache-directory /fast/scratch` to choose where temporary pyramids are built.
`--no-overwrite` refuses an existing destination; replacement is otherwise enabled
by default. JSON reports may contain source paths and metadata.

### Supported file conversions

| Source | `--type` | Representation | Default TIFF compression |
|---|---|---|---|
| Akoya mIF QPTIFF | `qptiff_mif` | Planar multiplex | LZW |
| Akoya Fusion multiplex QPTIFF | `qptiff_fusion` | Planar multiplex | LZW |
| Indica Labs/HALO mIF TIFF | `indica_mif` | Planar multiplex | LZW |
| Scalar/planar OME-TIFF | `ome_tiff` | Scalar or planar multiplex | LZW |
| Interleaved RGB OME-TIFF | `ome_tiff` | Interleaved RGB | JPEG |
| Akoya H&E QPTIFF | `qptiff_he` | Interleaved RGB | JPEG |
| Aperio SVS | `svs` | Interleaved RGB | JPEG |
| Leica SCN brightfield/H&E | `scn_he` | Interleaved RGB | JPEG |

OME-Zarr input additionally uses `--type ome_zarr`; its supported 0.4/0.5/0.6
images use the same conversions. Zarr output always has its own lossless defaults.
The retired `component` profile and `AkoyaComponentTiffReader` are removed in 0.22
without aliases. Legacy inForm component ROI TIFFs need a separately reviewed
workflow; they are not silently routed through another vendor profile.

Only the selected series is converted; `--series 0` is the default. For Leica SCN,
`--series` selects the zero-based `<image>` entry in the SCN collection, not the
embedded slide label. The SCN XML maps each image pyramid to its TIFF IFDs. RGB inputs
must have actual RGB samples or a supported JPEG decoding path that yields RGB.
Raw YCbCr/CIELAB samples are rejected, including on advertised reduced levels,
rather than silently relabeled. These are
explicit profiles, not a promise to convert every TIFF or every vendor variant.
[Calibration rules and source-specific options][supported-inputs] describe what
must be present or supplied.

An existing OME-TIFF can be normalized too:

```bash
omeify convert source.ome.tif --output normalized.ome.tif \
  --type ome_tiff --no-overwrite
```

**RGB uses lossy JPEG by default:** quality 90, 4:2:2 YCbCr encoding,
and 512 × 512 tiles. Grayscale, multiplex, and label images default to lossless
LZW with 1024 × 1024 tiles. These defaults apply to both CLI and Python writers,
including RGB OME-TIFF input and multi-image output. Three fluorescence channels
do not count as RGB.

Explicit compression and tile-size settings override these defaults. For RGB
JPEG, `444` uses RGB encoding; `422`, `420`, and `411` use YCbCr. To preserve
exact RGB pixel values, choose lossless compression, for example:

```bash
omeify convert slide.svs --output slide.ome.tif \
  --type svs --compression Deflate --no-overwrite
```

For an intentional numeric change, use `mutate`, for example on a planar
floating-point image:

```bash
omeify mutate processed.ome.tif --output compact.ome.tif \
  --type ome_tiff --dtype uint16 --range-mode auto \
  --no-overwrite --output-json compact.mutation.json
```

Automatic integer mapping can rescale each channel independently. It is not a
neutral format conversion or a cohort-normalization method. Float32 mantissa
trimming is a separate explicit option. See the [command-line reference][commands]
for the mapping rules, precision controls, channel renaming, and every option;
`omeify COMMAND --help` provides the installed command's help.

### Crop regions, or ask for their coordinates

```bash
omeify crop slide.ome.tiff -o Scratch/regions.ome.tiff --bounds 10000 5000 12048 7048

# Explicit visual mode sends one bounded overview to the configured model.
omeify inspect slide.ome.tiff -i --geojson \
  -q "Return only the right tissue; name it right." > right.geojson
omeify crop slide.ome.tiff -o Scratch/regions.ome.tiff --geojson right.geojson
```

Crop's `-o` is now the **exact destination filename**: both examples write
`Scratch/regions.ome.tiff`. All ROIs become separate images/series in that file,
in GeoJSON input order, including when names repeat. Series names combine the
padded input index and annotation name, such as `01 - right`; unnamed regions
use `01 - ROI`. Annotation names do not choose filenames by default.

Add `--shatter by_index` for one file per ROI (`regions-01.ome.tiff`, etc.), or
`--shatter by_name` for one file per name (`regions-right.ome.tiff`, etc.). Equal
names then share a file as separate series; unnamed objects use their indices.
A supplied TIFF extension is preserved, and a bare shattered prefix gets
`.ome.tiff`. Without shatter, no extension or suffix is added. This **0.20
breaking change** replaces `--naming`; see the [migration note](docs/regions.md#crop-020-migration).

Polygons become bounding rectangles, not masks. `--roi-size 2048 2048` on visual
inspection enforces exact full-resolution dimensions around the proposed
location. For mIF previews, `--channel NAME_OR_INDEX COLOR` can be repeated to
build a false-color composite (for example `--channel DAPI blue --channel panCK
magenta`), and scalar auto-display now clips at an upper quantile of 0.999 by
default so a tiny number of bright bins do not wash out the overview. The model
still estimates the location; it is not a tissue segmenter. Ordinary inspection
and metadata questions remain metadata-only.

See [cropping and visual region questions](docs/regions.md) for coordinate
conventions, preview/channel controls, privacy, piping, library use, and limits.

## Python library

### Write an image, then read a region

This complete example needs no scanner file or sibling application. The arrays
are synthetic; the names and pixel size are supplied explicitly. Use a fresh
output path when rerunning because the example refuses to overwrite one.

```python
from pathlib import Path

import numpy as np
from omeify import MultichannelImage, OMETiffReader, OMETiffWriter, PixelSize

path = Path("Scratch/omeify-demo.ome.tif")
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
    print(image.axes, image.shape, image.dtype)
    print(image.channel_names, image.pixel_size)
    print(image.levels)
    dapi_patch = image.get_by_name("DAPI").read_region(0, 128, 0, 128)

# The returned patch owns its pixels and survives the reader's context.
np.testing.assert_array_equal(dapi_patch, pixels[0, :128, :128])
```

Image meaning belongs to the Image; storage settings belong to the writer.
Constructors, including `.from_array()`, return unopened images, so use a context
manager or explicit `open()`/`close()`. Writers borrow an open image and leave it
open. Lazy channels and views require their parent's original open session;
`.asarray()` explicitly requests a whole image or channel. Array factories borrow
storage by default: do not mutate it during use, or pass `copy=True` for a snapshot.

The written file is usable without Omeify. For the small example above, ordinary
tifffile can read the complete base image directly:

```python
import tifffile

with tifffile.TiffFile("Scratch/omeify-demo.ome.tif") as tiff:
    base_pixels = tiff.series[0].asarray()
    print(tiff.series[0].axes, base_pixels.shape)
```

That last example deliberately materializes a small raster; use regional reads
instead for whole slides.

### What is available when an image is open?

The same consumer interface describes files, existing arrays, and external
regional providers. Metadata access does not decode the full raster.

| Surface | What it tells the application |
|---|---|
| `image.axes`, `image.shape`, `image.dtype`, `image.image_type` | Canonical array layout, dimensions, numeric representation, and intensity/RGB/label meaning. |
| `image.channel_names`, `image.channels` | Ordered logical channels. Handles expose IDs and available source metadata separately. Name lookup rejects ambiguity; fallback names are not inferred biomarkers. |
| `image.pixel_size` | Usable X/Y calibration as a `PixelSize`, or `None` when unavailable. Readable does not necessarily mean ready to write. |
| `image.levels[n]` | That level's shape, axes, spatial shape, and known calibration/downsampling. Unknown third-party level calibration stays unknown. |
| `image.read_region(y0, y1, x0, x1, ...)` | Owned NumPy pixels for half-open bounds in the selected level, without implicit clipping, padding, or rescaling. |

RGB has one logical channel containing three samples; it is not three scalar
fluorescence channels. A singleton channel stack can reopen as `YX` rather than
`CYX`; its channel handle still exposes a scalar plane. Never infer pyramid
calibration from ratios of rounded dimensions.

**Opening a file does not convert it.** Readers leave the source untouched,
even if it doesn't meet Omeify's stricter writing rules. Use `TiffInspector`
or a reader's `inspect()` to find missing metadata or conflicting calibration.
Writing requires valid calibration and checks the new output.

### Other image origins and products

For an existing `uint8` RGB array such as Mocktome's H&E raster, no storage
arguments are needed for the RGB defaults:

```python
from omeify import OMETiffWriter, PixelSize, RGBImage

with RGBImage.from_array(rgb, pixel_size=PixelSize(0.5, 0.5, "µm")) as image:
    report = OMETiffWriter("he.ome.tif").write(image)
    # JPEG quality 90 / 4:2:2, 512 × 512 tiles, automatic 2x pyramid.
```

Supply the array's actual physical pixel size. `compression="Deflate"` selects
lossless RGB storage; explicit settings in an existing application are never
overridden by a new default.

Use `RGBImage.from_array(...)` for interleaved RGB and `LabelImage.from_array(...)`
for categorical integer labels. Reopen labels with `OMETiffLabelReader`: label
meaning is explicit application knowledge, not something inferred from an integer
dtype. Nonzero background labels and object semantics need companion application
metadata; they are not standardized by an ordinary OME raster header.

`MultichannelImage.from_channels(...)` selects or assembles lazy scalar channels;
`image.with_metadata(...)` supplies deliberate calibration or naming overrides
without changing pixels. `OMEMultiSeriesWriter` writes named heterogeneous images
through `OMEImageSeries(name, image)`. A `TemporaryOMETiffWriter` owns a temporary
path's lifetime, using the same single-image writer internally.

An external provider can implement `ImageSource` and serve only requested
rectangles. There is no need to serialize an intermediate TIFF just to use that
image. Export still applies the same supported-dtype, calibration, metadata, and
encoding requirements. The provider must describe a fixed observation: repeated
and overlapping reads must agree. It owns its pixel-generation algorithm, not
Omeify.

See [Images and sources][image-sources] for ownership, coordinates, arrays,
providers, examples, and the 0.18 migration guide. The [Python reference][python-api]
includes conversion, mutation, and multi-series recipes.

## Interoperability with QuPath and other applications

Omeify targets ordinary OME-TIFF consumers, not a private reader ecosystem.
[QuPath's format guide][qupath-formats] recommends well-supported open formats
such as OME-TIFF and explains why recognizable image pyramids matter. Omeify's
tiled base planes, SubIFDs, OME channel mapping, and physical calibration are
intended to support that kind of downstream use.

Open the resulting `.ome.tif` in QuPath and check the intended series, channel
names, dimensions, and pixel size. Select compression and pixel types supported
by the receiving application. Format conformance is not a blanket compatibility
certification for every consumer version, codec, or multi-series product.
A label raster is an image product, not a QuPath project or an export of its
annotation objects.

**RGB storage is not an H&E stain declaration.** QuPath's image-type estimate
is separate from its recognition of `uint8 (rgb)` pixels. A saved choice or an
incorrect thumbnail-based estimate can still select Fluorescence. Choose
Brightfield (H&E) for known H&E data; Omeify does not fabricate acquisition or
stain metadata for every RGB image. Changing image type cannot repair an
incorrect JPEG color conversion.

The 0.18 releases corrected RGB JPEG color handling and established today's
defaults: quality 90 with 4:2:2 sampling (0.18.3), then 512-pixel tiles (0.18.4).
Upgrading Omeify does not change existing files. Re-export older affected files
from their original source, and choose lossless compression when exact values
matter. See the [JPEG policy and regression tests][jpeg-policy].

QuPath, Bio-Formats, and other tools do not have to reproduce Omeify's Python
classes to use the output. The [OME-TIFF specification][ome-tiff-spec] describes
the shared file format; the rules above describe Omeify's narrower use of it.

## Reproducibility and verification

**Predictable construction, explicit transformations, and recorded checks** are
the reproducibility promise, not identical file bytes on every run.

Before publication, TIFF writers check generated OME-XML against the local OME
XSD and minimum-header profile, then verify temporary TIFF structure, geometry,
encoding and representative pixels. Zarr writers check the generated image,
collection and manifest contracts, array structure, dense chunk inventory,
geometry and representative pixels. Each backend uses its own storage checks;
neither fabricates verification fields for the other format. A required missing
validator or failed check stops the write.

Pixel verification uses deterministic representative locations at the base and
every reduced level. Lossless comparisons check values under the documented
policy; JPEG samples are decoded, not claimed to match their original values.
Reports explicitly label this **sampled** verification. It is not an exhaustive
pixel audit, proof of correct source interpretation, or a tissue-quality check.
The [verification reference][verification] records the exact coverage and limits.

Keep the original acquisition data and study metadata, save conversion/mutation
reports, and retain the command or application parameters plus
`omeify version --json`. Pin the software environment for repeatable processing.
The header's generated UUID, version-bearing fields, and differences in encoders
can change file bytes; reports can also contain paths and run-specific timing.
Omeify does not promise byte-identical artifacts across runs or versions, and
omitting the UUID alone is not a byte-reproducibility switch.

Omeify writes in bounded regions rather than loading a whole slide, but it
needs temporary disk space. For TIFF output, a large 2× pyramid's reduced levels together
approach one third of the uncompressed base image size, plus file overhead.
Multi-image output needs space for every pyramid. The completed temporary TIFF
is created beside the destination. `--no-overwrite` needs hard-link support on
that filesystem; it fails rather than using an unsafe fallback.
Zarr stages the entire new directory beside its destination and publishes it
without replacement. See [I/O and failure behavior][io-behavior] and the
[Zarr publication contract](docs/ome_zarr.md#safe-writing-and-verification).

## Scope and further documentation

In 0.22, convert/mutate/crop accept `--output-format ome_tiff|ome_zarr`.
OME-Zarr input uses `--type ome_zarr`; inspection recognizes a local directory or
HTTP(S) root without that flag. `--group`, `--z-index`, and `--t-index` select a
specific Zarr image and plane. [0.22 report and geometry changes](docs/ome_zarr.md#migration)
are deliberate pre-1.0 changes, not compatibility aliases.


Omeify currently writes two-dimensional grayscale (`YX`), multiplex (`CYX`),
and RGB (`YXS`) images, with single Z and T positions. RGB requires three
`uint8` samples. Other readable TIFF layouts and pixel types may not be writable.
Omeify does not perform segmentation, registration, stain normalization,
object measurement, or experimental quality checks.

The current Image/writer API retains the 0.18 consolidation; earlier interfaces
are not kept as forwarding aliases. The [migration table][migration] covers those
older library changes. The [0.21 report migration](docs/reports.md#migration-from-020)
covers the earlier field/schema and CLI-output changes; the
[current inventory](docs/reports.md#current-external-contracts) includes both formats. This README describes
the current contract, not a 1.0 compatibility commitment.

[The detailed reference][reference] retains source-profile details, the complete
CLI walkthrough, Python recipes, writer architecture, and exact validation and
failure policies. [Images and sources][image-sources] explains the common library
contract and regional provider interface.

Optional `inspect -i` / `--question` metadata intelligence is advisory and separate
from image I/O. It requires the intelligence extra and a configured Sheetbend
source; ordinary conversion and mutation never use model output. Installation,
privacy scopes, evidence limits, and examples are in the
[metadata-intelligence reference][intelligence]. Adding `--geojson` explicitly
selects [visual region localization](docs/regions.md), which sends a bounded
image overview instead of a metadata packet. It never crops automatically.

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
[migration]: https://github.com/jason-weirather/omeify/blob/main/docs/image_sources.md#8-migration-from-017
[intelligence]: https://github.com/jason-weirather/omeify/blob/main/docs/reference.md#optional-metadata-intelligence
[license]: https://github.com/jason-weirather/omeify/blob/main/LICENSE
