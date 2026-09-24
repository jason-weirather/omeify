# omeify

**A predictable OME-TIFF home base for microscopy images.**

Omeify converts supported tissue-imaging formats into tiled, pyramidal OME-TIFFs
with explicit pixel interpretation, calibrated geometry, and deliberately
controlled metadata. It also provides a Python library for reading images and
writing new products under the same rules.

Microscopy TIFFs can arrive with different channel layouts, calibration
conventions, pyramid structures, and vendor descriptions. Omeify's purpose is to
resolve those differences at the file boundary, rather than make every downstream
application interpret them again. Its destination is a documented, narrower
profile of **OME-TIFF**, not a new private file format.

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

Omeify **builds on [tifffile]**, rather than replacing it. Tifffile already reads
and writes OME-TIFF, supports tiled and pyramidal data, exposes metadata, and
provides regional access. Those capabilities are the foundation, not something
Omeify claims to have invented.

Use **tifffile directly** for general TIFF manipulation or a straightforward array
write where you want to choose and maintain the file's structure and metadata.
Use **Omeify** when you want to delegate a recurring microscopy-image contract:
explicit source interpretation, a controlled output header, calibration checks,
appropriate pyramids, bounded writing, and verification before installation.

The value is not a shorter spelling of `imwrite()`. It is that a supported vendor
file, a NumPy image, and a calibrated external image provider can all be written
through the same policy, without each application constructing its own OME-XML or
TIFF-plane mapping. A valid OME-TIFF written by another tool need not follow
Omeify's narrower profile; conversion can normalize it when that profile is useful.

## The OME-TIFF home base

**For a successful write, these are the construction rules you can rely on.**
Image content and selected storage options still vary; the representation and the
rules for describing it do not depend on which supported input supplied it.

| Aspect | Omeify's output contract |
|---|---|
| Container | A self-contained, little-endian BigTIFF with OME 2016-06 XML. No external OME companion file is needed to locate the written pixels. |
| Image layout | Two-dimensional scalar `YX`, planar multiplex `CYX`, or interleaved `uint8` RGB `YXS`. Scalar channels are distinct from the three samples of one logical RGB channel. |
| Numeric fidelity | `convert` preserves the base image's numeric dtype and scale. Lossless encoding preserves base sample values; JPEG re-encodes them. Deliberate dtype or float-precision changes belong to `mutate` or an explicitly configured library operation. |
| Calibration | X and Y physical pixel sizes are required for writing and serialized in **µm**. OME sizes and TIFF resolution tags are checked against the intended calibration. An override must be supplied explicitly when source calibration is unavailable. |
| Pyramids | Tiled reduced resolutions are rebuilt as SubIFDs, not copied from a vendor pyramid. Mean reduction uses a defined 2x policy; explicitly typed label images require nearest-neighbor reduction and lossless storage. |
| Metadata | A new OME-XML header is generated from the permitted fields below. Arbitrary source descriptions are not copied wholesale. Extra product names or structured provenance enter only through their documented interfaces. |
| Installation | OME schema/profile validation and output readback checks precede atomic installation. A failed write or verification does not replace an existing destination with an incomplete TIFF. |

Pyramid construction is automatic until the image fits within one output tile,
unless an exact reduced-level count is requested. A small image or a request for
zero reduced levels can produce a base-only file. Source pyramids are not extra
channels. See the [input and output reference][reference] for physical plane
layout, supported dtypes, compression defaults, calibration precedence, and scratch
requirements.

These are **format and construction guarantees**, not certification of the
experiment. Raster verification is sampled, not an exhaustive full-slide
comparison. Correctly encoding a supplied pixel size cannot prove the microscope
was calibrated correctly. See [reproducibility and verification](#reproducibility-and-verification).

## Metadata minimization and MITI alignment

Omeify constructs a new OME-XML document instead of copying arbitrary source metadata. Its
minimum-information policy is guided by:

[Schapiro D, Yapp C, Sokolov A, et al. **MITI Minimum Information guidelines for highly
multiplexed tissue images.** *Nature Methods.* 2022;19:262-267.
doi:10.1038/s41592-022-01415-4.](https://doi.org/10.1038/s41592-022-01415-4)

The 2022 MITI publication describes a broader dataset-level metadata standard and specifies
OME-TIFF for raster image data. Omeify applies that minimum-information philosophy narrowly to the OME-TIFF file boundary:
retain the information required to interpret the raster, make every additional field deliberate,
and leave biospecimen, reagent, acquisition, instrument, processing, and analysis records to
companion metadata systems rather than synthesizing them inside the image header.

Only a small set of source information is eligible to cross that boundary automatically:

| Source information | Omeify policy | Why it is retained or rejected |
|---|---|---|
| Pixel dtype and image geometry | Preserved by `convert`; dtype changes occur only through explicit mutation | The numeric representation and dimensions are intrinsic to interpreting the raster, and conversion should not silently change them. |
| Channel names | Read from supported source metadata, normalized according to the selected profile, and optionally renamed explicitly | Logical channel identity is needed to interpret planar multiplex data. |
| Physical pixel size | Read from supported source metadata or standard TIFF calibration, or supplied as an explicit override | Spatial calibration is required by omeify's normalized output contract and is important for quantitative interpretation. |
| RGB ICC profile | Preserved when present on supported RGB input | Color interpretation can depend on the source color profile. |
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
| `Image/@Name` | Omitted for ordinary output; required and caller supplied for multi-series output | Avoids carrying arbitrary source image names while still giving multi-series files an explicit navigation identity. |
| Root `UUID` | Generated by default; optional | Gives the generated OME document a file-level identifier when desired. |
| Root `Creator` | Generated from the omeify version | Records the software that constructed the OME metadata rather than asserting source acquisition provenance. |
| Empty `LightPath` | Generated without inventing acquisition metadata | Satisfies the supported OME channel structure without fabricating instrument details. |
| Pyramid `MapAnnotation` | Generated and linked when reduced levels exist | Describes the reconstructed pyramid hierarchy to OME-aware consumers. |
| ICC profile | Preserved for RGB when supplied | Retains source color-management information when it is relevant to pixel interpretation. |
| Multi-series JSON provenance | Optional, caller supplied, canonicalized, and linked from every image | Allows downstream applications to attach explicit structured provenance without opening the door to arbitrary copied source metadata. |

A passing Omeify header assessment means this **image-header profile** passed,
not that an entire dataset satisfies MITI. Retain original acquisition records
and link companion specimen, reagent, processing, and analysis metadata through
your study's own data-management system. Minimal means sufficient and deliberate
at this boundary, not that the experiment's other metadata is unimportant.

**Metadata minimization is not deidentification.** Omeify does not inspect pixels
for burned-in identifiers. Permitted channel names, caller-supplied provenance,
and diagnostic reports can also contain sensitive values. Review them before
sharing; inspection deliberately reveals information that conversion may omit.

## Installation

Omeify requires **Python 3.10 or newer** for its ordinary image-I/O features.
These examples use the **0.18 library API**. From the corresponding repository
checkout:

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
python -m pytest -q
```

The required `ome-schema` distribution supplies a local OME XSD. Writing requires
that validation to run; a missing validator is not treated as a successful check.

## Command line

Start with a file, identify its source profile, and produce the shared
representation. The commands keep different responsibilities explicit:

| Command | Responsibility |
|---|---|
| `omeify inspect` | Examine TIFF layout and metadata without loading the whole raster. It is not restricted to conversion profiles and does not rewrite the file. |
| `omeify convert` | Normalize a supported image's layout and metadata, preserve numeric dtype and scale, rebuild pyramids, and verify the output. Lossy compression remains an explicit encoding policy. |
| `omeify crop` | Export pixel-coordinate or GeoJSON rectangles as named OME-TIFF products; group equal names as separate series. |
| `omeify mutate` | Use the same writer for an intentional float-to-integer or float32-precision change, with a report of the transformation. |
| `omeify version` | Report Omeify's version; `--json` also records the image-I/O dependency versions. |

`convert` and `mutate` require a destination file via **`--output` / `-o`**.
The input remains positional; a second positional output path is not accepted.
`--output-json` separately selects a report file. `inspect` still prints to
stdout by default and accepts an optional `--output` / `-o` for its report.

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

| Source | `--type` | Representation | Default compression |
|---|---|---|---|
| Akoya mIF QPTIFF | `qptiff_mif` | Planar multiplex | LZW |
| Akoya Fusion multiplex QPTIFF | `qptiff_fusion` | Planar multiplex | LZW |
| Indica Labs/HALO mIF TIFF | `indica_mif` | Planar multiplex | LZW |
| Akoya component TIFF | `component` | Scalar or planar multiplex | LZW |
| Scalar/planar OME-TIFF | `ome_tiff` | Scalar or planar multiplex | LZW |
| Interleaved RGB OME-TIFF | `ome_tiff` | Interleaved RGB | JPEG |
| Akoya H&E QPTIFF | `qptiff_he` | Interleaved RGB | JPEG |
| Aperio SVS | `svs` | Interleaved RGB | JPEG |

Only the selected series is converted; `--series 0` is the default. These are
explicit profiles, not a promise to convert every TIFF or every vendor variant.
[Calibration rules and source-specific options][supported-inputs] describe what
must be present or supplied.

An existing OME-TIFF can be normalized too:

```bash
omeify convert source.ome.tif --output normalized.ome.tif \
  --type ome_tiff --no-overwrite
```

**RGB defaults are lossy.** With no storage overrides, RGB uses **JPEG quality 90,
4:2:2 YCbCr encoding, and 512 × 512 tiles**. This is the same policy for CLI
conversion, Python `convert()`, ordinary and temporary image writers, and RGB
series in multi-image output, including RGB OME-TIFF input. Scalar, multiplex,
and label images retain lossless LZW and 1024 × 1024 tiles. Defaults follow the
image's declared meaning, not its filename or simply having three channels.

`--compression` / `compression=` and `--tile-size` / `tile_size=` override those
defaults. Explicit JPEG `444` remains true RGB encoding; `422`, `420`, and `411`
use YCbCr with matching TIFF tags. To avoid a lossy encoding step, choose a
lossless option explicitly, for example:

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
omeify crop slide.ome.tiff -o Scratch/slide --bounds 10000 5000 12048 7048

# Explicit visual mode sends one bounded overview through Sheetbend.
omeify inspect slide.ome.tiff -i --geojson \
  -q "Return only the right tissue; name it right." > right.geojson
omeify crop slide.ome.tiff -o Scratch/slide --geojson right.geojson
```

Crop's `-o` is a filename prefix: the examples produce `slide-01.ome.tiff` or
`slide-right.ome.tiff`. Default naming uses object names when present; `--naming
index` always uses padded indices. Equal names share a file with one series per
ROI. Polygons become bounding rectangles, not masks. `--roi-size 2048 2048` on
visual inspection enforces exact full-resolution dimensions around the proposed
location. The model still estimates that location; it is not a tissue segmenter.
Ordinary inspection and metadata questions remain metadata-only.

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

**Opening is not conversion.** Reading a vendor file or a third-party OME-TIFF
exposes its supported image representation; it neither rewrites that file nor
certifies it against the writer's full output contract. Use `TiffInspector` or a
file reader's `inspect()` for source diagnostics. A successful inspection can
report missing fields or a calibration mismatch. The writer is stricter at the
point of export and requires usable calibration.

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

Omeify 0.18.2 makes JPEG color-space selection explicit in the shared writer,
including CLI conversion, ordinary/temporary library writes, and RGB series in
multi-image files. Existing files are not changed by upgrading. Re-export from
the original SVS or the original RGB array, then open the new output in QuPath.
Version 0.18.3 changed automatic RGB storage to JPEG 90 / 4:2:2, including
library writes and RGB OME-TIFF conversions that previously defaulted to LZW.
Version 0.18.4 changes the automatic RGB tile size from 256 to 512 pixels.
Choose lossless compression explicitly for reference rasters
or exact-value workflows. See the [JPEG policy and regression tests][jpeg-policy]
for details.

QuPath, Bio-Formats, and other tools do not have to reproduce Omeify's Python
classes to use the output. The [OME-TIFF specification][ome-tiff-spec] describes
the shared file format; the rules above describe Omeify's narrower use of it.

## Reproducibility and verification

**Predictable construction, explicit transformations, and recorded checks** are
the reproducibility promise, not identical file bytes on every run.

The writer validates generated XML against the local OME 2016-06 XSD and the
bundled header profile. It then checks the temporary TIFF's layout, dtype,
channel/plane mapping, pyramid structure, calibration, and relevant ICC/provenance
fields before installing it. Both public writers share this machinery. A missing
schema validator or failed required check prevents successful installation.

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

Writing is bounded but not free of temporary storage. Reduced levels are staged
on disk; their aggregate pixel payload approaches one third of the uncompressed
base for a large two-dimensional 2x pyramid, with file/tile overhead in addition.
Multi-series products require scratch for all their pyramids, and the final
encoded temporary file lives beside the destination. A no-overwrite install needs
same-filesystem hard-link support and fails rather than silently using an unsafe
fallback. See [I/O and failure behavior][io-behavior].

## Scope and further documentation

The current image model is two-dimensional `YX`, `CYX`, or RGB `YXS`, with
singleton Z/T. RGB writing requires three `uint8` samples. Not every readable
array dtype or TIFF layout is a writable product. Omeify does not perform
segmentation, registration, stain normalization, object measurement, or
experimental quality assurance.

The **0.18 library API is a breaking consolidation**; earlier writer and metadata
interfaces are not kept as forwarding aliases. Check the
[migration table][migration] when upgrading an existing application. This
README describes the current contract, not a 1.0 compatibility commitment.

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
