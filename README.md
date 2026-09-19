# omeify

`omeify` is a Python toolkit for storage-independent microscopy image access and the OME-TIFF
file boundary. Its main goal is to make reading, writing, and normalizing microscopy images
predictable while keeping tight control over which metadata enters the resulting file.

In practice, omeify focuses on three jobs:

1. **Access images and write OME-TIFF deliberately.** A shared image/source contract supports
   real files, NumPy arrays, and externally implemented procedural sources. Bounded readers and
   single-image/heterogeneous multi-series writers provide tiled, pyramidal OME-TIFF output.
2. **Convert supported microscopy formats into OME-TIFF.** Conversion uses explicit source
   profiles rather than guessing from file extensions, normalizes layout, rebuilds pyramids, and
   verifies the written result before installation.
3. **Minimize and control OME metadata.** Omeify constructs a new OME-XML document from a small,
   explicit vocabulary instead of copying arbitrary vendor metadata into the output.

The normal `convert` workflow is fidelity-first: it preserves the source numeric dtype and does not
rescale intensities or explicitly cast pixel values. With lossless compression, pixel values remain
exact; brightfield profiles that use JPEG intentionally re-encode RGB pixels. The separate `mutate`
workflow exists for cases where changing dtype, numeric scale, or retained float precision is
scientifically intended and should be explicit rather than hidden inside conversion.

Omeify's metadata policy is guided by the minimum-information approach described in
[Schapiro *et al.*, **MITI Minimum Information guidelines for highly multiplexed tissue images**,
*Nature Methods* 19, 262-267 (2022)](https://doi.org/10.1038/s41592-022-01415-4). MITI is broader
than an OME-TIFF header: it also covers biospecimen, reagent, acquisition, instrument, processing,
and analysis metadata. Omeify applies the minimum-information principle specifically at the image
file boundary. It preserves or generates the metadata needed to interpret the OME-TIFF, and it does
not invent experimental facts or carry unrelated source text forward merely because it was present
in the input file.

> **Deidentification boundary:** metadata minimization avoids carrying vendor descriptions,
> filenames, user names, scanner identifiers, dates, and other arbitrary source text into the
> generated OME-XML. It does not inspect pixels for burned-in labels or other identifying content.
> Inspection reports are diagnostic and may expose source metadata.

## Interfaces

Omeify is intended to be useful both from the command line and as a Python library. The command
line is the fastest way to inspect and normalize files, so it is introduced first; the public Python
API exposes the same readers, writers, conversion workflows, and validation boundaries for
applications that need to build on omeify.

### Command line

The CLI uses explicit subcommands with deliberately different responsibilities:

| Command | Purpose | What makes it distinct |
|---|---|---|
| `omeify inspect` | Inspect TIFF structure and metadata | Format-agnostic TIFF diagnostics. It does not require an omeify conversion profile and can inspect any TIFF layout understood by tifffile without materializing the complete raster. |
| `omeify convert` | Normalize a supported source into OME-TIFF | Fidelity-first conversion. It preserves the source numeric dtype and scale, does not perform intensity rescaling or an explicit dtype cast, rebuilds pyramids, minimizes metadata, and verifies the output. |
| `omeify mutate` | Create an intentionally pixel-mutated OME-TIFF | Uses the same normalized writer path as `convert`, but explicitly changes either numeric dtype or float32 precision. Integer dtype mutation reports its per-channel mapping and anticipated quantization loss; float32 precision mutation preserves dtype and exponent range while reducing stored fraction precision. |
| `omeify version` | Report omeify and dependency versions | Provides a compact version string or a JSON description of the image-I/O software stack for reproducibility and diagnostics. |

Start by inspecting an unfamiliar TIFF:

```bash
omeify inspect input.tif
```

Convert a planar multiplex image without changing its dtype or intensity scale:

```bash
omeify convert input.qptiff output.ome.tif \
  --type qptiff_mif \
  --cache-directory /fast/scratch
```

Convert an H&E whole-slide image with the brightfield JPEG defaults:

```bash
omeify convert input.svs output.ome.tif \
  --type svs
```

Normalize an existing OME-TIFF into the omeify output contract:

```bash
omeify convert source.ome.tif normalized.ome.tif \
  --type ome_tiff
```

When changing the numeric representation is intentional, use `mutate` rather than hiding that
operation inside conversion. Integer dtype mutation remains available:

```bash
omeify mutate source.ome.tif compact.ome.tif \
  --type ome_tiff \
  --dtype uint16 \
  --range-mode auto \
  --output-json compact.mutation.json
```

For processed float32 data that should remain float32 while carrying less numerical precision:

```bash
omeify mutate source.ome.tif compact-float.ome.tif \
  --type ome_tiff \
  --float32-mantissa-bits 11 \
  --compression LZW
```

Retaining 11 stored float32 fraction bits provides 12 bits of binary significand precision while
preserving the float32 exponent range. This is a useful precision target for processed data derived
from nominal 12-bit acquisitions such as Akoya PhenoImager HT Extended Range output.

Use `omeify COMMAND --help` for the complete option set. Detailed command behavior is documented
in the [command-line reference](#command-line-reference).

### Python library

The same boundaries are public Python APIs. `convert()` and `mutate()` mirror the command-line
workflows; `TiffInspector` provides programmatic inspection; format-specific readers expose bounded
regional I/O; and `OMETiffWriter` / `OMEMultiSeriesWriter` provide the normalized writer contracts.
See the [Python API](#python-api) section for examples.

## Installation

Omeify requires Python 3.10 or newer.

From a repository checkout:

```bash
python -m pip install .
```

For optional metadata intelligence, use Python 3.13+ on Linux or macOS and install:

```bash
python -m pip install -e '.[intelligence]'
```

The `intelligence` extra adds `sheetbend[llm]>=0.5.0,<0.6`; it does not change the
Python 3.10+ baseline or dependencies of an ordinary installation. Requesting the
extra on an older Python is an installation error, not an empty-extra fallback.
When installing from a package index, the corresponding form is
`python -m pip install 'omeify[intelligence]'`. Sheetbend must be available from
your configured package index or already installed from its repository.

For development:

```bash
python -m pip install -e '.[dev]'
pytest
ruff check .
```

The package is licensed under Apache-2.0. The authoritative package version is the static
`[project].version` value in `pyproject.toml`. Installed code reads distribution metadata; direct
source-tree imports fall back to the neighboring `pyproject.toml`.

## Scope and capabilities

- Storage-independent `ImageSource` backends and semantic multichannel, RGB, and label images
- NumPy and OME-TIFF backends, context-managed borrowing/ownership, and immutable level descriptors
- Image-aware streaming single-image, temporary, and multi-series writers
- Explicit readers for supported vendor TIFFs and OME-TIFF
- Bounded regional access without materializing a whole slide
- Canonical planar multichannel, interleaved RGB, and label-image representations
- One ordinary writer for a single homogeneous OME Image
- One multi-series writer for named heterogeneous OME Images
- One shared internal engine for writer validation, pyramid construction, TIFF encoding,
  verification, scratch cleanup, and atomic installation
- Explicit pixel mutation: deterministic float-to-integer dtype conversion with a quantitative per-channel loss report, plus float32 mantissa-precision trimming
- TIFF and OME metadata inspection without decoding the complete raster
- OME 2016-06 schema validation and a bundled MITI-aligned header profile

Omeify owns generic image access and the image-file boundary. It does not own segmentation,
overlapping inference tiles,
object reconciliation, patch scheduling, region measurements, or image-analysis policy.

## Supported inputs

### Planar multiplex images

| Input profile | `--type` | Canonical output | Default compression |
|---|---|---|---|
| Akoya mIF QPTIFF | `qptiff_mif` | `CYX` planar | LZW |
| Akoya Fusion multiplex QPTIFF | `qptiff_fusion` | `CYX` planar | LZW |
| Indica Labs/HALO mIF TIFF | `indica_mif` | `CYX` planar | LZW |
| Planar OME-TIFF | `ome_tiff` | `CYX` or `YX` | LZW |
| Akoya component TIFF | `component` | `CYX` or `YX` | LZW |

Planar output uses one grayscale top-level IFD per physical plane,
`SamplesPerPixel=1`, and `Interleaved=false`. Supported scalar dtypes are `uint8`, `uint16`,
`uint32`, `int8`, `int16`, `int32`, `float32`, and `float64`.

### Brightfield RGB images

| Input profile | `--type` | Canonical output | Default compression |
|---|---|---|---|
| Akoya H&E QPTIFF | `qptiff_he` | `YXS` interleaved RGB | JPEG |
| Aperio SVS | `svs` | `YXS` interleaved RGB | JPEG |
| Interleaved RGB OME-TIFF | `ome_tiff` | `YXS` interleaved RGB | LZW |

RGB output is restricted to interleaved `uint8` data with three contiguous samples. The OME model
uses `SizeC=3`, one logical `Channel` with `SamplesPerPixel=3`, `Interleaved=true`, and one physical
TIFF plane.

### Physical pixel calibration

A generated OME-TIFF requires usable X and Y physical pixel sizes.

- Akoya QPTIFF readers use `PixelSizeMicrons` from the QPI description.
- Fusion multiplex calibration is checked across every full-resolution channel page.
- Indica mIF uses standard TIFF `XResolution`, `YResolution`, and `ResolutionUnit` tags.
- Aperio SVS uses `MPP` and warns before falling back to TIFF resolution tags.
- OME-TIFF uses OME physical-size metadata, then falls back to TIFF resolution tags only when
  both OME X and Y calibration values are absent.
- Component TIFF uses an explicit override when supplied, otherwise standard TIFF resolution tags.

For multi-page inputs, calibration present on one page may apply to pages where those tags are
absent. Any other page that explicitly provides usable calibration must agree. Partial OME
physical-size metadata is not combined with TIFF tags; supply an explicit override instead.

TIFF resolution values are interpreted as pixels per physical unit, converted to micrometers for
inch or centimeter source units, and rounded to six significant digits to avoid carrying rational
encoding noise into OME metadata. This reader normalization is separate from the
inspection consistency check, which uses the original rational tags without that rounding.

At the OME serialization boundary, omeify expresses every supported physical pixel size in
**µm** while preserving the physical quantity. For example, `500 nm`, `0.0005 mm`, and `0.5 µm`
all serialize as `PhysicalSizeX="0.5" PhysicalSizeXUnit="µm"`. Unit aliases such as `um` and
Greek-mu `μm` are normalized to the micro sign spelling `µm`. This is a metadata representation
policy, not a resampling or calibration change. TIFF `XResolution`/`YResolution` remain encoded
independently using the writer's existing centimeter density convention.

When no usable calibration remains, provide an explicit override:

```bash
omeify convert component.tif component.ome.tif \
  --type component \
  --pixel-size-x 0.5068 \
  --pixel-size-y 0.5068 \
  --pixel-size-unit µm
```

## Canonical OME-TIFF output

### Container and plane layout

Omeify writes little-endian BigTIFF. The generated OME-XML therefore declares
`BigEndian="false"`. Converting a big-endian source changes the byte encoding, not the numeric
sample type or value.

Every full-resolution physical plane is a top-level IFD. Reduced resolutions are tiled SubIFDs
marked as reduced-resolution images. A planar `CYX` image has one physical plane per channel. An
interleaved RGB image has one physical plane containing three samples. Pyramid SubIFDs are not
additional OME planes.

`SignificantBits` normally records the full storage width of the output dtype, such as 8 for
`uint8`, 16 for `uint16`, and 32 for ordinary `float32`. When float32 mantissa precision is
explicitly trimmed, `Type` remains `float` while `SignificantBits` records the IEEE-754 bits left
meaningful by that transform: one sign bit, eight exponent bits, and the requested retained
fraction bits. For example, retaining 11 float32 fraction bits writes `SignificantBits=20` while
providing 12 bits of binary significand precision. The distinction matters: this remains float32
storage with the float32 exponent range; it is not a 20-bit acquisition or a 20-bit integer image.

### Pyramid policy

Source pyramids are ignored. Every reduced level is rebuilt from the preceding level using one of
two explicit policies:

- `mean`: deterministic local 2x mean downsampling with nearest-even integer rounding
- `nearest`: nearest-neighbor sampling, required for label images

Automatic pyramid construction continues until the image fits within one output tile. Use
`--pyramid-levels` or the corresponding writer argument to request an exact number of reduced
levels. RGB downsampling is applied independently to all three samples while preserving `YXS`.

### Compression policy

Planar conversion profiles default to lossless LZW. Akoya H&E QPTIFF and Aperio SVS default to
JPEG because lossless whole-slide RGB files can be exceptionally large. Normalizing an existing
OME-TIFF defaults to LZW so the operation does not introduce a new lossy encoding step.

The brightfield JPEG defaults are:

- quality `90`
- 4:4:4 chroma sampling, represented to tifffile as `(1, 1)`

Available JPEG sampling choices are `444`, `422`, `420`, and `411`. The TIFF tile size must satisfy
the selected JPEG sampling alignment. Lossless LZW, Deflate, ZSTD, and uncompressed output remain
available for RGB.

The ordinary single-image writer permits JPEG only for `uint8` RGB. The multi-series writer also
permits JPEG for explicitly selected non-label `uint8` visualization series. Label series always
require lossless compression.

### UUID, Software tag, and ICC profile

The OME root UUID is generated by default and may be omitted with `--omit-uuid` or
`display_uuid=False`.

The TIFF `Software` tag defaults to `omeify <version>`. Applications using omeify as a Python
library may supply `software=` to identify the product generator. The command line intentionally
does not expose this override. Changing the TIFF Software tag does not change the OME root
`Creator`, which continues to identify the omeify version that generated the OME metadata.

A source ICC profile is retained for RGB output when present. Arbitrary vendor descriptions and
other free text are not copied.

## Command-line reference

Omeify uses explicit subcommands:

```text
omeify convert   Normalize a supported source into OME-TIFF
omeify mutate    Create an explicitly pixel-mutated OME-TIFF
omeify inspect   Inspect TIFF structure and metadata
omeify version   Show package and dependency versions
```

The former pre-subcommand form is not accepted or forwarded.

### `omeify convert`

Akoya mIF QPTIFF:

```bash
omeify convert input.qptiff output.ome.tif \
  --type qptiff_mif \
  --compression LZW \
  --tile-size 1024 \
  --downsample mean \
  --cache-directory /fast/scratch
```

Akoya Fusion multiplex QPTIFF, preferring `Biomarker` and falling back to `Name`:

```bash
omeify convert fusion.qptiff fusion.ome.tif \
  --type qptiff_fusion \
  --channel-name-field auto
```

Use `--channel-name-field name` or `--channel-name-field biomarker` to require that exact source
field. Both values remain available in the conversion report. Only the selected, optionally
renamed value becomes the minimized OME channel name.

Indica Labs/HALO mIF TIFF:

```bash
omeify convert halo-mif.tif halo-mif.ome.tif \
  --type indica_mif
```

The Indica profile reads channel names and full-resolution IFD mappings from its `<indica>`
ImageDescription. The source pyramid is not copied.

Aperio SVS with an explicit JPEG policy:

```bash
omeify convert slide.svs slide.ome.tif \
  --type svs \
  --jpeg-quality 92 \
  --jpeg-subsampling 444
```

Only the selected TIFF series is converted. Series 0 is the default. Use `--series` when
`omeify inspect` shows that the desired full-resolution image is elsewhere.

#### Channel renaming

Renaming is explicit and has one authoritative key mode.

Name mode interprets JSON keys as normalized source channel names:

```json
{
  "DAPI": "DNA",
  "CD3": "CD3e"
}
```

```bash
omeify convert input.ome.tif output.ome.tif \
  --type ome_tiff \
  --rename-channels-json renames.json \
  --rename-channels-by name
```

Index mode interprets JSON keys as zero-based integer strings:

```json
{
  "0": "DNA",
  "1": "PanCK",
  "2": "CD3"
}
```

```bash
omeify convert input.ome.tif output.ome.tif \
  --type ome_tiff \
  --rename-channels-json renames.json \
  --rename-channels-by index
```

The selected mode is authoritative. A source channel literally named `"0"` remains a name in name
mode. Malformed or mixed mappings fail rather than being guessed.

Common conversion options:

```text
--series N                 Source TIFF series; default 0
--compression NAME         LZW, Deflate, ZSTD, JPEG, or Uncompressed
--jpeg-quality N           JPEG quality from 1 through 100; default 90
--jpeg-subsampling MODE    444, 422, 420, or 411; default 444
--tile-size N              Square TIFF tile size; default 1024
--pyramid-levels N         Exact reduced-level count; automatic when omitted
--downsample METHOD        mean or nearest
--cache-directory PATH     Temporary pyramid scratch location
--workers N                Parallel TIFF compression workers
--omit-uuid                Omit the optional OME root UUID
--output-json PATH         Write the structured conversion report
-v                         Compact stages and progress
-vv                        Timestamped debug logging and tracebacks
```

### `omeify mutate`

`mutate` owns explicit pixel-value changes. It never edits the input in place, and exactly one
mutation must be selected:

- `--dtype uint8|uint16` performs the existing float-to-unsigned-integer mutation.
- `--float32-mantissa-bits N` keeps float32 storage but rounds the stored fraction to `N` bits.

Integer dtype mutation scans the source by bounded regions, selects one fixed mapping per
full-resolution channel, and reports the anticipated quantization loss:

```bash
omeify mutate halo-float.tif compact.ome.tif \
  --type indica_mif \
  --dtype uint16 \
  --range-mode auto \
  --output-json compact.mutation.json
```

Float32 precision mutation is intended for processed floating-point rasters whose computational
precision exceeds the meaningful precision of the measurement. The recommended starting point for
processed Akoya PhenoImager HT data is 11 retained fraction bits:

```bash
omeify mutate halo-float.ome.tif halo-float-compact.ome.tif \
  --type ome_tiff \
  --float32-mantissa-bits 11 \
  --compression LZW \
  --output-json halo-float-compact.mutation.json
```

`--float32-mantissa-bits 11` retains 12 bits of binary significand precision. The output remains
float32 and keeps the float32 exponent range, while lower fraction bits are rounded using nearest,
ties-to-even rounding. Base pixels are precision-trimmed before encoding, rebuilt pyramid levels
are rounded to the same precision, and OME `SignificantBits` is written as 20
(1 sign + 8 exponent + 11 retained fraction bits). This is lossy by design and belongs in `mutate`,
not fidelity-preserving `convert`.

For integer dtype mutation, the three range modes have distinct meanings:

- `auto` first evaluates unit-preserving nearest-integer rounding. It accepts that mapping when the
  rounded range fits and the source has strong integer-lattice evidence or the measured normalized
  RMSE stays within `--auto-max-normalized-rmse`. Otherwise it preserves zero and maps the exact
  observed maximum to the target maximum. No finite source value is clipped.
- `preserve` explicitly applies nearest-integer rounding without rescaling. It fails when the
  rounded range does not fit the requested dtype.
- `full` maps the exact observed minimum and maximum to the full target range. It can represent
  negative values, but it changes the numeric zero and must be selected deliberately.

The mapping is always:

```text
output = clip(rint((source - offset) / quantum), dtype range)
```

`rint` uses nearest-even rounding. The inverse approximation is
`source ≈ output * quantum + offset`.

The integer dtype mutation report records, per channel:

- exact minimum, maximum, zero count, negative count, and non-finite counts
- deterministic percentiles and integer-lattice diagnostics
- exact unit-rounding error and the automatic loss-budget decision
- selected offset and source-units-per-code quantum
- the reason for the selected mapping
- theoretical and sampled error, normalized RMSE, clipping count, and code usage

Automatic per-channel scaling preserves ordering within a channel but changes raw comparability
between independently scaled channels or slides. A cohort that requires common intensity units
should reuse fixed mappings instead of selecting a new automatic mapping per slide.

Important mutation options:

```text
--dtype TYPE                    Convert to uint8 or uint16; exclusive with mantissa trimming
--float32-mantissa-bits N       Retain N float32 fraction bits; range 0..22
--range-mode MODE               auto, preserve, or full; integer dtype mutation only
--compression NAME              LZW, Deflate, ZSTD, or Uncompressed
--output-json PATH              Write the structured mutation report
```

### `omeify inspect`

Inspection reads TIFF directories and metadata without materializing the complete raster. It works
for any TIFF layout understood by tifffile, not only conversion inputs.

```bash
omeify inspect image.tif
omeify inspect image.tif --detail 2
omeify inspect image.tif --detail 3 --max-text-length 500
omeify inspect image.tif --json --output inspection.json
```

Detail levels are cumulative:

```text
0  file and series summaries
1  pyramid levels plus OME image, dimension, channel, and calibration metadata
2  TIFF pages and lightweight frames
3  TIFF tags and parsed XML or text ImageDescription values
```

The JSON representation is defined by:

```text
omeify/schemas/tiff_inspection.schema.json
```

Inspection reports TIFF-tag-derived and OME-declared calibration separately. For OME-TIFF,
channel names, dimensions, dimension order, and OME physical sizes are parsed from OME-XML rather
than guessed from TIFF pages. Series metadata is associated through local `TiffData` ranges and
full-resolution geometry, not an assumption that TIFF series and OME Image indices coincide.
Ambiguous or incomplete associations remain unassigned. Readers use this same association and
refuse an unassigned selected series rather than guessing channel identities or calibration.
Metadata-only OME Images do not occupy TIFF-series positions; `series_names` follows TIFF-series
order. Reader-owned opens do not follow external OME companion-file references. Inspection also
reports the bundled MITI-header assessment and additional OME metadata outside omeify's minimized vocabulary.

#### Deterministic calibration agreement

Every inspection, including detail 0 and runs without intelligence, compares **full-resolution
X/Y** calibration where a local OME Image can be mapped unambiguously to the series' physical
IFDs. The result is stored in `report["calibration"]`, defined by the packaged
`omeify/schemas/calibration.schema.json` contract (version 1.0), and rendered under each series.

TIFF `XResolution` and `YResolution` are **pixels per length**; OME `PhysicalSizeX` and
`PhysicalSizeY` are **length per pixel**. TIFF's centimeter encoding and OME's µm encoding are
compatible: 20,000 pixels/cm equals 0.5 µm/pixel. Omeify writes TIFF
`ResolutionUnit=CENTIMETER` while canonicalizing OME `PhysicalSizeX`/`PhysicalSizeY` to explicit
`µm`. The different unit spellings describe the same physical calibration and are checked after
conversion to a common unit.

The check converts both axes independently to µm/pixel using original TIFF rational values.
Its documented tolerance is **1e-5 relative (10 parts per million), with zero absolute tolerance**:
`abs(a - b) <= 1e-5 * max(abs(a), abs(b))`. This accommodates the existing six-significant-digit
source normalization; it is not an estimate of microscope measurement uncertainty.

| Result | Meaning |
|---|---|
| `consistent` | Every checked base-plane X/Y value agrees and all base-plane entries were checked. |
| `mismatch` | At least one comparable X or Y value differs beyond the numerical tolerance. It does not decide which declaration is correct. |
| `partial` | Some comparisons agree, but other values or entries cannot be compared. |
| `not_comparable` | There is no usable comparison, for example no OME header, missing/nonphysical units, or an ambiguous mapping. It does **not** mean the image is uncalibrated. |

The check reads real IFDs behind lightweight frames rather than inheriting their keyframe's
calibration. It does not fill missing values from another axis or page. An absent TIFF
`ResolutionUnit` uses TIFF's **inch** default and is flagged as defaulted; explicit unit 1 gives
no absolute scale. Absent physical-size units in OME **2016-06** use its **µm** default and are
flagged too. This does not add missing attributes to the source or make omeify's explicit-unit
MITI-header requirement pass. Unknown units remain non-comparable.

This is deliberately a base-plane check. SubIFDs have different sampling scales and are never
compared directly against base OME sizes. It does not infer a resampling factor for arbitrary
third-party pyramids from rounded dimensions. The shared writer independently verifies its
known base and `2**level` pyramid scales against the intended pixel size before atomic install,
including odd-sized images. Internal agreement cannot prove the original acquisition calibration
was correct.

Inspection bounds this check to **4,096 base-plane entries per file**, not per series, and an
**8,388,608-character OME-XML limit** (bytes for byte-valued XML). The report retains page counts, skipped comparisons,
original rational values, normalized lengths, unit-default flags, mapping reasons, per-axis
relative differences, and scan-limit flags. Incomplete coverage never becomes `consistent`.
Unresolved companion UUIDs are not followed or matched by filename. These checks do not perform
full OME mapping validation, read raster pixels, edit images, or change inspection's exit status:
a valid diagnostic report can describe a mismatch. Writer verification, in contrast, rejects a
mismatch before installing the output.

> **Inspection is not deidentification.** Text and JSON reports may contain source paths,
> filenames, TIFF tags, vendor XML, scanner fields, or other identifying values. Review them before
> sharing.

XML comments and processing instructions are retained as diagnostic nodes without being mistaken
for OME model elements. Ordinary text rendering escapes terminal/control and bidirectional
formatting characters while preserving scientific Unicode such as µm. JSON retains the original
source strings; display escaping is not deidentification.

#### Optional metadata intelligence

```bash
omeify inspect image.tif -i
omeify inspect image.tif --intelligence --json --output inspection.json
```

`-i` / `--intelligence` adds one consolidated, advisory metadata summary at the end
of the tree, or an `intelligence` member in JSON. It does not replace the ordinary
TIFF/OME diagnostics or their deterministic validation results. Installing the
extra alone never enables inference. `convert` and `mutate` do not accept this
flag or use model output to alter images.

The summary looks for dates, identifiers, embedded paths and filenames, acquisition
and scanner details, channel/marker labels, physical calibration and units, and
software/processing provenance. The model selects useful metadata records and
interprets them; Python copies each selected record's exact value and quotation
into the report. The model does not have to reproduce paths, dates, XML-decoded
text, or Unicode correctly to preserve that evidence. Internal OME object IDs are
distinguished from specimen or person identifiers in the requested interpretation. Dates are not normalized into
an invented timezone or an assumed acquisition date.

Omeify uses the existing Sheetbend configuration, not a second endpoint/key store:
`${XDG_CONFIG_HOME:-$HOME/.config}/sheetbend/config.toml`, with Sheetbend's
`SHEETBEND_CONFIG` override. The default ordered scopes are **institutional, then
local**. Sheetbend selects the source and its configured default model; the
connection requires declared `json_schema` and `system_prompt` capabilities and
uses `application="omeify", tool="inspect"` for `sheetbend top`. Its configured
reasoning default is honored. There are no automatic probes, retries, source/model
fallbacks, or config changes.

Narrow to local processing, or resolve an ambiguous registry explicitly:

```bash
omeify inspect image.tif -i --intelligence-scope local
omeify inspect image.tif -i --intelligence-source laboratory
```

Repeat `--intelligence-scope` to supply an ordered preference; supplied values
replace the default pair. An explicit source name must still satisfy the scopes.
The `external` scope is available only by explicitly including it. Scope labels
are configuration declarations, not proof of institutional approval or privacy.
All `--intelligence-*` options require `-i`.

**Evidence collection is independent of `--detail` and `--max-text-length`.** It
walks physical TIFF directories, including SubIFDs and IFDs behind lightweight
frames, without decoding raster pixels. XML attributes/text and JSON metadata are
flattened into records; XML comments are treated as data too. Duplicate field/value
pairs share a record with occurrence counts and up to eight source locations.
Sources are cited by TIFF directory and metadata field. OME Image indices in XML
paths are not assumed to equal TIFF series indices, and file-wide OME metadata is
not incorrectly attributed only to the first stored plane.

The inspector also adds bounded, explicitly `origin="computed"` evidence records from its
**same deterministic calibration report**. These records contain numeric results, mapping and
coverage information, not file paths or arbitrary diagnostic exception text. They are labeled
as computed evidence in the text tree, rather than presented as strings embedded in the TIFF.
The system prompt explains reciprocal TIFF densities, centimeter/inch conversions, OME lengths,
unit defaults, and the difference between a true mismatch and insufficient information. The
model's prose remains advisory and does not replace or alter the deterministic verdict.

The model receives these source and computed records, not the file's current path/name, filesystem
timestamps, general inspection warnings, pixels, or attachments. Paths already
embedded in metadata are intentionally included. It gets no callable tools or
permission to open embedded paths/URLs. XML DTDs and known binary/pixel blocks are
omitted; external XML entities are never expanded. No prompt or response log is
written by omeify.

The default **32,000-character serialized-record budget** includes record locations
and JSON escaping, including computed calibration evidence, but excludes the small coverage
object, instructions, response schema, and output. Computed records receive at most one quarter
of that budget, prioritizing mismatch results before partial/non-comparable/consistent results.
`computed_records_available` and `computed_records_included` make omissions visible; the complete
deterministic results remain in the local inspection report. It is not a token/context-window guarantee. To change it:

```bash
omeify inspect image.tif -i --intelligence-max-chars 64000
```

The generated-response allowance defaults to **8,192 tokens** and is passed to the selected
model through Sheetbend. It is independent of the metadata character budget and can be raised
per request when the endpoint has sufficient context capacity:

```bash
omeify inspect image.tif -i --intelligence-max-output-tokens 16384
```

Increasing either limit cannot exceed the selected endpoint's total context window. A larger
output allowance can help avoid truncating schema-constrained JSON, while a larger metadata
packet consumes more of that same context.

Collection prioritizes likely identifiers, dates, paths and scientific fields,
with round-robin selection across directory/OME Image groups. This is a bounded
summary, not an exhaustive metadata scanner: long values are split into
contiguous 2,048-character excerpts with 128-character overlap, with their character
ranges recorded in the source locations. Excerpt records are flagged as truncated
even when all excerpts fit the packet. Collection stops at 4,096 directories, 20,000
unique records or an 8 MiB tag-value scan budget, and skips individual tags larger
than 1 MiB. Binary/large numeric arrays and encoded XML payloads are omitted.
Coverage reports record selection, truncation, unreadable/oversized tags and scan
limits. These limits bound this collector, not memory already used by tifffile
when opening/parsing a file. A larger endpoint context can justify increasing the
record budget, but does not remove the independent scan limits.

The authoritative contracts are packaged JSON Schemas:

```text
omeify/schemas/tiff_inspection.schema.json          inspection schema 1.4
omeify/schemas/metadata_intelligence.schema.json    summaries 1.1; questions 1.2
```

The packaged schema defines both the final `summary` and the model's smaller
`model_response` selection contract. **Prompt protocol 2.0 selects records instead
of asking the model to transcribe their values and quotations.** Each finding
contains `record_id`, `category`, `label`, and `interpretation`; the overview and
cautions contain `text` and `record_ids`. Every reference in the model-facing
schema is restricted to an enum of the IDs actually supplied in this request.
IDs are matched literally, not by list position or a guessed nearby record.

For grammar-based OpenAI-compatible servers, the model-facing schema is fully
inlined and keeps structural constraints (object shape, required fields, enums,
and `additionalProperties`) while omitting regex, length, and array-count bounds.
The complete selection schema is validated locally before materialization. Omeify
then checks each selected ID against the packet, copies its entire value into
the final evidence quotation, and uses that same value for the corresponding
finding. The materialized summary and complete report are validated against the
full packaged report schema. Source identity, allowed scopes, coverage and the
evidence catalog also come from omeify, not from the model.

This removes exact-copy errors without fuzzy matching, Unicode normalization,
path rewriting, accepting fabricated quotations, or silently discarding findings.
Repeated IDs within a statement are deduplicated without changing their order.
A composite/plain-text metadata record remains a complete record or collected
excerpt rather than a model-selected substring; the interpretation can point out
its interesting content. XML entity spellings are decoded by the collector, and
that extracted text is preserved exactly. JSON contains the complete supplied
value; `--max-text-length` still bounds its pretty-print preview.

Broad summary reports use intelligence schema **1.1**, with `prompt_version` **2.1** identifying
record selection plus calibration guidance. Schema 1.0 reports with prompt 1.0/2.0 still validate;
records without an `origin` field retain their original meaning as source metadata.
A failed request, invalid JSON, unknown reference, or violated local constraint
is an error, not an empty “all clear” summary. Diagnostics identify the response
field/index without echoing source values or model prose. The CLI exits nonzero
and leaves an existing output report untouched. No retry, repair request, or
source/model fallback is added. Run without `-i` for ordinary inspection.

Exact source values and valid record references do **not** establish that the
model chose the relevant record, interpreted it correctly, or found everything.
Prose, classifications, and cautions remain advisory.

> **Not deidentification or scientific validation.** A summary can miss information
> or misinterpret genuine evidence. “Not reported” applies only to the supplied
> metadata and the model's response. Reports themselves may expose identifying
> data; no burned-in labels, raster content or tissue quality were examined.

#### Ask a focused image question

```bash
omeify inspect image.ome.tif -i --question "Can you give me a channel list?"
omeify inspect image.ome.tif -i -q "What pixel size and dtype does this image use?"
omeify inspect image.ome.tif -i -q "Do the TIFF and OME calibrations agree?" --json
```

`--question TEXT` / `-q TEXT` requires `-i` / `--intelligence`. Without `-i`, it is a usage
error, not an implicit opt-in to inference. Empty/whitespace-only questions and questions over
4,096 characters are rejected before collection or source selection. The image path is still
required. The question is sent to the configured endpoint and retained in the report, so do not
put information in it that the selected source is not authorized to receive.

Question mode makes **one** Sheetbend request instead of the broad summary request. The normal
inspection tree, including the selected `--detail`, is followed by a focused **Question / Answer**
block; the extended metadata-intelligence overview/findings/categories are not generated or
printed. Ordinary `-i` without a question is unchanged. Answers use plain text and application-
formatted lists, wrap to terminal width (up to 100 columns, 88-column fallback), and escape
terminal controls. `--max-text-length` does not truncate the answer or its selected values.
Progress goes to stderr; the report goes to stdout or `--output`. `--json` remains one JSON
object, with exact unwrapped question, answer, selected values, evidence, and coverage.

The initial question interface knows only **metadata and file/layout statistics**: channel
labels, dimensions, dtype, compression tags, calibration and its deterministic checks, file
size, series/IFD/level counts, and estimated base-array bytes from shape and dtype. It does not
scan raster pixels, measure intensities or histograms, count cells, or assess tissue/staining
quality. Statistics already embedded in metadata are declarations, not measurements verified
by this command. Missing information should produce a qualified or unavailable answer.

For requested lists and exact values, the model selects evidence records and omeify copies
their values locally. This preserves channel names and Unicode without trusting model
transcription. The model still chooses records, labels them, and writes the interpretation;
valid references do not prove that its choices or explanation are correct. A brief evidence-ID
line is printed with the answer; full locations and exact quotations are retained in JSON.

Question mode uses the same source, scopes, capability checks, character/token settings, error
handling, and no-retry/no-fallback policy as ordinary intelligence. Within each source group,
records matching words in the question are prioritized. File/layout statistics are generated
from an allowlist of existing inspection fields, independent of display detail; current file
paths/names and general diagnostic messages are not added to the packet. These computed records
share the existing quarter-budget allowance with calibration evidence. Up to 4,096 series
statistics are considered; scan and budget omissions remain visible in coverage. The question
has its own 4,096-character limit and is outside the serialized-record character budget, but
still consumes endpoint context. No images, tools, shell commands, or new dependencies are added.

Question results use intelligence schema **1.2**, prompt **3.0**, and the mutually exclusive
`question` / `answer` alternative to `summary`. Broad summaries still emit schema 1.1 / prompt
2.1; older summary reports continue to validate. The enclosing inspection schema remains 1.4.

### `omeify version`

```bash
omeify version
omeify version --json
```

The JSON form includes the Python, tifffile, imagecodecs, NumPy, Click, lxml, jsonschema, and
`ome-schema` versions.

## Python API

### Conversion

```python
from omeify import convert

report = convert(
    "input.qptiff",
    "output.ome.tif",
    input_type="qptiff_mif",
    rename_channels={"FITC": "PanCK"},
    rename_channels_by="name",
    compression="LZW",
    tile_size=1024,
    cache_directory="/fast/scratch",
)
```

Python index mappings use integer keys:

```python
report = convert(
    "fusion.qptiff",
    "fusion.ome.tif",
    input_type="qptiff_fusion",
    channel_name_field="auto",
    rename_channels={0: "DNA"},
    rename_channels_by="index",
)
```

Writer APIs receive final ordered channel names and do not accept rename mappings. Source naming
policy remains separate from OME serialization.

### Mutation

Integer dtype mutation:

```python
from omeify import mutate

report = mutate(
    "source.ome.tif",
    "compact.ome.tif",
    input_type="ome_tiff",
    dtype="uint16",
    range_mode="auto",
)

for channel in report["dtype_mutation"]["channels"]:
    print(channel["channel_name"], channel["mapping"])
```

Float32 precision mutation:

```python
report = mutate(
    "source.ome.tif",
    "compact-float.ome.tif",
    input_type="ome_tiff",
    float32_mantissa_bits=11,
    compression="LZW",
)

print(report["float_precision_mutation"])
```

### Metadata intelligence

Inference is an explicit library operation, never a side effect of constructing,
printing or serializing an ordinary inspector:

```python
from omeify import TiffInspector

inspector = TiffInspector("image.ome.tif")
result = inspector.summarize_metadata()  # One request, institutional then local.
print(result["summary"]["findings"])
print(inspector.render_text())          # Reuses the attached result; no new request.
inspection_json = inspector.to_json()   # Includes evidence records and coverage.
assert inspector.validation_errors() == ()
```

Use the same method for one focused question:

```python
result = inspector.summarize_metadata(question="List the channels in metadata order.")
print(result["answer"]["status"])  # answered, partial, or unavailable
print(inspector.render_text())    # Ordinary inspection plus Question / Answer block
```

Each call refreshes the metadata and replaces the previous intelligence result; rendering
never makes a second request. With a question, the low-level inference payload separates
`question` from the untrusted `metadata` packet instead of interpolating it into the system prompt.

An application can supply its own registry while retaining the same restrictions:

```python
from sheetbend import Registry

result = inspector.summarize_metadata(
    registry=Registry.from_file(),
    source_name="laboratory",
    allowed_scopes=["institutional", "local"],
    max_metadata_chars=32000,
    max_output_tokens=8192,
)
```

`model_name=` optionally selects another configured model on the selected source.
The output-token setting defaults to 8,192 and is passed through LLM; actual token
accounting, context limits and reasoning behavior depend on the configured model.
No temperature, seed or model-specific reasoning policy is imposed by omeify.
Calling `summarize_metadata()` again deliberately makes a new request. Path-backed
inspectors refresh local diagnostics and metadata together before summarization;
`from_tiff()` reuses the caller-owned handle without closing it. Inspector-owned
opens disable external OME companion-file loading.

To inspect exactly which metadata records would be supplied, without inference or
optional dependencies:

```python
import tifffile
from omeify.intelligence import collect_metadata

with tifffile.TiffFile("image.ome.tif", _multifile=False) as tiff:
    packet = collect_metadata(tiff, max_chars=32000)

print(packet["coverage"])
# Inspect packet["records"] locally before any transmission.
```

The standalone collector does not enumerate series to create calibration context, which could
open companion files in a caller-owned tifffile handle. To reproduce the inspector's packet
locally on one handle:

```python
with tifffile.TiffFile("image.ome.tif", _multifile=False) as tiff:
    inspector = TiffInspector.from_tiff(tiff, file_path="image.ome.tif")
    packet = collect_metadata(tiff, calibration=inspector.report["calibration"])
```

For question-mode evidence without making an inference request:

```python
with tifffile.TiffFile("image.ome.tif", _multifile=False) as tiff:
    inspector = TiffInspector.from_tiff(tiff, file_path="image.ome.tif")
    packet = collect_metadata(
        tiff, calibration=inspector.report["calibration"], inspection=inspector.report,
        question="List the channels.",
    )
```

`omeify.intelligence.summarize_metadata(packet, question="List the channels.")` answers that
question using the supplied packet. `metadata_question_schema(record_ids=...)` exposes its
model-facing answer contract, with the same bound references and compact grammar projection
as the summary schema.

`omeify.intelligence.summarize_metadata(packet)` is the explicit inference step;
it returns the same schema-defined result that the inspector attaches.
`metadata_summary_schema(record_ids=[record["id"] for record in packet["records"]])`
exposes the exact model-facing schema for that packet; calling it without IDs
returns the unbound selection schema for offline inspection. This is the selection
protocol, not the materialized report summary. Caller-constructed packets must
use records of at most 2,048 characters, as `collect_metadata()` already does;
oversized records are rejected before inference rather than clipped silently.
Base tests use a model double; optional tests exercise real
Sheetbend selection/capability checks and the actual LLM/SDK adapter against a
synthetic loopback server. These tests do not measure a model's scientific accuracy.

### Pixel size

`PixelSize` is the immutable calibration value shared by readers, writers, and conversion profiles:

```python
from omeify import PixelSize

pixel_size = PixelSize(0.5068, 0.5068, "µm")
assert pixel_size.to_tuple() == (0.5068, 0.5068, "µm")
assert PixelSize.from_tuple(pixel_size.to_tuple()) == pixel_size

level_one_size = pixel_size.scaled(2)
nanometers = pixel_size.converted_to("nm")
```

The value object does not impose a micrometer-only policy: callers may construct and convert
`PixelSize` values in any supported physical-length unit. OME serialization is the policy boundary
and converts that quantity to canonical `µm`; readers return `None` only when no usable calibration
is available.

### Storage-independent images and sources (0.17)

`ImageSource` describes and serves pixel regions. `Image`, `MultichannelImage`, `RGBImage`, and
`LabelImage` provide the semantic interface. The backend can be `ArraySource`, `OMETiffSource`,
or an external provider. No source is required to invent a filesystem path. Mocktome-specific
simulation and biological truth stay in Mocktome, not in Omeify.

```python
import numpy as np
from omeify import MultichannelImage, OMETiffWriter, PixelSize

array = np.zeros((3, 1024, 1536), dtype=np.float32)
with MultichannelImage.from_array(
    array, axes="CYX", channel_names=("DAPI", "CD3", "CD20"),
    pixel_size=PixelSize(0.5, 0.5, "µm"), copy=False,
) as image:
    patch = image.read_region(100, 356, 200, 456, channels=[0, 2])
    dapi = image.get_by_name("DAPI").read_region(100, 356, 200, 456)
    report = OMETiffWriter(
        "signals.ome.tif", compression="Deflate", tile_size=256, overwrite=False,
    ).write(image)
```

Use `RGBImage.from_array(rgb, ...)` for interleaved `YXS` and
`LabelImage.from_array(labels, ...)` for categorical integer `YX`. The three semantic image
kinds can also wrap procedural sources; none is tied to NumPy or TIFF. Array wrapping is
file-free, **not** an assertion that the underlying array was generated lazily.

A procedural provider subclasses `ImageSource`, supplies immutable `ImageMetadata`, and implements
`_read_region(y0, y1, x0, x1, *, level, channels)`. Omeify handles bounds/channel normalization,
returned shape/dtype checks, byte-order normalization, owned regional results, and resource
lifetime. Provider metadata discovery must not render pixels. Repeated/overlapping reads and
channel subsets must agree: a source is one fixed observation, not a new random exposure per read.

```python
from omeify import MultichannelImage, OMETiffSource

with MultichannelImage(OMETiffSource("signals.ome.tif")) as image:
    print(image.channel_names, image.level_descriptors)
    patch = image.read_region(100, 356, 200, 456)
```

`OMETiffReader(path)` remains supported directly, including its inspection and lazy-channel APIs.
It now also exposes `.metadata`, `.level_descriptors`, `.as_image()` and a borrowing `.source`
adapter. Native TIFF `.levels` objects remain available for compatibility. New generic consumers
should use `.level_descriptors`, `.level_shape(level)`, `.pixel_size_at_level(level)`, and
`.level_downsample(level)`. Sampling scales are explicit or unknown, never guessed from rounded
pyramid dimensions.

Images own their backend by default. Use `owns_source=False` to borrow an already-open source.
Writers always borrow images; keep the image's `with` block active until the write finishes.
Closing/reopening invalidates old Channel and writer adapters instead of silently rebinding them.
Nested contexts on one composed image/source are rejected. `from_array()` returns an opened
image and also works with `with`. Borrowed input arrays must not be mutated during observation;
`copy=True` explicitly takes a complete snapshot.

`OMETiffWriter(path, ...).write(image)` and `.write_image(image, level=0)` infer metadata from an
open Image. `OMEImageSeries.from_image(name, image)` supplies the same source to heterogeneous
output. `TemporaryOMETiffWriter.write(image)` and `write_ometiff(path, image=image, ...)` use the
same engine. Existing array and `PlaneReaderSource` writer paths remain available. Output
pyramids are still rebuilt, calibration is still mandatory for writing, label output stays
lossless/nearest, and RGB writing stays uint8. No intensity scaling is hidden in source adaptation.
Local source-file dependencies are protected from accidental overwrite, including link aliases.

The provider contract, runnable procedural example, and notebook recipes are in
[Images without files](docs/image_sources.md). From the repository root:

```python
from examples.image_sources import run_example
result = run_example("Scratch/omeify_sources", overwrite=False)
```

This release adds the Omeify contract, not Mocktome's regional renderer or a GUI. Zarr/PNG
backends and spatial view wrappers remain future adapters. Source-level thread safety is not
assumed. Downstream packages with an `omeify<0.17` pin need that constraint reviewed for this
upgrade; path-only application entry points do not automatically become Image-aware.

### Readers and lazy channels

`pixel_size_at_level(0)` uses the ordinary OME/fallback calibration policy. Reduced levels use
consistent physical TIFF resolution tags on their own pages, normalized to µm. When those tags
are unavailable it returns `None` with a warning, rather than deriving scale from rounded image
dimensions or assuming a third-party pyramid is 2x. Omeify-written levels have explicit tags for
the writer's `2**level` sampling scale, including odd-sized and one-pixel-wide images.

The supported conversion profiles have corresponding readers:

```text
AkoyaMIFQPTiffReader
AkoyaFusionQPTiffReader
AkoyaHEQPTiffReader
AperioSVSReader
AkoyaComponentTiffReader
IndicaMIFTiffReader
OMETiffReader
```

Readers are context-managed and expose the same bounded plane contract consumed by the writers.
Reading channel metadata does not decode the complete image:

```python
from omeify import OMETiffReader

with OMETiffReader("image.ome.tif") as ome:
    print(ome.pixel_size)
    print(ome.series_name)
    print(ome.series_names)

    dapi = ome[0]
    panck = ome.get_by_name("PanCK")
    same_dapi = ome.get_by_id(dapi.id)

    patch = dapi.read_region(10_000, 11_024, 20_000, 21_024)
    full_dapi = dapi.array  # explicit whole-channel materialization
```

`get_by_name()` fails when a name is ambiguous. `get_by_id()` performs exact lookup. Inputs without
channel IDs receive deterministic normalized IDs such as `Channel:0:0`, while the `Channel` object
records whether the ID was generated.

`read_region()` supports planar `CYX`, grayscale `YX`, and interleaved `YXS` layouts.
`asarray(level=N)` is the explicit whole-level escape hatch. Logical OME channels are distinct from
stored samples: RGB has one logical channel named `RGB` and three stored samples.

Akoya Fusion exposes source `Name` and `Biomarker` fields independently:

```python
from omeify import AkoyaFusionQPTiffReader

with AkoyaFusionQPTiffReader(
    "fusion.qptiff",
    channel_name_field="auto",
) as fusion:
    print(fusion.pixel_size)
    print(fusion.channel_names)
    print(fusion[0].source_metadata["name"])
    print(fusion[0].source_metadata["biomarker"])
    patch = fusion.get_by_name("CD3").read_region(0, 1024, 0, 1024)
```

A generic label raster is available through `OMETiffLabelReader`. It validates integer storage and
provides bounded image access without inventing biological semantics or an unreliable generic
`label_count`.

### Single-image writer

`OMETiffWriter` writes one homogeneous OME Image and is the standards-enforcing path used by
`convert` and `mutate`:

```python
import numpy as np
from omeify import OMETiffWriter, PixelSize

image = np.zeros((3, 4096, 4096), dtype=np.uint16)
report = OMETiffWriter(
    "image.ome.tif",
    image_type="multichannel",
    channel_names=("DAPI", "PanCK", "CD3"),
    pixel_size=PixelSize(0.5, 0.5, "µm"),
).write(image)
```

For callers that explicitly need the low-level writer rather than the `mutate` workflow,
`float32_mantissa_bits=N` remains available as a Python writer control:

```python
report = OMETiffWriter(
    "image.ome.tif",
    image_type="multichannel",
    channel_names=("DAPI",),
    pixel_size=PixelSize(0.5, 0.5, "µm"),
    compression="LZW",
    float32_mantissa_bits=11,
).write(image)
```

The option keeps float32 storage and exponent range while rounding every base and pyramid pixel to
`N` stored fraction bits using nearest, ties-to-even rounding. `N=23` preserves full float32
precision. For user-facing file mutation, prefer `omeify mutate --float32-mantissa-bits N` so the
operation is explicit in the workflow and structured report.

Float32 precision trimming accepts either input byte order, preserves NaN/infinity bit patterns
and signed zeros, and does not mutate the caller's array. If rounding a finite extreme would
produce infinity, it raises `OverflowError` before output installation. It does not silently
saturate, rescale, or replace the pixel; retaining all 23 fraction bits remains a no-trimming path.

Use `image_type="rgb"` for `YXS uint8` RGB and `image_type="label"` for one integer `YX` label
raster. `image_type` is writer policy, not a private TIFF tag. OME-TIFF itself does not intrinsically
distinguish label values from intensity values.

An open `Image` can be passed to `.write(image)` without repeating its metadata; the writer
requests bounded regions and never calls `image.asarray()`. Existing advanced sources implement
the small `PlaneReaderSource` protocol and call `write_source()`.
Conversion readers use this path directly, so the CLI and Python conversion helper do not maintain
a separate writer implementation.

### Heterogeneous multi-series writer

`OMEMultiSeriesWriter` writes several named OME Images into one file. Each `OMEImageSeries` may
specify its own dtype, image type, channel names, pixel size, compression, and downsampling policy:

```python
import numpy as np
from omeify import OMEImageSeries, OMEMultiSeriesWriter, PixelSize

pixel_size = PixelSize(0.5, 0.5, "µm")
series = (
    OMEImageSeries.from_array(
        "Normalized signal",
        np.zeros((2, 4096, 4096), dtype=np.float32),
        image_type="multichannel",
        channel_names=("A", "B"),
        pixel_size=pixel_size,
    ),
    OMEImageSeries.from_array(
        "Object labels",
        np.zeros((4096, 4096), dtype=np.uint32),
        image_type="label",
        channel_names=("Object labels",),
        pixel_size=pixel_size,
    ),
    OMEImageSeries.from_array(
        "Preview",
        np.zeros((4096, 4096), dtype=np.uint8),
        image_type="multichannel",
        channel_names=("Preview",),
        pixel_size=pixel_size,
        compression="JPEG",
    ),
)

report = OMEMultiSeriesWriter(
    "derived.ome.tif",
    compression="Deflate",
    software="example-application 1.0",
).write(
    series,
    provenance={
        "schema": "example.provenance/1",
        "software": {"name": "example-application", "version": "1.0"},
        "parameters": {"threshold": 0.25},
    },
)
```

Series names must be unique. `OMEImageSeries.from_image(name, image)` borrows any open semantic
Image and infers its specification without reading pixels. Array-backed series may use NumPy
memory maps. Advanced callers use
`OMEImageSeries.from_source()` with the same `PlaneReaderSource` boundary as the ordinary writer.

The multi-series writer accepts the same `float32_mantissa_bits=` precision control. A float32
precision setting is applied only to float32 series; integer, label, RGB, and float64 series retain
their source precision.

The optional provenance mapping must contain finite JSON-serializable values. It is serialized once
as canonical JSON, stored in a namespaced OME `MapAnnotation`, and linked from every OME Image.
Omeify does not interpret application-specific provenance fields.

### Convenience and temporary writers

`write_ometiff()` accepts one `CYX` array, a list of `YX` arrays, or lazy `Channel` objects through
`channels=`. Alternatively, use `image=` with an open semantic Image and inferred metadata:

```python
from omeify import PixelSize, write_ometiff

write_ometiff(
    "output.ome.tif",
    channels=image,
    channel_names=("DAPI", "PanCK", "CD3"),
    pixel_size=PixelSize(0.5, 0.5, "µm"),
)
```

Lazy channels are adapted directly to the regional source contract. Their `.array` properties are
not touched merely because they were passed to the convenience function.

`TemporaryOMETiffWriter` owns only temporary-path lifecycle:

```python
from omeify import PixelSize, TemporaryOMETiffWriter

with TemporaryOMETiffWriter(
    channel_names=("DAPI", "PanCK"),
    pixel_size=PixelSize(0.5, 0.5, "µm"),
) as writer:
    writer.write(image[:2])
    temporary_path = writer.path

# The temporary OME-TIFF and its pyramid cache are removed here.
```

## Writer architecture

Omeify exposes two public writer contracts because they describe different products:

- `OMETiffWriter` accepts one homogeneous image specification.
- `OMEMultiSeriesWriter` accepts an ordered collection of named, potentially heterogeneous image
  specifications.

They are not separate implementations. Both normalize their public inputs into the same internal
prepared-image model and use one shared writer engine for:

1. source-plane validation
2. compression normalization and JPEG alignment checks
3. image-type-specific lossy-compression policy enforcement
4. pyramid shape planning
5. bounded temporary pyramid construction
6. tiled BigTIFF base and SubIFD encoding
7. common container, storage, decoding, and lossless spot-value verification
8. temporary scratch cleanup
9. atomic installation of the verified output

The public writers differ only where their contracts require it: input modeling, generated OME
metadata, allowed lossy series, multi-image provenance, and the shape of the returned report. This
keeps TIFF construction behavior centralized while preserving clear public APIs for genuinely
different outputs.

## Metadata minimization and MITI alignment

Omeify constructs a new OME-XML document instead of copying arbitrary source metadata. The guiding
publication is the MITI minimum-information work cited in the introduction:

[Schapiro D, Yapp C, Sokolov A, et al. **MITI Minimum Information guidelines for highly
multiplexed tissue images.** *Nature Methods.* 2022;19:262-267.
doi:10.1038/s41592-022-01415-4.](https://doi.org/10.1038/s41592-022-01415-4)

MITI describes a broader dataset-level metadata standard and recommends OME-TIFF for raster image
data. Omeify applies that minimum-information philosophy narrowly to the OME-TIFF file boundary:
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

## Validation and verification

Before a temporary output replaces the destination, omeify checks:

- OME 2016-06 XML schema validity using a local schema
- the bundled MITI-aligned header profile
- BigTIFF identity and TIFF/OME byte-order agreement
- dtype and `SignificantBits`
- channel and `SamplesPerPixel` organization
- `TiffData` physical-plane mapping
- output axes, full-resolution shape, and top-level IFD count
- tiled storage, compression, SubIFDs, and reduced-resolution flags
- rebuilt pyramid dimensions and linked pyramid annotations
- re-read OME physical pixel sizes against the intended X/Y calibration, with explicit units
- re-read TIFF resolution/unit tags on every base and SubIFD plane against the intended calibration
  and the writer's known `2**level` sampling scale, including odd-sized pyramids
- ICC profile preservation when supplied
- representative-pixel decoding at the base and every pyramid level, including lossy output
- bitwise representative full-resolution values for lossless output, after byte-order normalization
- representative lossless pyramid values against downsampling of the preceding decoded level
- caller-supplied multi-series provenance and annotation links

Successful single- and multi-series writer reports add `ome_physical_sizes_match_specs`,
`tiff_calibration_matches_specs`, `pyramid_calibration_matches_specs`,
`calibration_ifds_checked`, and `calibration_relative_tolerance` to their verification section.
The calibration readback checks do not require any additional raster scan.

Both writer reports also include `verification.pixel_verification` (version 1.0), which records
`mode="sampled"`, decoded and compared point counts, checked plane/level counts, and comparison
policies. The deterministic sample is top-left, center, and bottom-right, deduplicated for tiny
planes. These checks are not exhaustive and do not certify unsampled pixels. Base comparisons
include NaN payloads and signed zeros; nearest-neighbor pyramid comparisons are also bitwise.
Mean-pyramid comparisons use numerical equality with NaNs equal at matching positions because
arithmetic does not promise preservation of NaN payloads. JPEG levels are decoded but not value-
compared against the preceding JPEG level, which is not their original uncompressed reference.
The source reader is not an independent oracle; regression tests additionally compare complete
small rasters against independently constructed arrays. No file checksums or full-slide
verification scans are added.

Float64 mean downsampling retries only overflowing groups of finite contributors using binary
scaling before summation. Ordinary means retain their existing arithmetic; NaNs and infinities
in source groups retain IEEE arithmetic behavior rather than being silently repaired.

The conversion report keeps OME schema validity, MITI-header validity, and binary-output
verification as distinct structured sections.

## I/O, scratch space, and failure behavior

Full-resolution sources are read through bounded strips or tiles. Omeify does not materialize a
whole slide merely to convert or write it. Temporary reduced levels are uncompressed tiled BigTIFF
files built one level at a time. Peak RAM is governed primarily by a small number of image tiles
plus the largest source segment decoded by tifffile.

A complete 2x pyramid requires temporary disk space of approximately one third of the uncompressed
base raster. Multi-series output stages the reduced levels for every series, so scratch planning
must account for their combined uncompressed pyramid footprint. Use `cache_directory=` or
`--cache-directory` to place that work on an appropriate local disk.

Final output is written to a temporary file beside the destination. The writer validates OME
metadata before construction, checks the temporary TIFF's structure and bounded raster samples
after encoding, and installs the output atomically only after those checks pass. A failed write
or verification does not destroy an existing valid output.

With `overwrite=True`, installation atomically replaces the destination. With `overwrite=False`
(or CLI `--no-overwrite`), installation uses an atomic same-filesystem hard link and refuses any
existing destination entry, including a dangling symlink or a file created by another writer
during processing. On a filesystem without hard-link support this mode fails rather than falling
back to a race-prone replacement or non-atomic copy. Temporary output and pyramid scratch are
cleaned up on these failures.

Grayscale TIFF segment decoding preserves singleton spatial axes, including one-row strips and
one-row terminal strips. A decoded segment must cover its entire promised in-bounds rectangle;
a short or misplaced segment is an error, not implicitly padded missing data. Explicit sparse
TIFF segments continue to read as zeros.

## Current limitations

- Supported image models are two-dimensional `YX`, planar `CYX`, and interleaved `YXS` RGB.
  Non-singleton Z and T workflows require an explicit future plane-selection contract.
- RGB writing is restricted to interleaved `uint8` samples.
- Mutation supports planar floating-point input to `uint8` or `uint16`, or float32-to-float32 precision trimming through retained mantissa bits.
- Metadata minimization does not detect identifying text embedded in pixels.
- `inspect` reports source metadata and must be reviewed before sharing.

## License

Omeify is licensed under the Apache License 2.0. See `LICENSE` for the complete terms.
