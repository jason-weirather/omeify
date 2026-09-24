# Omeify reference

Detailed options, library recipes, and implementation policies for the current
Omeify image/file contract. Start with the [README](../README.md) for the OME-TIFF
home base, quickstarts, the tifffile comparison, and the
[metadata/MITI tables](../README.md#metadata-minimization-and-miti-alignment).
For the common Image interface, regional providers, lifetime rules, and migration,
see [Images and sources](image_sources.md).

[Installation](#installation) · [Supported inputs](#supported-inputs) ·
[Output layout](#canonical-ome-tiff-output) · [CLI](#command-line-reference) ·
[Python API](#python-api) · [Verification](#validation-and-verification) ·
[I/O and failure behavior](#io-scratch-space-and-failure-behavior)

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
omeify convert component.tif --output component.ome.tif \
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

`convert` and `mutate` take one positional `INPUT_PATH` and require an output
file through `--output FILE` or `-o FILE`:

```bash
omeify convert input.ome.tif -o output.ome.tif --type ome_tiff
omeify mutate input.ome.tif -o output.ome.tif --type ome_tiff --dtype uint16
```

The former positional destination is no longer accepted. Omitting the output
option is a usage error before image processing begins; `--output-json` cannot
substitute for it. That option remains an optional report destination, separate
from the OME-TIFF. `inspect` retains its optional `--output` / `-o` report file
and defaults to stdout. Python `convert()` and `mutate()` signatures are unchanged.

### `omeify convert`

Akoya mIF QPTIFF:

```bash
omeify convert input.qptiff --output output.ome.tif \
  --type qptiff_mif \
  --compression LZW \
  --tile-size 1024 \
  --downsample mean \
  --cache-directory /fast/scratch
```

Akoya Fusion multiplex QPTIFF, preferring `Biomarker` and falling back to `Name`:

```bash
omeify convert fusion.qptiff --output fusion.ome.tif \
  --type qptiff_fusion \
  --channel-name-field auto
```

Use `--channel-name-field name` or `--channel-name-field biomarker` to require that exact source
field. Both values remain available in the conversion report. Only the selected, optionally
renamed value becomes the minimized OME channel name.

Indica Labs/HALO mIF TIFF:

```bash
omeify convert halo-mif.tif --output halo-mif.ome.tif \
  --type indica_mif
```

The Indica profile reads channel names and full-resolution IFD mappings from its `<indica>`
ImageDescription. The source pyramid is not copied.

Aperio SVS with an explicit JPEG policy:

```bash
omeify convert slide.svs --output slide.ome.tif \
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
omeify convert input.ome.tif --output output.ome.tif \
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
omeify convert input.ome.tif --output output.ome.tif \
  --type ome_tiff \
  --rename-channels-json renames.json \
  --rename-channels-by index
```

The selected mode is authoritative. A source channel literally named `"0"` remains a name in name
mode. Malformed or mixed mappings fail rather than being guessed.

Common conversion options:

```text
-o, --output FILE          Required destination OME-TIFF file
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
omeify mutate halo-float.tif --output compact.ome.tif \
  --type indica_mif \
  --dtype uint16 \
  --range-mode auto \
  --output-json compact.mutation.json
```

Float32 precision mutation is intended for processed floating-point rasters whose computational
precision exceeds the meaningful precision of the measurement. The recommended starting point for
processed Akoya PhenoImager HT data is 11 retained fraction bits:

```bash
omeify mutate halo-float.ome.tif --output halo-float-compact.ome.tif \
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
-o, --output FILE               Required destination OME-TIFF file
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

Install and configure the optional dependency as described under
[Installation](#installation); it is separate from ordinary image I/O.

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

### Images and pixel sources (0.18)

The library has one image contract. `MultichannelImage`, `RGBImage`, and `LabelImage`
use an `ImageSource` to obtain pixels. File readers compose that same source
machinery; they do not have a separate image-level read path. Arrays enter through
explicit named constructors. External providers implement `ImageSource` in their
own package. Omeify contains no Mocktome-specific code or dependency.

**0.18 is a breaking library consolidation.** The old writer
synonyms, public plane-source adapters, expensive `.array` property, and competing
level/metadata access paths are removed. Current notebook patterns, provider
requirements, and a complete migration table are in [Images and sources](image_sources.md).

| Pattern | Library call |
|---|---|
| Open real images | `with OMETiffReader(path) as image:` or an explicit vendor reader |
| Open categorical labels | `with OMETiffLabelReader(path) as labels:` |
| Wrap intensity arrays | `MultichannelImage.from_array(array, axes="CYX", ...)` |
| Wrap RGB or label arrays | `RGBImage.from_array(...)` / `LabelImage.from_array(...)` |
| Serve procedural pixels | `with MultichannelImage(CustomSource(...)) as image:` |
| Select/assemble lazy scalar channels | `MultichannelImage.from_channels([...])` |
| Change metadata without changing pixels | `image.with_metadata(...)` |
| Write one image | `OMETiffWriter(path, ...).write(image)` |
| Write heterogeneous images | `OMEMultiSeriesWriter(path, ...).write(series)` |

Constructors do not open resources. Use `with` or explicit `open()`/`close()`.
Writers borrow the image and leave it open. Lazy channels and derived images are
bound to their parents' open sessions; they do not quietly reconnect after a
parent is closed and reopened. Regional NumPy results own their data and survive
context exit. `copy=False` array construction deliberately borrows large data;
keep it unchanged during use. `copy=True` takes a full independent snapshot.

### Readers and lazy channels

```python
from omeify import MultichannelImage, OMETiffReader, OMETiffWriter

with OMETiffReader("image.ome.tif") as image:
    print(image)
    print(image.pixel_size, image.channel_names)
    print(image.series_name, image.series_names)
    print(image.levels)  # Generic ImageLevel descriptors, never TIFF directory objects.

    dapi = image.get_by_name("DAPI")
    panck = image.get_by_name("PanCK")
    patch = dapi.read_region(10_000, 11_024, 20_000, 21_024)
    # dapi.asarray() is the explicit whole-channel materialization operation.

    with MultichannelImage.from_channels([dapi, panck]) as selected:
        report = OMETiffWriter(
            "selected.ome.tif", compression="Deflate", overwrite=False,
        ).write(selected)
```

`image[index]`, `get_by_name(name)`, and `get_by_id(id)` look up different channel
identities. Ambiguous names fail. Channels retain normalized IDs, source IDs, and
source metadata separately. Expensive reads are methods, not properties.

`read_region(y0, y1, x0, x1, level=0, channels=None)` uses half-open integer bounds
in the selected level. It supports `CYX`, `YX`, and RGB `YXS`. Channel selection
always selects **logical channels**: RGB has one channel containing three samples.
Read that channel and index the returned patch to select individual RGB samples.
`channel_count` counts logical channels; `sample_count` counts stored samples.
`image_type` distinguishes multichannel intensities, RGB, and categorical labels.

`image.levels[n]` supplies that level's axes, shape, pixel size, and declared
`downsample_yx`. Do not derive pixel size from rounded level dimensions. Unknown
third-party level calibration remains unknown. All advertised levels are checked
at file open without decoding pixels. Non-singleton Z/T is not a supported image
product; use `TiffInspector` for metadata inspection of such files.

The supported conversion profiles retain explicit reader constructors:

```text
AkoyaMIFQPTiffReader
AkoyaFusionQPTiffReader
AkoyaHEQPTiffReader
AperioSVSReader
AkoyaComponentTiffReader
IndicaMIFTiffReader
OMETiffReader
OMETiffLabelReader
```

Akoya Fusion still exposes source `Name` and `Biomarker` independently through the
channel's `source_metadata`. Source-native storage axes/shape are available as
`native_axes` and `native_shape` on file readers. `reader.inspect()` returns the
file inspector; `print(reader)` is now a compact image summary rather than the
complete inspection tree. The `omeify inspect` CLI output is unchanged.

### Single-image writing and metadata

Image metadata belongs to the Image; storage settings belong to the writer.

```python
import numpy as np
from omeify import MultichannelImage, OMETiffWriter, PixelSize

array = np.zeros((3, 512, 768), dtype=np.float32)
with MultichannelImage.from_array(
    array, axes="CYX", channel_names=("DAPI", "PanCK", "CD3"),
    pixel_size=PixelSize(0.5, 0.5, "µm"), copy=False,
) as image:
    report = OMETiffWriter(
        "image.ome.tif", compression="Deflate", tile_size=256, overwrite=False,
    ).write(image)
```

Use `RGBImage.from_array(rgb, ...)` for `YXS` RGB and
`LabelImage.from_array(labels, ...)` for categorical integer `YX`. Wrapping a
NumPy array does not imply that generation of the array was lazy. A custom
provider can instead generate only requested rectangles through `ImageSource`.

Writer inputs are open Images, not arrays plus a second metadata declaration.
`write(image, level=N)` materializes a chosen resolution as the new output base.
It never invokes `image.asarray()` or `channel.asarray()`. It rebuilds pyramids
through the shared bounded TIFF engine, including temporary scratch levels.

For intentional calibration or final-name changes:

```python
from omeify import OMETiffReader, OMETiffWriter, PixelSize

with OMETiffReader("input.ome.tif") as image:
    with image.with_metadata(pixel_size=PixelSize(0.5, 0.5, "µm")) as calibrated:
        OMETiffWriter("calibrated.ome.tif", compression="Deflate").write(calibrated)
```

This creates a borrowed metadata view. It does not mutate pixels, edit the input
file, or infer an experimental fact. Calibration remains mandatory for writing.
RGB writing stays uint8. Labels remain lossless with nearest-neighbor pyramids.

The writer's explicit `float32_mantissa_bits=N` storage control remains available.
It preserves float32 storage and exponent range while rounding retained fraction
bits at the base and every rebuilt level. NaN/infinity payloads and signed zeros
retain the existing fidelity policy; finite overflow raises rather than silently
saturating. For a user-facing file mutation, prefer `mutate()` or its CLI command
so the numeric change is explicit in the workflow and its report.

### Heterogeneous and temporary products

Single-image and multi-series output are distinct products, not synonyms. Each
multi-series entry names one open Image, with an optional per-series compression,
downsampling policy, or selected level.

```python
import numpy as np
from omeify import LabelImage, OMEImageSeries, OMEMultiSeriesWriter, PixelSize, RGBImage

size = PixelSize(0.5, 0.5, "µm")
rgb = np.zeros((512, 768, 3), dtype=np.uint8)
labels = np.zeros((512, 768), dtype=np.uint32)
with (
    RGBImage.from_array(rgb, pixel_size=size) as brightfield,
    LabelImage.from_array(labels, pixel_size=size) as objects,
):
    report = OMEMultiSeriesWriter(
        "products.ome.tif", compression="Deflate", overwrite=False,
    ).write((
        OMEImageSeries("H&E", brightfield),
        OMEImageSeries("Objects", objects),
    ), provenance={"schema": "example.provenance/1", "parameters": {"seed": 7}})
```

`OMEImageSeries(name, image)` is the only series-construction API. Keep all images
open through the write. Series names must be unique. Optional provenance is
canonical finite JSON stored in a namespaced MapAnnotation and linked from every
OME Image. Per-series JPEG remains restricted to non-label uint8 visualization
products; source type and dtype are never changed implicitly.

`TemporaryOMETiffWriter` has a distinct responsibility: temporary-path lifetime.
It delegates encoding to the same ordinary writer.

```python
from omeify import OMETiffReader, TemporaryOMETiffWriter

with OMETiffReader("input.ome.tif") as image:
    with TemporaryOMETiffWriter(compression="Deflate") as temporary:
        temporary.write(image)
        temporary_path = temporary.path
        # Consume the file here. It is removed on context exit.
```

### Mocktome and external image providers

For existing Mocktome 0.4 results, wrap `mif.data`, `he.rgb`, and the chosen
section label array with the corresponding semantic `.from_array()` constructor.
Use the section's pixel size and the rendered mIF channel names. This avoids an
intermediate TIFF without claiming to make Mocktome's eager rendering lazy.

A genuinely regional Mocktome provider belongs in Mocktome: subclass
`ImageSource`, publish `ImageMetadata`, and implement `_read_region()`. Omeify
validates bounds, selected logical channels, returned shape/dtype, and lifetime.
The provider owns deterministic rendering, halos, and scientific truth.

Runnable notebook imports from the repository root:

```python
from examples.image_sources import run_example
result = run_example("Scratch/omeify_sources")  # Procedural pattern, no backing full array.

# Requires a separately installed Mocktome, not a new Omeify dependency.
from examples.mocktome_io import run_example
result = run_example("Scratch/mocktome_omeify")
```

See [the migration guide](image_sources.md#8-migration-from-017) before
upgrading downstream packages. In particular, Tilework must use explicit level
calibration rather than its shape-ratio fallback, and Cadastre's old
`PlaneReaderSource`/`OMEImageSeries.from_source` output route needs migration.
These are not silently supported by a second API in Omeify 0.18.

## Writer architecture

Omeify exposes two public writer contracts because they describe different products:

- `OMETiffWriter` accepts one open Image.
- `OMEMultiSeriesWriter` accepts an ordered collection of named, potentially heterogeneous Images.

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

The [README metadata tables](../README.md#metadata-minimization-and-miti-alignment)
define which source information can cross into the generated header, which fields
Omeify requires or generates, and why. That is an image-header profile, not a
certification of dataset-level MITI completeness or deidentification.

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
