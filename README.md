# omeify

`omeify` reads supported microscopy TIFF-family formats and writes predictable, tiled,
pyramidal OME-TIFF. It preserves the native pixel dtype and never rescales or casts during normal
conversion. Pixel values remain exact with lossless output and are approximate when JPEG is
selected. Omeify rebuilds pyramids from the full-resolution raster and constructs a minimized OME
header instead of copying arbitrary source metadata.

> **Deidentification boundary:** metadata minimization avoids carrying vendor descriptions,
> filenames, user names, scanner identifiers, dates, and other arbitrary source text into the
> generated OME-XML. It does not inspect pixels for burned-in labels or other identifying content.
> Inspection reports are diagnostic and may expose source metadata.

## What omeify provides

- Explicit readers for supported vendor TIFFs and OME-TIFF
- Bounded regional access without materializing a whole slide
- Canonical planar multichannel, interleaved RGB, and label-image representations
- One ordinary writer for a single homogeneous OME Image
- One multi-series writer for named heterogeneous OME Images
- One shared internal engine for writer validation, pyramid construction, TIFF encoding,
  verification, scratch cleanup, and atomic installation
- Deterministic dtype mutation with a quantitative per-channel loss report
- TIFF and OME metadata inspection without decoding the complete raster
- OME 2016-06 schema validation and a bundled MITI-aligned header profile

Omeify owns the image-file boundary. It does not own segmentation, overlapping inference tiles,
object reconciliation, patch scheduling, region measurements, or image-analysis policy.

## Installation

Omeify supports Python 3.10 through 3.14.

```bash
python -m pip install .
```

For development:

```bash
python -m pip install -e '.[dev]'
pytest
ruff check .
```

The package is licensed under Apache-2.0. The authoritative package version is the static
`[project].version` value in `pyproject.toml`. Installed code reads distribution metadata; direct
source-tree imports fall back to the neighboring `pyproject.toml`.

## Quick start

Inspect a TIFF before choosing a conversion profile:

```bash
omeify inspect input.tif
```

Convert a planar multiplex image:

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

Create an explicit integer representation of a floating-point multiplex image:

```bash
omeify mutate source.ome.tif compact.ome.tif \
  --type ome_tiff \
  --dtype uint16 \
  --range-mode auto \
  --output-json compact.mutation.json
```

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
encoding noise into OME metadata.

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

`SignificantBits` records the full storage width of the output dtype, such as 8 for `uint8`, 16 for
`uint16`, and 32 for `float32`.

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

The TIFF `Software` tag defaults to `omeify <version>`. Applications using omeify as their file
writer may supply `--software` or `software=` to identify the product generator. This does not
change the OME root `Creator`, which continues to identify the omeify version that generated the
OME metadata.

A source ICC profile is retained for RGB output when present. Arbitrary vendor descriptions and
other free text are not copied.

## Command-line interface

Omeify uses explicit subcommands:

```text
omeify convert   Normalize a supported source into OME-TIFF
omeify mutate    Create an explicitly dtype-mutated OME-TIFF
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
--software TEXT            Override the TIFF Software tag
--output-json PATH         Write the structured conversion report
-v                         Compact stages and progress
-vv                        Timestamped debug logging and tracebacks
```

### `omeify mutate`

`mutate` currently converts planar `float32` or `float64` channels to `uint8` or `uint16`. It never
edits the input in place. The source is scanned by bounded regions, one fixed mapping is selected
per full-resolution channel, and transformed regions are passed through the same ordinary writer
used by conversion.

```bash
omeify mutate halo-float.tif compact.ome.tif \
  --type indica_mif \
  --dtype uint16 \
  --range-mode auto \
  --output-json compact.mutation.json
```

The three range modes have distinct meanings:

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

The mutation report records, per channel:

- exact minimum, maximum, zero count, negative count, and non-finite counts
- deterministic percentiles and integer-lattice diagnostics
- exact unit-rounding error and the automatic loss-budget decision
- selected offset and source-units-per-code quantum
- the reason for the selected mapping
- theoretical and sampled error, normalized RMSE, clipping count, and code usage

Automatic per-channel scaling preserves ordering within a channel but changes raw comparability
between independently scaled channels or slides. A cohort that requires common intensity units
should reuse fixed mappings instead of selecting a new automatic mapping per slide.

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
than guessed from TIFF pages. Inspection also reports the bundled MITI-header assessment and
additional OME metadata outside omeify's minimized vocabulary.

> **Inspection is not deidentification.** Text and JSON reports may contain source paths,
> filenames, TIFF tags, vendor XML, scanner fields, or other identifying values. Review them before
> sharing.

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

The value object does not impose a micrometer-only policy. OME serialization validates that its
unit is a supported physical-length unit. Readers return `None` only when no usable calibration is
available.

### Readers and lazy channels

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

Use `image_type="rgb"` for `YXS uint8` RGB and `image_type="label"` for one integer `YX` label
raster. `image_type` is writer policy, not a private TIFF tag. OME-TIFF itself does not intrinsically
distinguish label values from intensity values.

Advanced sources implement the small `PlaneReaderSource` protocol and call `write_source()`.
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

Series names must be unique. Array-backed series may use NumPy memory maps. Advanced callers use
`OMEImageSeries.from_source()` with the same `PlaneReaderSource` boundary as the ordinary writer.

The optional provenance mapping must contain finite JSON-serializable values. It is serialized once
as canonical JSON, stored in a namespaced OME `MapAnnotation`, and linked from every OME Image.
Omeify does not interpret application-specific provenance fields.

### Convenience and temporary writers

`write_ometiff()` accepts one `CYX` array, a list of `YX` arrays, or lazy `Channel` objects:

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
resource is:

> Schapiro D, Yapp C, Sokolov A, et al. MITI minimum information guidelines for highly multiplexed
> tissue images. *Nature Methods.* 2022;19:262-267. doi:10.1038/s41592-022-01415-4.

The normalized header is validated against:

```text
omeify/schemas/miti_ome_tiff_header.schema.json
```

The schema implements the MITI OME-TIFF header minimums plus structural requirements needed to
interpret the generated file mechanically.

| Header value | Omeify policy |
|---|---|
| `Image ID`, `Pixels ID` | Required generated identifiers |
| `BigEndian`, `DimensionOrder`, `Interleaved` | Required and checked against TIFF storage |
| `PhysicalSizeX`, `PhysicalSizeY`, units | Required |
| `SizeX`, `SizeY`, `SizeC`, `SizeZ`, `SizeT` | Required |
| `Type`, `SignificantBits` | Required and checked against stored samples |
| Channel `ID`, `Name`, `SamplesPerPixel` | Required |
| `TiffData IFD`, `PlaneCount` | Required and checked against physical top-level IFDs |
| `Image/@Name` | Omitted for ordinary output; required and caller supplied for multi-series output |
| Root `UUID` | Generated by default; optional |
| Root `Creator` | Generated from the omeify version |
| Empty `LightPath` | Generated without inventing acquisition metadata |
| Pyramid `MapAnnotation` | Generated and linked when reduced levels exist |
| ICC profile | Preserved for RGB when supplied |
| Multi-series JSON provenance | Optional, caller supplied, canonicalized, and linked from every image |

MITI companion records for biospecimens, reagents, acquisition, instruments, processing, and
analysis remain important to a complete dataset but are outside the minimized OME-TIFF header.
Omeify does not synthesize experimental facts that were not supplied through an explicit supported
contract.

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
- ICC profile preservation when supplied
- representative-pixel decoding for lossy output
- exact representative full-resolution values for lossless output
- caller-supplied multi-series provenance and annotation links

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
metadata before construction, verifies the complete temporary TIFF after encoding, and replaces the
destination atomically only after those checks pass. A failed write or verification does not
destroy an existing valid output.

## Current limitations

- Supported image models are two-dimensional `YX`, planar `CYX`, and interleaved `YXS` RGB.
  Non-singleton Z and T workflows require an explicit future plane-selection contract.
- RGB writing is restricted to interleaved `uint8` samples.
- Dtype mutation currently supports planar floating-point input to `uint8` or `uint16`.
- Metadata minimization does not detect identifying text embedded in pixels.
- `inspect` reports source metadata and must be reviewed before sharing.

## License

Omeify is licensed under the Apache License 2.0. See `LICENSE` for the complete terms.
