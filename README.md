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
- Provide context-managed Python readers for every supported conversion source plus OME-TIFF labels
- Provide one standards-enforcing OME-TIFF writer used by both the Python API and `convert`
- Keep image analysis, segmentation, patch generation, stitching, and region measurements outside omeify

## Supported inputs

### Planar multiplex images

- Akoya mIF QPTIFF
- Akoya Fusion multiplex QPTIFF, with explicit `Name`, `Biomarker`, or `auto` channel naming
- Indica Labs/HALO mIF TIFF with an `<indica>` ImageDescription
- Planar OME-TIFF, including HALO exports that already contain OME metadata
- Akoya component TIFF
- Native `uint8`, `uint16`, `uint32`, `int8`, `int16`, `int32`, `float32`, and
  `float64` pixels

The planar profile writes `CYX`: one grayscale top-level IFD per logical channel,
`SamplesPerPixel=1`, and `Interleaved=false`.

### Brightfield RGB images

- Akoya H&E QPTIFF from PhenoImager HT or PhenoCycler Fusion
- Aperio SVS
- Interleaved RGB OME-TIFF
- Interleaved `uint8` RGB with source axes `YXS`, `SamplesPerPixel=3`, and contiguous samples

The RGB profile writes one top-level interleaved RGB IFD, not three grayscale pages. Its OME
model uses `SizeC=3`, one `Channel` with `SamplesPerPixel=3`, `Interleaved=true`, and one
physical TIFF plane in `TiffData`.

Akoya physical pixel size is read from `PixelSizeMicrons` in the QPI description. Fusion
multiplex inputs are checked across every full-resolution channel page. If expected QPI
calibration is missing, standard TIFF `XResolution`, `YResolution`, and `ResolutionUnit` tags are
used as a fallback and a warning is emitted; inconsistent calibration still fails. Fusion `Name`
and `Biomarker` values are both retained in the reader/source metadata, while an explicit policy
selects the one used as the normalized OME channel name. Indica mIF channel names and the
channel/level-to-IFD mapping are read from the `<indica>` ImageDescription. TIFF resolution tags
are the expected Indica physical-scale source, so using them does not emit a warning. Aperio uses
`MPP` when available and warns before falling back to TIFF resolution tags. OME-TIFF similarly
falls back with a warning when its OME `PhysicalSizeX`/`PhysicalSizeY` metadata is incomplete.
Component TIFF uses an explicit override when supplied, otherwise standard TIFF resolution tags
when available. TIFF resolution-derived values are normalized to micrometers for centimeter or
inch source units and rounded to six significant digits. If no usable calibration source remains,
conversion requires an explicit pixel-size override. A source ICC profile is preserved when
present. Arbitrary vendor descriptions, filenames, user names, scanner identifiers, dates, and
other free text are not copied into the output OME-XML.

OME-TIFF is also a first-class conversion input. `omeify convert --type ome_tiff` reads the
source pixels and the minimum structural metadata needed to interpret them, then writes a fresh
OME header through the same standards-enforcing writer used for every other input. This is useful
for taking a valid but non-canonical OME-TIFF and producing one that satisfies the omeify MITI
header profile while dropping arbitrary extra OME metadata. The conversion report records the
source OME header's MITI status before normalization.

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

`omeify` is a command group. The supported interface always includes a subcommand:

```text
omeify convert   Convert a supported source into standardized OME-TIFF
omeify inspect   Summarize any TIFF as a text tree or schema-backed JSON
omeify version   Show the omeify version
```

The former pre-subcommand form is not accepted or forwarded.

### Convert

Planar Akoya mIF:

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

Use `--channel-name-field name` or `--channel-name-field biomarker` to require that exact
source field. Both source values remain available in the conversion report; only the selected,
optionally renamed value becomes the minimized OME channel name.

Akoya H&E QPTIFF using the brightfield JPEG defaults:

```bash
omeify convert H32_HE.qptiff H32_HE.ome.tif \
  --type qptiff_he \
  --tile-size 1024
```

Indica Labs/HALO planar mIF TIFF:

```bash
omeify convert halo-mif.tif halo-mif.ome.tif \
  --type indica_mif \
  --compression LZW
```

The Indica profile reads channel names and IFD mappings from the `<indica>` ImageDescription.
The source pyramid is not copied; omeify reads the full-resolution channel IFDs declared as
`level="0"` and rebuilds the output pyramid through the normal writer path.

Aperio SVS with an explicit quality setting:

```bash
omeify convert CMU-1.svs CMU-1.ome.tif \
  --type svs \
  --jpeg-quality 92 \
  --jpeg-subsampling 444
```

Normalize an existing OME-TIFF into the omeify contract:

```bash
omeify convert source.ome.tif normalized.ome.tif \
  --type ome_tiff
```

HALO exports that already contain OME metadata use `--type ome_tiff`. HALO/Indica planar mIF
TIFFs with an `<indica>` ImageDescription use the explicit `--type indica_mif` profile.

Channel renaming is deliberately explicit. Name mode interprets JSON keys as normalized source
channel names:

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

Index mode uses zero-based integer strings because JSON object keys are strings:

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

The mode is authoritative. In `name` mode a source channel literally named `"0"` is still a
name; in `index` mode JSON keys such as `"0"` are validated and converted to integer index `0`.
Malformed mappings fail instead of being guessed. Component TIFFs use standard TIFF resolution
tags when available. An explicit override uses the same `PixelSize` vocabulary as the Python API:

```bash
omeify convert component.tif component.ome.tif \
  --type component \
  --pixel-size-x 0.5068 \
  --pixel-size-y 0.5068 \
  --pixel-size-unit µm
```

Useful conversion options:

```text
--compression NAME        LZW, Deflate, ZSTD, JPEG, or Uncompressed
--jpeg-quality N          JPEG quality from 1 through 100; default 90
--jpeg-subsampling MODE   444, 422, 420, or 411; default 444
--pyramid-levels N        Explicit subresolution count; auto by default
--channel-name-field      name, biomarker, or auto for Fusion QPTIFF
--rename-channels-json    JSON map used with --rename-channels-by
--rename-channels-by      name or index
--pixel-size-x/y/unit     Explicit physical-size override; default unit µm
--omit-uuid               Omit the optional OME root UUID
--workers N               TIFF compression workers
--no-checksums            Skip the final whole-file checksum pass
```

Only the selected TIFF series is converted. The default is series 0, which is the baseline
whole-slide series for the supported QPTIFF and SVS examples. Use `--series` only when inspection
shows that the desired full-resolution image is elsewhere.

The conversion report keeps separate `ome`, `miti_header`, and `verification` sections so XML
validity, header-profile validity, and binary-image verification remain distinct. Pixel size is
serialized in reports as `[x, y, unit]`.

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

`inspect` reports pixel size calculated directly from standard TIFF `XResolution`,
`YResolution`, and `ResolutionUnit` tags when those tags provide usable physical calibration.
If the tags are absent or do not define a physical unit, the text summary reports `N/A` without
adding a warning. For an OME-TIFF, inspection separately parses the OME header rather than
guessing channel names, dimension sizes, dimension order, or OME physical pixel sizes from TIFF
pages alone. This makes the OME-declared and TIFF-tag-derived calibrations independently visible.
Inspection also reports whether each OME
header satisfies the bundled omeify MITI header profile, lists missing or invalid fields, and
identifies additional OME metadata outside omeify's minimized output vocabulary. Additional
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

### Conversion

The Python conversion API mirrors the CLI with one `convert()` helper. The former parallel
`omeify.inputs` conversion classes have been removed; source interpretation now lives only in the
`omeify.io` reader classes, and every conversion is written through `OMETiffWriter`.

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

Fusion channel interpretation and index-based renaming remain explicit:

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

Python index mappings use actual integer keys. JSON has no integer object keys, so the CLI accepts
`{"0": "DNA"}` and converts that key to integer `0` only when `--rename-channels-by index` was
selected. Writer APIs do not accept rename mappings at all: they receive the final ordered
`channel_names`, keeping source renaming separate from OME serialization.

Existing OME-TIFFs use the same helper:

```python
report = convert(
    "source.ome.tif",
    "normalized.ome.tif",
    input_type="ome_tiff",
)
```

If an otherwise usable source lacks physical calibration, an explicit `PixelSize` can be supplied
as an override. Standard TIFF resolution tags are used as a fallback where available rather than
inventing calibration.

### Pixel size

`PixelSize` is the immutable physical-calibration value used by readers, writers, and conversion
profiles:

```python
from omeify import PixelSize

pixel_size = PixelSize(0.5068, 0.5068, "µm")
assert pixel_size.to_tuple() == (0.5068, 0.5068, "µm")
assert PixelSize.from_tuple(pixel_size.to_tuple()) == pixel_size

level_one_size = pixel_size.scaled(2)
nanometers = pixel_size.converted_to("nm")
```

The value object itself does not impose a micrometer policy. OME serialization validates that its
unit is a supported physical-length unit. A reader returns `None` only when the source genuinely
contains no usable calibration rather than inventing one. Standard TIFF resolution tags are
interpreted as pixels per physical unit and converted to micrometers for centimeter or inch units;
the derived pixel size is rounded to six significant digits to avoid carrying rational-encoding
noise into OME metadata.

### Readers and lazy channels

The supported conversion sources now have corresponding I/O readers:

```text
AkoyaMIFQPTiffReader
AkoyaFusionQPTiffReader
AkoyaHEQPTiffReader
AperioSVSReader
AkoyaComponentTiffReader
IndicaMIFTiffReader
OMETiffReader
```

These readers own source-format interpretation and implement the same random-access plane contract
consumed by the writer. The conversion helper therefore does not maintain a second source parser.
Readers expose source channel names and metadata; channel renaming happens only at the conversion
boundary or by supplying replacement final names to a writer.

`OMETiffReader` keeps the TIFF open and exposes ordered lazy `Channel` objects. Looking at channel
metadata does not decode the full image:

```python
from omeify import OMETiffReader

with OMETiffReader("output.ome.tif") as ome:
    print(ome)  # same default tree as: omeify inspect output.ome.tif
    print(ome.pixel_size)

    dapi = ome[0]
    panck = ome.get_by_name("PanCK")
    same_dapi = ome.get_by_id(dapi.id)

    print(dapi.index, dapi.id, dapi.name, dapi.dtype, dapi.shape)
    patch = dapi.read_region(10_000, 11_024, 20_000, 21_024)
    full_dapi = dapi.array  # explicit whole-channel materialization
```

`get_by_name()` fails if a name occurs more than once. `get_by_id()` performs exact ID lookup.
Inputs without channel IDs receive deterministic normalized IDs such as `Channel:0:0`, while the
`Channel` object records whether that ID was generated.

`read_region` on the parent reader continues to support common planar `CYX`, grayscale `YX`, and
interleaved `YXS` layouts. `asarray(level=N)` remains the explicit whole-level escape hatch.
Logical OME channels are distinct from stored samples: RGB has one logical channel named `RGB`
with three stored samples named red, green, and blue.

Akoya Fusion QPTIFF has its own lazy reader:

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

Fusion `auto` prefers `Biomarker` and falls back to `Name`. Explicit `name` or `biomarker` mode
fails if the requested field is absent. Pixel calibration is checked across every channel page.

For custom workflows, readers can be handed directly to the same writer path without materializing
the whole image:

```python
from omeify import AkoyaMIFQPTiffReader, write_ometiff

with AkoyaMIFQPTiffReader("input.qptiff") as source:
    assert source.pixel_size is not None
    write_ometiff(
        "output.ome.tif",
        channels=source.channels,
        pixel_size=source.pixel_size,
    )
```

A label raster uses the same virtual-access boundary without inventing biological semantics:

```python
from omeify import OMETiffLabelReader

with OMETiffLabelReader("cells.ome.tif") as labels:
    print(labels.pixel_size)
    block = labels.read_region(10_000, 11_024, 20_000, 21_024)
```

There are no separate `CellLabelImage` or `TissueLabelImage` types. `LabelImage` validates integer
storage and provides image access only. It intentionally does not expose region properties or a
`label_count`: an exact generic label count requires scanning the raster, and `max(label)` is not
a reliable count when IDs are sparse. `scikit-image` is deliberately not an omeify dependency.

### OME-TIFF writer

`OMETiffWriter` is the single standards-enforcing output implementation used by both the Python
API and `omeify convert`:

```python
import numpy as np
from omeify import OMETiffWriter, PixelSize

image = np.zeros((3, 4096, 4096), dtype=np.uint16)
report = OMETiffWriter(
    "image.ome.tif",
    image_type="multichannel",
    channel_names=["DAPI", "PanCK", "CD3"],
    pixel_size=PixelSize(0.5, 0.5, "µm"),
).write(image)
```

Use `image_type="rgb"` for `YXS` `uint8` RGB data. RGB is written as one logical OME channel
with `SamplesPerPixel=3`. Use `image_type="label"` for one integer `YX` label raster; label
pyramids use nearest-neighbor downsampling and lossless compression. The image-type argument is
writer policy, not a private TIFF tag. OME-TIFF itself does not intrinsically distinguish label
values from intensity values.

Advanced streaming sources can implement the small `PlaneReaderSource` contract and call
`write_source`. The public `convert()` helper uses that path directly with the source readers, so
there is no second private writer or conversion-only I/O stack drifting away from the public API.

### Convenience writers

`write_ometiff()` accepts one `CYX` NumPy array, a list of `YX` arrays, or lazy omeify `Channel`
objects:

```python
from omeify import PixelSize, write_ometiff

write_ometiff(
    "output.ome.tif",
    channels=image,
    channel_names=["DAPI", "PanCK", "CD3"],
    pixel_size=PixelSize(0.5, 0.5, "µm"),
)
```

```python
with OMETiffReader("source.ome.tif") as source:
    assert source.pixel_size is not None
    write_ometiff(
        "selected.ome.tif",
        channels=[
            source.get_by_name("DAPI"),
            source.get_by_name("PanCK"),
        ],
        pixel_size=source.pixel_size,
    )
```

Lazy `Channel` objects are adapted directly to the writer's region-based source contract. Their
`.array` properties are not touched merely because they were passed to the convenience function.
Channel names are inferred from those objects unless replacement names are supplied.

`TemporaryOMETiffWriter` wraps the same writer and owns only temporary-path lifecycle:

```python
from omeify import PixelSize, TemporaryOMETiffWriter

with TemporaryOMETiffWriter(
    channel_names=["DAPI", "PanCK"],
    pixel_size=PixelSize(0.5, 0.5, "µm"),
) as writer:
    writer.write(image[:2])
    temporary_path = writer.path
    # temporary_path exists here

# the temporary OME-TIFF and its pyramid cache are gone here
```

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
