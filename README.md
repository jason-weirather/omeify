# omeify

`omeify` converts supported multiplexed tissue images into standardized, deidentified, tiled, pyramidal OME-TIFF. It is deliberately opinionated: preserve the full-resolution scientific pixels, reconstruct only the metadata needed to interpret them, rebuild the image pyramid, and verify the resulting file before it replaces the destination.

## Design goals

- Produce a predictable OME-TIFF representation from heterogeneous TIFF-family inputs
- Preserve the native pixel dtype and full-resolution pixel values
- Minimize carried-forward metadata while retaining the information required to interpret the image
- Rebuild pyramid levels from the full-resolution image rather than trusting source pyramids
- Keep peak memory bounded by processing strips and tiles instead of materializing a whole slide
- Fail on unsupported image layouts or dtypes rather than silently coercing data

## Supported inputs

The current planar mIF path supports:

- Akoya mIF QPTIFF
- HALO planar mIF TIFF / OME-TIFF
- Akoya component TIFF
- Native `uint8`, `uint16`, `uint32`, `int8`, `int16`, `int32`, `float32`, and `float64` pixels
- Lossless LZW, Deflate, ZSTD, or uncompressed output
- JPEG output for `uint8` images when lossy compression is explicitly desired

Unsupported dtypes fail before conversion. `omeify` does not automatically cast or rescale unsupported pixel data.

The current mIF profile is planar: one grayscale sample per channel with `SizeZ=1` and `SizeT=1`. The H&E class remains importable but intentionally raises `NotImplementedError`; the H&E iteration will use the conventional interleaved RGB representation rather than three grayscale channel pages. That RGB work will also require a separate or generalized header-validation profile because `SamplesPerPixel=3` changes the relationship between logical channels and `SizeC`.

## Output conventions

The mIF writer emits one little-endian BigTIFF series. The TIFF byte order is `<`, and the generated OME-XML therefore declares `BigEndian="false"`. Converting a big-endian source to this representation changes byte encoding, not the numeric dtype or pixel values.

`SignificantBits` is the full storage width of the output pixel type: 8 for `uint8`, 16 for `uint16`, 32 for `float32`, and so on.

Each full-resolution channel is a top-level IFD. Rebuilt lower resolutions are tiled SubIFDs marked as reduced-resolution images. `TiffData IFD="0"` maps the full-resolution planes beginning at the first IFD; pyramid SubIFDs are not additional OME planes.

The OME root UUID is generated and retained by default. It can be omitted explicitly with `--omit-uuid`.

## MITI-aligned metadata minimization

`omeify` constructs a new OME-XML header instead of copying arbitrary source metadata into the output. The goal is to retain the minimum information needed to unambiguously interpret the pixels in the OME-TIFF while avoiding unnecessary vendor, acquisition, or potentially identifying metadata.

The guiding resource for MITI (Minimum Information about highly multiplexed Tissue Imaging) is:

> **Schapiro D, Yapp C, Sokolov A, et al.** MITI minimum information guidelines for highly multiplexed tissue images. *Nat Methods.* 2022;19:262–267. doi:10.1038/s41592-022-01415-4.

The normalized header record is validated against the bundled JSON Schema:

```text
omeify/schemas/miti_ome_tiff_header.schema.json
```

This schema implements the intent of the MITI OME-TIFF header minimums for the image representation written by `omeify`. It is intentionally a little stronger than a literal transcription of the MITI table: fields such as `Pixels ID`, `Interleaved`, `SignificantBits`, channel IDs, and TIFF IFD mapping are also checked because they make the generated OME-TIFF self-consistent and mechanically interpretable.

The published MITI header definition does not enumerate every valid OME scalar pixel type. `omeify` accepts the scalar numeric types that the current mIF writer can preserve exactly and never changes scientific data merely to satisfy an incomplete enumeration.

### Header fields

| Field | Requirement | Why it is retained |
|---|---|---|
| `Image ID` | Required | Provides the internal OME identifier for the image object. |
| `Pixels ID` | Required | Provides the internal OME identifier for the pixel object referenced by the image. |
| `BigEndian` | Required | Declares the byte order of the written TIFF so numeric values can be decoded correctly. It must agree with the actual TIFF byte order. |
| `DimensionOrder` | Required | Defines how Z, C, and T planes map onto the TIFF plane sequence. The current planar mIF profile writes `XYZCT`. |
| `Interleaved` | Required by the omeify mIF profile | Declares that channels are stored as separate planar samples rather than interleaved samples. The current mIF profile requires `false`. |
| `PhysicalSizeX`, `PhysicalSizeY` and units | Required | Converts pixel coordinates into physical distance. Without this calibration, spatial measurements in tissue have no physical scale. |
| `PhysicalSizeZ` and unit | Required when `SizeZ > 1` | Provides the corresponding physical calibration for a Z stack. The current planar mIF profile has `SizeZ=1`. |
| `SizeX`, `SizeY` | Required | Defines the full-resolution raster dimensions. |
| `SizeC`, `SizeZ`, `SizeT` | Required | Defines the dimensional shape of the image and therefore how many logical planes must exist. |
| `Type` | Required | Defines the numeric representation of each pixel. `omeify` preserves the supported native dtype and fails rather than casting an unsupported dtype. |
| `SignificantBits` | Required by the omeify profile | Records the storage width of the output pixel type and is checked against `Type`. |
| Channel `ID` | Required by the omeify profile | Gives each logical channel a unique OME identifier. |
| Channel `Name` | Required | Preserves the human-readable identity of each image channel. |
| Channel `SamplesPerPixel` | Required by the omeify mIF profile | Describes the planar channel layout. It is `1` for every channel in the current mIF writer. |
| `TiffData IFD` | Required by the omeify profile | Connects the OME plane model to the actual TIFF IFD sequence. The current writer starts at IFD 0. |
| `TiffData PlaneCount` | Required | States how many full-resolution planes belong to the image and is checked against `SizeC × SizeZ × SizeT`. |

For the current planar mIF representation, those fields are sufficient to reconstruct what is actually stored in the file: the raster dimensions, pixel type and byte encoding, physical scale, channel identities, logical dimension ordering, and the mapping from OME planes to TIFF IFDs. Experimental context such as biospecimen identifiers, antibody clone and lot, staining protocol, instrument configuration, acquisition date, and downstream analysis history is valuable metadata, but it is not required to decode or spatially interpret the image raster itself and belongs in the associated experimental records.

The MITI header table also recommends a free-text `Comment`. `omeify` does not synthesize one or copy arbitrary source comments into the output. A comment is not needed to interpret the raster, and free-text source metadata is exactly the kind of material that metadata minimization is intended to avoid carrying forward. Conversion provenance is instead captured by the generated `Creator` value and the conversion report.

### Additional OME metadata written by omeify

A few fields are deliberately written in addition to the minimum header record:

| Metadata | Policy | Purpose |
|---|---|---|
| Root `UUID` | Generated by default; optional with `--omit-uuid` | Gives the OME document a globally unique identifier without carrying a source identifier forward. |
| Root `Creator` | Generated | Records the `omeify` version that created the OME-XML. |
| `Image Name` | Generated from the input profile | Provides a simple human-readable image label such as `WholeSlideMIF`. |
| Empty `LightPath` | Retained for each channel | Preserves the current channel structure without inventing acquisition metadata that is not known. |
| Pyramid `MapAnnotation` + `AnnotationRef` | Generated when a pyramid is present | Describes the rebuilt resolution levels and links that annotation to the image. The TIFF SubIFD structure remains the actual pyramid storage. |

This is metadata minimization, not metadata invention. `omeify` carries forward only the small set of source facts it needs, principally the image geometry, native dtype, physical pixel size, and channel names, then generates the OME/TIFF structural metadata for the file it actually writes.

MITI also defines biospecimen, reagent, acquisition, instrument, processing, analysis, and other companion metadata. Those records remain important to a complete MITI dataset, but they are outside the purpose of the OME-TIFF header generated by this package.

> **Deidentification note:** rebuilding a minimal OME header avoids carrying arbitrary source metadata into the output. It does not inspect the image pixels themselves for burned-in labels or other identifying content.

## Validation and verification

Before an output replaces the destination, `omeify` checks both metadata and TIFF structure. Validation includes:

- OME 2016-06 XML schema validation
- The bundled MITI-aligned header profile
- BigTIFF and TIFF/OME byte-order agreement
- Native dtype and `SignificantBits`
- `TiffData` plane mapping
- Channel and top-level IFD counts
- Pyramid dimensions, tiled storage, SubIFDs, and reduced-resolution flags
- The linked pyramid annotation
- Spot checks of full-resolution pixel values for lossless output

Channel names are taken from the source when available. If a source does not provide one, the current reader can fall back to a generic `Channel N` label. For quantitative mIF workflows, meaningful source channel names are strongly preferred because a syntactically valid placeholder cannot recover the biological identity of an unknown channel.

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

The conversion report includes separate `ome`, `miti_header`, and `verification` sections so XML validity, header-profile validation, and binary-image verification remain distinct.

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
