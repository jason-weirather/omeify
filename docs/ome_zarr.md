# OME-Zarr: one image API, two output contracts

Omeify 0.22 reads **OME-Zarr 0.4, 0.5 and 0.6**. Its canonical writer produces
**OME-Zarr 0.5 on Zarr v3**, under the documented `omeify.ome_zarr/1` profile.
OME-TIFF remains the default output; choose Zarr explicitly. This is a restricted
use of an open format, not a requirement for a private Omeify reader.

## Installation and dependencies

```bash
# From the checkout, in a separate image-I/O environment:
python -m pip install .

# Development plus independent NGFF model tests (Python 3.12+):
python -m pip install -e '.[dev,interop]'
python -m pip check
```

Python **3.11+** is the baseline for the entire package. `zarr-python >=3.1.3` and
`fsspec[http]` are required dependencies, so a normal installation includes local
and HTTP(S) Zarr access. There is no `zarr` extra or TIFF-only installation mode.
The installer resolves dependencies compatible with the running Python version.
Preserve the earlier **0.21.0** checkpoint for applications that still need Python
3.10. The independent `interop` test extra requires Python 3.12+; it does not raise
the ordinary package minimum.

`zarr-python` owns arrays, codecs and chunk access. `fsspec` owns HTTP transport.
Omeify owns interpretation, selection, metadata policy, image semantics and
verified construction. Zarr uses ordinary imports at the backend's point of use,
so TIFF-only operations do not initialize its array/codec stack. Delayed import
is a startup-cost decision, not an optional installation mode; missing required
dependencies remain installation errors. There is no optional-backend loader or
repeated runtime version check. No Dask, viewer, segmentation or model-inference
framework is added to the runtime. `ome-zarr-models` is an independent **test**
consumer, not another runtime domain-model layer.

## Why the writing home base is 0.5

0.5 and 0.6 both use Zarr v3. Sharding, compression and chunk access are storage
choices, not advantages unlocked by 0.6. The newer NGFF version adds named
coordinate systems, transformation relationships and scenes. Our output is a
selected calibrated raster, not a registered scene, and scale plus translation
expresses its required geometry.

The integration tests build distinct 0.4, 0.5 and 0.6 stores from independently
specified metadata. They test actual reads, plane selection, coordinate mappings,
normalization and HTTP access. The optional independent `ome-zarr-models` tests
validate all three fixture versions and read the actual canonical 0.5 output.
Those tests do not establish universal browser compatibility or prove that 0.6
is broken. We retain 0.5 because its simpler image representation meets this
product's needs without requiring every consumer to understand the newer spatial
model. There is no hidden 0.6 writer or dependency-default-driven format switch.

No application-wide QuPath/Vizarr certification is claimed. New independent
consumer tests may justify a future explicit output-profile revision; upgrading
a dependency alone must not change the format being written.

## Reading contract

| Aspect | Supported behavior |
|---|---|
| Metadata/storage versions | 0.4 on Zarr v2; 0.5 and 0.6 on Zarr v3 |
| Image selection | A multiscale image group, a multiscale entry, or an ordered `bioformats2raw.layout=3` image collection |
| Plate/well/scene containers | Inspect their metadata; explicitly select a child image with `group` for pixel access |
| Axes | Conventional, explicitly named `t`, `c`, `z`, `y`, `x`; spatial order can vary |
| Dimensionality | Select one 2D observation; non-singleton Z/T requires an explicit index |
| Intrinsic geometry | Positive scale and optional translation; 0.6 identity and scale/translation sequence |
| Image types | Scalar/multichannel, explicitly declared uint8 RGB, categorical integer labels |
| Scalar pixel types | uint8/16/32, int8/16/32, float32/64; no inferred numeric cast |
| Stores | Local directories and plain HTTP(S) store roots; transport options are library-only |
| Input encoding | Zarr-supported installed codecs, including Zarr v3 sharded arrays; no codec implementation is copied into Omeify |
| Regional reads | Preserve channel ordering and duplicates, half-open bounds, native dtype and owned output arrays |

0.5 requires array `dimension_names` to agree with its NGFF axes. 0.6's
array-to-intrinsic transform defines the ordered mapping; array dimension labels
are not treated as a second 0.5-style coordinate declaration.

`OMEZarrReader` does not guess RGB from three channels, apply contrast to returned
pixels, resample registration, stitch scenes, scan for a convenient image, or
follow metadata paths outside the selected store. Custom axis names, array-valued
transform parameters, reflected intrinsic grids, arbitrary affine/nonlinear
intrinsic transforms and Z-downsampled selected-plane pyramids fail explicitly.
This is not a claim to implement every feature of the three NGFF specifications.

A 0.6 image can have additional spatial relationships beyond its intrinsic raster.
The reader exposes the intrinsic pixels and flags the additional context. Export
then fails rather than silently discarding the relationships. Reopening with
`intrinsic_only=True` (CLI `--intrinsic-only`) deliberately selects a standalone
intrinsic image product. Crops, channel assembly and metadata views retain this
restriction; they cannot accidentally turn a blocked export into an unqualified
success. The inspector retains the raw relationship metadata. No scene graph is
rendered or reconstructed by this profile.

### Python access

```python
from omeify import OMEZarrReader, OMEZarrWriter

with OMEZarrReader("source.ome.zarr") as image:
    print(image.ngff_version, image.axes, image.shape)
    print(image.channel_names, image.pixel_size, image.origin_yx)
    patch = image.read_region(100, 356, 200, 456, channels=[0, 2])
    with image.crop(100, 356, 200, 456) as region:
        OMEZarrWriter("region.ome.zarr").write(region)
```

The coordinates must fit the actual image. `patch` owns its data after closing the
source. Lazy channel and crop handles do not. Writer instances borrow an open
Image; no intermediate TIFF is required. Use `OMEZarrLabelReader` to explicitly
interpret an unmarked integer image as labels. Source-declared label or RGB
semantics cannot be contradicted by an override.

```python
with OMEZarrReader("plate.ome.zarr", group="A/1/0", t=2, z=4) as plane:
    patch = plane.read_region(0, 256, 0, 256)
```

Those Z/T indices are examples, not defaults. An absent axis cannot be selected.
Collections expose `series_names`; `series=N` selects a declared image in its
stored order, not a group discovered by sorting arbitrary keys.

## Coordinate conventions

Region bounds retain Omeify's **integer pixel-edge** convention. Physical origins
are explicitly the position of the **center** of pixel `(0,0)`, in **µm**, not a
scanner stage measurement. `image.origin_yx` is the base origin; every `ImageLevel`
has its own `origin_yx`, `downsample_yx`, and `offset_yx`.

For each axis, a level-edge coordinate maps to a base-edge coordinate as:

```text
base_edge = level_edge * downsample + offset
level_origin = base_origin + base_pixel_size * (offset + (downsample - 1) / 2)
```

These are two descriptions of the same grid. Scale never comes from a ratio of
rounded raster dimensions. An odd-sized 33-pixel image downsampled by two has
17 pixels and a sampling step of two, not 33/17.

The writer rebuilds 2x pyramids using the shared deterministic mean algorithm, or
nearest-neighbor for labels. Mean levels use the corresponding nominal center
translation and zero edge offset. Nearest levels preserve the sampled centers,
so their edge offset is `(1 - downsample) / 2`. Odd trailing mean bins use the
available samples, retaining the original mean arithmetic. These are nominal
regular-grid coordinates, not per-bin centers of mass at an incomplete edge.

A crop shifts a **known** origin by its integer source offset times pixel size.
An unknown source origin remains unknown in the Image; the Zarr writer establishes
a local frame with base center `(0,0)`, without inventing stage coordinates. The
file-level crop workflow separately records source-relative ROI offsets. Use
`image.with_metadata(origin_yx=(y_um, x_um))` to establish an explicit physical
origin without changing pixels.

When writing a located image to TIFF, Omeify records the origin and known pyramid
sampling policy in a linked, controlled `omeify.geometry/1` MapAnnotation. Its
reader recovers that record through the authoritative OME Image mapping. It does
not mislabel the origin as acquisition `Plane/PositionX/Y`. Ordinary TIFF readers
still see standard pixels and physical pixel sizes; interpreting this supplemental
origin requires understanding the annotation. Arbitrary external TIFF annotations
are not imported as geometry.

Preview rendering currently uses only pyramid levels whose edge origin matches
the base. Shifted levels fall back to a suitable aligned level or the base rather
than silently moving overlays. This can make previews of shifted input pyramids
slower, but does not change regional pixel reads or export values.

## Canonical output profile

| Aspect | Required or chosen policy |
|---|---|
| NGFF | Version 0.5, explicitly selected by Omeify |
| Zarr | Version 3, regular chunks, little-endian multibyte samples; byte order is immaterial for 8-bit samples |
| Layout | `yx` scalar/labels; `cyx` scalar channels or explicitly declared RGB components |
| Chunks | Default 512x512 spatial size; one stored component per chunk; explicit spatial override |
| Compression | Lossless Zstd level 3 without frame checksum; Gzip level 6 or Uncompressed are explicit alternatives |
| Sharding | Not emitted by this first profile; input sharding remains readable |
| Pyramids | Rebuilt from the selected input level, with common numerical reduction and precision policy |
| Calibration | Required X/Y sizes; NGFF unit spelling is `micrometer`, while the Python API displays µm |
| Metadata | Fresh allowlisted image, axis, channel, pyramid and geometry declarations |
| Display | Small deterministic samples of the coarsest level supply suggested ranges; no full-slide contrast scan and no pixel normalization |
| Empty chunks | Explicitly written; absent planned output chunks are verification failures |
| Publication | New local directory only, staged and verified before no-replace publication |

All eight scalar numeric types use lossless storage. Intentional float32 mantissa
trimming still changes values and must be understood as a numerical mutation,
not made lossless merely because the resulting bytes use lossless compression.
RGB also uses lossless Zarr storage by default; TIFF's JPEG default does not carry
across. TIFF LZW/Deflate/JPEG, JPEG quality, TIFF scratch paths, Software tags and
UUID flags do not silently become Zarr settings.

Ordinary intensity output needs no private metadata. NGFF stores RGB as three
component planes with red/green/blue display colors. A small supplemental `omeify`
attribute records that they belong to one logical RGB channel, retains its name,
and preserves an existing ICC profile. No implicit ICC transform is applied.
Independent consumers can read the planes but may not honor that supplemental
color profile or logical-channel declaration.

Standalone label rasters have integer pixels, nearest pyramids, and a supplemental
categorical declaration including the background value. They are not automatically
attached to a parent image's `labels` hierarchy and are not feature tables or
annotation objects. Existing standard `image-label` inputs are recognized. A
third-party consumer of standalone label output may require explicit label-mode
selection. Label IDs and pixel values are never renumbered.

### Metadata minimization

Omeify builds new JSON instead of copying source dictionaries. Unrelated groups,
array attributes, source filenames, dates, scanner descriptions, free-text patient
fields, source display settings and arbitrary annotation properties are not
copied. Selected channel names and explicitly requested collection/ROI names are
retained. Input names still require review. Only deliberate collection provenance
is accepted, with a 1 MiB bound; it is not a source-metadata passthrough.

Physical geometry needed to interpret the selected product is not private clutter
to silently discard. Additional geometry requires explicit intrinsic-only export.
Full source metadata remains visible in `inspect`, which **does not minimize** it.
Diagnostic reports can contain sensitive paths and values and should not be
published casually. Image pixels are not screened for identifying content.
Metadata minimization is not deidentification or study-wide MITI compliance.

## Convert, mutate, crop and inspect

```bash
# TIFF/vendor input -> the same canonical Zarr product.
omeify convert slide.qptiff -o slide.ome.zarr \
  --type qptiff_mif --output-format ome_zarr --no-overwrite -v

# Zarr 0.4/0.5/0.6 -> fresh canonical 0.5 output, not a metadata copy.
omeify convert foreign.ome.zarr -o normalized.ome.zarr \
  --type ome_zarr --output-format ome_zarr --no-overwrite

# Zarr -> TIFF through the existing encoder and real OME-XSD validator.
omeify convert normalized.ome.zarr -o normalized.ome.tif \
  --type ome_zarr --compression Deflate --no-overwrite

# Intentional float-to-integer mutation, preserving scale with unit rounding.
omeify mutate floats.ome.zarr -o integers.ome.zarr \
  --type ome_zarr --output-format ome_zarr --dtype uint16 --range-mode preserve

# Intentional float32 precision mutation.
omeify mutate floats.ome.zarr -o compact.ome.zarr \
  --type ome_zarr --output-format ome_zarr --float32-mantissa-bits 11

# One ordered crop collection, or explicitly split stores.
omeify crop normalized.ome.zarr -o regions.ome.zarr \
  --type ome_zarr --output-format ome_zarr --geojson regions.geojson
omeify crop normalized.ome.zarr -o region.ome.zarr \
  --type ome_zarr --output-format ome_zarr --geojson regions.geojson --shatter by_index

# Metadata only, including unsupported metadata as diagnostics.
omeify inspect normalized.ome.zarr
omeify inspect normalized.ome.zarr --detail 3 --json -o inspection.json
```

`convert` and `mutate` have the same keyword `output_format="ome_zarr"` in Python.
The same source selectors apply to all image workflows. Native `crop()` retains
its existing GeoJSON rectangles, naming/grouping and per-output transaction rules.
`OMEZarrCollectionWriter` accepts the same `OMEImageSeries` entries used for TIFF
collections. `TemporaryOMEZarrWriter` handles temporary-directory lifetime.

Inspection has its own `ZarrInspector` with `report`, `render_text()`, `to_json()`
and `validation_errors()`, also available through `reader.inspect()`. It never
decodes pixels or traverses every chunk merely to compute a byte size. The
metadata-profile result describes the selected image's **attributes**, not a
whole-store audit or universal NGFF validation. Missing/unsupported interpretation
remains explicit in diagnostics.

Ordinary Zarr metadata intelligence (`inspect -i`) is deliberately not added by
this change: it errors before any model request. Existing explicit visual preview
and GeoJSON workflows can consume a Zarr Image using `--type ome_zarr`; their
existing opt-in transmission policy is unchanged.

## Remote access

```python
from omeify import OMEZarrReader

with OMEZarrReader("https://example.org/datasets/tonsil-v1/image.ome.zarr") as image:
    patch = image.read_region(0, 256, 0, 256, channels=0)
```

That URL is illustrative, not a hosted Omeify dataset. An ordinary static HTTP
server can serve an unsharded store. No directory listing, complete download or
persistent disk cache is required. A read transfers metadata and encoded chunks
needed for its region. Each reader has a bounded 64 MiB encoded-byte cache;
`cache_bytes=0` disables it and `image.clear_cache()` evicts it.

Library `storage_options` are passed deliberately to fsspec for transport settings.
Use plain store-root URLs, not credentials or signed query parameters embedded in
the URL. Those options are not serialized into operation reports. HTTP sessions
are owned and closed per reader. Other URI schemes, ZIP/reference stores and
remote writing are not advertised by this implementation.

Missing third-party chunks retain Zarr's legitimate fill-value semantics. HTTP
permission errors, transport failures and corrupt compressed chunks are not
converted into background pixels. We cannot infer whether a missing third-party
chunk was deliberately sparse or accidentally omitted during upload. Generated
output is therefore dense and checked before publication.

Local metadata/array references must stay beneath the store root, including
resolved symlinks. Metadata discovery is bounded to 8 MiB per node and 32 MiB per
session, with image/level/channel count limits. These are interpretation/cache
bounds, not a process-wide memory cap or protection against arbitrarily oversized
chunks from an untrusted source. The transport may receive an object before its
metadata-size limit can be checked. Read only trusted data and installed codecs.

## Safe writing and verification

Zarr writes are local and **create-new only**. Existing directories, including
empty ones or symlinks, are refused. `--overwrite` cannot authorize replacing a
Zarr store. A workflow's default TIFF overwrite permission is not such an
instruction. Directory replacement is not quietly implemented as delete-and-copy.

The writer stages beside the destination, closes encoding resources, verifies
metadata and the dense file/chunk inventory, and decodes representative pixels
at each level. Linux/macOS no-replace rename primitives or Windows no-replace
rename publish the result. Unsupported publication primitives/filesystems fail
rather than falling back to a race-prone sequence. Existing destinations survive
failed writes and destination races. This is namespace atomicity, not a power-loss
durability guarantee or a distributed cloud transaction.

The pixel checks are sampled, not an exhaustive integrity scan. Full-resolution
checks are bitwise, including signed zeros and NaN payloads; mean pyramids use
numeric equality with NaNs equal. The source reader is not an independent oracle.
Regression tests compare complete small arrays against independent expected data.
No scientific quality, calibration accuracy, or sharing clearance is certified.

Image and report publication remain separate transactions. A report failure after
a completed image does not delete that image. A shattered batch may retain earlier
successful stores if a later output fails.

## Migration

0.22 includes Zarr I/O in every installation without changing ordinary TIFF
storage defaults or numeric mutation algorithms. The shared image descriptor gains physical origins
and level offsets; existing callers need not supply them. There is no long/wide
parallel image API or file-only wrapper forced onto remote stores.

Convert report: `omeify.convert/2`; mutation: `omeify.mutate/2`; crop:
`omeify.crop/4`. Direct TIFF writer report identifiers stay unchanged. New Zarr
records use `omeify.zarr_write/1`, `omeify.zarr_collection/1`, and
`omeify.zarr_inspect/1`. Readers of the operation reports should accept the new
explicit format variants. `miti_header` is null for Zarr. Zarr source `size_bytes`
is null instead of triggering a full store traversal. The output size is measured
from the locally generated store. Actual array shape, chunks, codecs and native
layout are available in Zarr inspection; normalized RGB `YXS` is the Image's
regional-read layout, not the stored `CYX` component layout.

## Tests and references

```bash
python -m pytest -q tests/test_ome_zarr*.py
python -m pytest -q
```

Run `.[dev]` in Python 3.11 for the ordinary suite, including real TIFF codecs,
the OME XSD, all three NGFF readers, the canonical writer, and loopback HTTP tests.
Missing required dependencies fail rather than silently removing format coverage.
In a separate Python 3.12+ environment, install `.[dev,interop]` to include the
independent NGFF models. An absent optional model package is an explicit skip;
an installed but incompatible or broken model package must fail.

The [manual development checks](development.md) define the minimum-dependency and
newer-Python runs, plus installed-wheel smoke tests. They do not require GitHub
Actions or any other CI service. Skipped optional integrations do not certify
interoperability with those consumers.

- [OME-Zarr 0.4](https://ngff.openmicroscopy.org/specifications/0.4/index.html)
- [OME-Zarr 0.5](https://ngff.openmicroscopy.org/specifications/0.5/index.html)
- [OME-Zarr 0.6](https://ngff.openmicroscopy.org/specifications/0.6/index.html)
- [Zarr v3 storage](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html)
- [zarr-python storage](https://zarr.readthedocs.io/en/stable/user-guide/storage/)
- [Independent ome-zarr-models](https://ome-zarr-models-py.readthedocs.io/)
