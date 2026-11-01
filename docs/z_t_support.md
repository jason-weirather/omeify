# Z/T source sessions (0.23)

This addition keeps **Image two-dimensional**. `Image`, `ImageSource`, channel
handles, `read_region()`, `asarray()`, and ordinary `writer.write(image)` retain
`YX`, `CYX`, and `YXS` behavior. A new **ImageDataset source session** describes
native Z/T axes and owns shared storage resources; it is not another renderer or
a multidimensional subclass of Image. No additional package or runtime framework
is required.

The 0.23 release adds Z/T support to Omeify without changing the existing 2D
Image contract. Required real Zarr, TIFF codec, and OME-XSD tests are part of the
ordinary test suite, not optional integrations. See [development and release
checks](development.md) for supported Python/dependency environments, installed-wheel
verification, and independent interoperability tests. Synthetic test fixtures do
not establish compatibility with every third-party acquisition.

## Open once, select ordinary Images

```python
from omeify import open_dataset

with open_dataset("input.ome.tif", input_type="ome_tiff") as dataset:
    print(dataset.axes, dataset.shape)  # Logical TCZYX, TZYXS, or TZYX
    print(dataset.native_axes, dataset.native_shape)  # Original storage
    print(dataset.axis_info("Z"), dataset.axis_info("T"))

    # Explicit indices are required for each non-singleton Z/T axis.
    with dataset.plane(z=12, t=3) as image:
        print(image.axes, image.shape)   # CYX, YX, or YXS, never TCZYX
        patch = image.read_region(0, 128, 0, 128, channels=[0])
        identity = image.plane_selection

    # Same source handle, metadata, and cache; no reopen or dataset copy.
    with dataset.plane(z=13, t=3) as next_image:
        next_patch = next_image.read_region(0, 128, 0, 128, channels=[0])
```

Substitute `input_type="ome_zarr"` for a local Zarr directory or a plain
HTTP(S)/S3 store root. Existing remote options, authentication rules and path
containment checks remain in force. `group=` explicitly selects a Zarr child
image. `series=` selects a declared image/series, not a Z plane.

Constructors are lazy. Entering the dataset discovers metadata without decoding
pixels. `plane()` returns an unopened, immutable, borrowed Image. Closing that
Image does not close or clear the dataset. Keep the owning dataset open until
all its consumers have finished. Closing/reopening invalidates old views.
Returned NumPy patches own their values and remain usable after closure.

`plane_selection` contains Z, T, `native_level` and a session-local identity.
It is useful for render/cache request identities, not a persistent accession.
The identity propagates through ordinary single-observation metadata/crop/channel
views. It is absent where an ordinary Image does not have a dataset selection.

Existing reader calls remain valid:

```python
from omeify import OMETiffReader, OMEZarrReader

with OMETiffReader("input.ome.tif", z=12, t=3) as image:
    print(image.axis_info("Z"), image.source_levels)
    with image.plane(z=13, t=3) as other:
        patch = other.read_region(0, 64, 0, 64)
```

Both OME readers remain Images representing the constructor's fixed selection.
They expose a borrowed `dataset` session for navigation. `open_dataset()` is the
metadata-first entry point when dimensions are not yet known. There is no mutable
`set_z()`/`set_t()`, and neither API silently selects index zero for an ordinary
ambiguous 2D read. Absent axes are distinct from explicitly declared singleton
axes; do not pass an index for an absent axis.

## Source axes and native pyramid levels

`AxisInfo` contains `name`, `size`, `kind`, `unit`, `scale`, and `origin`.
`coordinate(index)` returns a coordinate only when scale and origin are both
known. Physical spatial quantities are normalized to micrometers, time to
seconds. Unknown calibration/origin remains unknown. Metadata-only origin zero
in a generated output establishes a local frame, not an acquisition measurement.

OME `PhysicalSizeZ` and `TimeIncrement` describe regular sampling. Numeric OME
Plane declarations are exposed separately with
`dataset.acquisition_plane(z=..., t=..., channel=...)`. Values retain their
declared units, including standard defaults. They are not silently fitted to a
regular grid or reinterpreted as intrinsic pixel origins. Duplicate, nonfinite,
or out-of-range acquisition records fail when that surface is requested.

`dataset.levels` describes native stored levels, with source axes, shapes,
calibration and `plane_preserving`. `image.levels` remains the ordinary 2D level
surface. These two index spaces can differ. The dataset API deliberately uses
`native_level=` when indexing a stored resolution; `Image.read_region(level=...)`
continues indexing the selected Image's own **plane-preserving** resolutions.

`dataset.axes`/`dataset.shape` are **canonical logical block axes**, regardless of
input format and source axis order:

| Pixel kind | Logical dataset axes | Corresponding 2D `Image.axes` |
|---|---|---|
| Multiplex/scalar with a logical C axis | `TCZYX` | `CYX` |
| Single scalar or categorical labels | `TZYX` | `YX` |
| Interleaved RGB (one logical channel) | `TZYXS` | `YXS` |

The time and depth axes remain present with size one in this normalized dataset
layout even when absent in the source. `read_block()` returns arrays in the same
logical order; channel subsets change the C extent without moving that axis.
`dataset.native_axes`/`dataset.native_shape` describe the **original physical
storage** (including arbitrary TIFF dimension orders and permitted Zarr axis
permutations). Borrowed metadata/selection views retain the original storage
identity; their logical `shape` describes the selected view. Source-native level
descriptors remain available through `dataset.levels`.

OME-TIFF writers use their existing specified T/C/Z plane ordering and per-plane
XY layout. OME-Zarr stack writers emit canonical *stored* `tczyx` arrays for NGFF
interoperability, including RGB encoded as three stored C components and labels
encoded with a singleton stored C. The public logical API represents RGB as S
samples, not three independent biomarker channels. Native and logical axes are
reported separately rather than falsely asserting the file uses a nonstandard
NGFF S axis.

* `dataset.plane(z=..., t=..., native_level=0)` includes only reduced XY levels that
  retain the same Z/T sampling.
* `dataset.plane(z=..., t=..., native_level=N)` for N > 0 explicitly selects a native
  coarse observation. Its Z index is **local to level N**; the Image has one level.
* `dataset.locate_plane(z=..., t=..., native_level=N)` reports the nearest native Z sample
  to a base-plane position, including the displacement. It does not resample,
  change the current plane, or claim a known reduction kernel for external data.

Thus XYZ-pyramidal Zarr input is readable without pretending that a Z-averaged
coarse voxel is an exact optical section. T and C sampling must still agree
across Zarr levels. Arbitrary registration, negative/reflected intrinsic grids,
scenes and custom axis names are not added. Existing XY transform/unknown-offset
rules, including conservative TIFF SubIFD scale recovery, remain in force.

The TIFF adapter uses an operational `TiffData` plane map independent of the
inspector's 4,096-plane audit budget. It honors dimension order and explicit
plane reordering and checks actual selected IFD/SubIFD metadata. This is not a
promise that scanning a million TIFF directory references is constant-time.
Companion-file loading remains disabled; multi-file OME-TIFF is not supported.

## Explicit blocks and selections

```python
with open_dataset("input.zarr", input_type="ome_zarr") as dataset:
    block = dataset.read_block(
        0, 64, 0, 64, z=slice(10, 13), t=2,
        channels=[1, 0, 1], max_bytes=16 * 1024 * 1024,
    )
    # Output axes: TCZYX / TZ YXS / TZYX, including singleton T/Z.
    # Dataset.read_block always agrees with dataset.axes.

    with dataset.select(z=slice(10, 20), t=slice(0, 5),
                        channels=[1, 0], bounds=(100, 356, 200, 456)) as selected:
        print(selected.shape)
```

Selections use half-open integer bounds and unit-stride ranges. `select()` keeps
all of an omitted Z/T axis; `plane()`, `iter_planes()` and `read_block()` require
explicit selection of a non-singleton axis. Conversion and mutation keep the
full extent of an omitted axis; other 2D read workflows still select one plane.
Rectangular/channel/Z/T selections expose only their newly selected base level;
output pyramids are rebuilt. **Metadata-only** dataset views preserve every native
resolution and the exact-plane compatible XY pyramid. X/Y and Z/T sampling
recalibrations use declared native level ratios and known coordinate offsets;
unknown phase and absent physical origins remain unknown. No cropped whole
volume is materialized.

`read_block()` bounds the returned allocation, not upstream codec scratch or
source chunk size. Reads still return owned, native-endian arrays. Channel order
and duplicates are retained. RGB remains one indivisible logical channel.

Arrays can supply synthetic or application data without a file intermediate:

```python
import numpy as np
from omeify import AxisInfo, ImageDataset, PixelSize

values = np.arange(2 * 3 * 5 * 32 * 48, dtype=np.uint16).reshape(2, 3, 5, 32, 48)
with ImageDataset.from_array(
    values, axes="TCZYX", channel_names=("A", "B", "C"),
    pixel_size=PixelSize(0.5, 0.5, "µm"),
    axis_info=(AxisInfo("Z", 5, "space", "µm", 2.0),
               AxisInfo("T", 2, "time", "s", 30.0)),
) as dataset:
    with dataset.plane(z=2, t=1) as image:
        np.testing.assert_array_equal(image.asarray(), values[1, :, 2])
```

`copy=False` borrows arrays and requires callers not to mutate them while open.
`image_type="rgb"` requires an S axis of length three, not an inferred C=3;
labels require categorical integer values and one logical channel.

## Dataset sources versus 2D ImageSource

`ImageSource` remains the established public 2D provider interface: its
`read_region()` validates one Image's coordinates and returns owned YX/CYX/YXS
arrays. `ImageDataset` instead requires an internal, structurally checked
`DatasetSource` protocol (`dataset_levels`, `plane_metadata`, `read_plane_region`,
and session lifecycle). It does **not** require a source to expose an ambiguous
2D `read_region()` while Z/T remain unselected.

The file-backed TIFF and Zarr source classes implement **both** capabilities,
serving existing fixed-plane readers and new dataset sessions. Native array-backed
dataset sources and borrowed dataset views inherit `DatasetSourceBase` only.
`dataset.plane()` is the explicit bridge: it builds an ordinary 2D `ImageSource`
and hence an ordinary `Image`. This keeps existing custom 2D providers and the
reader/writer contract unchanged.

Both interfaces share a **private `_SourceSession` lifecycle** for acquiring and
releasing resources, validating startup, session generations and context safety.
The base is intentionally metadata-neutral and does not expose a pixel-read API.
`ImageSource` alone requires a fixed 2D `ImageMetadata` descriptor. A
dataset-only source instead publishes ordered native `DatasetLevel` descriptors;
these hold the validated geometry and plane-description templates from which
`plane_metadata(z=..., t=..., level=...)` supplies the selected observation.
It does not also carry a redundant `_metadata`/`metadata` record. Borrowed
dataset views preserve their parent session without owning its file or store.

`OMETiffReader` and `OMEZarrReader` remain convenience **single-observation
Images**, not a separate decoding implementation: each delegates to the same
underlying TIFF/Zarr source's `read_plane_region()`. Their `.dataset` property
borrows that very source and cache rather than opening another backend. Closing
the reader ends the physical source session and invalidates its borrowed views.
Direct `open_dataset()` is the preferred metadata-first entry point for Z/T
navigation, while ordinary procedural 2D `ImageSource` providers remain first
class with no Z/T requirements. No new runtime dependencies are introduced.

## Cache ownership and resource limits

A file/store session shares its cache across selected Z/T planes and levels.
TIFF entries are decoded physical segments. Zarr entries include native decoded
chunks, so a chunk spanning channels or several Z/T positions can be reused
without decoding it again. Zarr continues to own codecs, shard interpretation and
range I/O; Omeify does not implement its own codec stack.

`cache_bytes` is an aggregate retained-payload allowance, default 64 MiB.
TIFF uses the allowance for decoded segments; Zarr divides it between encoded
responses (one quarter) and decoded native blocks (three quarters). Speculative
encoded responses are not admitted into the encoded cache. At most a quarter of
the decoded tier is reserved for speculative entries; they cannot evict demand
entries. Large entries bypass retention. `cache_bytes=0` disables retention.

The limit does **not** include metadata, caller-owned output arrays, native
codec temporaries, Python bookkeeping, or running allocations. A huge source
chunk/strip can still be expensive to decode. This is not a process RSS or HTTP
transfer cap. Source content must remain immutable for a session; publish remote
data at immutable versioned prefixes. Cross-session disk caching is not added.

`cache_info()` exposes local hit/load/wait/eviction/residency counters; Zarr also
reports encoded reads and bytes. `clear_cache()` cancels pending predictions and
waits for current source access before eviction. Concurrent identical block
requests share a cache load. Source accesses are conservatively serialized where
needed for backend safety. Ordinary reads do not start speculative workers.

## Optional read-ahead

```python
from omeify import PlaneRequest, PrefetchPolicy, open_dataset

with open_dataset("input.zarr", input_type="ome_zarr") as dataset:
    request = PlaneRequest((100, 356, 200, 456), z=12, t=3, channels=(0, 1))
    futures = dataset.hint_navigation(
        request, axis="Z", direction=1, count=1,
        policy=PrefetchPolicy(max_pending=2, max_bytes=16 * 1024 * 1024),
    )
    # An interactive application normally keeps processing input here.
    # Explicit test/batch callers may wait and observe speculative errors:
    for future in futures:
        future.result()
    print(dataset.prefetch_info(), dataset.cache_info())
```

`prefetch([PlaneRequest(...)])` accepts explicit upcoming regions. The request
field `native_level` follows the dataset API, not the 2D Image `level` keyword.
`hint_navigation()` predicts only adjacent Z/T viewports at the caller's requested
level/channels. `hint_pan()` predicts leading XY strips. Each submission replaces
older pending predictions. Request count and estimated decoded native-block work
are bounded. A request beyond those limits raises rather than silently expanding
them. Default readers remain demand-only.

One worker is created lazily. Demanded dataset reads have priority over pending
speculation; an already running native operation is not preemptible. Cancellation
is cooperative, and closure joins the worker before closing the source. External
ImageSource implementations must opt into worker use. Prefetch failures remain
in completion futures and a bounded diagnostic history; they never become valid
background pixels or erase errors in a later demanded read.

The initial predictor is deterministic and caller-directed. Adaptive learning,
network-byte quotas, process-wide memory admission, persistent caches and
throughput/latency guarantees are **not** implemented. There are no AI calls,
additional model permissions, or pixel transmissions to a model.

## Whole-stack writers and XYZ construction

Use `write_dataset()` for an explicitly selected dataset. `write(image)` still
writes only its 2D observation. The writers borrow open datasets and do not
close them or flatten their Z/T axes into image series.

```python
from omeify import OMEZarrWriter, OMETiffWriter, open_dataset

with open_dataset("input.ome.tif", input_type="ome_tiff") as dataset:
    OMETiffWriter("stack.ome.tif", compression="Deflate", overwrite=False,
                  tile_size=128, pyramid_levels=2).write_dataset(dataset)
    OMEZarrWriter("stack.zarr", tile_size=128, pyramid_levels=3).write_dataset(
        dataset, pyramid_axes="auto", chunk_z=1,
        max_block_bytes=64 * 1024 * 1024,
    )
```

Both low-level `write_dataset()` writers require calibrated XY. Unknown Z spacing
and time intervals remain unknown: the TIFF header omits the corresponding optional
OME fields, and NGFF records unitless index axes rather than pretending they are
micrometers/seconds. **Higher-level `convert` and `mutate` refuse multi-Z output
without physical Z spacing unless the caller explicitly declares
`--z-spacing-ordinal`.** Explicit `--z-spacing-size` and optional
`--z-spacing-unit` override physical Z spacing using the same length units as XY;
the default input unit is µm. The physical value is canonicalized to µm at the
writer boundary. This gate is a workflow policy, not a new requirement for
`open_dataset()`, `dataset.plane()`, or direct `write_dataset()`. XYZ automatic
physical-anisotropy pyramids still require calibrated Z, even if ordinal mode
was declared. Low-level metadata overrides remain borrowed views:
`dataset.with_metadata(pixel_size=..., z_spacing_um=..., time_increment_s=...)`.
Regular intrinsic Z/T origins are kept separately from acquisition measurements.

TIFF uses the existing tiled BigTIFF writer engine, one full-resolution IFD per
T/C/Z plane, and each plane's XY SubIFDs. Lazy plane-reader sequences bound live
handles. Existing compression, ICC handling, float precision, scratch, metadata
validation and atomic-publication rules are reused. Multi-file output is absent.

Zarr writes fresh NGFF 0.5/Zarr v3 arrays with explicit stored `tczyx` axes and
logical RGB/label declarations. It streams one working block at a time, writes
dense chunks, verifies structure and representative pixels, and publishes only a
new local directory. `chunk_z` sets chunk depth; T and C chunk extents are one.
Output sharding is not added. Chunk size and pyramid reduction remain independent.

Pyramid policies are explicit:

| Policy | Behavior |
|---|---|
| `xy` (default) | Independent XY reduction, retaining all Z/T samples |
| `xyz` | Reduce Z and XY spatial sampling, never T/C/RGB samples |
| `auto` | Require calibrated Z/Y/X; reduce finer spatial sampling first, delaying coarse Z |
| `pyramid_factors=((1,2,2),(2,2,2))` | Explicit per-level ZYX factors, each 1 or 2; mutually exclusive with level count |

Explicit factors reducing Z require `pyramid_axes="xyz"` or `"auto"`.
Mean reduction sums a complete XYZ bin and rounds integers once, ties to even;
it does not round intermediate XY planes before averaging Z. Incomplete edge
bins use their actual contributors. Nearest reduction is compulsory for labels.
Pure XY reduction delegates to the established 2D arithmetic. Known nominal
voxel-center transforms accompany each generated level. Source pyramids are
rebuilt rather than copied. Temporal reduction is not a pyramid option.

A requested reduction must change at least one spatial extent. Once a Z, Y, or X
axis reaches length one, automatically planned steps leave that axis unchanged;
explicit factors of two on singleton axes are rejected. Requests for more levels
than the permitted dimensions can support fail instead of writing duplicate
resolutions or falsely increasing their sampling. XYZ floating-point means retain
IEEE NaN/infinity behavior and recover finite float64 accumulation overflow.

`max_block_bytes` preflights an estimate of working data and padded output chunks;
upstream codec scratch remains outside this estimate. The Zarr stack encoder is
serial at the outer block loop; Zarr owns internal I/O. Its ordinary writer
`max_workers` setting does not create an additional stack thread pool.

Whole-stack output curates one optional per-plane acquisition measurement:
**`DeltaT`**, normalized to seconds and associated with its exact logical
T/C/Z observation. It may describe irregular sampling or channel-dependent
acquisition delays. A known `TimeIncrement` is sufficient when no `DeltaT` is
recorded; missing timing does not invent an acquisition clock or prevent ordinal
T browsing. For TIFF, generated OME `Plane` nodes carry `DeltaT` and explicit
units; a post-write check confirms their coordinates and values. For NGFF 0.5,
which does not offer a per-plane timestamp axis, a small, validated
`omeify.plane_delta_t/1` attribute retains the same sparse T/C/Z/seconds records
without falsely claiming an irregular uniform axis transform. An Omeify reader
recovers that extension; another NGFF viewer may ignore it. No source
`ExposureTime` or `PositionXYZ` is copied, and their mere presence cannot block
conversion. Their acquisition context belongs in external curated records.

The output contracts are the required OME XSD and
`miti_ome_tiff_header.schema.json` for TIFF,
`ome_zarr_dataset_profile.schema.json` for NGFF attributes, and
`stack_geometry.schema.json` for intrinsic Z/T coordinate origins;
`dataset_write_report.schema.json` describes both outputs.

## CLI scopes and mutation

```bash
# Whole selected series is preserved by default (no all-planes flag).
omeify convert input.ome.tif -o full.zarr --type ome_tiff \
  --output-format ome_zarr --pyramid-axes auto --chunk-z 1

# Restrict Z only; omitted T remains at its complete source extent.
omeify convert input.zarr -o subset.ome.tif --type ome_zarr \
  --z-range 10 30 --compression Deflate --no-overwrite

# For a source without a physical Z step, explicitly authorize index-only Z.
omeify convert ordinal.zarr -o ordinal.ome.tif --type ome_zarr \
  --z-spacing-ordinal --compression Deflate

# Or supply a measured/known Z step with an independent physical unit.
omeify convert ordinal.zarr -o calibrated.ome.tif --type ome_zarr \
  --z-spacing-size 500 --z-spacing-unit nm --compression Deflate

# Restrict both Z and T to export an ordinary 2D observation.
omeify convert input.ome.tif -o plane.ome.tif --type ome_tiff \
  --t-index 3 --z-index 12 --no-overwrite

# One fixed numerical mutation mapping per channel across all Z/T by default.
omeify mutate float.zarr -o compact.zarr --type ome_zarr \
  --output-format ome_zarr --dtype uint16 --range-mode full
```

The opt-in `--preserve-rois` (Python `preserve_rois=True`) is limited to
OME-TIFF -> OME-TIFF operations that retain the complete source Z/T extent and
unchanged X/Y/C dimensions. Linked ROIs, including plane-specific TheZ/TheT/TheC,
are retained without coordinate transformation and checked during TIFF readback.
Partial Z/T selections are refused with this flag, rather than misleadingly
reassigning ROI plane coordinates. Without the flag, source ROIs are omitted.
OME-Zarr is not an ROI-preserving output in this release.

Omitted Z/T selectors retain every plane of the selected image/series. A single
index restricts that axis while retaining the other; two indices select a 2D
observation. A range is half-open and can be combined with an index on the other
axis. Singleton/absent Z/T input retains the established 2D conversion path and
report schema. `--all-planes` and `--z-spacing-um` are not supported. Other workflows
requiring one 2D Image still reject unselected non-singleton axes. With multi-Z
output, missing physical Z spacing fails before writing unless the user chooses
`--z-spacing-ordinal`; a physical override is `--z-spacing-size VALUE`
`[--z-spacing-unit UNIT]` (default µm). The two choices are mutually exclusive.
An explicit ordinal claim is an interpretation decision and is recorded in the
operation report, not an invented instrument measurement. Other stack flags are
`--time-increment-s`, `--pyramid-axes`, repeated `--pyramid-step Z Y X`, `--chunk-z`,
and `--max-block-mib`. Incompatible TIFF/Zarr-only options fail.

Integer mutation reuses the established mapping algorithms but scans the entire
selected T/Z extent for each channel. The scan's bounded deterministic sample
layout is recorded. This is not per-plane normalization. Explicit float32
mantissa trimming also applies to full stacks and rebuilt levels.

Library `dataset.select(bounds=..., channels=...)` provides rectangular stack
cropping. **Whole-stack polygon crop, concat, extract-rgb and heterogeneous stack
collections are not added to their CLI workflows.** Existing 2D workflows retain
their prior scope. Stack operation and writer reports have separate versioned
schemas; consumers should dispatch on `schema`, not assume the old 2D report.

## Tests and outstanding release gates

```bash
python -m pip install -e '.[dev]'
python -m pip check
python -m pytest -q tests/test_dataset*.py tests/test_volume_pyramid.py
python -m pytest -q
python -m ruff check --select E9,F63,F7,F82 .
python -m build
```

Required real Zarr tests cover 0.4/0.5/0.6, axis permutations, chunks spanning
T/C/Z, sharded input, XYZ selection, and loopback HTTP reuse. Public writer tests
require actual codecs and OME-XSD validation; missing dependencies fail, not skip.
The separate TIFF-engine tests use actual uncompressed files below the public
XSD boundary and explicitly do not claim public writer validation.

Independent tiny arrays encode T/C/Z positions. Tests cover all six OME dimension
orders, reversed TiffData mappings, 4,097 planes, borrowed lifetimes, cache
coalescing/eviction, explicit prefetch, odd XYZ reductions, numeric overflow,
fixed stack mutation and fresh metadata. The existing minimum-Python, wheel,
optional independent NGFF-model/Bio-Formats and real public dataset gates remain
necessary. No remote latency or end-to-end application performance benchmark is implied.
