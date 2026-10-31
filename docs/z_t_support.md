# Z/T source sessions (development branch)

This addition keeps **Image two-dimensional**. `Image`, `ImageSource`, channel
handles, `read_region()`, `asarray()`, and ordinary `writer.write(image)` retain
`YX`, `CYX`, and `YXS` behavior. A new **ImageDataset source session** describes
native Z/T axes and owns shared storage resources; it is not another renderer or
a multidimensional subclass of Image. No additional package or runtime framework
is required.

This is an unreleased branch implementation. The mandatory real Zarr, codec and
OME-XSD tests must pass in a complete development environment before release.
The dependency-limited implementation environment did not establish those
integration results. See the accompanying implementation test report for the
actual runs, not an implied compatibility certification.

## Open once, select ordinary Images

```python
from omeify import open_dataset

with open_dataset("input.ome.tif", input_type="ome_tiff") as dataset:
    print(dataset.axes, dataset.shape)
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

`plane_selection` contains Z, T, source-level index and a session-local identity.
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

`dataset.levels` describes native stored levels, with full source axes, shapes,
calibration and `plane_preserving`. `image.levels` remains the ordinary 2D level
surface. These two index spaces can differ.

* `dataset.plane(z=..., t=..., level=0)` includes only reduced XY levels that
  retain the same Z/T sampling.
* `dataset.plane(z=..., t=..., level=N)` for N > 0 explicitly selects a native
  coarse observation. Its Z index is **local to level N**; the Image has one level.
* `dataset.locate_plane(z=..., t=..., level=N)` reports the nearest native Z sample
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
    # Output axes: TZ + the plane's axes, including singleton T/Z.

    with dataset.select(z=slice(10, 20), t=slice(0, 5),
                        channels=[1, 0], bounds=(100, 356, 200, 456)) as selected:
        print(selected.shape)
```

Selections use half-open integer bounds and unit-stride ranges. `select()` keeps
all of an omitted Z/T axis; `plane()`, `iter_planes()` and `read_block()` require
explicit selection of a non-singleton axis. The CLI also requires explicit scope.
Selection and metadata-override dataset views expose one base level; output
pyramids are rebuilt. No cropped whole volume is materialized.

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

`prefetch([PlaneRequest(...)])` accepts explicit upcoming regions.
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

Both outputs require calibrated XY; non-singleton Z requires physical Z spacing.
Missing time intervals are not invented. Explicit overrides are borrowed views:
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

`max_block_bytes` preflights an estimate of working data and padded output chunks;
upstream codec scratch remains outside this estimate. The Zarr stack encoder is
serial at the outer block loop; Zarr owns internal I/O. Its ordinary writer
`max_workers` setting does not create an additional stack thread pool.

Whole-stack export **currently refuses any OME Plane acquisition measurements**
(DeltaT, ExposureTime or PositionXYZ) rather than dropping them or inventing a
regular grid. This is conservative, including regularly spaced measurements.
Reading those planes and explicit 2D export remain possible. Supporting retained
per-plane measurement output is unfinished work, not claimed here.

## CLI scopes and mutation

```bash
# Explicit complete dataset, fresh normalized Zarr, anisotropy-aware pyramid.
omeify convert input.ome.tif -o full.zarr --type ome_tiff \
  --output-format ome_zarr --all-planes --pyramid-axes auto --chunk-z 1

# Explicit T/Z subrange, with calibrated output supplied by the source.
omeify convert input.zarr -o subset.ome.tif --type ome_zarr \
  --t-range 0 4 --z-range 10 30 --compression Deflate --no-overwrite

# One plane also works for OME-TIFF in existing convert/crop/mutate workflows.
omeify convert input.ome.tif -o plane.ome.tif --type ome_tiff \
  --t-index 3 --z-index 12 --no-overwrite

# One fixed intensity mapping per channel across the selected T/Z extent.
omeify mutate float.zarr -o compact.zarr --type ome_zarr \
  --output-format ome_zarr --all-planes --dtype uint16 --range-mode full
```

`--all-planes` is exclusive with Z/T indices and ranges. With ranges, explicitly
select every remaining non-singleton axis. No-scope calls follow the previous
2D workflow; they never silently export plane zero or the complete volume.
Additional stack flags are `--z-spacing-um`, `--time-increment-s`,
`--pyramid-axes`, repeated `--pyramid-step Z Y X`, `--chunk-z`, and
`--max-block-mib`. Incompatible TIFF/Zarr-only options fail.

Integer mutation reuses the established mapping algorithms but scans the entire
selected T/Z extent for each channel. The scan's bounded deterministic sample
layout is recorded. This is not per-plane normalization. Explicit float32
mantissa trimming also applies to full stacks and rebuilt levels.

Library `dataset.select(bounds=..., channels=...)` provides rectangular stack
cropping. **Whole-stack polygon crop, concat, extract-rgb and heterogeneous stack
collections are not added to their CLI workflows.** Existing 2D workflows retain
their prior scope. Stack operation and writer reports have separate versioned
schemas; consumers should dispatch on `schema`, not assume the old 2D report.

## twhy integration boundary

This patch changes Omeify, not twhy key bindings or its render pump. Keep one
`open_dataset()` alive per opened source/series. Discover Z/T lengths from its
axes. When navigating, create a new plane Image and render session, preserve
camera/display settings, and use an immutable request identity including the
selection. Keep the owning dataset alive until old render work is finished.

Do not flush Omeify's cache on ordinary channel/level/plane changes: its budget
is now shared. twhy's existing `ViewSession` contains such eager cache clearing
and needs a companion adjustment. Continue guarding against stale completed
frames, and distinguish native coarse XYZ observations from exact sections.
Optional caller-directed lookahead can then target the current viewport.

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
necessary. No remote latency benchmark or twhy end-to-end certification is implied.
