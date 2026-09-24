# Images and sources: Omeify 0.18

Omeify has one library input model: an **Image** backed by an **ImageSource**.
File readers, array images, and procedural images use the same lifecycle,
regional reads, logical channels, and writer bridge. Omeify does not import
Mocktome or contain simulator-specific dispatch.

This is a deliberate breaking library cleanup. Conversion/mutation policies
and report structure are unchanged. Independently, the CLI now requires
`--output` / `-o` for the destination of `convert` and `mutate`; their Python
call signatures are unchanged. There are no deprecated forwarding APIs. File opening
now validates the complete advertised level description; inconsistent metadata
can fail earlier than a later regional read in the former reader implementation.

## Supported patterns

| Task | Canonical library operation |
|---|---|
| Read an OME intensity/RGB series | `with OMETiffReader(path, series=0) as image:` |
| Read categorical labels | `with OMETiffLabelReader(path, series=0) as labels:` |
| Read a vendor image | Its explicit reader, for example `AkoyaFusionQPTiffReader(path)` |
| Describe an existing intensity array | `MultichannelImage.from_array(array, axes="CYX", ...)` |
| Describe RGB or labels | `RGBImage.from_array(...)` or `LabelImage.from_array(...)` |
| Implement a procedural backend | Subclass `ImageSource`; implement `_read_region()` |
| Use that backend | `with MultichannelImage(source) as image:` (or RGB/label wrapper) |
| Select/reorder/assemble lazy scalar channels | `MultichannelImage.from_channels(channels, ...)` |
| Override calibration/names without changing pixels | `image.with_metadata(...)` |
| Read a rectangle | `image.read_region(y0, y1, x0, x1, level=0, channels=...)` |
| Materialize deliberately | `image.asarray(level=0)` or `channel.asarray(level=0)` |
| Write one image | `OMETiffWriter(path, ...).write(image, level=0)` |
| Write heterogeneous named images | `OMEMultiSeriesWriter(path, ...).write(series)` |
| Name an image in that collection | `OMEImageSeries(name, image, ...)` |
| Manage temporary-file lifetime | `with TemporaryOMETiffWriter(...) as writer:` |
| Inspect file structure | `reader.inspect()` or independently `TiffInspector(path)` |

Image metadata is available on the image itself. There is no public
`image.metadata` record duplicating those properties. `ImageMetadata` is the
provider's validated, immutable descriptor, not a second consumer API.
Physical TIFF planes are private encoder details, not an alternative provider
contract. The storage backend is not exposed through `image.source`.

## 1. Read, select channels, and write

Run in a kernel with this checkout installed. These examples assume the input
has a channel called DAPI and at least two scalar channels.

```python
from pathlib import Path
from omeify import MultichannelImage, OMETiffReader, OMETiffWriter

out = Path("Scratch/omeify_library")
out.mkdir(parents=True, exist_ok=True)

with OMETiffReader("input.ome.tif") as image:
    print(image)
    print(image.channel_names)
    print(image.shape, image.axes, image.dtype)
    print(image.pixel_size)
    print(image.levels)

    height, width = image.levels[0].spatial_shape
    patch = image.read_region(0, min(256, height), 0, min(256, width), channels=[0, 1])
    dapi = image.get_by_name("DAPI")
    dapi_patch = dapi.read_region(0, min(256, height), 0, min(256, width))

    # Select/reorder using lazy handles, not full-channel arrays.
    with MultichannelImage.from_channels([image[1], dapi]) as selected:
        report = OMETiffWriter(
            out / "selected.ome.tif",
            compression="Deflate",
            tile_size=512,
            overwrite=False,
        ).write(selected)

# patch and dapi_patch own their pixels and remain usable.
# image, selected, and dapi cannot perform further reads outside their lifetimes.
```

`from_channels()` can also assemble scalar channels from different open images.
It verifies matching dtype, selected-level spatial shape, and calibration (with
unit conversion). It does not prove that two samples are registered. Registration
and biological correspondence remain the caller's responsibility.

The result has one `CYX` level, even for one selected channel. Select `level=N`
when constructing it to use that resolution as the new base. Keep every parent
image open until use of the assembled image finishes. No full array is created.
Lists and tuples are both ordinary channel selections; strings are not.

An integer channel index, a unique name, and an exact OME channel ID are different
lookup operations: `image[index]`, `get_by_name(name)`, and `get_by_id(id)`.
Duplicate names are allowed but name lookup rejects ambiguity. Channel metadata is
read-only; rename with `with_metadata()` or `from_channels(channel_names=...)`.

`ImageMetadata.channel_metadata` and `channel.source_metadata` are recursively
immutable snapshots, not live views of caller-owned dictionaries. Records accept
plain Python `None`, `bool`, `int`, `float`, and `str` values, string-keyed mappings,
and lists/tuples. Mappings are copied into read-only proxies; lists/tuples become
tuples, recursively. Reusing one input container is allowed, but reference cycles,
more than 64 nested containers, non-string keys, arrays, and arbitrary objects are
rejected. Convert unsupported metadata explicitly before constructing a descriptor;
no value is silently stringified, and non-finite float declarations are not repaired.
This rule applies to channel metadata, not borrowed image pixels or ICC profiles.
The conversion/mutation reports still contain detached ordinary JSON containers.

## 2. Arrays from notebooks and other libraries

```python
import numpy as np
from omeify import MultichannelImage, OMETiffWriter, PixelSize

array = np.zeros((3, 512, 768), dtype=np.float32)
with MultichannelImage.from_array(
    array,
    axes="CYX",
    channel_names=("DAPI", "CD3", "CD20"),
    pixel_size=PixelSize(0.5, 0.5, "µm"),
    copy=False,
) as image:
    patch = image.read_region(0, 128, 0, 128)
    report = OMETiffWriter(
        "Scratch/signals.ome.tif", compression="Deflate", overwrite=False,
    ).write(image)
```

Constructors return **unopened** images, including `from_array()`. Use `with`, or
explicit `open()` and `try/finally: close()` when a notebook must keep an image
open across cells. Do not leave a live lazy viewer attached to an image whose
context has exited. File-free access does not necessarily mean lazy generation:
this example's full array already exists.

`copy=False` deliberately borrows large arrays; do not mutate or resize them
while observed. `copy=True` takes a full independent snapshot. Non-contiguous
NumPy views and memory maps work without copying the entire array by default.
Closing does not close a caller-owned memory map. A retained array is released
when its owning Python references are released, not merely because a context
ends. Masked arrays require an explicit mask/data policy upstream.

Declare `axes="YX"` for a single scalar intensity plane, and `axes="CYX"` for a
channel stack. Array size is not used to guess whether three dimensions mean RGB.
RGB and label constructors fix their respective `YXS` and `YX` layouts.

A list of separately stored arrays need not be stacked into a large copy. Wrap
each scalar array with `MultichannelImage.from_array(..., axes="YX")`, keep those
images open using `contextlib.ExitStack`, and assemble their `image[0]` handles
with `from_channels()`.

## 3. Current Mocktome outputs: mIF, H&E, and oracle labels

This is executable with Mocktome 0.4's existing eager API. It does **not** claim
that Omeify adds lazy sectioning or new `section.mif_image()` methods to Mocktome.

```python
from pathlib import Path
from mocktome import Mocktome, SpectralAcquisitionSpec, tonsil_like_recipe
from omeify import (
    LabelImage, MultichannelImage, OMEImageSeries, OMEMultiSeriesWriter,
    OMETiffWriter, PixelSize, RGBImage,
)

out = Path("Scratch/mocktome_omeify")
out.mkdir(parents=True, exist_ok=True)
recipe = tonsil_like_recipe(
    width_um=128, height_um=128, depth_um=32,
    microns_per_pixel=0.5, seed=7,
)
engine = Mocktome(recipe)
volume = engine.build_volume()
section = engine.section(volume, z_um=14, thickness_um=4)
mif = engine.render_mif(
    section, acquisition=SpectralAcquisitionSpec(seed=101, retain_spectral=False),
)
he = engine.render_he(section)
size = PixelSize(recipe.microns_per_pixel, recipe.microns_per_pixel, "µm")

with (
    MultichannelImage.from_array(
        mif.data, axes="CYX", channel_names=mif.channel_names,
        pixel_size=size, copy=False,
    ) as signals,
    RGBImage.from_array(he.rgb, pixel_size=size, copy=False) as brightfield,
    LabelImage.from_array(section.nucleus_labels, pixel_size=size, copy=False) as nuclei,
):
    report = OMETiffWriter(
        out / "mif.ome.tif", compression="Deflate", overwrite=False,
    ).write(signals)

    product_report = OMEMultiSeriesWriter(
        out / "paired_products.ome.tif", compression="Deflate", overwrite=False,
    ).write((
        OMEImageSeries("H&E", brightfield),
        OMEImageSeries("Nuclear truth", nuclei),
    ))
```

For a complete runnable example, including full small-image readback checks:

```python
from examples.mocktome_io import run_example
result = run_example("Scratch/mocktome_omeify")
```

Launch the notebook from the repository root to import `examples`. Mocktome must
be installed separately. It is not an Omeify dependency. Increase `width_um` and
`height_um` only after budgeting for Mocktome's eager intermediate arrays; wrapping
those arrays does not remove that allocation.

Oracle labels are still application truth. Omeify sees categorical integer
pixels, not cell geometry, cell counts, or projection policies. Nonzero
`background_label` is application metadata, not a standard OME-TIFF semantic tag;
record it in application provenance and supply it when reopening labels.

## 4. Implementing a genuinely regional provider

```python
import numpy as np
from omeify import ImageMetadata, ImageSource, MultichannelImage, PixelSize

class RampSource(ImageSource):
    def __init__(self):
        super().__init__(ImageMetadata(
            axes="CYX",
            shape=(2, 100_000, 120_000),
            dtype=np.dtype("float32"),
            channel_names=("X", "Y"),
            pixel_size=PixelSize(0.5, 0.5, "µm"),
        ))

    def _read_region(self, y0, y1, x0, x1, *, level, channels):
        shape = (y1 - y0, x1 - x0)
        fields = (
            np.arange(x0, x1, dtype=np.float32)[None, :],
            np.arange(y0, y1, dtype=np.float32)[:, None],
        )
        return np.stack([np.broadcast_to(fields[c], shape) for c in channels])

with MultichannelImage(RampSource()) as image:
    patch = image.read_region(20_000, 20_256, 50_000, 50_256)
    print(image.shape, patch.shape)
```

Only the requested rectangle exists. The smaller `examples/image_sources.py`
uses this mechanism to write a 3072-by-2048 procedural image and an RGB/label
multi-series file:

```python
from examples.image_sources import run_example
result = run_example("Scratch/omeify_sources", overwrite=False)
```

For a future Mocktome provider, the image's source may query a spatial index and
render intersecting geometry. That implementation belongs in Mocktome. Declare
`image_type="rgb"`, `axes="YXS"`, shape `(H, W, 3)` for H&E, or
`image_type="label"`, `axes="YX"`, integer dtype for oracle labels. Use the
matching semantic image wrapper. No TIFF byte stream is required.

Providers must implement `_read_region`, not bypass the common public validation
method. `_open()`/`_close()` are optional resource hooks. Metadata discovery must
not generate pixels. If a backend depends on local files, expose `backing_paths`
so writers can protect them from replacement, including symlink/hard-link aliases.

A source is a **fixed observation**, not a new exposure every time it is read.
Repeated reads, overlaps, reordered requests, and channel subsets must agree.
Coordinate-stable randomness, necessary filter halos, neighbor contributions,
and retained unmixing contributors are the provider's responsibility. Saving and
restoring specimen truth also belongs to the simulator, not this I/O interface.
Writers may request pixels repeatedly during pyramid construction and verification.

The contract promises regional requests, not inexpensive generation by an arbitrary
backend. It does not impose a global cache, simulator worker pool, thread-safety
promise, automatic normalization, or a spatial registration model.

## 5. Coordinates, channels, and levels

Bounds are half-open **integer pixel-edge coordinates in the selected level**.
Negative, reversed, fractional, boolean, and out-of-bounds requests fail. Empty
spatial requests return an empty correctly shaped array without calling the
backend. Returned arrays own their data and use native byte order, without a
numeric dtype cast. Signed zeros and NaN payloads are preserved by byte swapping.

| Layout | Full regional result | Meaning of `channels=` |
|---|---|---|
| `CYX` | `(selected_C, H, W)` | Logical channel indices; order and duplicates retained |
| `YX` | `(H, W)` | Only logical channel zero |
| `YXS` | `(H, W, 3)` | Only logical channel zero, containing all three RGB samples |

A single-channel OME-TIFF may reopen as `YX` even when supplied as singleton
`CYX`, because TIFF readers may squeeze that axis. Its logical channel is still
scalar `YX`; use the channel handle when comparing such round trips.

**RGB sample selection is not a channel selection.** To inspect red after a bounded
read, use `rgb_patch[..., 0]`. `image[0]` is the logical RGB channel. This is now the
same for OME-TIFF, vendor RGB, and array/procedural images.

Use the single resolution surface:

```python
level = image.levels[1]
print(level.shape, level.axes, level.spatial_shape)
print(level.pixel_size, level.downsample_yx)
```

`ImageLevel` contains explicit sampling information. A 5-by-7 raster reduced
2x has shape 3-by-4 and still has a 2x sampling step. Do not calculate sampling
scale from shape ratios. Unknown third-party level calibration stays `None`.
Metadata-only file open now validates all advertised levels, without decoding
pixels. Source levels describe one observation, not different microscope exposures.

`level_arrays=` and `levels=` on array construction describe existing multiscale
arrays; no hidden pyramid is generated. File `.native_axes`/`.native_shape`
describe original storage only. Native TIFF directory objects belong to file
inspection, not the generic `.levels` interface.

## 6. Metadata changes and writing policy

```python
with OMETiffReader("uncalibrated.ome.tif") as raw:
    with raw.with_metadata(pixel_size=PixelSize(0.5, 0.5, "µm")) as calibrated:
        OMETiffWriter("calibrated.ome.tif", compression="Deflate").write(calibrated)
```

`with_metadata()` creates a borrowed view, not an in-place edit. It supports final
ordered channel names, physical calibration, and an RGB ICC profile. `None` keeps
the existing value. Known level scales are recalibrated; unknown scales stay
unknown. It does not infer or remove missing experimental information.

Writers only take storage settings: compression, tiling, pyramid count, scratch
location, workers, output precision, software tag, UUID and overwrite policy.
Image axes, dtype, channel names and calibration belong to the Image. Writing an
uncalibrated image fails; dtype conversion and intensity normalization are not
implicit. RGB writing remains uint8; labels require lossless storage and nearest
pyramids. Explicit float32 mantissa trimming remains available.

Since 0.18.3, omitted compression and tile size resolve per image: RGB uses
**lossy JPEG quality 90 / 4:2:2 and 256 × 256 tiles**, while scalar, multiplex,
and label images use lossless LZW and 1024 × 1024 tiles. This is shared by
conversion, ordinary/temporary writes, and mixed-series writes, including RGB
OME-TIFF inputs. Pass `compression="Deflate"` (or another lossless codec) for
exact RGB samples. In mixed products, per-series compression overrides the
writer setting; explicit writer settings override automatic defaults.
Existing examples that explicitly select Deflate stay lossless.

A writer is an operation object; it opens and closes its real file resources
inside `write()`. It therefore does not need an otherwise empty outer `with`.
Temporary files do have a caller-visible lifetime:

```python
from omeify import TemporaryOMETiffWriter

with MultichannelImage.from_array(array, axes="CYX", pixel_size=size) as image:
    with TemporaryOMETiffWriter(compression="Deflate") as temporary:
        temporary.write(image)
        use_path = temporary.path
        # The path exists here; consume it before leaving the context.
```

Writers borrow images and leave them open on success or failure. They may evict
source decode caches between passes; eviction must not change pixels or close
resources. Source pyramids are rebuilt from the selected input level, not copied.
Temporary reduced TIFFs still consume scratch space. Bounded RAM does not mean
no temporary files.

Single-image and heterogeneous writers share the existing encoder, pyramid
arithmetic, metadata validation, precision handling and atomic installation.
`convert()` and `mutate()` now use this same Image write path. Their user-facing
operations remain distinct; CLI destinations use `--output` / `-o`.

## 7. Lifetime and ownership

An image owns its supplied backend by default. `owns_source=False` borrows an
already-open source. File-origin constructors own the file source. Metadata and
channel views own only their view backend and borrow parents.

Closing is idempotent. Old lazy handles do not reconnect to a reopened image.
Views must be recreated for the new session. Nested contexts on the same image
or source are rejected so an inner exit cannot silently close an outer scope.
Use separate borrowing wrappers when scopes genuinely differ.

No implicit `__array__` conversion exists. Whole-image reads are explicit methods.
A viewer requires an open source for future requests; a returned NumPy patch does
not. Keep source contexts around live viewers/analyses, or close them explicitly
when the interactive work is finished.

## 8. Migration from 0.17

| Removed/changed | Current form |
|---|---|
| `writer.write(array)` with writer-side image metadata | Semantic `.from_array(...)`, then `.write(image)` |
| `writer.write_image(image)` | `.write(image)` |
| Public `write_source` / `PlaneReaderSource` | External `ImageSource` implementation, wrapped as Image |
| `write_ometiff(...)` | `OMETiffWriter(path, ...).write(image)` |
| `OMEImageSeries.from_array/from_source/from_image` | `OMEImageSeries(name, image, ...)` |
| Public `ArraySource`, `OMETiffSource`, `ReaderImageSource` adapters | Named array constructors, file readers, or external `ImageSource` |
| `reader.as_image()` | Reader already is an Image |
| `image.metadata`, `image.source` | Image's direct metadata/read API; provider descriptor is separate |
| `channel.array` | `channel.asarray()` |
| `level_descriptors`, native TIFF `.levels` | `.levels` is always `ImageLevel` descriptors |
| `level_shape(n)`, `pixel_size_at_level(n)`, `level_downsample(n)` | `levels[n].shape`, `.pixel_size`, `.downsample_yx` |
| `level_count`, `is_pyramidal` | `len(image.levels)`, `len(image.levels) > 1` |
| `logical_channel_count`, `logical_channel_names`, `size_c` | `channel_count`, `channel_names`, `sample_count` |
| `is_rgb`, `is_label` | `image_type == "rgb"` / `"label"` |
| Writer `output_path` attribute | `writer.path` |
| `analyze_dtype_mutation(source, channel_names=..., source_dtype=...)` | `analyze_dtype_mutation(image, ...)`; metadata comes from the image |
| `reader.series_count` | `len(reader.series_names)` |
| `reader.inspection` / `inspection_report` | `reader.inspect()` / `reader.inspect().report` |
| `reader.tiff`, `reader.series`, physical `plane_readers` | Explicit file inspection; ordinary pixels use Image reads |
| Implicitly open `.from_array()` | Context manager or explicit `.open()` |
| `channels=[2, 0]` on RGB | Read logical channel 0, select RGB samples in returned patch |

Source-native axes remain available as `native_axes` and `native_shape` on file
readers. Generic axes/shape always describe the normalized product. Conversion
report fields still distinguish source layout from normalized layout.

### Tilework and Cadastre need coordinated migration

The available Tilework 6586040 adapter uses `is_rgb` and calls
`pixel_size_at_level()` with a fallback based on rounded image dimensions.
Change it to `image_type == "rgb"` and `reader.levels[level].pixel_size`, and
**remove that shape-ratio fallback**. Do not let a missing removed method quietly
select different calibration. Its existing bounded CYX/YXS-to-YXC conversion can
stay. Constructed Omeify Images can enter the same adapter as file readers; there
is no reason to create a simulator-specific Tilework branch.

The available Cadastre acfb391 output code creates `OMEImageSeries.from_source`
from `ArtifactSource` and `VisualizationStackSource`, typed as `PlaneReaderSource`.
Those providers need the new `ImageSource` descriptor/regional interface and an
open semantic wrapper; series construction becomes `OMEImageSeries(name, image)`.
Its existing bounded artifact reads, contour halos, JPEG display policy, and
scientific postprocessing should not be changed merely to migrate I/O.

Cadastre's tissue workflow and local-DAPI preparation also reopen paths. Migrating
its writer is not the same as enabling end-to-end file-free input. The latter
needs a deliberate Image-accepting workflow and lifetime for all stages that
reread the observation. Keep its CLI opening real files at the boundary.

This patch changes Omeify only. These sibling observations are based on the
available snapshots, not an assertion that their latest checkouts were modified
or tested. Migrate their call sites and dependency bounds before upgrading their
production environments. The same removed APIs may affect ttwhy, Fieldwork,
CellGate, or notebook helpers; search those callers rather than adding shims here.
