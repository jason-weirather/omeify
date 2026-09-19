# Images without files: Omeify 0.17

Omeify now separates **image meaning** from **pixel origin**. The semantic classes
are `MultichannelImage`, `RGBImage`, and `LabelImage`. Their backend is an
`ImageSource`. A source need not have a path, a TIFF byte stream, or a complete
array. No Mocktome-specific implementation or dependency lives in Omeify.

The single-image and heterogeneous writers accept these images through the
existing bounded plane-reader engine. The converter, metadata-minimization rules,
TIFF decoding, pyramid arithmetic, and atomic installation remain shared.

## 1. Notebook: wrap an array produced by any program

```python
from pathlib import Path
import numpy as np
from omeify import MultichannelImage, OMETiffWriter, PixelSize

out = Path("Scratch/image_sources")
out.mkdir(parents=True, exist_ok=True)
array = np.zeros((3, 1024, 1536), dtype=np.float32)

with MultichannelImage.from_array(
    array,
    axes="CYX",
    channel_names=("DAPI", "CD3", "CD20"),
    pixel_size=PixelSize(0.5, 0.5, "µm"),
    copy=False,
) as image:
    print(image)
    print(image.channel_names)
    patch = image.read_region(100, 356, 200, 456, channels=[0, 2])
    dapi = image.get_by_name("DAPI").read_region(100, 356, 200, 456)

    report = OMETiffWriter(
        out / "signals.ome.tif",
        tile_size=256,
        compression="Deflate",
        overwrite=False,
    ).write(image)
```

`patch` is `(2, 256, 256)` and `dapi` is `(256, 256)`. They remain usable after the
context exits; their contents do not alias source storage. This example borrows
the original array, so it is **file-free, not a lazy simulation**.

`from_array()` returns an already-open image for ordinary notebook use. A `with`
block is still useful for giving it a clear lifetime. `copy=False` avoids copying
the full array at construction. The caller must not mutate or resize a borrowed
array during an observation. `copy=True` explicitly takes a complete snapshot.
Closing an array source does not close a caller-owned NumPy memory map.

`PixelSize` requires X, Y, and a unit. Omit it for an uncalibrated image that you
only need to read. Writing still requires usable calibration, either in the image
or supplied explicitly to the writer. It is never guessed.

## 2. RGB and labels use the same pattern

```python
from omeify import RGBImage, LabelImage, PixelSize

size = PixelSize(0.5, 0.5, "µm")
rgb = np.zeros((1024, 1536, 3), dtype=np.uint8)
label_array = np.zeros((1024, 1536), dtype=np.uint32)

with RGBImage.from_array(rgb, pixel_size=size) as he:
    rgb_patch = he.read_region(100, 356, 200, 456)
    assert rgb_patch.shape == (256, 256, 3)
    assert he.logical_channel_count == 1
    assert he.sample_count == 3

with LabelImage.from_array(label_array, pixel_size=size) as labels:
    label_patch = labels.read_region(100, 356, 200, 456)
    assert label_patch.dtype == np.uint32
    assert label_patch.shape == (256, 256)
```

Label values are categorical identifiers, not intensities or an inferred object
count. No cell/nucleus/vessel knowledge is added to Omeify. Label-image writes
retain lossless storage and nearest-neighbor pyramids. A nonzero
`background_label=` may describe an in-memory label image, but ordinary OME-TIFF
does not encode that application policy. Supply it again when reopening with
`OMETiffLabelReader`, or record it in application provenance.

RGB means three interleaved samples, not three independent fluorescent stains.
A three-dimensional array passed to `MultichannelImage.from_array()` defaults to
`CYX`; it is never silently interpreted as RGB. Generic access supports real
numeric dtypes (and boolean intensity arrays); the TIFF writer retains its narrower
supported dtype rules, including **uint8-only RGB output**. No automatic conversion
is introduced. Masked arrays require an explicit mask/data policy upstream.

## 3. Real OME-TIFFs remain ordinary readers

Existing code keeps working:

```python
from omeify import OMETiffReader

with OMETiffReader("real.ome.tif") as image:
    patch = image.read_region(100, 356, 200, 456, channels=[0, 2])
    print(image.metadata)          # Immutable canonical ImageMetadata.
    print(image.level_descriptors)
```

The explicit composition form is also available:

```python
from omeify import Image, MultichannelImage, LabelImage, OMETiffSource

with MultichannelImage(OMETiffSource("real.ome.tif")) as image:
    patch = image.read_region(100, 356, 200, 456, channels=[0, 2])

# Auto-select RGBImage or MultichannelImage from the source's declared semantics.
with Image.from_source(OMETiffSource("real.ome.tif")) as image:
    print(type(image).__name__)

# A scalar TIFF is not automatically known to contain categorical labels.
with LabelImage(OMETiffSource("labels.ome.tif", image_type="label")) as labels:
    patch = labels.read_region(100, 356, 200, 456)
```

`OMETiffSource` reuses the existing OME readers and TIFF decoder. It is not a
second TIFF implementation. Its `path` belongs to that backend. The generic
`Image` and `ImageSource` do not require one.

`reader.as_image()` creates an opened semantic facade borrowing an already-open
reader. `ReaderImageSource` is the explicit adapter, also usable with the existing
vendor readers. `reader.source` exposes a session-bound borrowing adapter.

```python
with OMETiffReader("real.ome.tif") as reader:
    with reader.as_image() as image:
        patch = image.read_region(100, 356, 200, 456)
    assert reader.is_open          # The borrowing facade did not close it.
```

## 4. A genuinely on-demand source

External libraries should subclass `ImageSource`, declare `ImageMetadata`, and
implement `_read_region`. They need no TIFF tags, OME-XML, or plane-reader code.
This is the minimal contract for a procedural backend:

```python
import numpy as np
from omeify import ImageMetadata, ImageSource, MultichannelImage, PixelSize

class RampSource(ImageSource):
    def __init__(self):
        super().__init__(ImageMetadata(
            axes="CYX",
            shape=(2, 100_000, 120_000),
            dtype=np.dtype("float32"),
            image_type="multichannel",
            channel_names=("X", "Y"),
            pixel_size=PixelSize(0.5, 0.5, "µm"),
        ))

    def _read_region(self, y0, y1, x0, x1, *, level, channels):
        height, width = y1 - y0, x1 - x0
        x = np.arange(x0, x1, dtype=np.float32)[None, :]
        y = np.arange(y0, y1, dtype=np.float32)[:, None]
        fields = (x, y)
        return np.stack([
            np.broadcast_to(fields[c], (height, width)) for c in channels
        ])

with MultichannelImage(RampSource()) as image:
    patch = image.read_region(20_000, 20_256, 50_000, 50_256)
    print(image.shape)             # (2, 100000, 120000)
    print(patch.shape)             # (2, 256, 256)
```

That entire advertised image never exists. Only the requested rectangle is
calculated. The base class normalizes requests, handles empty regions, validates
returned shape/dtype, normalizes byte order without numeric casting, and returns
caller-owned contiguous arrays. Implement `_read_region`, not a replacement for
the defensive public `read_region` method.

Override `_open` and `_close` only when the backend needs handles/workers/caches.
A file backend may assign its descriptor to `self._metadata` in `_open`.
Metadata discovery must not decode/render pixels. `clear_cache()` is optional and
must not change the scientific observation or close its source.

For a simulated RGB product use `axes="YXS"`, `image_type="rgb"`, and shape
`(height, width, 3)`. For oracle labels use `axes="YX"`, `image_type="label"`, and
an integer dtype. The returned data must match those declared semantics.

### The provider's scientific obligations

A source describes one fixed observation during its open lifetime. Repeated reads,
overlapping regions, channel subsets, and different read orders must agree. A new
random exposure is a new observation, not a consequence of panning a viewer.
Use absolute coordinates or another equivalent deterministic construction for
noise and procedural textures. A selected output channel must still account for
unrequested latent contributors if the declared acquisition couples channels.

Spatial filters require the provider to evaluate its necessary halo and return
only the requested center. Neighbors with centers outside a region may contribute
inside it. Omeify neither knows nor guesses a simulator's physical support.

Writing may read a region more than once: pyramid construction, base writing,
and lossless spot verification all consume the same source. A non-repeatable
provider can fail verification. Numerical contract tests cannot establish whether
a custom provider has modeled biology or optics correctly.

The interface guarantees *regional requests*, not cheap execution inside an
arbitrary backend. No parallel-read safety is assumed; consumers must serialize
access unless their backend explicitly provides a stronger guarantee. There is
no universal cache, simulator worker pool, global normalization, or hidden
whole-image scan in the new facade.

## 5. Coordinates, axes, channels, and ownership

Bounds are **half-open integer pixel-edge coordinates in the selected level**:
`[y0:y1, x0:x1]`. Pixel centers are `(index + 0.5)` in that level. Bounds outside
the raster, reversed bounds, fractional indices, and boolean indices are rejected;
there is no implicit clipping or padding. Empty spatial regions return empty arrays
without calling the provider.

| Layout | `read_region` result | Meaning of `channels=` |
|---|---|---|
| `CYX` | `(selected_C, height, width)` | Logical channel indices; requested order and duplicates preserved. |
| `YX` | `(height, width)` | Only channel zero. |
| `YXS` | `(height, width, selected_S)` | RGB sample indices, preserving historical `OMETiffReader` behavior. |

For full RGB, omit `channels=`. The logical RGB channel is `image[0]`, which returns
all three samples. Existing vendor-reader RGB conveniences remain unchanged;
`reader.as_image()` provides the unified sample-selection behavior. Lazy `Channel`
access uses YX for scalar channels and YXS for the logical RGB channel.

There is no implicit NumPy `__array__` implementation. `asarray(level=...)` and
`channel.array` are deliberate whole-level/channel materializations. Writers do
not invoke either for an Image input.

Image wrappers own their source by default. Borrow a shared source explicitly:

```python
source = RampSource()
with source:
    with MultichannelImage(source, owns_source=False) as image:
        patch = image.read_region(0, 256, 0, 256)
    assert source.is_open
```

Closing is idempotent. Closed images cannot be read. Reopening creates a new
session: old Channel objects and writer-plane adapters are rejected rather than
silently rebound. Do not nest contexts on the same composed image or source;
use separate borrowing wrappers. Metadata describes the data, while session
generation is only a lifetime token, not a content hash or persistent specimen ID.

A backend depending on local files must expose those paths through
`backing_paths`. Image-aware writers reject replacing a declared source file,
including symlink/hard-link aliases. This is separate from OME metadata and never
causes source paths to be copied into the output header.

## 6. Explicit multiresolution metadata

Use `image.level_descriptors`, `image.level_shape(level)`,
`image.pixel_size_at_level(level)`, and `image.level_downsample(level)`.
The last returns `(scale_y, scale_x)` or `None`.

`ImageLevel` carries the index, canonical axes/shape, optional calibration, and
optional declared `downsample_yx`. Base scale is `(1, 1)`. Do not infer sampling
scale from rounded dimensions: a 5-by-7 raster reduced 2x becomes 3-by-4 but still
has a 2x sampling step. Missing calibration on a third-party TIFF pyramid remains
unknown. Source-backed images expose descriptors through `.levels` too; existing
readers keep their native TIFF `.levels` objects for compatibility. New consumers
should use `.level_descriptors` instead of TIFF-specific attributes.

All levels share a coordinate origin and axis directions; this interface does not
encode arbitrary affine registration. Available levels describe the same image,
not separate microscope acquisitions. An independently simulated higher-resolution
exposure should be a different source.

`ArraySource` supports supplied reduced arrays with `level_arrays=` and a complete
`levels=` descriptor sequence, including level zero. It does not generate a
pyramid at construction or guess reduced-level calibration.

## 7. One writer for any open Image

```python
from omeify import OMETiffWriter

with MultichannelImage(RampSource()) as image:
    # This would evaluate/write the entire declared image tile-by-tile.
    # Use the smaller examples/image_sources.py fixture for a quick test.
    report = OMETiffWriter(
        "Scratch/virtual.ome.tif",
        tile_size=1024,
        compression="Deflate",
        overwrite=False,
    ).write(image)
```

The writer infers image type, channel names, pixel calibration, and RGB ICC profile
from the image. An explicitly conflicting `image_type` or `axes` is rejected.
Explicit channel-name and pixel-size overrides remain possible and do not mutate
the source. It writes the selected `level=0` by default; selecting another level
uses that level's calibration as the new output base.

**Input pyramids are not copied.** Output pyramids are rebuilt by the existing
writer from the selected source level. The existing scratch-disk footprint remains;
streaming avoids a full base array in RAM, not all temporary storage. Labels still
require lossless compression and nearest reduction. RGB writing remains uint8.
NaN payloads and signed zero retain the existing fidelity/precision rules.

The writer borrows the image and leaves it open after success or failure. Keep
its context active until writing finishes. `write_source()` and `PlaneReaderSource`
remain available unchanged for existing advanced callers. They are the lower-level
writer bridge, not a second generic image architecture.

The existing output-path-first API is preserved:
`OMETiffWriter(path, ...).write(image)`. Earlier design sketches using a static
`OMETiffWriter.write(image, path)` were pseudocode, not the implemented signature.

For temporary files:

```python
from omeify import TemporaryOMETiffWriter

with MultichannelImage.from_array(array, pixel_size=PixelSize(.5, .5, "µm")) as image:
    with TemporaryOMETiffWriter(compression="Deflate") as writer:
        writer.write(image)
        temporary_path = writer.path
        # Consume the temporary file here, while this context remains active.
```

The convenience function also supports
`write_ometiff(path, image=image, ...)`. It is mutually exclusive with its existing
`channels=` array/channel-list argument.

## 8. Heterogeneous products in one OME-TIFF

```python
from omeify import OMEImageSeries, OMEMultiSeriesWriter, RGBImage, LabelImage

with (
    RGBImage.from_array(rgb, pixel_size=size) as he,
    LabelImage.from_array(label_array, pixel_size=size) as labels,
):
    report = OMEMultiSeriesWriter(
        "Scratch/products.ome.tif", compression="Deflate", overwrite=False,
    ).write((
        OMEImageSeries.from_image("H&E", he),
        OMEImageSeries.from_image("Nuclear labels", labels),
    ))
```

`from_image` discovers metadata but reads no pixels. It borrows the current image
session until `.write()` completes. Existing `from_array` and `from_source` APIs
are retained. Single- and multi-series output still use the same shared engine.
Application provenance remains explicit; simulator truth is not smuggled into
ordinary image headers.

## 9. Run the supplied example in Jupyter

From the repository root, in a kernel with the patched Omeify installed:

```python
from examples.image_sources import run_example
result = run_example("Scratch/omeify_sources", overwrite=False)
```

It writes a 3072-by-2048 procedural three-channel image and a separate RGB/label
multi-series file, and checks a full-resolution region against its file readback.
It needs neither Matplotlib nor a GUI. Restart an existing kernel after upgrading.

For Mocktome's *current eager results*, wrap `mif.data`, `he.rgb`, and the selected
truth-label array using the corresponding `from_array` factory. Supply channel
names and the section's calibration. This immediately removes the need for an
intermediate TIFF, but does not make Mocktome rendering lazy.

True Mocktome regional synthesis and its `mif_image`/`he_image`/`label_image`
conveniences must be implemented in Mocktome against this contract. They are not
methods added by this Omeify patch. Zarr/PNG backends, widgets, ImageView/crop
wrappers, arbitrary Z/T selection, and simulator-specific caching remain separate.

Version 0.17.0 retains the existing file reader, writer, converter, and CLI entry
points. Downstream package constraints such as `omeify<0.17` must be reviewed before
upgrading those environments. A consumer that accepts only a filename still needs
an image-input route; the new contract cannot silently change that consumer's API.
