# Cropping and visual region questions (0.20)

Two explicit operations compose through **image-pixel GeoJSON**:

```text
inspect -i -q ... --geojson  ->  full-resolution pixel rectangles  ->  crop --geojson ...
```

The visual command proposes locations. The crop command does not use a model;
it validates geometry and writes pixels through Omeify's existing writer.
Ordinary inspection and metadata questions remain metadata-only.

## Crop known coordinates

```bash
omeify crop slide.ome.tiff -o Scratch/regions.ome.tiff \
  --bounds 10000 5000 12048 7048 -v
```

This writes exactly `Scratch/regions.ome.tiff`, a 2048 by 2048 rectangle. CLI bounds are
**X0 Y0 X1 Y1**, integer, half-open, in the selected series' **level-zero pixels**.
They are not `x,y,width,height`, physical units, or pyramid-level coordinates.
No source pixels are rescaled or masked. Output encoding can still be lossy.

`--type` defaults to `ome_tiff`; supported vendor profiles are the same as
`convert`. `--series` defaults to zero. All channels of the selected series are
exported. Scalar/label output retains lossless LZW / 1024-pixel tiles; RGB retains
JPEG quality 90 / 4:2:2 / 512-pixel tiles. Use `--compression Deflate` for exact
RGB sample values. The output pyramid is rebuilt from the cropped base.

Physical pixel calibration is preserved and remains required for export.
`--pixel-size-x`, `--pixel-size-y`, and optionally `--pixel-size-unit` supply an
explicit override. Declare a categorical OME series with `--labels` so the
writer uses lossless storage and nearest-neighbor pyramids. Labels are not
renumbered. Storage, scratch, workers, and verbosity options appear in `crop --help`.

## Crop GeoJSON

```bash
omeify crop slide.ome.tiff -o Scratch/regions.ome.tiff --geojson regions.geojson -v
omeify crop slide.ome.tiff -o Scratch/regions.ome.tiff --geojson regions.geojson --shatter by_index
omeify crop slide.ome.tiff -o Scratch/regions.ome.tiff --geojson regions.geojson --shatter by_name
```

Exactly one of `--bounds` or `--geojson` is required. Without `--shatter`,
`-o` / `--output` is the **exact destination filename**, not a directory. One or
many ROIs go into that one file. No extension, index, or model-generated suffix
is added. Prefer an explicit `.ome.tif` or `.ome.tiff` filename for viewers.
The crop report goes to stdout; progress goes to stderr.

Supported inputs are a Feature, FeatureCollection, bare Polygon/MultiPolygon,
polygon GeometryCollection, or a Feature array. `--geojson -` reads stdin.
GeoJSON positions are `[x,y]`, with origin at the top left of the full-resolution
image, x rightward, y downward. This is the
[QuPath image-coordinate convention](https://qupath.readthedocs.io/en/stable/docs/advanced/exporting_annotations.html),
**not geographic RFC 7946 longitude/latitude**. Geographic CRS declarations,
3D positions, null geometries, points, and lines are not supported.

Every object becomes its **bounding rectangle**. Polygon interiors, holes,
and disjoint parts are not masked; a multipart Feature stays one ROI covering
its parts. A top-level GeometryCollection supplies one ROI per member.
Envelope/coordinate validation is not polygon-topology validation.

Example:

```json
{
  "type": "FeatureCollection",
  "features": [
    {"type": "Feature", "properties": {"name": "left"},
     "geometry": {"type": "Polygon", "coordinates": [
       [[100,100],[700,100],[700,600],[100,600],[100,100]]
     ]}},
    {"type": "Feature", "properties": {"name": "right"},
     "geometry": {"type": "Polygon", "coordinates": [
       [[900,100],[1500,100],[1500,600],[900,600],[900,100]]
     ]}}
  ]
}
```

With `-o Scratch/regions.ome.tiff`, the default writes **one file** containing
series 0 named `01 - left` and series 1 named `02 - right`. Each ROI remains an
independent image, retaining its own dimensions and pyramid; ROIs are not packed
into a channel stack, stitched into a mosaic, or enclosed in one union rectangle.

**Input order is authoritative.** The first GeoJSON member becomes series 0,
the second becomes series 1, and so on. Names, Feature IDs, `properties.index`,
spatial location, and area do not sort the output. The prefix in a series name
is its **one-based input index**, padded to at least two places, with more digits
when the total ROI count requires them. `properties.name` supplies the label; a bare geometry may carry `name`.
Classification labels are not substituted for names. An unnamed region gets
`NN - ROI`. Duplicate annotation names remain separate, uniquely indexed series.

### Explicit file splitting

`--shatter` takes one required value and writes only the split files, not an
additional combined file:

| Mode | Files for the example above | Contents |
|---|---|---|
| No `--shatter` | `regions.ome.tiff` | All ROIs in input order |
| `--shatter by_index` | `regions-01.ome.tiff`, `regions-02.ome.tiff` | One ROI per file, regardless of names |
| `--shatter by_name` | `regions-left.ome.tiff`, `regions-right.ome.tiff` | One file per name; equal names share a file |

Name grouping compares names after stripping surrounding whitespace. Unnamed
objects receive individual files using their padded input indices. Groups follow
their first occurrence, with members retained in original input order. For example,
`right, left, right` becomes two files: the `right` file contains input ROIs 1 and
3, and the `left` file contains ROI 2. Their series names retain those **global
input indices**, while `output_series` is zero-based within each output file.

A supplied `.ome.tiff`, `.ome.tif`, `.tiff`, or `.tif` extension is preserved;
shatter inserts its suffix immediately before it. With a bare prefix, shatter
adds `.ome.tiff`. Thus `sample.he.ome.tiff` becomes `sample.he-01.ome.tiff`,
not `sample.he.ome.tiff-01.ome.tiff`. No path is derived from a region name except
in explicit `by_name` mode.

Unsafe filename characters become hyphens **only in `by_name` mode**. Distinct
names that collide after sanitization or case folding are rejected, including
collisions with an unnamed object's index. Empty unsafe names or names with more
than 120 UTF-8 bytes after sanitization are rejected too. Omit shatter or select
`by_index` to avoid making filenames depend on those names. Series labels retain
the annotation text rather than the filename slug.

Float geometry bounds round outward: floor the minima, ceil the maxima. The
default rejects out-of-bounds ROIs; `--clip` opts into intersection with the image.
An empty intersection always fails. Every ROI and all destination paths are
preflighted before writing pixels. Overwrite is on by default; `--no-overwrite`
refuses existing entries, including dangling symlinks. Input image/GeoJSON aliases
and colliding output paths are protected. Each file installs atomically, but a
shattered batch is not a multi-file transaction: completed files remain after a
later I/O failure. The default combined file is installed as a single atomic
product. Existing unrelated files, including old shattered outputs, are not deleted.

These are named crop products written through `OMEMultiSeriesWriter`, including
single-ROI files. Each file carries generated crop provenance: input series and
shape, original requested bounds, exported bounds, one-based input indices,
zero-based output series indices, original names, output series names, the shatter
mode, and source XY offsets. A crop's local `(0,0)` maps to that source offset. No source
filename or arbitrary source annotation properties are copied into this metadata.
Retain the original GeoJSON for polygon geometry and other annotation properties.

## Crop 0.20 migration

The new default is deliberately **one requested file**, replacing name-based
file splitting. The `--naming` option and Python `naming=` argument are removed;
no compatibility alias can silently retain the old default. Use
`--shatter by_name` / `shatter="by_name"` for the former name-grouped export or
`--shatter by_index` / `shatter="by_index"` for per-ROI files.

Python's second parameter is now `output_path` rather than `output_base`.
Positional calls still work, but without shatter they write the exact requested
path. Change bare prefixes to complete filenames for combined output. Crop
reports and embedded crop provenance use `omeify.crop/2`; the report includes
`output_path` and `shatter` instead of `naming`, and each region adds `output_name`.
No image, geometry, or previously written file is modified by upgrading.

## Ask for visual GeoJSON

Install the existing intelligence extra in a Python 3.13+ environment:

```bash
python -m pip install -e '.[intelligence]'
```

The configured Sheetbend model must actually accept images and declare `vision`,
`json_schema`, and `system_prompt` capabilities. Add `vision = true` to its existing
capability declaration only when supported by that deployment. An architecture
with vision support can still be deployed as a text-only endpoint. No alternate
endpoint settings, model download, automatic probe, retry, or fallback are added.
The configured model and reasoning defaults remain Sheetbend's responsibility.

```bash
omeify inspect slide.ome.tiff -i --geojson \
  -q "Return only the right tissue, with the name right." > right.geojson

omeify crop slide.ome.tiff -o Scratch/regions.ome.tiff --geojson right.geojson -v
```

`inspect --geojson` requires both `-i` and `-q`. Stdout contains **only one
FeatureCollection**, without the usual inspection tree or prose answer. Progress
is on stderr. The existing `--output` / `-o` can also save the GeoJSON; an existing
report is not touched when the request or validation fails. Explicit metadata
`--detail`, `--max-text-length`, and `--intelligence-max-chars` do not apply and
are rejected in this mode.

For a fixed-size ROI:

```bash
omeify inspect slide.ome.tiff -i --geojson \
  --roi-size 2048 2048 \
  -q "Choose one ROI centered on the outer edge of the right tissue; name it edge." \
  > edge.geojson
```

**`--roi-size WIDTH HEIGHT` enforces exact full-resolution dimensions in Python.**
The model chooses the location, not the final dimensions. A box crossing the
image boundary is shifted minimally inward rather than shrunk; that shift is
recorded. A requested size larger than the image fails, rather than silently
reducing it. Fixed-size ROIs are not guaranteed to contain only tissue.

Natural-language dimensions also work through the model response: a question
such as "a 2048px x 2048px ROI on the tissue edge" asks the model to select that
fixed size. Python enforces the size the model returns. The explicit flag is the
authoritative route when those dimensions must not depend on language parsing.
It overrides any size returned by the model. A vague "small ROI" remains an
approximate model-selected bounding rectangle.

A direct pipeline is supported:

```bash
set -o pipefail
omeify inspect slide.ome.tiff -i --geojson \
  -q "Return only the right tissue; name it right." \
  | omeify crop slide.ome.tiff -o Scratch/regions.ome.tiff --geojson - -v
```

Generated GeoJSON carries a foreign `omeify` member with series and dimensions.
Crop checks those against the selected source. This catches coordinate-space
mistakes, **not** accidental substitution of a different image with the same
shape. For nonzero series or vendor input, supply matching `--series` / `--type`
to both commands.

## What the model sees

One PNG overview covers the selected image, without padding or rotation. Default
maximum edge is **1536 pixels**, adjustable with `--preview-size` from 128 through
2048. Encoded attachments are capped at 16 MiB. Preview context supplies the base
width/height, preview width/height, axis orientation, selected channel, display
mapping, source level, and exact canvas-to-base transform. Channel-name context
is capped at 256 characters, with truncation reported. No current path/name,
metadata audit, ICC payload, or full-resolution raster is sent.

A known, calibrated pyramid is preferred. Sampling is never inferred from ratios
of rounded pyramid dimensions. Unknown/inappropriate pyramid scales fall back
to the base. The selected level is read in regions no larger than 1024 by 1024,
accumulating display-only binned means into the bounded canvas. A non-pyramidal
slide can therefore take substantial local I/O, but is not uploaded in full.
The TIFF decoder may still need to decode a complete underlying strip/tile for
a regional request; this is not a process-wide memory guarantee.

RGB uses all three samples with no stain normalization or ICC transform.
Scalar/mIF defaults to a uniquely named DAPI channel (case-insensitive), otherwise
channel zero. `--preview-channel N` chooses another scalar channel by zero-based
index. Repeat `--channel NAME_OR_INDEX COLOR` to build a false-color composite
preview for mIF, for example `--channel DAPI blue --channel panCK green`; colors
accept common names, `#RRGGBB`, or `R,G,B`. `--preview-range LOW HIGH` supplies
an explicit display range for one-channel previews only. Otherwise Omeify uses a
display-only minimum-to-upper-quantile mapping, with `--preview-quantile` defaulting
to **0.999** so the brightest 0.1% of overview bins do not dominate the preview.
Non-finite samples are omitted and reported. None of these display controls change
pixels later exported by `crop`; crops still contain all channels. A DAPI overview
shows nuclear signal, **not a definitive tissue mask**.

The model returns a compact, schema-constrained selection: names, normalized
`[x0,y0,x1,y1]` boxes on a **0..1000 scale per axis**, and optional fixed pixel
sizes. Python validates the full local schema, coordinates, and sizes; maps
coordinates to level zero; rounds edges; and constructs closed GeoJSON polygons.
The model does not author the final GeoJSON structure. Its compact server schema
uses the existing grammar-compatible projection; full bounds and lengths are
checked locally. Malformed responses, reversed boxes, non-finite numbers, unknown
fields, or out-of-range coordinates fail; there is no silent JSON repair.

A valid `unavailable` answer produces an empty FeatureCollection with an explicit
status/message, not a fabricated whole-image ROI. Crop rejects an empty input.
The `omeify` member records provenance and the preview description; each Feature
is marked approximate and retains the normalized box and any size/edge shift.
Interpretations and messages are advisory. A small ROI might span only a few
preview pixels, so exact dimensions do **not** imply accurate edge localization.
Review the proposed rectangles before consequential analysis. This does not add
segmentation, zoom-in refinement, tissue-boundary tracing, or diagnostic claims.

The usual ordered scopes are institutional then local; external transmission
requires explicit inclusion. For local-only use, add `--intelligence-scope local`.
Scope labels are declarations, not authorization. Visual mode sends **image
pixels**, which can contain burned-in identifiers; question/channel text and the
returned GeoJSON can be sensitive too. Ordinary `inspect -i -q` without
`--geojson` still sends metadata only.

## Python use

```python
from omeify import OMETiffReader, OMETiffWriter, crop
from omeify.visual_intelligence import locate_regions

with OMETiffReader("slide.ome.tiff") as image:
    # Library crop follows read_region's Y0,Y1,X0,X1 ordering, unlike CLI XYXY.
    with image.crop(5000, 7048, 10000, 12048) as roi:
        OMETiffWriter("one-roi.ome.tiff", compression="Deflate").write(roi)
    regions = locate_regions(
        image, "Choose the right tissue edge.", roi_size=(2048, 2048),
        allowed_scopes=("institutional", "local"),
    )

report = crop("slide.ome.tiff", "Scratch/regions.ome.tiff", geojson=regions)
# Optional per-ROI files, still using the same writer and source coordinates:
split_report = crop(
    "slide.ome.tiff", "Scratch/split.ome.tiff", geojson=regions, shatter="by_index",
)
```

`Image.crop` constructs an unopened borrowed view, requiring strict integer,
in-bounds level-zero coordinates. It owns no parent resources and inherits
image type, dtype, channels, calibration, and RGB ICC profile. It has one level;
writers rebuild crop-local pyramids. Keep the parent open through all uses of the
view. The parent remains open after the view closes; stale views cannot reconnect
to a later parent session. A view alone does not store the source offset in OME
metadata; the file-level `crop()` workflow records it in crop provenance.

## Focused tests

```bash
python -m pytest -q tests/test_crop.py tests/test_visual_regions.py \
  tests/test_intelligence.py tests/test_intelligence_questions.py
```

Offline tests cover lazy crop lifetimes, exact pixel slices, combined/shattered
ordering, literal output filenames, indexed series labels, grouping and path
preflight, normalized-coordinate mapping, fixed-size edges, display binning,
known/unknown pyramid sampling, and stdout/request boundaries. Controlled model
and PNG doubles do not measure model accuracy or validate an actual endpoint.
Real crop readback requires the normal local OME schema dependency. The optional
loopback adapter test requires Sheetbend/LLM/SDK/imagecodecs and checks actual
PNG attachment serialization against a synthetic server, not a real model.
