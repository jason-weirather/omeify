# Reports and terminal output

Omeify 0.22 keeps full library reports while making routine image-writing CLI
output compact. This is a pre-1.0 contract cleanup, not a compatibility alias layer.
Both storage formats use the common Image API and explicit geometry model.
The current hardening pass does not change pixel arithmetic or storage defaults;
it tightens interpretation/metadata checks and makes inspection scope explicit.

## Zarr integration (0.22)

OME-Zarr has its own storage and inspection records, not fabricated TIFF IFDs.
`omeify.zarr_write/1`, `omeify.zarr_collection/1`, and `omeify.zarr_inspect/2`
are validated by packaged schemas. Convert/mutate report version 2 selects the
appropriate TIFF or Zarr output/verification variant. Crop version 4 accepts both
collection writers; the embedded crop provenance stays `omeify.crop_provenance/1`.
`miti_header` is null for Zarr, never a false claim of TIFF-header compliance.
Input Zarr `size_bytes` is null: discovery does not traverse an entire store to
measure it. Source selectors and physical origins are explicit. A source store
and the intended output directory are protected in their entirety from report
writes. Explicit storage credentials are never copied to reports.

See [the Zarr contract](ome_zarr.md) for publication, version, and capability limits.

## CLI presentation

These rules apply equally to `convert`, `concat`, `mutate`, and `crop`:

| Selection | Output |
|---|---|
| No verbosity flag | One short completion line; warnings/errors remain visible. |
| `-v` | Compact stages on stderr, with tqdm progress on a terminal. Redirected logs use plain lines. |
| `-vv` | Timestamped diagnostic lines and tracebacks, without animated bars or a JSON dump. |
| `-vvv` | Diagnostic stderr plus a full JSON report on stdout unless an explicit report destination is supplied. |
| `--output-json FILE` | Complete UTF-8 JSON in that file at any verbosity; a short completion line on stdout. |
| `--output-json -` | Exactly one complete JSON document on stdout, with no completion sentence. |

The full report can include generated OME-XML, source metadata, detailed mutation
statistics, paths, and timing. Keeping it explicit does not disable verification
or change the image computation. Python workflows continue to return full reports.

```bash
omeify convert input.ome.tif -o normalized.ome.tif --type ome_tiff -v
omeify convert input.ome.tif -o normalized.ome.tif --type ome_tiff \
  --output-json conversion.json
omeify crop input.ome.tif -o regions.ome.tif --geojson regions.json \
  --output-json - > crop-report.json
```

`inspect` still prints its inspection tree by default; `--json` and `--output`
retain their meanings. Visual `--geojson` remains JSON-only on stdout. Inference
progress describes the operation, without library-brand announcements. Dependency
configuration and actual source/model provenance remain available.

Terminal rendering escapes control and bidirectional formatting characters while
preserving ordinary Unicode such as `µm`. This is display safety, not alteration of
report values or deidentification. JSON uses JSON escaping and preserves the values.

## Current external contracts

Schemas are installed as package resources under `omeify/schemas/`. They are the
authoritative serialized structures; reports remain ordinary dictionaries/lists.
References resolve against packaged resources, never by fetching their `$id` URLs.

| Schema resource | Product / current identifier |
|---|---|
| `write_report.schema.json` | Single image or temporary writer: `omeify.write/1` |
| `multi_series_report.schema.json` | Named heterogeneous images, including `concat`: `omeify.multi_series/1` |
| `conversion_report.schema.json` | Conversion: `omeify.convert/2` |
| `mutation_report.schema.json` | Integer or float32 precision mutation: `omeify.mutate/2` |
| `crop_report.schema.json` | Crop export, including nested writer reports: `omeify.crop/4` |
| `crop_provenance.schema.json` | Embedded crop geometry/provenance: `omeify.crop_provenance/1` |
| `tiff_inspection.schema.json` | TIFF inspection: `schema_version="2.0"` |
| `zarr_inspection.schema.json` | Zarr inspection, including discovery metadata snapshots: `omeify.zarr_inspect/2` |
| `zarr_write_report.schema.json` | Zarr single-image/temporary writer: `omeify.zarr_write/1` |
| `zarr_collection_report.schema.json` | Zarr collection writer report: `omeify.zarr_collection/1` |
| `ome_zarr_profile.schema.json` | Canonical NGFF 0.5 image attributes: `omeify.ome_zarr/1` policy |
| `ome_zarr_collection.schema.json` | Canonical NGFF 0.5 collection-root attributes, not the writer report |
| `ome_zarr_manifest.schema.json` | Canonical `OME/` ordered series manifest attributes |
| `ome_geometry.schema.json` | Linked TIFF image-coordinate record: `omeify.geometry/1` |
| `metadata_intelligence.schema.json` | Summary or focused metadata answer: `schema_version="2.0"` |
| `calibration.schema.json` | Deterministic calibration comparison: version `1.0` |
| `visual_answer.schema.json` | Visual answer: `omeify.visual_answer/1`; also defines its model response |
| `visual_regions.schema.json` | Pixel-coordinate GeoJSON: `omeify.visual_regions/1` inside `omeify` |
| `region_intelligence.schema.json` | Model's normalized region-selection response, not exported pixel coordinates |
| `preview.schema.json` | Display overview and canvas-to-image coordinate context |
| `report_common.schema.json` | Shared definitions used by operation reports |
| `miti_ome_tiff_header.schema.json` | The generated image-header profile, not dataset-level MITI certification |

Example of validating an emitted report locally:

```python
from omeify.reports import validate_report

validate_report(report, "conversion_report.schema.json")
```

`load_schema(name)` returns an independently editable copy of the schema resource.
Editing that copy does not change Omeify's validation. Internal compiled validators
reuse the same installed definitions; there is no independently authored domain-model
layer. Geometry relationships and physical calibration checks that require image
knowledge remain Python checks alongside the structural schemas.

Required OME-XSD validation uses the installed `ome-schema` distribution. An explicit
`OMESchemaValidator(schema_location=...)` remains supported. No adjacent-tifffile
schema search, network fallback, or missing-validator success state is provided.

## Migration from 0.20

Use the current fields, without deprecated aliases:

| Removed field | Current representation |
|---|---|
| `options.deidentify_ome` | `options.metadata_minimization` in single/multi writer reports and derived reports |
| `verification.significant_bits_matches_dtype` | `verification.significant_bits_matches_spec` |
| `conversion_stats.compression_ratio` | `conversion_stats.output_to_input_size_ratio` |
| `pyramid.level_shapes_cyx`, `pyramid.level_shapes_yxs` | `pyramid.axes` plus `pyramid.level_shapes` |
| `output_file.shape_cyx` | `output_file.axes` plus `output_file.shape` |
| `verification.channels_checked`, `verification.points_per_channel` | Physical `planes_checked` / `points_per_plane` and `pixel_verification` coverage |

RGB still has one logical channel and three samples. Removing ambiguous channel
verification counters does not remove channel metadata or coverage. `SignificantBits`
is still written and checked against the writer's specification. The checked value
can intentionally be smaller than dtype storage width after mantissa trimming.
Channel-rename mapping keys in JSON-compatible reports are strings, including indices.
Python's input API still accepts integer index keys with the explicit index mode.

Metadata-intelligence reports now require an explicit `origin` for every record
and explicit computed-record counts. Only current version 2.0 reports validate:
summary responses use prompt 2.1, question/answer responses prompt 3.0. These prompt
identifiers describe active protocols and are independent of report/package versions.
Retired schema/prompt combinations are not silently upgraded or reinterpreted.
TIFF inspection schema version 2.0 reflects that nested contract change, not a restriction to
writer-supported TIFF layouts. Ordinary inspection remains useful on unsupported inputs.

The 0.21 cleanup introduced crop report version 3; **current output is version 4**
to support either TIFF or Zarr collection reports. Embedded crop provenance
retains its separate `omeify.crop_provenance/1` identifier. Old reports are not
silently rewritten into current schemas.

## Stored metadata checks

The inventory above is the single current schema index. Operation-report schemas
validate report structure, not the truth of a claimed pixel check. Stored-metadata
schemas validate the image product's own controlled records:

* `ome_geometry.schema.json` is used when generating and reading a linked TIFF
  geometry annotation. Python additionally checks finite coordinates and the
  4 KiB UTF-8 bound. Duplicate JSON keys are rejected on read; the existing
  namespace and OME Image reference remain authoritative.
* `ome_zarr_profile.schema.json` fixes axes to ordered `yx` or `cyx`, requires
  scale followed by optional translation of matching rank, fixes channel scale
  to one/translation to zero, and constrains image-type extensions. Python checks
  consecutive dataset paths, display-range ordering, and supplemental value
  relationships. Array dimensions, pixels, encoding and physical consistency
  remain reader/writer checks, not facts inferred from schema success.
* Zarr collection-root and `OME/` manifest attributes are schema-checked before
  construction and on readback, with exact generated-value/order comparison.
  Child images use the ordinary image-attribute profile. These output profiles
  do not forbid unrelated valid metadata in a foreign input store.

Explicit collection provenance in either backend must be a mapping of finite,
acyclic plain Python JSON data with string keys (no object/key coercion), at most
64 nested containers and 1 MiB encoded
UTF-8. A declared `omeify.crop_provenance/1` object must also satisfy its schema.
Other caller-supplied provenance objects are intentionally not given invented
biological semantics; they are copied deliberately and need sharing review.

## Inspection report version 2 for Zarr

`omeify.zarr_inspect/2` adds required `metadata_nodes`, a list of
`{"group": relative_group_path, "attributes": raw_attributes}` snapshots collected
during bounded discovery. It preserves collection-root provenance, the consulted
`OME` manifest and declared child-image attributes rather than hiding them behind
the selected image. `attributes` still describes the selected group, while
`arrays` holds that image's original array metadata. Normalized `image`, `levels`
and `series` fields have strict application-owned schemas; raw source metadata
remains open because inspection must be able to describe foreign formats.

This is **not recursive discovery** of arbitrary sidecars, sibling groups or
unvisited ancestors. Selecting `group="1"` directly does not inspect the
collection above it. Inspect from the collection root to include its discovery
context. Reader-owned `image.inspect()` snapshots the already opened session and
does not reopen the source, fetch additional URLs, or keep its handles alive.
Ordinary text output flags additional group metadata; detail 3 shows bounded raw
snippets and JSON retains all collected snapshots. `whole_store_audited=false`
and `raster_pixels_examined=false` remain explicit.

## Metadata retention and sharing

Metadata minimization means constructing the chosen output vocabulary, **not
classifying every retained value as safe**. These distinctions apply to both
formats:

| Value or object | Boundary |
|---|---|
| Vendor descriptions, patient/scanner fields, arbitrary source dictionaries, array attributes and unrelated groups/sidecars | Not automatically copied into new image products |
| Channel names | Only selected fields in supported source structures; names are still free text requiring review |
| ROI/collection names, caller Software tag and explicit provenance | Deliberate retention, not automatic privacy approval; index-based filenames do not erase internal names |
| RGB ICC profile | Used only to map source RGB pixels into sRGB; never copied into output. A pinned sRGB ICC is the only emitted profile; malformed inputs fail |
| Physical origins and crop coordinates | Retained when needed for the explicitly selected product; not invented acquisition-stage measurements |
| Inspection and operation reports | Diagnostics may include original attributes, paths and allowed free text; these are not the minimized image artifact |
| Pixels | Preserved/transformed under the selected image operation; burned-in identifiers are not detected |

RGB writer reports include `image.color_management` (or the per-series value):
source-profile status, whether color pixels were transformed, rendering intent,
and SHA-256 of the fixed output ICC. This excludes raw source-profile bytes or
source ICC descriptive text. Geometry-only crop provenance is generated by
Omeify; caller-supplied ROI names and other deliberate text remain the
caller's responsibility.

`metadata_minimization=true` records application of the construction policy.
A MITI minimum-header pass permits additional OME metadata and reports it
separately; it is not a minimization certificate. A canonical Zarr attribute
profile match is not a whole-store privacy audit. Neither replaces independent
sharing review, and no validator proves source calibration or biological truth.

## Safety and failure behavior

Before expensive work, CLI report destinations are checked against image inputs,
image outputs, and supplied rename/GeoJSON input files. Crop checks every planned
shattered destination. Matching pathnames, symlink aliases, and existing hard-link
identities are rejected. Parent directory symlinks are allowed.

Report output is serialized before opening its destination, written to a temporary
sibling, closed, and installed atomically. Replacing an unprotected final-component
symlink replaces the link, not its target; replacing an unprotected hard link does
not truncate the other linked file. Temporary report files are cleaned on failure.
`--no-overwrite` applies to an explicitly named report as well as the image. It uses
an atomic hard-link install and fails on filesystems without that support.

**The image and report are not one transaction.** Required metadata validation
and backend-specific readback precede TIFF installation or Zarr publication. Final
operation-report construction/validation and report-file serialization can occur
after installation. A later report failure does not delete or roll back a valid
image; the command fails and the explicit report-output error identifies that the
image already completed. Shattered crop files are likewise installed individually,
not as an all-or-nothing multi-file transaction. Atomic installation is not a promise
of power-loss durability or protection against hostile concurrent directory rewrites.

`verification.pixel_verification` explicitly says `mode="sampled"` and
`all_pixels_checked=false`. Representative base and pyramid points are checked;
this is not an exhaustive image audit or proof of correct biological interpretation.
Successful structural report validation does not independently repeat those checks.

Metadata minimization is not deidentification. Reports and allowed image metadata
can contain identifying information, and pixels are not screened for burned-in text.
