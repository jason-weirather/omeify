# Reports and terminal output

Omeify 0.21 keeps full library reports while making routine image-writing CLI
output compact. This is a pre-1.0 contract cleanup, not a compatibility alias layer.
The image API, coordinate conventions, storage defaults, and numeric algorithms
are unchanged for supported inputs.

## CLI presentation

These rules apply equally to `convert`, `mutate`, and `crop`:

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
| `multi_series_report.schema.json` | Named heterogeneous images: `omeify.multi_series/1` |
| `conversion_report.schema.json` | Conversion: `omeify.convert/1` |
| `mutation_report.schema.json` | Integer or float32 precision mutation: `omeify.mutate/1` |
| `crop_report.schema.json` | Crop export, including nested writer reports: `omeify.crop/3` |
| `crop_provenance.schema.json` | Embedded crop geometry/provenance: `omeify.crop_provenance/1` |
| `tiff_inspection.schema.json` | Inspection: `schema_version="2.0"` |
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
Inspection version 2.0 reflects that nested contract change, not a restriction to
writer-supported TIFF layouts. Ordinary inspection remains useful on unsupported inputs.

Crop report version 3 replaces version 2 because its nested writer reports and
embedded provenance now have explicit current contracts. Embedded crop provenance
uses its own identifier rather than sharing the report's identifier. Crop filenames,
ROI order, coordinate conventions, bounds, and offsets are unchanged.

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

**The image and report are not one transaction.** Image encoding, required OME-XSD/
profile validation, and TIFF readback checks precede the TIFF's installation. Final
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
