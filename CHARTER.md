# Omeify Charter

Omeify is a dependable microscopy image I/O layer: it makes supported sources
predictable to read and constructs physically interpretable, metadata-minimized OME-TIFF and
OME-Zarr products that other software can use.

This charter records the principles guiding our design. It can evolve through
intentional discussion and review; it does not claim that every format or workflow
is supported. Implementation gaps are not reasons to quietly change its meaning.

## Completeness and portability

For supported conversions, Omeify aims to guarantee two kinds of completeness:

- **Format completeness:** the output is a self-contained, standard-compatible image
  whose encoded pixels, dimensions, channel/plane mapping, and storage organization
  can be decoded correctly by conforming readers. Validation and sampled readback
  check construction; they do not constitute an exhaustive pixel audit or certify
  every downstream application.
- **Image-interpretation completeness:** the output contains the smallest curated,
  validated description sufficient to interpret its numeric pixels, channel and
  RGB meaning, known spatial/temporal sampling, and coordinate relationships.
  Unknown physical quantities remain unknown, rather than being invented to make
  a header appear complete. Viewers must not be led to misinterpret axes, channels,
  color profiles, or physical distances. Identical default pseudocolors and
  auto-contrast across independent viewers are not guaranteed.

**Study/reproducibility completeness is not an image-output requirement.** Specimen,
reagent, instrument, acquisition-protocol, processing-history, and broader study
records may be necessary for reproducible science, but belong in deliberate linked
records, not wholesale copies of source metadata in every converted image.
Minimization means *sufficient* metadata, not the fewest fields regardless of meaning.

MITI guides the curated header contract, but format validity, Omeify's image-
interpretation contract, and strict compliance with the published MITI checklist
are separate assessments. A published conditional requirement cannot be declared
satisfied by an undocumented exception. When a Z axis has multiple ordered sections
and no reliable physical spacing, `convert` and `mutate` must refuse to export
a multi-Z product until the caller explicitly declares ordinal-only sampling or
supplies a positive physical Z spacing and its unit (µm by default). Missing
measurements do not prove intentional ordinal acquisition: the declaration is
an informed export decision, not an instrument fact. A known or explicitly
supplied physical spacing is written with an explicit unit and validated;
no arbitrary numeric step is invented. Index-only output remains OME-valid and
interpretation-complete for index-based use under Omeify's documented MITI-
guided exception, rather than a claim of literal satisfaction of the published
multi-Z physical-size checklist.

## Core rules

1. **Stay in the I/O niche.** Own image interpretation, regional access, inspection,
   explicit data transformations, and verified format construction. Segmentation,
   registration, stain normalization, biological measurement, model training, and
   interactive viewer policy belong to consumers. Optional extensions must not
   silently change the pixels or semantics of core image operations.

2. **Keep 2D simple and tested; add dimensions through datasets.** `Image` is
   the smaller, established, heavily regression-tested 2D observation API.
   `ImageDataset` supports Z/T and native resolutions without complicating that
   contract. `ImageSource` and `DatasetSource` are complementary, not old and new
   alternatives. Share source lifetime and decoding; explicit plane selection
   yields an ordinary Image. Image semantics belong to images and datasets;
   storage and publication settings belong to writers.

3. **Make dimensions predictable.** Public axes, shapes, and returned arrays must
   agree: Images use `YX`, `CYX`, or `YXS`; dataset blocks use `TZYX`, `TCZYX`, or
   `TZYXS` according to their logical pixel kind. Expose native storage separately.
   RGB is one logical channel with three samples, not three inferred biomarkers.
   Series, channels, Z, T, and pyramid resolutions are not interchangeable. Write
   documented standard-compatible layouts, not a private raster format.

4. **Convert to preserve; mutate to change values.** `convert` preserves the
   selected image/series and its complete Z/T extent by default, unless a caller
   explicitly selects a subset. `mutate` requires a deliberate numerical change
   to intensity or precision; selecting planes or regions is not a mutation.
   Reading one 2D observation still requires explicit indices for ambiguous Z/T
   axes. Never guess image types, RGB semantics, source profiles, or measurements.
   Document color management, lossy encoding, and resampling instead of claiming
   every conversion is bitwise lossless.

5. **Never invent geometry.** Distinguish indices, sampling, origins, and measured
   acquisition positions. Honor applicable standard-defined defaults without
   inventing missing values; unknown calibration or phase remains unknown.
   When an ordinal-only Z interpretation is explicitly authorized, preserve
   indices without pretending physical spacing or acquisition intent is known.
   Rounded raster dimensions do not establish a sampling transform. Dataset
   `native_level` indexes stored resolutions; Image `level` indexes that 2D
   observation's compatible pyramid. A Z-reduced coarse slice must never silently
   stand in for the exact optical section. Metadata-only views preserve compatible
   resolutions and pixels.

6. **Rebuild metadata deliberately.** Generate a constrained, validated output
   vocabulary instead of copying arbitrary source headers, annotations, paths, or
   ICC payloads. Retain necessary interpretation and explicitly supported
   provenance; do not silently discard geometry an export cannot represent.
   For time series, a regular `TimeIncrement` suffices when known; curate optional
   per-plane `DeltaT` where recorded to preserve actual acquisition timing,
   including irregular timing within nominal T points. Neither is required to
   navigate an ordinal time sequence. Detector exposure and microscope stage
   `PositionXYZ` belong in external acquisition records, not the minimized image.
   Never invent frame times. Assess format, image-interpretation, and strict
   external-standard conformance independently. Metadata minimization is not
   deidentification. Retained names, diagnostics, provenance, and image pixels
   still require review before sharing.

7. **Make resource ownership explicit.** Owners release their resources; borrowed
   views never close their parents. Closing is idempotent, failed opens clean up,
   and stale views cannot reconnect to a later session. Returned regional arrays
   own their pixels. Keep the source alive until its consumers and outstanding
   reads finish; immutable selections must not change under those reads.

8. **Optimize without changing meaning.** Work regionally; whole-image and
   whole-stack materialization must be deliberate. Share bounded caches across
   plane views. Speculative reads are opt-in, subordinate to demand, and cancellable
   where possible. Budgets must state what they exclude. Cache eviction, request
   order, and prefetch cannot change pixels or hide read failures. Honor declared
   sparse-fill semantics without treating transport or decoding errors as fill.

9. **Keep the core runtime focused.** Use established libraries for codecs,
   containers, and transport rather than duplicating them. Core TIFF and
   local/HTTP(S) Zarr operations must not require a JVM, viewer, segmentation
   framework, or optional S3/intelligence dependencies. Independent
   interoperability tests may use optional external consumers.

10. **Isolate heavyweight and experimental features.** Substantial dependencies
    belong in explicit installation extras: S3 via `omeify[s3]` and intelligence
    via `omeify[intelligence]`. Intelligence remains a work in progress; this
    charter makes no assumptions about its eventual features, contracts, or
    guarantees. Experimentation must stay within that optional capability,
    without altering core operations, public APIs, or their established behavior.
    The core must work independently of either extra.

11. **Publish safely and claim only what was checked.** Protect inputs and existing
    destinations, stage output, validate metadata and storage, and verify decoded
    samples before publication. Do not weaken overwrite guarantees when a platform
    cannot provide them. State sampled coverage and transaction boundaries.
    Required backend tests need real dependencies; optional skips and mocked tests
    do not establish interoperability. Preserve 2D behavior with regression tests.

## Changes and supporting contracts

The charter may change with an explicit rationale and maintainer review.
Changes to public APIs or output contracts must update documentation, tests, and
any required downstream migration together. Prefer one authoritative contract to
parallel implementations or indefinite compatibility shims.

Keep versions, tuning defaults, release status, and feature backlogs out of this
file. The [Image/source guide](docs/image_sources.md),
[Z/T guide](docs/z_t_support.md), [format reference](docs/reference.md),
[Zarr guide](docs/ome_zarr.md), [report/schema inventory](docs/reports.md), and
[development checks](docs/development.md) own those details.
