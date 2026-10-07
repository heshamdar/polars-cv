# Migrating from 0.33 to 0.34

0.34 moves polars-cv to **Polars 2.0**. The polars-cv API is unchanged: no
method, parameter or output dtype differs from 0.33. What changes comes from
Polars itself. This page lists what to check coming from 0.33; the
[changelog](../changelog.md) has the full list.

## Requirements

- **polars `>=2.0.0,<3.0`** (was `>=1.44.2,<2.0`). Every plugin call is
  registered as deterministic, which polars 1.x does not accept. A project
  that pins polars 1.x resolves polars-cv 0.33.x.

## Results that change

**Seeded bootstrap intervals differ from 0.33 for the same seed.** The
resample draw is `hash(slot) % n`, and Polars 2.0 changed `Expr.hash`'s values
(Polars guarantees hash stability only within a version). Intervals remain
reproducible for a given seed *and* Polars version, and the method is
unchanged, so they agree with 0.33's in distribution, not bit for bit. Re-pin
any test that snapshots seeded bounds.

**A tagged Parquet column read without importing polars-cv** now loads as
Polars' generic extension (`pl.Extension("polars_cv.point", ...)`, Polars
2.0's default for unknown extension types) instead of the plain struct. No data
is lost: `.ext.storage()` gives the struct, or set
`POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage` to read the struct
directly. Importing `polars_cv` first still gives the registered type, as
before. See [extension types](concepts/extension-types.md).

## Behaviour you get for free

- **Equal pipelines in one query run once.** Two pipeline expressions built
  separately but equal (same pipeline and sink) are merged by Polars' common
  subexpression elimination into one plugin call. To read several outputs of
  one pipeline, a [multi-output sink](composition/multi-output.md) is still the
  way to share everything but the encode.
- **A lazy `.collect()` streams by default** (Polars 2.0), and polars-cv's
  outputs and allocations take part in out-of-core spilling: memory the plugin
  allocates counts toward Polars' budget, and every sink survives a spill. An
  eager `DataFrame.with_columns` still runs in memory. See
  [streaming](concepts/streaming.md) for the budget variables.
- **Metrics' per-group scans stream natively**, since Polars 2.0 runs
  `.over()` windows in the streaming engine.

## Fixed along the way

- `froc_curve_lazy(thresholds=[1, ...])` with integer thresholds raised under
  Polars 2.0's stricter `is_in`; thresholds are now passed as Float64.
- An import-time segfault on macOS, introduced by this release's pyo3 0.29
  bump and fixed before it (no released wheel was affected).
