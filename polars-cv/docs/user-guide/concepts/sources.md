# Sources

Sources define how the input column should be interpreted before operations are applied.

## The `auto` Source (default)

`source()` defaults to `"auto"`, which picks the decode path from the input
column's Polars dtype:

| Column dtype | Resolves to |
|--------------|-------------|
| `String` | `file_path` |
| `List` | `list` |
| `Array` | `array` |
| `Binary` carrying the VIEW protocol magic | `blob` |
| `Binary` otherwise | `image_bytes` |

```python
from polars_cv import Pipeline

# Reads image bytes from a Binary column, or paths from a String column
pipe = Pipeline().source().resize(height=224, width=224)
```

The column dtype is constant across rows, so the resolution happens once per
batch — the default path adds no per-row cost. Options that apply to the
resolved format still apply to `auto`: `cloud_options` and concurrent remote
prefetch (over a `String` URL column), `decode_max_size`, and
`require_contiguous`.

Pass an explicit format when the dtype cannot be routed — a plain numeric
column raises and names the alternatives — or when you want to override the
inference (for example `source("raw", dtype="u8")` over a `Binary` column that
would otherwise be treated as encoded image bytes).

## Image Sources: `image_bytes` and `file_path`

Both `image_bytes` and `file_path` decode images with automatic format and dtype detection:

- PNG/JPEG usually decode to `u8`
- 16-bit PNG decodes to `u16`
- TIFF may decode to `u8`, `u16`, `f32`, or `f64`

Decoded images are treated as 3D buffers (`[H, W, C]`).

```python
from polars_cv import Pipeline

bytes_pipe = Pipeline().source("image_bytes")
path_pipe = Pipeline().source("file_path")
```

### Scaled Decoding: `decode_max_size`

Both image sources accept `decode_max_size=<n>`, asserting the pipeline needs at
most `n` pixels on the decoded long side. JPEG decoding then uses IDCT scaling
(1/8, 1/4, 1/2) to skip work — a large CPU and memory saving for curation passes.
Non-JPEG formats ignore the assertion and decode at full size.

```python
pipe = Pipeline().source("file_path", decode_max_size=256)
```

The chainable [`thumbnail(max_size)`](../operations/image-ops.md#thumbnail) method
is the explicit equivalent. Because a scaled decode + resize is not bit-identical
to a full decode + the same resize, this is an opt-in.

### Error Handling

By default, a failure while producing a row raises and aborts the whole query.
There are three complementary controls for tolerating per-row failures, from
narrowest to broadest.

**Source-level: `source(..., on_error="null")`** — scoped to decode failures.
A row that fails to *decode* yields null for the outputs that depend on that
source; the rest of the DataFrame still processes.

```python
# Skip undecodable images instead of aborting
pipe = Pipeline().source("image_bytes", on_error="null").resize(height=224, width=224)
```

**Graph-level: `.on_error(policy)`** — a pipeline-wide policy covering **any**
error that produces a row (source decode, operation execution, or output
encoding). It takes one of three values:

```python
# "raise" (default): the first failing row fails the expression with its error.
pipe = Pipeline().source("image_bytes").resize(height=224, width=224)

# "null": failing rows yield null for ALL of the graph's outputs; good rows are unaffected.
pipe = Pipeline().source("image_bytes").resize(height=224, width=224).on_error("null")

# "null_with_message": like "null", but the output becomes a struct with a
# reserved `_error` field carrying the failure message (null for good rows).
# A single-output pipeline becomes a two-field struct (`_output` + `_error`).
pipe = Pipeline().source("image_bytes").resize(height=224, width=224).on_error("null_with_message")
```

`.on_error()` is a graph-level setting: when pipelines are composed (via
`merge_pipe` or binary ops) they must all agree on the policy. It is also
mirrored on `LazyPipelineExpr`, so you can set it after `.pipe(...)`.

**Null parameters: `.on_null_param(policy)`** — scoped to nulls in the columns
backing [dynamic parameters](../operations/image-ops.md#dynamic-parameters).
A parameter given as a `pl.Expr` is read from an ordinary column, which may
contain nulls:

```python
# "raise" (default): a null parameter fails the whole expression.
pipe = Pipeline().source("image_bytes").resize(height=pl.col("h"), width=pl.col("h"))

# "null": rows whose parameter is null yield null; other rows are unaffected.
pipe = (
    Pipeline()
    .source("image_bytes")
    .resize(height=pl.col("h"), width=pl.col("h"))
    .on_null_param("null")
)
```

This is deliberately separate from `.on_error()`. Under `"null"` a null
parameter is not treated as an error at all, which means two things:

- Only the outputs that actually depend on the affected operation go null,
  rather than every output of the row.
- Decode, encode and genuine operation failures still raise. You do not have to
  go blind to real bugs in order to tolerate missing parameter values.

For a **fallback value** rather than a null, fill the null in the expression
itself — there is no policy for this because Polars already expresses it:

```python
pipe = Pipeline().source("image_bytes").resize(
    height=pl.col("h").fill_null(224), width=pl.col("w").fill_null(224)
)
```

Like `.on_error()`, this is graph-level, composed pipelines must agree, and it
is mirrored on `LazyPipelineExpr`. The `.contour` / `.point` / `.bbox`
namespaces bypass the graph engine, so they carry the same policy on the
accessor instead: `pl.col("c").contour.on_null("null").normalize(pl.col("w"), 100)`.

## Auto DType Behavior

For `image_bytes` and `file_path`, dtype is runtime-dependent, so the pipeline starts with dtype `auto`.
An `auto` source starts there too, since its decode path is not chosen until execution.

You can resolve dtype by:

- providing `dtype` in `source(...)`
- using an operation with deterministic output dtype (for example `normalize`, `threshold`, or `cast`)

```python
# Assert and enforce dtype at source
pipe = Pipeline().source("image_bytes", dtype="f32").resize(height=224, width=224)
```

## Planning-Time Requirement for `list`/`array` Sinks

`sink("list")` and `sink("array")` require known element dtype at planning time.

`list`/`array` sources — and an `auto` source over a `List`/`Array` column —
resolve their leaf dtype and rank from the Polars column when the plan sees the
input, so these sinks type-check without any extra help. An `auto` source over a
`Binary`/`String` (image) column does not: its dtype is only known after decode.

For image sources, resolve dtype before these sinks:

```python
# Option 1: source dtype
pipe = Pipeline().source("file_path", dtype="u8").resize(height=224, width=224)

# Option 2: cast in pipeline
pipe = Pipeline().source("image_bytes").resize(height=224, width=224).cast("f32")
```

## `file_path` and Cloud Access

`file_path` supports local paths (bare or `file://` URIs) and remote URIs (for
example `s3://`, `gs://`, `az://`, `http://`).
Use `CloudOptions` when credentials or provider settings are needed:

```python
from polars_cv import CloudOptions, Pipeline

options = CloudOptions(
    aws_region="us-east-1",
    aws_access_key_id="...",
    aws_secret_access_key="...",
)

pipe = Pipeline().source("file_path", cloud_options=options)
```

`cloud_options` also applies to the default `auto` source, since it can resolve
to `file_path` from a `String` column. It is ignored (with a warning) for
sources that never read remotely.

Remote requests are **signed by default**. To read from a public bucket without
credentials, opt into anonymous access explicitly with
`CloudOptions(anonymous=True)` (honored for S3, GCS, and Azure).

### Passing arbitrary backend options

`CloudOptions` exposes named fields for the common credentials, but any option
the underlying `object_store` backend understands can be supplied through
`storage_options`, keyed by that backend's native config names. Keys in
`storage_options` win over the named fields on collision.

```python
CloudOptions(storage_options={"aws_endpoint": "https://minio.local:9000"})
CloudOptions(storage_options={"google_application_credentials": "/path/adc.json"})
```

### GCS authentication matrix

| How | Option |
| --- | --- |
| Service-account JSON file | `gcs_service_account_key="/path/sa.json"` (or `storage_options={"google_service_account": ...}`) |
| Inline service-account JSON | `storage_options={"google_service_account_key": inline_json}` |
| Application Default Credentials file | `storage_options={"google_application_credentials": "/path/adc.json"}` |
| Federated / workload identity | automatic via `gcloud` (see below) |
| Token from a custom command | `token_command="my-broker get-token"` |
| Pre-obtained OAuth access token | `gcs_bearer_token=token` |
| Public bucket | `anonymous=True` |

Federated ADC (type `external_account` or `external_account_authorized_user`,
e.g. an OIDC identity exchanged through a workload/workforce identity pool)
cannot be parsed by `object_store`. polars-cv handles it by delegating to
`gcloud`: when it detects a federated ADC — from `GOOGLE_APPLICATION_CREDENTIALS`,
an explicit `google_application_credentials` option, or the well-known gcloud
path — it runs `gcloud auth application-default print-access-token` and uses the
resulting token, caching it until just before it expires. This needs the
`gcloud` CLI on `PATH`; set `POLARS_CV_DISABLE_GCS_FEDERATION=1` to disable it.
To source the token another way, set `token_command` to any shell command that
prints an access token (it takes precedence over the automatic `gcloud`
delegation), or pass `gcs_bearer_token` to supply one yourself.

`token_command` is provider-agnostic across the OAuth-bearer backends: the same
option works for **Azure** Blob Storage (e.g.
`az account get-access-token --resource https://storage.azure.com/ --query accessToken -o tsv`).
It does **not** apply to S3, which authenticates with SigV4 rather than a bearer
token — supplying it with an `s3://` source raises; use the `aws_*` fields or
`storage_options` for AWS credentials instead.

## Reading Bytes Without Decoding

The `file_path` source is two stages: fetch the bytes a path names, then decode
them as an image. `.cv.read_bytes()` is the first stage on its own — same fetch
mechanism, same credentials, same concurrency, no decode.

```python
import polars as pl

df.with_columns(raw=pl.col("path").cv.read_bytes())  # -> Binary
```

Bytes come back **verbatim**, so an encoded file survives the round trip
unchanged:

```python
assert df.with_columns(raw=pl.col("path").cv.read_bytes())["raw"][0] == Path(p).read_bytes()
```

That matters because decoding is one-way. Re-encoding a decoded JPEG never
reproduces the original file, and no image sink carries EXIF or ICC metadata —
so even a PNG loses information through a decode/encode round trip.
`read_bytes` is the only lossless path.

It also gives the [header-only metadata methods](../operations/metadata.md)
a way to reach remote files, since they take binary columns:

```python
df = df.with_columns(raw=pl.col("path").cv.read_bytes(cloud_options=options))
df = df.filter(pl.col("raw").cv.width() > 512)                  # header-only, no decode
df = df.with_columns(thumb=pl.col("raw").cv.pipe(pipe).sink("png"))
df.select("path", "raw")                                        # originals, untouched
```

The fetch happens once here and both the predicate and the pipeline read the
same column — where two separate expressions over a path column would each
fetch independently.

| Argument | Meaning |
|----------|---------|
| `cloud_options` | Credentials/settings for remote reads. Same `CloudOptions` (or dict) the `file_path` source takes. |
| `on_error` | `"raise"` (default) fails the query on the first unreadable path; `"null"` yields null for that row only. |

### Memory and streaming

Fetching is per plugin call — one morsel under the streaming engine. Within a
call, distinct remote paths are fetched concurrently up front (exactly as the
`file_path` source already does) and local files are read per row.

Under `.collect(engine="streaming")` a bytes column is therefore
morsel-bounded: if you filter on it and drop it, only a morsel's worth is
resident at a time, and projection pushdown skips it entirely when nothing
downstream uses it. It only becomes corpus-resident if you select it in the
final projection — which is the point when you want the originals.

Under the **default in-memory engine there is no such bound**: `with_columns`
materializes the whole column, so `read_bytes` over a million-path frame holds
every file at once. There is also no per-file size cap — whatever the path names
is read in full. Use `engine="streaming"`, or slice the frame, when the corpus
does not fit in memory.

!!! warning "Paths are not sandboxed"
    `read_bytes` reads whatever the column names, local or remote, with no
    allowlisting — the same caveat that applies to the `file_path` source, but
    over any file rather than only decodable images. Two edges are sharper here
    than for the source: any local file is returned verbatim rather than having
    to survive an image decode, and an `http://` path is fetched as-is, which
    reaches link-local addresses such as cloud instance-metadata endpoints. Use
    it with trusted path columns only.

## Contour Source and Shape Inference

The `contour` source rasterizes contour structs to binary mask buffers. You can specify dimensions explicitly or infer them from another pipeline expression.

```python
from polars_cv import Pipeline

# Explicit dimensions
mask_pipe = Pipeline().source("contour", width=200, height=200)

# Infer dimensions from an image pipeline
img = pl.col("image").cv.pipe(Pipeline().source("image_bytes").resize(height=200, width=200))
mask_pipe = Pipeline().source("contour", shape=img)
```

When using `shape=`, the contour mask dimensions automatically match the referenced pipeline's output size.

## Other Sources

- `auto` (default): infer the decode path from the column dtype (see above)
- `raw`: raw bytes, requires explicit `dtype`
- `blob`: self-describing binary VIEW protocol
- `list`/`array`: infer from Polars column type (or override with `dtype`)
- `contour`: rasterizes contour structs to mask buffers (see above)
