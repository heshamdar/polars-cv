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
There are two complementary controls for tolerating per-row failures.

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
