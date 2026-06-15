# Sources

Sources define how the input column should be interpreted before operations are applied.

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

You can resolve dtype by:

- providing `dtype` in `source(...)`
- using an operation with deterministic output dtype (for example `normalize`, `threshold`, or `cast`)

```python
# Assert and enforce dtype at source
pipe = Pipeline().source("image_bytes", dtype="f32").resize(height=224, width=224)
```

## Planning-Time Requirement for `list`/`array` Sinks

`sink("list")` and `sink("array")` require known element dtype at planning time.

For image sources, resolve dtype before these sinks:

```python
# Option 1: source dtype
pipe = Pipeline().source("file_path", dtype="u8").resize(height=224, width=224)

# Option 2: cast in pipeline
pipe = Pipeline().source("image_bytes").resize(height=224, width=224).cast("f32")
```

## `file_path` and Cloud Access

`file_path` supports local paths and remote URIs (for example `s3://`, `gs://`, `az://`, `http://`).
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

- `raw`: raw bytes, requires explicit `dtype`
- `blob`: self-describing binary VIEW protocol
- `list`/`array`: infer from Polars column type (or override with `dtype`)
- `contour`: rasterizes contour structs to mask buffers (see above)
