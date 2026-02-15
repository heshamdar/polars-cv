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

## Auto DType Behavior

For `image_bytes` and `file_path`, dtype is runtime-dependent, so the pipeline starts with dtype `auto`.

You can resolve dtype by:

- providing `dtype` in `source(...)`
- using an operation with deterministic output dtype (for example `normalize`, `threshold`, or `cast`)

```python
# Assert and enforce dtype at source
pipe = Pipeline().source("image_bytes", dtype="f32").resize(224, 224)
```

## Planning-Time Requirement for `list`/`array` Sinks

`sink("list")` and `sink("array")` require known element dtype at planning time.

For image sources, resolve dtype before these sinks:

```python
# Option 1: source dtype
pipe = Pipeline().source("file_path", dtype="u8").resize(224, 224)

# Option 2: cast in pipeline
pipe = Pipeline().source("image_bytes").resize(224, 224).cast("f32")
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

pipe = Pipeline().source("file_path", cloud_options=options).sink("numpy")
```

## Other Sources

- `raw`: raw bytes, requires explicit `dtype`
- `blob`: self-describing binary VIEW protocol
- `list`/`array`: infer from Polars column type (or override with `dtype`)
- `contour`: rasterizes contour structs to mask buffers
