# Extension Types

polars-cv's outputs are plain Polars structs: a point is `{x, y}`, a contour is
`{exterior, holes, is_closed}`, the numpy sink is `{data, dtype, shape, strides,
offset}`. A struct says what fields it has, not what it *is*. Arrow
**extension types** add that: a column tagged `polars_cv.point` is a point by
type, and keeps that identity through Parquet and IPC.

| Type | Class | Storage |
|------|-------|---------|
| `polars_cv.ndarray` | `NdArrayType` | `NUMPY_OUTPUT_SCHEMA` |
| `polars_cv.point` | `PointType` | `POINT_SCHEMA` |
| `polars_cv.contour` | `ContourType` | `CONTOUR_SCHEMA` |
| `polars_cv.bbox` | `BBoxType` | `BBOX_SCHEMA` |

The types are registered with Polars when you `import polars_cv`.

!!! warning
    Polars documents its extension-type API as unstable. polars-cv keeps its
    use of it in one module on each side, but behaviour may follow Polars
    changes between releases.

## Producing tagged columns

**Arrays** — `sink("ndarray")` is `sink("numpy")` with the `polars_cv.ndarray`
tag. Same rows, bytes and strides; `dtype="f16"` works the same way.

```python
import polars as pl
from polars_cv import NdArrayType, Pipeline, numpy_from_struct

pipe = Pipeline().source("image_bytes").resize(height=224, width=224)
out = df.select(t=pl.col("image").cv.pipe(pipe).sink("ndarray"))

assert isinstance(out.schema["t"], NdArrayType)
arr = numpy_from_struct(out["t"][0])
```

**Geometry** — tag any column that already has the canonical storage with
`.ext.to(...)`. It is a zero-copy relabel, and Polars rejects it if the storage
does not match exactly.

```python
from polars_cv import POINT_SCHEMA, PointType

points = pl.DataFrame(
    {"p": [{"x": 1.0, "y": 2.0}]}, schema={"p": POINT_SCHEMA}
).select(pl.col("p").ext.to(PointType()))
```

Geometry functions return plain structs; tag their output the same way if you
want to keep the type.

## Consuming tagged columns

Every polars-cv expression accepts a tagged column wherever it accepts the
plain struct, and computes exactly the same result — every `.point`,
`.contour`, `.bbox` and `.cv` accessor, and pipeline sources such as
`source("contour")`. `numpy_from_struct` and `show_images` read
`sink("ndarray")` output directly.

To check what a column is, test its dtype:

```python
isinstance(df.schema["p"], PointType)
```

This is a complete check. A column that carries the `polars_cv.point` name over
any other storage — say, written by another tool — comes back as Polars'
generic `pl.Extension`, never as `PointType`.

## Struct operations need `.ext.storage()`

Polars' struct operations do not see through a tag. On a tagged column these
fail:

| Operation | On a tagged column |
|-----------|-------------------|
| `.struct.field("x")` | error |
| `df.unnest(col)` | error |
| `.cast(<struct>)` | error |
| `pl.concat` with an untagged column | error |
| `.to_list()`, `.rows()`, indexing | work (plain dicts) |

Recover the struct first:

```python
df.select(pl.col("p").ext.storage().struct.field("x"))
```

This is why `sink("numpy")` stays untagged: tagging it would break existing
code that unnests or casts its output. Choose `sink("ndarray")` when you want
the type.

## Persistence

A tagged column written to Parquet reads back tagged in any process that has
imported `polars_cv`. A process that has not reads the plain struct — no data
is lost — and Polars prints a warning that the extension type is not
registered. Setting `POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_extension`
keeps the name as a generic extension instead.
