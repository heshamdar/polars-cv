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

## Arrow's tensor type

`sink("fixed_shape_tensor")` emits Arrow's canonical
[`arrow.fixed_shape_tensor`](https://arrow.apache.org/docs/format/CanonicalExtensions.html#fixed-shape-tensor):
each row is one flat `Array(dtype, n)` of its elements in row-major order, and
the shape is in the type's metadata. PyArrow reads it as a
`FixedShapeTensorArray`, and so does any other reader of the canonical type.
A `sink("array")` column holds the same values nested one `Array` per axis,
which only Polars reads as a tensor.

```python
out = df.select(t=pl.col("image").cv.pipe(pipe).sink("fixed_shape_tensor"))
out.schema["t"]
# Extension('arrow.fixed_shape_tensor', Array(UInt8, shape=(150528,)), '{"shape":[224,224,3]}')
```

`out.to_arrow()` hands PyArrow the column as its `FixedShapeTensorArray`, whose
`to_numpy_ndarray()` gives one `(n_rows, 224, 224, 3)` array.

Like `array`, it needs the full shape at planning time (`shape=[...]` on the
sink supplies it), and every row is copied once into the column's values. The
type is Arrow's, so polars-cv registers no class for it: Polars shows its
generic `pl.Extension`.

!!! note
    A row that is null cannot cross Parquet between Polars and PyArrow in any
    fixed-size list column, this one and `array` included: PyArrow (22) can
    neither write such a column nor read the one Polars writes. Polars itself
    round-trips it.

## Persistence

A tagged column written to Parquet reads back tagged in any process that has
imported `polars_cv`. A process that has not reads it as Polars' generic
extension of the same name (`pl.Extension("polars_cv.point", ...)`, Polars 2.0's
default) — no data is lost, and `.ext.storage()` gives the plain struct. Setting
`POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage` reads the plain struct
directly instead.
