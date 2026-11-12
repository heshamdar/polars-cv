"""``sink("tensor")``: the tensor struct, tagged ``polars_cv.ndarray``.

The sink's column is the struct ``{data, dtype, shape, strides, offset}`` with
an extension type on top, planned and executed alike. The tag is what says a
column holds tensors after a round trip; polars' struct operations need the
storage under it (``.ext.storage()``, see ``polars_cv.extension_types``).

There were three sinks: ``numpy`` and ``torch`` wrote the plain struct and
``ndarray`` the tagged one. They are one, always tagged, so what was compared
here against the plain sink is compared against NumPy's own result.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import polars_cv
from polars_cv import NdArrayType, Pipeline, numpy_from_struct

from ._schema_parity import assert_plan_equals_exec
from .conftest import make_image_png, plugin_required

pytestmark = plugin_required


def _images() -> pl.DataFrame:
    """RGB and RGBA rows with a null between them."""
    return pl.DataFrame(
        {
            "img": [
                make_image_png(6, 9, 3, seed=1),
                None,
                make_image_png(4, 5, 4, seed=2),
            ]
        },
        schema={"img": pl.Binary},
    )


def _tensors(pipe: Pipeline, **sink_kwargs: object) -> pl.Series:
    """*pipe* through ``sink("tensor")``, its planned type checked against
    the executed one."""
    return assert_plan_equals_exec(
        _images(), pl.col("img").cv.pipe(pipe).sink("tensor", **sink_kwargs)
    )


def _decoded() -> list[np.ndarray | None]:
    """The rows of :func:`_images` as PIL decodes them."""
    import io

    from PIL import Image

    return [
        None if data is None else np.asarray(Image.open(io.BytesIO(data)))
        for data in _images()["img"]
    ]


def _is_null_row(row: dict | None) -> bool:
    """A null row: a null value (or, read from older files, a struct whose
    fields are all null)."""
    return row is None or row["data"] is None


def _arrays(series: pl.Series) -> list[np.ndarray | None]:
    return [None if _is_null_row(row) else numpy_from_struct(row) for row in series]


#: name -> (the pipeline, the same operation on a decoded NumPy array).
PIPELINES = {
    "decode": (lambda: Pipeline().source("image_bytes"), lambda a: a),
    "grayscale_shape": (
        lambda: Pipeline().source("image_bytes").channel_select(0),
        lambda a: a[..., :1],
    ),
    # Non-contiguous outputs: the strided branch of the struct encoding.
    "transpose": (
        lambda: Pipeline().source("image_bytes").transpose([1, 0, 2]),
        lambda a: a.transpose(1, 0, 2),
    ),
    "flip": (lambda: Pipeline().source("image_bytes").flip([0]), lambda a: a[::-1]),
    "float": (
        lambda: Pipeline().source("image_bytes").cast("f32").scale(0.5),
        lambda a: a.astype(np.float32) * np.float32(0.5),
    ),
}


@pytest.mark.parametrize("name", sorted(PIPELINES))
def test_the_tensor_sink_is_the_storage_struct_tagged(name: str) -> None:
    tagged = _tensors(PIPELINES[name][0]())

    assert isinstance(tagged.dtype, NdArrayType)
    assert tagged.ext.storage().dtype == polars_cv.NUMPY_OUTPUT_SCHEMA


@pytest.mark.parametrize("name", sorted(PIPELINES))
def test_numpy_from_struct_reads_the_tagged_rows(name: str) -> None:
    pipe, reference = PIPELINES[name]
    for got, image in zip(_arrays(_tensors(pipe())), _decoded(), strict=True):
        if image is None:
            assert got is None
        else:
            np.testing.assert_array_equal(got, reference(image))


def test_numpy_from_struct_accepts_a_tagged_single_row_series() -> None:
    one = _tensors(PIPELINES["transpose"][0]()).slice(0, 1)
    assert isinstance(one.dtype, NdArrayType)
    np.testing.assert_array_equal(numpy_from_struct(one), numpy_from_struct(one[0]))


def test_a_null_input_row_is_a_null_value() -> None:
    tagged = _tensors(PIPELINES["decode"][0]())
    assert tagged[1] is None
    assert tagged.null_count() == 1 == tagged.ext.storage().null_count()
    assert not _is_null_row(tagged[0]) and not _is_null_row(tagged[2])


def test_the_f16_downcast_applies() -> None:
    tagged = _tensors(PIPELINES["float"][0](), dtype="f16")
    assert {row["dtype"] for row in tagged if not _is_null_row(row)} == {"float16"}
    for got, image in zip(_arrays(tagged), _decoded(), strict=True):
        if image is not None:
            want = PIPELINES["float"][1](image).astype(np.float16)
            np.testing.assert_array_equal(got, want)


def test_every_tensor_field_of_a_multi_output_struct_is_tagged() -> None:
    base = pl.col("img").cv.pipe(Pipeline().source("image_bytes")).alias("base")
    gray = base.pipe(Pipeline().grayscale()).alias("gray")
    expr = gray.sink({"base": "tensor", "gray": "tensor"})

    out = assert_plan_equals_exec(_images(), expr)

    fields = dict(zip(out.struct.fields, out.dtype.fields, strict=True))  # type: ignore[union-attr]
    assert isinstance(fields["base"].dtype, NdArrayType)
    assert isinstance(fields["gray"].dtype, NdArrayType)


def test_parquet_round_trip(tmp_path) -> None:
    tagged = _tensors(PIPELINES["transpose"][0]())
    path = tmp_path / "arrays.parquet"
    pl.DataFrame({"a": tagged}).write_parquet(path)

    back = pl.read_parquet(path)["a"]
    assert isinstance(back.dtype, NdArrayType)
    for got, want in zip(_arrays(back), _arrays(tagged), strict=True):
        if want is None:
            assert got is None
        else:
            np.testing.assert_array_equal(got, want)


def test_show_images_recognises_the_tag(capsys) -> None:
    """``format="auto"`` identifies a tensor column by its type."""
    tagged = _tensors(PIPELINES["decode"][0]())
    polars_cv.show_images(pl.DataFrame({"a": tagged}), "a")
    out = capsys.readouterr().out
    assert "numpy struct: shape=[6, 9, 3], dtype=uint8" in out
    assert "numpy struct: shape=[4, 5, 4], dtype=uint8" in out
