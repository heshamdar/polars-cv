"""``sink("ndarray")``: the numpy sink struct, tagged ``polars_cv.ndarray``.

The tagged sink is the plain ``sink("numpy")`` struct with a type on top, so
every assertion here is a comparison against ``sink("numpy")``: same rows, same
bytes, same strides, same nulls — plus the tag, planned and executed alike.
``sink("numpy")`` itself must not change; users' struct operations on it do not
see through a tag (see ``polars_cv.extension_types``).
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


def _both(pipe: Pipeline, **sink_kwargs: object) -> tuple[pl.Series, pl.Series]:
    """The same pipeline through ``sink("numpy")`` and ``sink("ndarray")``."""
    df = _images()
    plain = df.select(
        out=pl.col("img").cv.pipe(pipe).sink("numpy", **sink_kwargs)
    ).to_series()
    tagged = assert_plan_equals_exec(
        df, pl.col("img").cv.pipe(pipe).sink("ndarray", **sink_kwargs)
    )
    return plain, tagged


def _is_null_row(row: dict | None) -> bool:
    """A null input row, as the numpy struct encodes it: all fields null.

    (Not a null struct — that is ``sink("numpy")``'s existing encoding, and the
    tagged sink must match it rather than improve on it silently.)
    """
    return row is None or row["data"] is None


def _arrays(series: pl.Series) -> list[np.ndarray | None]:
    return [None if _is_null_row(row) else numpy_from_struct(row) for row in series]


PIPELINES = {
    "decode": lambda: Pipeline().source("image_bytes"),
    "grayscale": lambda: Pipeline().source("image_bytes").grayscale(),
    # Non-contiguous outputs: the strided branch of the struct encoding.
    "transpose": lambda: Pipeline().source("image_bytes").transpose([1, 0, 2]),
    "flip": lambda: Pipeline().source("image_bytes").flip([0]),
    "float": lambda: Pipeline().source("image_bytes").cast("f32").scale(0.5),
}


@pytest.mark.parametrize("name", sorted(PIPELINES))
def test_ndarray_is_the_numpy_struct_tagged(name: str) -> None:
    plain, tagged = _both(PIPELINES[name]())

    assert plain.dtype == polars_cv.NUMPY_OUTPUT_SCHEMA
    assert isinstance(tagged.dtype, NdArrayType)
    assert tagged.ext.storage().to_list() == plain.to_list()


@pytest.mark.parametrize("name", sorted(PIPELINES))
def test_numpy_from_struct_reads_the_tagged_rows(name: str) -> None:
    plain, tagged = _both(PIPELINES[name]())
    for got, want in zip(_arrays(tagged), _arrays(plain), strict=True):
        if want is None:
            assert got is None
        else:
            np.testing.assert_array_equal(got, want)


def test_numpy_from_struct_accepts_a_tagged_single_row_series() -> None:
    _, tagged = _both(PIPELINES["transpose"]())
    one = tagged.slice(0, 1)
    assert isinstance(one.dtype, NdArrayType)
    np.testing.assert_array_equal(numpy_from_struct(one), numpy_from_struct(one[0]))


def test_null_input_rows_encode_as_the_numpy_sink_does() -> None:
    plain, tagged = _both(PIPELINES["decode"]())
    assert _is_null_row(tagged[1])
    assert tagged[1] == plain[1]
    assert not _is_null_row(tagged[0]) and not _is_null_row(tagged[2])


def test_the_f16_downcast_applies() -> None:
    plain, tagged = _both(PIPELINES["float"](), dtype="f16")
    assert {row["dtype"] for row in tagged if not _is_null_row(row)} == {"float16"}
    assert tagged.ext.storage().to_list() == plain.to_list()


def test_ndarray_inside_a_multi_output_struct() -> None:
    base = pl.col("img").cv.pipe(Pipeline().source("image_bytes")).alias("base")
    gray = base.pipe(Pipeline().grayscale()).alias("gray")
    expr = gray.sink({"base": "ndarray", "gray": "numpy"})

    out = assert_plan_equals_exec(_images(), expr)

    fields = dict(zip(out.struct.fields, out.dtype.fields, strict=True))  # type: ignore[union-attr]
    assert isinstance(fields["base"].dtype, NdArrayType)
    assert fields["gray"].dtype == polars_cv.NUMPY_OUTPUT_SCHEMA


def test_parquet_round_trip(tmp_path) -> None:
    _, tagged = _both(PIPELINES["transpose"]())
    path = tmp_path / "arrays.parquet"
    pl.DataFrame({"a": tagged}).write_parquet(path)

    back = pl.read_parquet(path)["a"]
    assert isinstance(back.dtype, NdArrayType)
    for got, want in zip(_arrays(back), _arrays(tagged), strict=True):
        if want is None:
            assert got is None
        else:
            np.testing.assert_array_equal(got, want)


def test_the_plain_numpy_sink_is_unchanged() -> None:
    """``sink("numpy")`` keeps emitting the untagged struct: tagging it would
    break ``.struct.field`` / ``unnest`` / casts in existing user code."""
    plain, _ = _both(PIPELINES["decode"]())
    assert plain.dtype == polars_cv.NUMPY_OUTPUT_SCHEMA
    assert not isinstance(plain.dtype, pl.datatypes.BaseExtension)


def test_show_images_recognises_the_tag(capsys) -> None:
    """``format="auto"`` identifies an ndarray column by its type."""
    _, tagged = _both(PIPELINES["decode"]())
    polars_cv.show_images(pl.DataFrame({"a": tagged}), "a")
    out = capsys.readouterr().out
    assert "numpy struct: shape=[6, 9, 3], dtype=uint8" in out
    assert "numpy struct: shape=[4, 5, 4], dtype=uint8" in out
