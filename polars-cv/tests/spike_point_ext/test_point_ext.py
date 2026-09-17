"""SPIKE (throwaway): tests answering the three feasibility questions for an
Arrow extension type on ``polars_cv.point``.

Not part of the guarded suite — the extension-type API is documented as
unstable and this path is a throwaway probe. Run explicitly:

    uv run pytest tests/spike_point_ext/
"""

from __future__ import annotations

import polars as pl
import pytest

pytest.importorskip("polars_cv._lib", reason="requires the compiled plugin")

from polars_cv._spike_point_ext import (  # noqa: E402
    POINT_EXT_NAME,
    PointXY,
    point_ext,
    point_ext_translate,
)


def _is_point_ext(dtype: pl.DataType) -> bool:
    """True iff the dtype is the ``polars_cv.point`` extension (not decayed).

    Tolerant of which class the reconstructed dtype presents as (the built-in
    ``Extension`` vs. the registered ``PointXY``) — an unstable-API detail; the
    load-bearing check is the extension name.
    """
    ext_name = getattr(dtype, "ext_name", None)
    try:
        return callable(ext_name) and ext_name() == POINT_EXT_NAME
    except Exception:
        return False


def test_q1_construct_and_roundtrip_keeps_tag() -> None:
    """Q2a: a tagged column survives collect and its storage is recoverable."""
    df = pl.DataFrame({"a": [1.0, 3.0], "b": [2.0, 4.0]})
    out = df.select(pt=point_ext("a", "b"))

    assert _is_point_ext(out.schema["pt"]), out.schema["pt"]

    storage = df.select(s=point_ext("a", "b").ext.storage()).to_series()
    assert storage.dtype == pl.Struct({"x": pl.Float64, "y": pl.Float64})
    assert storage.to_list() == [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}]


def test_q2_plugin_op_receives_and_returns_tag() -> None:
    """Q2b: the tag reaches *inside* the plugin (ext() succeeds there) and the
    output is still tagged ``polars_cv.point``."""
    df = pl.DataFrame({"a": [1.0, 10.0], "b": [2.0, 20.0]})
    out = df.select(moved=point_ext_translate(point_ext("a", "b"), dx=5.0, dy=-1.0))

    assert _is_point_ext(out.schema["moved"]), out.schema["moved"]

    moved_storage = (
        df.select(
            m=point_ext_translate(point_ext("a", "b"), dx=5.0, dy=-1.0).ext.storage()
        )
        .to_series()
        .to_list()
    )
    assert moved_storage == [{"x": 6.0, "y": 1.0}, {"x": 15.0, "y": 19.0}]


def test_q3_non_point_input_fails_at_schema_resolution() -> None:
    """Q3: a plain ``{x, y}`` struct (untagged) is rejected with a clear type
    error, not a confusing query-time parse failure."""
    df = pl.DataFrame({"a": [1.0], "b": [2.0]})
    plain = pl.struct(pl.col("a").alias("x"), pl.col("b").alias("y"))

    with pytest.raises(Exception) as excinfo:
        df.select(point_ext_translate(plain, dx=1.0, dy=1.0))

    assert "polars_cv.point" in str(excinfo.value)


def test_host_type_identity() -> None:
    """The registered host type reports the expected name and storage."""
    t = PointXY()
    assert t.ext_name() == POINT_EXT_NAME
    assert t.ext_storage() == pl.Struct({"x": pl.Float64, "y": pl.Float64})
