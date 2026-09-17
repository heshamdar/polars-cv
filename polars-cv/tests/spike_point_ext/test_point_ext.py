"""SPIKE (throwaway): tests for the Arrow extension-type spike.

Covers the three feasibility questions for ``polars_cv.point``, lazy first-use
registration, and a second isolated type (``polars_cv.ndarray``) that tags the
numpy sink struct.

Not part of the guarded suite — the extension-type API is documented as
unstable and this path is a throwaway probe. Run explicitly:

    uv run pytest tests/spike_point_ext/
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import polars as pl
import pytest

pytest.importorskip("polars_cv._lib", reason="requires the compiled plugin")

from polars_cv._spike_ext import is_extension_named  # noqa: E402
from polars_cv._spike_ndarray_ext import (  # noqa: E402
    NDARRAY_EXT_NAME,
    NdArray,
    ndarray_ext,
    ndarray_ext_identity,
    numpy_from_ext,
)
from polars_cv._spike_point_ext import (  # noqa: E402
    POINT_EXT_NAME,
    PointXY,
    point_ext,
    point_ext_translate,
)


def _is_point_ext(dtype: pl.DataType) -> bool:
    """True iff the dtype is the ``polars_cv.point`` extension (not decayed)."""
    return is_extension_named(dtype, POINT_EXT_NAME)


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


def test_import_is_lazy() -> None:
    """Importing a spike type module must NOT load the compiled ``.so``; the
    first real use is what triggers it. Run in a subprocess so module state is
    clean (other tests import ``_lib``)."""
    code = textwrap.dedent(
        """
        import sys
        import polars_cv._spike_point_ext as m
        assert "polars_cv._lib" not in sys.modules, "import eagerly loaded _lib"
        m.point_ext("a", "b")  # first use -> ensure_registered -> _lib import
        assert "polars_cv._lib" in sys.modules, "first use did not load _lib"
        print("LAZY_OK")
        """
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "LAZY_OK" in r.stdout


# --- Second isolated type: polars_cv.ndarray (tags the numpy sink struct) ---


def test_ndarray_roundtrip_contiguous() -> None:
    """A tagged ndarray column round-trips and the tag-aware reader reconstructs it."""
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    s = ndarray_ext(a)
    assert is_extension_named(s.dtype, NDARRAY_EXT_NAME), s.dtype
    np.testing.assert_array_equal(numpy_from_ext(s), a)


def test_ndarray_roundtrip_transposed() -> None:
    """A non-contiguous (transposed) array survives the tag round-trip by value."""
    a = np.arange(6, dtype=np.uint8).reshape(2, 3).T
    s = ndarray_ext(a)
    assert is_extension_named(s.dtype, NDARRAY_EXT_NAME), s.dtype
    np.testing.assert_array_equal(numpy_from_ext(s), a)


def test_ndarray_identity_op_keeps_tag() -> None:
    """The plugin identity op receives the tag and returns it still tagged."""
    a = np.arange(4, dtype=np.float32).reshape(2, 2)
    df = pl.DataFrame({"t": ndarray_ext(a)})
    out = df.select(same=ndarray_ext_identity(pl.col("t")))
    assert is_extension_named(out.schema["same"], NDARRAY_EXT_NAME), out.schema["same"]
    np.testing.assert_array_equal(numpy_from_ext(out.to_series()), a)


def test_ndarray_host_type_identity() -> None:
    """The ndarray host type reports the expected name and numpy-struct storage."""
    from polars_cv import NUMPY_OUTPUT_SCHEMA

    t = NdArray()
    assert t.ext_name() == NDARRAY_EXT_NAME
    assert t.ext_storage() == NUMPY_OUTPUT_SCHEMA
