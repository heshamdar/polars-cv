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

from polars_cv._spike_bbox_ext import (  # noqa: E402
    BBOX_EXT_NAME,
    BBox,
    bbox_ext,
    bbox_ext_identity,
)
from polars_cv._spike_contour_ext import (  # noqa: E402
    CONTOUR_EXT_NAME,
    Contour,
    contour_ext,
    contour_ext_identity,
)
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


# --- Geometry family: contour + bbox ---


def test_bbox_roundtrip_and_op_keep_tag() -> None:
    """A tagged bbox column round-trips and survives the identity op still tagged."""
    df = pl.DataFrame({"x": [1.0], "y": [2.0], "w": [3.0], "h": [4.0]})
    out = df.select(b=bbox_ext("x", "y", "w", "h"))
    assert is_extension_named(out.schema["b"], BBOX_EXT_NAME), out.schema["b"]
    assert df.select(
        s=bbox_ext("x", "y", "w", "h").ext.storage()
    ).to_series().to_list() == [{"x": 1.0, "y": 2.0, "width": 3.0, "height": 4.0}]

    passed = df.select(same=bbox_ext_identity(bbox_ext("x", "y", "w", "h")))
    assert is_extension_named(passed.schema["same"], BBOX_EXT_NAME), passed.schema[
        "same"
    ]


def test_bbox_host_type_identity() -> None:
    from polars_cv.geometry.schemas import BBOX_SCHEMA

    t = BBox()
    assert t.ext_name() == BBOX_EXT_NAME
    assert t.ext_storage() == BBOX_SCHEMA


def test_contour_roundtrip_and_op_keep_tag() -> None:
    """A tagged contour column round-trips (nested storage intact) and survives the op."""
    s = contour_ext([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
    assert is_extension_named(s.dtype, CONTOUR_EXT_NAME), s.dtype

    storage = s.ext.storage()
    row = storage.struct.unnest().row(0, named=True)
    assert row["exterior"] == [
        {"x": 0.0, "y": 0.0},
        {"x": 10.0, "y": 0.0},
        {"x": 10.0, "y": 10.0},
        {"x": 0.0, "y": 10.0},
    ]
    assert row["holes"] == []

    df = pl.DataFrame({"c": s})
    out = df.select(same=contour_ext_identity(pl.col("c")))
    assert is_extension_named(out.schema["same"], CONTOUR_EXT_NAME), out.schema["same"]


def test_contour_host_type_identity() -> None:
    from polars_cv.geometry.schemas import CONTOUR_SCHEMA

    t = Contour()
    assert t.ext_name() == CONTOUR_EXT_NAME
    assert t.ext_storage() == CONTOUR_SCHEMA


# --- Persistence: what survives a Parquet round trip ---


def test_parquet_persistence(tmp_path) -> None:
    """A tagged column persisted to Parquet keeps its tag when read back by a
    process that has the type registered, and its storage data is intact even for
    a reader that does not — documenting the Parquet/Iceberg persistence behavior."""
    df = pl.DataFrame({"a": [1.0, 3.0], "b": [2.0, 4.0]}).select(kp=point_ext("a", "b"))
    path = tmp_path / "kp.parquet"
    df.write_parquet(path)

    # Reader WITH the type registered (this process): tag reconstructed.
    back = pl.read_parquet(path)
    assert is_extension_named(back.schema["kp"], POINT_EXT_NAME), back.schema["kp"]
    assert back.select(pl.col("kp").ext.storage()).to_series().to_list() == [
        {"x": 1.0, "y": 2.0},
        {"x": 3.0, "y": 4.0},
    ]

    # Reader WITHOUT polars_cv imported (subprocess): storage data must be intact.
    # Whether the tag survives as a generic extension or decays to the struct is
    # recorded (printed) rather than asserted — it is the documented degradation.
    code = textwrap.dedent(
        f"""
        import polars as pl
        s = pl.read_parquet(r"{path}")["kp"]
        name = getattr(s.dtype, "ext_name", None)
        tagged = callable(name)
        storage = s.ext.storage() if tagged else s
        print("DTYPE", repr(s.dtype))
        print("VALUES", storage.to_list())
        assert storage.to_list() == [{{"x": 1.0, "y": 2.0}}, {{"x": 3.0, "y": 4.0}}]
        print("OK")
        """
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout
