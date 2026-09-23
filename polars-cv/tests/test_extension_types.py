"""polars-cv's Arrow extension types: declaration, registration, and the one
way into the plugin.

Four types tag the structs polars-cv already emits — ``polars_cv.ndarray``
(the numpy/torch sink struct), ``polars_cv.point``, ``polars_cv.contour`` and
``polars_cv.bbox`` — so a consumer can identify a column by its type instead of
by sniffing field names.

The facts pinned here, and why each is load-bearing:

- The Python classes and the Rust ``ExtType`` agree on every name and storage
  (``test_python_types_match_the_rust_declaration``). Rust owns the storage
  layouts (``geom_schema``, ``output::numpy_output_dtype``); Python restates
  them only so ``import polars_cv`` can register types without loading the
  compiled extension, and this parity check is what keeps that copy honest.
- An instance of a polars-cv type *always* has canonical storage. A column
  that merely carries the name over other storage reconstructs as polars'
  generic ``Extension`` rather than posing as ours.
- ``import polars_cv`` registers every type, so a tagged Parquet column keeps
  its tag on read.
- ``polars_cv._plugin.call`` pins polars to the exact extension file Python
  imports and hands the plugin *storage*, so no Rust function ever sees an
  extension dtype. That it is the *only* way in is pinned separately, by the
  structural scan in ``test_plugin_entry_point.py``.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

import polars as pl
import pytest

import polars_cv
from polars_cv import extension_types as ext
from polars_cv.geometry.schemas import BBOX_SCHEMA, CONTOUR_SCHEMA, POINT_SCHEMA

from .conftest import plugin_required

_PROJECT = Path(__file__).resolve().parents[1]


def _run(code: str) -> subprocess.CompletedProcess[str]:
    """Run *code* in a fresh interpreter, so import-time state is clean."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        cwd=_PROJECT,
    )


# ---------------------------------------------------------------------------
# Declaration
# ---------------------------------------------------------------------------


def test_the_four_types_are_declared_once() -> None:
    assert [t.NAME for t in ext.EXTENSION_TYPES] == [
        "polars_cv.ndarray",
        "polars_cv.point",
        "polars_cv.contour",
        "polars_cv.bbox",
    ]


def test_storages_are_the_published_schemas() -> None:
    """The types tag the schemas polars-cv already publishes, not copies."""
    assert ext.NdArrayType.STORAGE == polars_cv.NUMPY_OUTPUT_SCHEMA
    assert ext.PointType.STORAGE == POINT_SCHEMA
    assert ext.ContourType.STORAGE == CONTOUR_SCHEMA
    assert ext.BBoxType.STORAGE == BBOX_SCHEMA


def test_types_are_exported_from_the_package() -> None:
    for t in ext.EXTENSION_TYPES:
        assert getattr(polars_cv, t.__name__) is t


@pytest.mark.parametrize("cls", ext.EXTENSION_TYPES, ids=lambda c: c.NAME)
def test_instance_reports_name_and_storage(cls) -> None:
    t = cls()
    assert t.ext_name() == cls.NAME
    assert t.ext_storage() == cls.STORAGE
    assert t.ext_metadata() is None


@plugin_required
def test_python_types_match_the_rust_declaration() -> None:
    """Names, order and storage agree with ``ExtType`` in both directions."""
    from polars_cv import _lib

    rust = [(name, empty.dtype) for name, empty in _lib.extension_types()]
    python = [(t.NAME, t.STORAGE) for t in ext.EXTENSION_TYPES]
    assert python == rust


# ---------------------------------------------------------------------------
# An instance of our type means canonical storage
# ---------------------------------------------------------------------------


def _point_column(dtype: pl.DataType) -> pl.DataFrame:
    return (
        pl.DataFrame({"x": [1.0, 3.0], "y": [2.0, 4.0]})
        .select(
            p=pl.struct("x", "y").cast(pl.Struct({"x": pl.Float64, "y": pl.Float64}))
        )
        .select(pl.col("p").ext.to(dtype))
    )


def test_canonical_column_reconstructs_as_our_type() -> None:
    df = _point_column(ext.PointType())
    assert isinstance(df.schema["p"], ext.PointType)
    assert df.select(pl.col("p").ext.storage()).to_series().to_list() == [
        {"x": 1.0, "y": 2.0},
        {"x": 3.0, "y": 4.0},
    ]


def test_generic_extension_by_our_name_with_canonical_storage_is_ours() -> None:
    """Spelling the name by hand yields our type when the storage is right."""
    df = _point_column(pl.Extension("polars_cv.point", POINT_SCHEMA))
    assert isinstance(df.schema["p"], ext.PointType)


def test_our_name_over_other_storage_is_not_our_type() -> None:
    """A malformed tag must not pose as ours.

    polars panics if ``ext_from_params`` raises, so the class answers a
    mismatch with the generic ``Extension`` instead: the column still exists,
    but nothing that checks for ``PointType`` will accept it.
    """
    f32 = pl.Struct({"x": pl.Float32, "y": pl.Float32})
    df = pl.select(
        p=pl.struct(
            pl.lit(1.0, pl.Float32).alias("x"), pl.lit(2.0, pl.Float32).alias("y")
        ).ext.to(pl.Extension("polars_cv.point", f32))
    )
    dtype = df.schema["p"]
    assert not isinstance(dtype, ext.PointType)
    assert isinstance(dtype, pl.Extension)
    assert dtype.ext_storage() == f32


def test_our_name_with_metadata_is_not_our_type() -> None:
    df = _point_column(pl.Extension("polars_cv.point", POINT_SCHEMA, "v2"))
    assert not isinstance(df.schema["p"], ext.PointType)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_import_registers_every_type_without_loading_the_plugin() -> None:
    r = _run(
        """
        import sys
        import polars as pl
        import polars_cv
        from polars_cv.extension_types import EXTENSION_TYPES
        assert "polars_cv._lib" not in sys.modules, "import loaded the plugin"
        for t in EXTENSION_TYPES:
            assert pl.get_extension_type(t.NAME) is t, t.NAME
        print("OK")
        """
    )
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_registration_is_idempotent() -> None:
    ext.register_extension_types()
    ext.register_extension_types()
    for t in ext.EXTENSION_TYPES:
        assert pl.get_extension_type(t.NAME) is t


def test_a_foreign_claim_on_our_name_is_refused(monkeypatch) -> None:
    """Another class registered under a ``polars_cv.`` name is a conflict.

    Registering over it would silently change how that code's columns decode;
    leaving it would silently change ours. Neither is ours to decide quietly.
    """

    class Foreign(pl.datatypes.BaseExtension):
        pass

    real = pl.get_extension_type
    monkeypatch.setattr(
        pl,
        "get_extension_type",
        lambda name: Foreign if name == "polars_cv.point" else real(name),
    )
    with pytest.raises(RuntimeError, match="polars_cv.point"):
        ext.register_extension_types()


def test_parquet_round_trip_keeps_the_tag(tmp_path) -> None:
    """A tagged column written to Parquet reads back tagged in a polars-cv
    process, and as its plain storage (with polars' warning) in one that never
    imported polars-cv — the documented degradation, not data loss."""
    path = tmp_path / "points.parquet"
    _point_column(ext.PointType()).write_parquet(path)

    back = pl.read_parquet(path)
    assert isinstance(back.schema["p"], ext.PointType)

    r = _run(
        f"""
        import warnings
        import polars as pl
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = pl.read_parquet(r"{path}")["p"]
        assert s.dtype == pl.Struct({{"x": pl.Float64, "y": pl.Float64}}), s.dtype
        assert s.to_list() == [{{"x": 1.0, "y": 2.0}}, {{"x": 3.0, "y": 4.0}}]
        print("OK")
        """
    )
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


# ---------------------------------------------------------------------------
# The one way into the plugin
# ---------------------------------------------------------------------------


@plugin_required
def test_plugin_path_is_the_file_python_imports() -> None:
    """polars must load the same extension file ``import polars_cv._lib`` loads.

    Given a directory, polars takes the first ``.so`` ``iterdir()`` returns;
    beside a stale second build that is an arbitrary one of the two.
    """
    from polars_cv import _plugin

    spec = importlib.util.find_spec("polars_cv._lib")
    assert spec is not None and spec.origin is not None
    assert Path(_plugin.plugin_path()).resolve() == Path(spec.origin).resolve()


@plugin_required
def test_plugin_receives_storage_not_the_tag() -> None:
    """A tagged input reaches the plugin as its storage, so the op computes on
    it exactly as it would on the untagged column."""
    tagged = _point_column(ext.PointType())
    plain = tagged.select(pl.col("p").ext.storage())

    got = tagged.select(pl.col("p").point.translate(1.0, 1.0)).to_series()
    want = plain.select(pl.col("p").point.translate(1.0, 1.0)).to_series()
    assert got.dtype == want.dtype == POINT_SCHEMA
    assert got.to_list() == want.to_list()


def test_call_rejects_non_expression_arguments() -> None:
    from polars_cv import _plugin

    with pytest.raises(TypeError, match="pl.Expr"):
        _plugin.call("point_translate", args=["p"])  # type: ignore[list-item]
