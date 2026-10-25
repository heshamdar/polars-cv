"""Dtype spellings have one authority: ``dtype_table!`` in ``view-buffer/src/core/dtype.rs``."""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl
import pytest

from tests._discovery import (
    requires_checkout,
    rust_sources,
)
from tests._dtype_ratchet import dispatch_offenders

#: A structural guard: it checks the *shape* of the codebase rather than the
#: numerical behaviour of a pipeline. `-m structural` is the lane pre-commit
#: runs; see `tests/AGENTS.md`.
pytestmark = pytest.mark.structural


def _load_script(name: str):
    """Import a ``scripts/`` module by path.

    The generators are not an importable package, and putting ``scripts/`` on
    ``sys.path`` at module scope would reorder every import in this file.
    """
    import importlib.util

    path = Path(__file__).resolve().parent.parent / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@requires_checkout
def test_dtype_names_module_is_current() -> None:
    """``_dtype_names.py`` must match what ``gen_dtype_names.py`` produces.

    Python cannot read `dtype_table!` at runtime — `numpy_name` does not cross
    the FFI, and `numpy_from_struct` has to work with no compiled extension —
    so the spellings are generated and checked in. Regenerate-and-diff is what
    keeps that copy honest; without it the file is just a fourth hand-written
    dtype table, which is what it replaced.
    """
    gen = _load_script("gen_dtype_names")
    generated = gen.generate()
    committed = Path(gen._OUT_PATH)
    assert committed.exists(), (
        "_dtype_names.py is missing; run python scripts/gen_dtype_names.py"
    )
    assert committed.read_text() == generated, (
        "_dtype_names.py is out of date. Run: python scripts/gen_dtype_names.py"
    )


def test_the_numpy_allowlist_holds_no_character_codes() -> None:
    """The names must be numpy's *spelled* dtypes, not its character codes.

    This is the specific defect the generated module replaced: the hand-written
    allowlist admitted ``"u8"``/``"i8"``/``"f2"``, where numpy reads ``"u8"`` as
    ``uint64`` — the opposite of this project's ``u8``. A caller hand-building a
    struct got a silent reinterpretation of the bytes. Pinned as its own
    assertion because a regenerated file would happily carry them back if the
    generator ever read the wrong column.
    """
    from polars_cv._dtype_names import NUMPY_NAMES
    from polars_cv._types import DType

    collisions = NUMPY_NAMES & {m.value for m in DType}
    assert not collisions, (
        f"these numpy names collide with engine dtype names: {sorted(collisions)}. "
        f"numpy reads 'u8' as uint64 and this project reads it as uint8."
    )
    assert all(not name[-1].isdigit() or len(name) > 3 for name in NUMPY_NAMES), (
        f"character-code-shaped entries in NUMPY_NAMES: {sorted(NUMPY_NAMES)}"
    )


def _dtype_table_rows() -> list[tuple[str, str, int, str]]:
    """``dtype_table!``'s rows, parsed by the generator that also reads them.

    The parse lives in ``scripts/gen_dtype_names.py`` because that script has to
    read the table anyway, and two parsers of one table is the shape these
    guards exist to catch. Imported rather than re-implemented.
    """
    return _load_script("gen_dtype_names").dtype_table_rows()


def test_engine_dtype_names_match_the_generated_table() -> None:
    """``_types.DType`` must spell exactly what ``dtype_table!`` spells.

    ``DType`` is generated from view-buffer's ``naming::REGISTRY`` (via
    ``enum_catalog.json``), and the catalogue is checked against the built
    extension — which needs the extension. Editing Python is exactly when the
    extension is stale, so the check that matters most runs in the plugin-free
    lane, against the generated module. This is also what gives ``SHORT_NAMES`` a reader: a generated
    constant nothing reads is a fourth dtype table with extra steps.
    """
    from polars_cv._dtype_names import SHORT_NAMES, WIRE_CODES
    from polars_cv._types import DType

    assert SHORT_NAMES == {member.value for member in DType}, (
        "polars_cv._types.DType has drifted from dtype_table!: "
        f"only in DType={sorted({m.value for m in DType} - SHORT_NAMES)}, "
        f"only in dtype_table!={sorted(SHORT_NAMES - {m.value for m in DType})}"
    )
    # And the wire codes cover the same names, which is what makes WIRE_CODES
    # usable as the authority `display.py` is pinned to below.
    assert set(WIRE_CODES) == SHORT_NAMES


def test_display_reads_the_generated_wire_code_table() -> None:
    """``display.py`` must not carry its own copy of the wire codes.

    It used to, guarded by a regex over its own source -- the weakest of the
    three guard kinds `CLAUDE.md` ranks, and one a reformat silently defeats.
    It now imports ``NUMPY_BY_WIRE_CODE``, which `gen_dtype_names.py` emits
    from ``dtype_table!`` under a regenerate-and-diff, so this only has to
    check that it still reads the table rather than re-deriving it.
    """
    import polars_cv.display as display_mod

    source = Path(display_mod.__file__).read_text()
    assert "NUMPY_BY_WIRE_CODE" in source, (
        "display.py no longer reads the generated wire-code table"
    )
    assert not re.search(r"dtype_map\s*=\s*\{", source), (
        "display.py declares a local wire-code map again; dtype_table! is the "
        "authority and _dtype_names.py is how Python reads it"
    )


def test_display_rejects_unknown_wire_codes() -> None:
    """An unrecognised dtype code must raise, not render as uint8. Guessing
    reinterprets the payload and produces a plausible but meaningless image."""
    import polars_cv.display as display_mod

    blob = bytearray(b"VIEW" + bytes(60))
    blob[6] = 200  # not a dtype code
    blob[7] = 2
    with pytest.raises(ValueError, match="unknown VIEW dtype code"):
        display_mod._view_to_png(bytes(blob))


# Functions that map to a *subset* of DType variants on purpose. The ratchet
# fails on any unlisted partial rather than skipping it: every blind spot it has
# had was something it chose to skip, so an unrecognised shape must be an
# explicit, reviewable entry here rather than silence.
ALLOWED_PARTIAL_VARIANT_MAPS = frozenset(
    {
        # Arrow FFI import supports a subset by design and rejects the rest
        # with "Unsupported Arrow type" rather than guessing, so the missing
        # arms are an honest refusal, not silent drift.
        "view-buffer/src/interop/arrow_ffi.rs::from_arrow_ffi",
    }
)


def test_no_second_dtype_spelling_table() -> None:
    """Ratchet: dtype names are declared in ``dtype_table!`` only.

    Some dispatch genuinely has to name every dtype: matching a string to a
    Polars ``DataType``, or to a macro arm that needs the variant as a literal
    token. Those are correspondences, not naming decisions, and cannot be
    derived away. What must not happen is one of them disagreeing with the
    authority — a missing arm or a stray ``"f16"`` is how a dtype ends up
    handled in nine places and forgotten in the tenth.

    So this does not forbid the tables; it requires any dtype dispatch to name
    *exactly* the ten that ``dtype_table!`` declares. New code that only needs
    the name should still use ``DType::short_name`` / ``numpy_name`` /
    ``wire_code`` rather than adding an eleventh site.

    Two earlier versions of this test were unsound, so the technique matters:

    - Checking per *file* let ``encode.rs``'s two dispatches cover for each
      other, so deleting an arm from one passed.
    - Checking per brace-matched ``match`` block fixed that but was
      non-monotonic: with a "six or more names must be all ten" threshold,
      dropping four arms failed while dropping five *passed*, because the
      block fell under the threshold. Damage concealed itself by growing. The
      brace matcher was also defeatable by an unbalanced ``{`` inside a string
      literal (``polars_bail!("expected one of {")``) and by a brace in the
      scrutinee (``match Key { id: 1 } {``), either of which merged sibling
      blocks and restored the per-file hole.

    This version does no brace matching and has no threshold. It groups arms
    by enclosing function and requires each function's dtype-arm set to be
    exactly the ten. Multiplicity is checked too: a function holding two
    dispatches names every dtype twice, so dropping one arm makes the counts
    uneven even though the *set* is still complete.

    Two limits, stated because a guard that overstates its reach is how this
    test went wrong twice already:

    - It checks that the arm *keys* are all present, never that a key maps to
      the right thing. ``"u32" | "u64" => build_typed_list_u32(..)`` folds two
      dtypes onto one builder and passes: the set is complete. Only a test that
      executes the dispatch can catch that.
    - It reads ``match`` arms. A dtype table written as an if/else chain or a
      ``HashMap`` literal is invisible. Arm keys and ``DType::`` variant keys
      can be found without parsing Rust, and the two parsing attempts above are
      what produced two unsound versions.
    """
    root = Path(__file__).resolve().parents[2]
    allowed = {
        # The authority itself.
        root / "view-buffer" / "src" / "core" / "dtype.rs",
        # Pins the frozen VIEW codes literally, on purpose.
        root / "view-buffer" / "tests" / "dtype_single_authority.rs",
        # `SinkDType`: half precision exists only as the tensor sinks'
        # encode-time downcast (the engine has no f16), so its name is
        # deliberately outside `dtype_table!`. The file holds nothing else.
        root / "polars-cv" / "src" / "formats" / "sink_dtype.rs",
    }
    expected_pairs = {
        (variant, short) for variant, short, _code, _numpy in _dtype_table_rows()
    }

    offenders = []
    for path in rust_sources():
        if path in allowed:
            continue
        offenders += dispatch_offenders(
            path.read_text(),
            str(path.relative_to(root)),
            expected_pairs,
            ALLOWED_PARTIAL_VARIANT_MAPS,
        )

    assert not offenders, (
        "these dtype dispatches do not agree with dtype_table! in "
        f"view-buffer/src/core/dtype.rs: {offenders}"
    )


def test_dtype_name_for_covers_the_whole_table() -> None:
    """``dtype_name_for`` must name every dtype ``dtype_table!`` declares.

    This replaces a guard that compared a *private copy* of the correspondence,
    kept inside the metrics contour matcher, against the Rust table. The copy is
    gone -- ``polars_cv.dtype_name_for`` reads the generated ``NUMPY_TO_SHORT``
    -- so what is left to check is coverage: every engine dtype must be
    reachable from some Polars type, or a column the engine can hold has no way
    to be named at a source.
    """
    import polars as pl

    from polars_cv import dtype_name_for

    expected = {short for _variant, short, _code, _numpy in _dtype_table_rows()}
    reached = set()
    for name in dir(pl):
        candidate = getattr(pl, name)
        if not isinstance(candidate, type) or not issubclass(candidate, pl.DataType):
            continue
        try:
            reached.add(dtype_name_for(candidate))
        except (ValueError, TypeError):
            continue
    assert expected <= reached, (
        "dtype_name_for cannot name every dtype_table! row: "
        f"unreachable={sorted(expected - reached)}"
    )


def test_contour_source_accepts_boolean_masks() -> None:
    """A ``List(Boolean)`` mask is the documented shape for ``gt_col``.

    Regression guard. It briefly raised, on the theory that Boolean is not a
    buffer element type and declaring one was a silent lie. The source decoder
    *casts* rather than reinterprets (``convert_row_values`` in graph/decode.rs),
    so the declared dtype was always honest and boolean masks worked; the
    rejection broke them.
    """
    from polars_cv.metrics._matching._contour import _detect_source_info

    info = _detect_source_info({"mask": pl.List(pl.List(pl.Boolean))}, "mask")
    assert info.kwargs["dtype"] == "u8"

    # And through the public accessor the matcher now reads it from.
    from polars_cv import dtype_name_for

    assert dtype_name_for(pl.Boolean) == "u8"


def test_contour_source_rejects_types_with_no_buffer_meaning() -> None:
    """Types whose cast would fail or lose meaning must still be refused up
    front, rather than becoming an f32 source that fails later and deeper."""
    from polars_cv.metrics._matching._contour import _detect_source_info

    for leaf in (pl.String, pl.Duration):
        with pytest.raises(ValueError, match="no meaningful buffer"):
            _detect_source_info({"mask": pl.List(pl.List(leaf))}, "mask")
