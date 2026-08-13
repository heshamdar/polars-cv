"""Tests for the dtype ratchet itself.

``test_no_second_dtype_spelling_table`` guards the tree. Nothing guarded *it*,
and it was wrong seven times — each time silently, passing while covering less
than the version it replaced. Twice the regression was introduced while fixing
the previous one.

Every one of those seven is a fixture here. A checker that stops catching a
known defect now fails a test, instead of quietly widening its blind spot. The
good fixtures matter just as much: three of the seven rewrites false-positived
on correct, ``rustfmt``-clean code, which is how a ratchet gets weakened or
deleted by the next person who hits it.

These call the same ``dispatch_offenders`` the repo-wide test calls. A fixture
exercising a copy of the logic would prove nothing about the guard.
"""

from __future__ import annotations

import pytest

from tests._dtype_ratchet import dispatch_offenders

#: Every test here is a structural guard: it checks the *shape* of the codebase
#: -- registries, authorities, removed surfaces, documented vocabularies --
#: rather than the numerical behaviour of a pipeline. `-m structural` is the
#: lane pre-commit runs; see `tests/AGENTS.md`. Note that the lane as a whole
#: does need the compiled extension: many structural facts are only observable
#: through the FFI, and those tests fail rather than skip without it.
pytestmark = pytest.mark.structural

PAIRS = {
    ("U8", "u8"),
    ("I8", "i8"),
    ("U16", "u16"),
    ("I16", "i16"),
    ("U32", "u32"),
    ("I32", "i32"),
    ("U64", "u64"),
    ("I64", "i64"),
    ("F32", "f32"),
    ("F64", "f64"),
}
NAMES = {n for _, n in PAIRS}
VARIANTS = {v for v, _ in PAIRS}

_ALL_ARMS = "\n".join(f'        "{n}" => build_{n}(),' for n in sorted(NAMES))


def _key_dispatch(names: list[str], fn: str = "dispatch") -> str:
    arms = "\n".join(f'        "{n}" => build_{n}(),' for n in names)
    return f"pub fn {fn}(s: &str) -> S {{\n    match s {{\n{arms}\n        _ => d(),\n    }}\n}}"


def _value_table(pairs: list[tuple[str, str]], ty: str = "DType") -> str:
    arms = "\n".join(f'        {ty}::{v} => "{n}",' for v, n in pairs)
    return (
        f"pub fn name_of(d: {ty}) -> &'static str {{\n"
        f"    match d {{\n{arms}\n    }}\n}}"
    )


_COMPLETE_PAIRS = [(v, v.lower()) for v in sorted(VARIANTS)]


# Each entry is a defect the ratchet has failed to catch at some point.
BAD_FIXTURES = [
    pytest.param(
        _key_dispatch(sorted(NAMES - {"u64"})),
        "missing",
        id="one-arm-missing",
    ),
    pytest.param(
        _key_dispatch(sorted(NAMES - {"u64", "i64", "u32", "i32", "u16"})),
        "missing",
        # Defect 3: a ">=6 names must be all ten" threshold made damage
        # self-concealing — dropping four arms failed, dropping five passed.
        id="five-arms-missing-was-under-threshold",
    ),
    pytest.param(
        _key_dispatch(sorted(NAMES))
        + "\n"
        + _key_dispatch(sorted(NAMES - {"f64"}), "second"),
        "missing",
        # Defect 2: checked per file, so a complete dispatch covered for an
        # incomplete one elsewhere in the same file.
        id="second-dispatch-incomplete-same-file",
    ),
    pytest.param(
        "pub fn two(s: &str) -> S {\n    match s {\n"
        + _ALL_ARMS
        + "\n        _ => d(),\n    }\n    match s {\n"
        + "\n".join(f'        "{n}" => other_{n}(),' for n in sorted(NAMES - {"f64"}))
        + "\n        _ => d(),\n    }\n}",
        "disagree",
        # Two dispatches in one function: the set stays complete, so only the
        # multiplicity check catches the dropped arm.
        id="two-dispatches-one-fn-uneven",
    ),
    pytest.param(
        _value_table(
            [(v, n) if v != "U64" else (v, "u32") for v, n in _COMPLETE_PAIRS]
        ),
        "does not agree",
        # Defect 5: only arm keys were counted, so a variant-to-name table was
        # invisible — a coverage loss versus the version it replaced.
        id="value-side-table-wrong-name",
    ),
    pytest.param(
        _value_table(
            [(v, n) if v != "U64" else (v, "u32") for v, n in _COMPLETE_PAIRS],
            ty="VbDType",
        ),
        "does not agree",
        # Defect 7: `\bDType::` cannot match inside `VbDType::`, the alias this
        # repo actually uses, so the same table written against it was invisible.
        id="value-side-table-via-type-alias",
    ),
    pytest.param(
        _key_dispatch(sorted(NAMES - {"u64"})).replace(
            "        _ => d(),",
            '        _ => d(), // "u64" => build_u64(), removed, see #123',
        ),
        "missing",
        # Defect 6-adjacent: a trailing comment before the first `=>` counted as
        # an arm key, so the comment explaining a removal restored its coverage.
        id="removed-arm-mentioned-in-trailing-comment",
    ),
    pytest.param(
        _key_dispatch(sorted(NAMES) + ["f16"]),
        "unknown",
        id="unknown-dtype-name",
    ),
    pytest.param(
        "pub fn amb(d: DType) -> &'static str {\n    match d {\n"
        '        DType::U8 => pick("u8", "f32"),\n    }\n}',
        "several dtype names",
        id="arm-yields-two-names",
    ),
    pytest.param(
        "pub fn to_dtype(t: PolarsType) -> Option<DType> {\n    match t {\n"
        + "\n".join(
            f"        PolarsType::{v} => Some(DType::{v}),"
            for v in sorted(VARIANTS - {"U16", "I16"})
        )
        + "\n        _ => None,\n    }\n}",
        "not all ten",
        # Defect 8: a foreign-type -> DType table is invisible to both the key
        # and name checks. Five arms were deletable from two such tables in the
        # tree with a fully green suite.
        id="foreign-variant-to-dtype-table-incomplete",
    ),
    pytest.param(
        _value_table(
            [(v, n) for v, n in _COMPLETE_PAIRS if v not in ("U32", "U64")]
            + [("U32", "u64"), ("U64", "u32")]
        ),
        "wrong pairs",
        # Permutation blindness: two swapped rows leave both sets complete, so
        # comparing sets rather than pairs passed while every dtype was misnamed.
        id="value-side-table-rows-swapped",
    ),
    pytest.param(
        "#[cfg(test)]\nmod t { fn helper() {} }\n\n"
        + _key_dispatch(sorted(NAMES - {"f64"}), "after_tests"),
        "missing",
        # A production fn placed after the test module must still be scanned.
        id="production-fn-after-test-module",
    ),
    pytest.param(
        "#[cfg(test)]\nfn helper(\n    s: &str,\n) -> u8 {\n    match s {\n"
        '        "u8" => 1,\n        _ => 0,\n    }\n}\n\n'
        + _key_dispatch(sorted(NAMES - {"f64"}), "after_wrapped_cfg_fn"),
        "missing",
        # The pair of the good fixture above: skipping the wrapped `#[cfg(test)]
        # fn` correctly must not also swallow the production code following it.
        # Widening the strip until nothing false-positives is the other way to
        # get a guard that catches nothing.
        id="production-fn-after-cfg-test-fn-with-wrapped-signature",
    ),
]

# Correct code that must NOT be reported. Three rewrites false-positived here.
GOOD_FIXTURES = [
    pytest.param(_key_dispatch(sorted(NAMES)), id="complete-key-dispatch"),
    pytest.param(_value_table(_COMPLETE_PAIRS), id="complete-value-table"),
    pytest.param(_value_table(_COMPLETE_PAIRS, ty="VbDType"), id="complete-via-alias"),
    pytest.param(
        "pub fn is_dtype(s: &str) -> bool {\n    match s {\n"
        '        "u8" | "i8" | "u16" | "i16" | "u32" | "i32" | "u64" | "i64" | "f32"\n'
        '        | "f64" => true,\n        _ => false,\n    }\n}',
        # rustfmt wraps long alternations; the earlier names land on a line with
        # no `=>`. Reported as missing nine names before the rejoin.
        id="rustfmt-wrapped-alternation",
    ),
    pytest.param(
        "#[cfg(test)]\nmod helpers {\n    fn parse(s: &str) -> u8 {\n        match s {\n"
        + "\n".join(f'            "{n}" => 1,' for n in sorted(NAMES - {"f64", "u64"}))
        + "\n            _ => 0,\n        }\n    }\n}",
        # A partial dispatch inside a test module is not production drift.
        id="incomplete-dispatch-inside-test-module",
    ),
    pytest.param(
        "#[cfg(test)]\nmod first {\n    fn a() {}\n}\n\n"
        "#[cfg(test)]\nmod second {\n    fn parse(s: &str) -> u8 {\n        match s {\n"
        + "\n".join(f'            "{n}" => 1,' for n in sorted(NAMES - {"f64"}))
        + "\n            _ => 0,\n        }\n    }\n}",
        # Only the *first* test module used to be skipped, so a helper in the
        # second was scanned as production. `polars-cv/src/execute.rs` has two;
        # the tree was one committed helper away from a false positive.
        id="incomplete-dispatch-in-second-test-module",
    ),
    pytest.param(
        # `view-buffer/src/geometry/rasterize.rs` is shaped exactly like this: a
        # `#[cfg(test)] fn` whose signature wraps, then the test module. Brace
        # depth was still 0 at the end of the fn's opening line, so the skip
        # ended there and resumed at the next column-0 item -- the test module,
        # which was then scanned as production.
        "#[cfg(test)]\nfn helper(\n    c: &Contour,\n) -> u8 {\n    0\n}\n\n"
        "#[cfg(test)]\nmod tests {\n    fn parse(s: &str) -> u8 {\n        match s {\n"
        + "\n".join(f'            "{n}" => 1,' for n in sorted(NAMES - {"f64"}))
        + "\n            _ => 0,\n        }\n    }\n}",
        id="incomplete-dispatch-after-cfg-test-fn-with-wrapped-signature",
    ),
    pytest.param(
        "pub fn get_i64(&self, r: usize) -> Res {\n    match col {\n"
        '        TypedCol::U64(ca) => Some(x.map_err(|_| self.cast_err(r, "i64"))?),\n'
        '        TypedCol::Bool(_) => return Err(self.cast_err(r, "i64")),\n    }\n}',
        # A dtype named in an arm *body* is an error message, not a dispatch.
        id="dtype-name-in-arm-body-only",
    ),
    pytest.param("pub fn nothing() -> u8 { 7 }", id="no-dtypes-at-all"),
]


@pytest.mark.parametrize("source,expect_substring", BAD_FIXTURES)
def test_ratchet_flags_known_defects(source: str, expect_substring: str) -> None:
    """Each past blind spot must still be caught."""
    offenders = dispatch_offenders(source, "fixture.rs", PAIRS)
    assert offenders, "ratchet did not flag a defect it is supposed to catch"
    assert any(expect_substring in o for o in offenders), (
        f"flagged, but not for the expected reason: wanted {expect_substring!r}, "
        f"got {offenders}"
    )


@pytest.mark.parametrize("source", GOOD_FIXTURES)
def test_ratchet_accepts_correct_code(source: str) -> None:
    """Correct code must not be reported.

    False positives are not harmless here: a ratchet that fires on
    ``rustfmt``-clean code gets weakened or deleted by whoever hits it next.
    """
    offenders = dispatch_offenders(source, "fixture.rs", PAIRS)
    assert not offenders, f"false positive on correct code: {offenders}"


def test_dropping_any_single_arm_is_caught() -> None:
    """Monotonicity: sensitivity must not fall off as damage grows.

    The threshold version failed at four missing arms and passed at five, so a
    larger defect hid better than a small one. Every prefix is checked here.
    """
    ordered = sorted(NAMES)
    for drop in range(1, len(ordered)):
        kept = ordered[drop:]
        offenders = dispatch_offenders(_key_dispatch(kept), "fixture.rs", PAIRS)
        assert offenders, f"dropping {drop} arm(s) was not caught"
