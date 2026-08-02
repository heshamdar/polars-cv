"""The dtype-dispatch ratchet's logic, separated so it can be tested itself.

``test_no_second_dtype_spelling_table`` runs this over the repo. ``test_dtype_
ratchet_fixtures`` runs it over committed known-bad and known-good snippets.
Both import from here, so the fixtures exercise the code that actually guards
the tree — a fixture testing a copy would prove nothing about the guard.

That second test exists because this checker has been wrong seven times, and
each time the wrongness was silent: it kept passing while covering less. Every
past defect is a fixture below, so re-introducing one fails the suite instead
of quietly widening the blind spot again.
"""

from __future__ import annotations

import re
from collections import Counter

# A dtype short name as it appears in Rust source, e.g. `"u8"`, `"f32"`.
# `f8`/`f16` are matched deliberately: they are not dtypes, so a dispatch
# naming one is drift worth reporting rather than something to skip.
DTYPE_LIT = re.compile(r'"([ui](?:8|16|32|64)|f(?:8|16|32|64))"')

# A `DType` variant path. The leading `\w*` is load-bearing: this repo aliases
# the type (`use view_buffer::DType as VbDType`), and a `\bDType::` pattern
# cannot match inside `VbDType::` because there is no word boundary there. A
# whole variant-to-name table written against the alias was invisible.
DTYPE_VARIANT = re.compile(r"\b\w*DType::(\w+)\b")

FN_START = re.compile(
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:default\s+)?(?:const\s+)?"
    r"(?:async\s+)?(?:unsafe\s+)?(?:extern\s+\"[^\"]*\"\s+)?fn\s+(\w+)"
)


def _strip_line_comment(line: str) -> str:
    """Drop a trailing ``//`` comment, ignoring ``//`` inside a string."""
    idx = line.find("//")
    while idx != -1:
        if line[:idx].count('"') % 2 == 0:
            return line[:idx]
        idx = line.find("//", idx + 2)
    return line


def _logical_lines(text: str):
    """Yield arms as single lines, rejoining rustfmt's wrapped alternation.

    ``rustfmt`` splits a long ``"u8" | "i8" | ... | "f64" => ...`` across lines,
    stranding the earlier names on a line with no ``=>``. Without rejoining, a
    correctly formatted ten-name dispatch reads as one missing nine names.
    """
    buf = ""
    for raw in text.splitlines():
        line = _strip_line_comment(raw)
        stripped = line.strip()
        if buf and (stripped.startswith("|") or buf.rstrip().endswith("|")):
            buf += " " + stripped
        else:
            if buf:
                yield buf
            buf = line
    if buf:
        yield buf


def _production_source(source: str) -> str:
    """Strip block comments and test modules, keeping post-test production fns.

    Test modules name dtypes freely in assertions. Everything at indent 0 after
    the test module is production code and is kept: an earlier version reported
    that region as unscannable, which false-positived on any ``#[cfg(test)]``
    helper that happened to hold a complete dispatch.
    """
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    production, sep, after_tests = source.partition("#[cfg(test)]")
    if not sep:
        return production
    keep, in_top_fn = [], False
    for line in after_tests.splitlines():
        if FN_START.match(line) and not line.startswith((" ", "\t")):
            in_top_fn = True
        if in_top_fn:
            keep.append(line)
            if line.startswith("}"):
                in_top_fn = False
    return production + "\n" + "\n".join(keep)


def dispatch_offenders(
    source: str, label: str, expected_names: set[str], expected_variants: set[str]
) -> list[str]:
    """Report dtype dispatches in ``source`` that disagree with the authority.

    Checks both halves of an arm, because a dtype table can be written either
    way round and only checking keys lost coverage the previous version had:

    - arm *keys* that are dtype names (``match s { "u8" => .. }``);
    - arms keyed by a ``DType`` variant that *yield* a name
      (``match d { DType::U8 => "u8" }``).

    Two limits, stated because overstating this test's reach is how it went
    wrong repeatedly. It checks that the names are all *present*, never that a
    name maps to the right thing — ``"u32" | "u64" => build_u32`` passes. And
    it reads ``match`` arms, so an if/else chain or a ``HashMap`` literal is
    invisible.
    """
    offenders: list[str] = []
    current = "<file scope>"
    keyed: dict[str, list[str]] = {}
    named: dict[str, list[tuple[str, str]]] = {}

    for line in _logical_lines(_production_source(source)):
        m = FN_START.match(line)
        if m:
            current = m.group(1)
        if "=>" not in line:
            continue
        key, _, value = line.partition("=>")

        # An arm key that is a dtype name. `cast_err(row_idx, "i64")` in an arm
        # *body* names a dtype without dispatching on it, so only the key side
        # counts here.
        for lit in DTYPE_LIT.findall(key):
            keyed.setdefault(current, []).append(lit)

        # An arm keyed by a DType variant that yields a dtype name.
        variant_keys = [v for v in DTYPE_VARIANT.findall(key) if v in expected_variants]
        value_names = DTYPE_LIT.findall(value)
        if len(variant_keys) == 1 and len(value_names) == 1:
            named.setdefault(current, []).append((variant_keys[0], value_names[0]))
        elif len(variant_keys) == 1 and len(value_names) > 1:
            offenders.append(
                f"{label}::{current} maps {variant_keys[0]} to several dtype "
                f"names ({value_names}); this checker cannot tell which is the "
                f"dispatch, so split the arm"
            )

    for fn_name, names in keyed.items():
        counts = Counter(names)
        if set(counts) != expected_names:
            offenders.append(
                f"{label}::{fn_name} dispatches on dtype names "
                f"(missing={sorted(expected_names - set(counts))}, "
                f"unknown={sorted(set(counts) - expected_names)})"
            )
        elif len(set(counts.values())) != 1:
            # A function holding two dispatches names every dtype twice, so
            # uneven counts mean one of them is short an arm even though the
            # set is complete. Checking per file let the two cover each other.
            offenders.append(
                f"{label}::{fn_name} holds more than one dtype dispatch and "
                f"they disagree: { {n: c for n, c in sorted(counts.items())} }"
            )

    for fn_name, pairs in named.items():
        got_variants = {v for v, _ in pairs}
        got_names = {n for _, n in pairs}
        if got_variants != expected_variants or got_names != expected_names:
            offenders.append(
                f"{label}::{fn_name} maps DType variants to names but does not "
                f"agree with dtype_table! "
                f"(missing variants={sorted(expected_variants - got_variants)}, "
                f"missing names={sorted(expected_names - got_names)}, "
                f"unknown names={sorted(got_names - expected_names)})"
            )

    return offenders
