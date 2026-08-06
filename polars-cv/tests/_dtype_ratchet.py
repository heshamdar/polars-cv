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
DTYPE_VARIANT = re.compile(r"\b(?:\w*DType|Self)::(\w+)\b")

# An arm key of the form `SomeEnum::Variant`, capturing both parts.
ENUM_KEY = re.compile(r"\b(\w+)::(\w+)")

# Foreign spellings of a dtype, as other enums name the same concept:
# `DataType::UInt16` and `TypedBufferData::U16` both mean our `U16`.
_FOREIGN_DTYPE_NAME = {
    "uint8": "U8",
    "int8": "I8",
    "uint16": "U16",
    "int16": "I16",
    "uint32": "U32",
    "int32": "I32",
    "uint64": "U64",
    "int64": "I64",
    "float32": "F32",
    "float64": "F64",
}


def _as_dtype_variant(name: str) -> str | None:
    """Normalise a foreign enum variant name to our `DType` variant, or None.

    `UInt16` -> `U16`, `Float32` -> `F32`, `U16` -> `U16`. Anything else (an op
    name, a colour type like `L8`) is not a dtype spelling.
    """
    low = name.lower()
    if low in _FOREIGN_DTYPE_NAME:
        return _FOREIGN_DTYPE_NAME[low]
    if re.fullmatch(r"[uif](?:8|16|32|64)", low):
        return low[0].upper() + low[1:]
    return None


FN_START = re.compile(
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:default\s+)?(?:const\s+)?"
    r"(?:async\s+)?(?:unsafe\s+)?(?:extern\s+\"[^\"]*\"\s+)?fn\s+(\w+)"
)


def _strip_line_comment(line: str) -> str:
    """Drop a trailing ``//`` comment, ignoring ``//`` inside a literal.

    Scans rather than counting quotes: the parity heuristic this replaces
    miscounted on an escaped quote (``"a \\" b"``) and on the char literal
    ``\'"\'``, leaving the comment in place so its text was read as arm keys —
    which restored the coverage of an arm the comment was explaining away.
    """
    out, i, in_str, in_chr = [], 0, False, False
    while i < len(line):
        c = line[i]
        if (in_str or in_chr) and c == "\\":
            out.append(line[i : i + 2])
            i += 2
            continue
        if c == '"' and not in_chr:
            in_str = not in_str
        elif c == "'" and not in_str:
            in_chr = not in_chr
        elif c == "/" and not in_str and not in_chr and line[i : i + 2] == "//":
            break
        out.append(c)
        i += 1
    return "".join(out)


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


def _literals_removed(line: str) -> str:
    """A line with its comment, string and char literals blanked out.

    For brace counting only: a `{` inside `"{"`, `'{'` or a trailing comment
    must not move the depth.
    """
    bare = _strip_line_comment(line)
    bare = re.sub(r'"(?:[^"\\]|\\.)*"', "", bare)
    return re.sub(r"'(?:[^'\\]|\\.)'", "", bare)


def _production_source(source: str) -> str:
    """Strip block comments and every ``#[cfg(test)]`` item, keeping the rest.

    Test modules name dtypes freely in assertions, so scanning one reports a
    deliberately partial test helper as production drift. Production code that
    *follows* a test module is kept: an early version dropped it, which is the
    opposite error.

    Each attributed item is skipped by brace depth from its own attribute,
    rather than by partitioning on the first ``#[cfg(test)]`` and guessing
    where the module ends. That guess was wrong two ways, both live in this
    tree:

    - A file with two test modules (``polars-cv/src/execute.rs``) had the
      second scanned as production, because only the first was skipped.
    - A ``#[cfg(test)] fn`` whose signature wraps
      (``view-buffer/src/geometry/rasterize.rs``) ended the skip on its opening
      line, because brace depth was still 0 there — the body's ``{`` sits on
      the line with the return type. The scan then resumed at the next column-0
      item, which in that file is the test module.

    Both routes end in the same place: test code scanned as production. Neither
    fired at the time, which is the point — the tree was one committed test
    helper away from a guard that reports correct code, and a guard that does
    that gets weakened by whoever hits it next.
    """
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    lines = source.splitlines()
    kept: list[str] = []
    i = 0
    while i < len(lines):
        if "#[cfg(test)]" not in lines[i]:
            kept.append(lines[i])
            i += 1
            continue
        # Skip the attributed item. Start counting from the remainder of the
        # attribute's own line so `#[cfg(test)] fn f() { .. }` on one line is
        # handled like the multi-line form.
        rest = lines[i].split("#[cfg(test)]", 1)[1]
        depth, opened = 0, False
        while True:
            bare = _literals_removed(rest)
            depth += bare.count("{") - bare.count("}")
            opened = opened or "{" in bare
            # `#[cfg(test)] use foo::bar;` and `#[cfg(test)] mod t;` have no
            # block to close.
            if (opened and depth <= 0) or (not opened and ";" in bare):
                i += 1
                break
            i += 1
            if i >= len(lines):
                break
            rest = lines[i]
    return "\n".join(kept)


def dispatch_offenders(
    source: str,
    label: str,
    expected_pairs: set[tuple[str, str]],
    allow_partial: frozenset[str] = frozenset(),
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
    expected_names = {n for _, n in expected_pairs}
    expected_variants = {v for v, _ in expected_pairs}

    offenders: list[str] = []
    current = "<file scope>"
    keyed: dict[str, list[str]] = {}
    named: dict[str, list[tuple[str, str]]] = {}
    to_variant: dict[str, set[str]] = {}

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

        # An arm that *yields* a DType variant. Covers the shape both real
        # Polars-type tables use (`DataType::UInt16 => Some(DType::U16)`),
        # which neither the key check nor the name check can see: the key is a
        # foreign enum and the value is a variant, not a literal. Five of ten
        # arms were deletable from two such tables with a fully green suite.
        # Only count it when the *key* is a foreign enum whose variant names a
        # dtype -- that is what a correspondence table looks like. It excludes
        # `DType::F64 => DType::F32` (a promotion, not a table) and
        # `ComputeOp::Relu => Some(DType::F32)` (one op's working dtype), both
        # of which are legitimately partial and were false-positived by an
        # earlier, blunter version of this rule.
        if any(v in expected_variants for v in DTYPE_VARIANT.findall(value)):
            for enum_name, variant in ENUM_KEY.findall(key):
                if enum_name.endswith("DType") or enum_name == "Self":
                    continue
                normalised = _as_dtype_variant(variant)
                if normalised in expected_variants:
                    to_variant.setdefault(current, set()).add(normalised)
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
        # Compared as pairs, not two independent sets: with set comparison a
        # table that swaps two rows (`U64 => "u32"`, `U32 => "u64"`) has both
        # sets complete and passes, while mapping every dtype to the wrong name.
        if set(pairs) != expected_pairs:
            wrong = sorted(set(pairs) - expected_pairs)
            offenders.append(
                f"{label}::{fn_name} maps DType variants to names but does not "
                f"agree with dtype_table! "
                f"(missing variants={sorted(expected_variants - got_variants)}, "
                f"missing names={sorted(expected_names - got_names)}, "
                f"wrong pairs={wrong})"
            )

    for fn_name, got in to_variant.items():
        if got == expected_variants or f"{label}::{fn_name}" in allow_partial:
            continue
        # Fail closed. Every previous version of this checker skipped what it
        # did not recognise, and every one of its blind spots was something it
        # skipped. An incomplete variant map is either drift or a deliberate
        # subset; the deliberate ones are few and belong in `allow_partial`
        # with a reason, where a reviewer can see them.
        offenders.append(
            f"{label}::{fn_name} maps to DType variants but not all ten "
            f"(missing={sorted(expected_variants - got)}). If the subset is "
            f"deliberate, add {label}::{fn_name!r} to ALLOWED_PARTIAL_VARIANT_MAPS "
            f"in test_sanitation.py with a reason."
        )

    return offenders
