#!/usr/bin/env python
"""Fail if a symbol a phase deleted appears anywhere in the tracked sources.

The typed-op migration (``TYPED_OPS_PLAN.md``) ends every phase by deleting
what it replaced. Compiler errors catch a deleted Rust item that is still
*called*; they do not catch a comment, a doc page, a Python string or a test
that still names it, and those are how a removed mechanism quietly comes back
or keeps being documented. This is the phase exit gate for that: each entry
names a symbol, the reason it is gone, and (rarely) the files that may still
mention it on purpose.

Matching is whole-word (``\\b``), over files tracked by git, so build output,
virtualenvs and untracked scratch files are never read. History files are
exempt: the changelog, the review ledger and the plan itself record removals
by name.

Limits: this is a textual scan (see CLAUDE.md, "prefer compiler exhaustiveness
> runtime assertion > source scanning"). It is used only for what the compiler
cannot see — references in prose, strings and other languages — and each
symbol must be distinctive enough that a whole-word match means the symbol.

Usage::

    python scripts/check_removed_symbols.py      # exit 1 on any hit
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Files that record removals by name and are therefore never scanned.
HISTORY_FILES: frozenset[str] = frozenset(
    {
        "CHANGELOG.md",
        "CODE_REVIEW_FINDINGS.md",
        "EXTENSION_TYPES_PLAN.md",
        "TYPED_OPS_PLAN.md",
        "polars-cv/docs/changelog.md",
        "polars-cv/scripts/check_removed_symbols.py",
        "polars-cv/tests/test_removed_symbols.py",
    }
)


@dataclass(frozen=True)
class Removed:
    """One deleted symbol: what it was, why it is gone, where it may remain."""

    symbol: str
    reason: str
    allowed_in: frozenset[str] = field(default_factory=frozenset)


#: Every removed symbol, grouped by the change that removed it. A phase of the
#: typed-op plan appends its list here as its exit gate.
REMOVED: tuple[Removed, ...] = (
    # CR-32: a call runs its rows in parallel, so the single-thread warning's
    # advice ("ran on one thread", use streaming) became false.
    Removed("engine_warning", "CR-32: the single-thread engine warning was deleted"),
    Removed("CallGuard", "CR-32: the engine warning's per-call tracker"),
    Removed(
        "POLARS_CV_SILENCE_ENGINE_WARNING",
        "CR-32: the engine warning it silenced is gone",
        # The tombstone clears it so a leftover setting cannot mask a warning.
        allowed_in=frozenset({"polars-cv/tests/test_removed_surfaces.py"}),
    ),
    Removed(
        "POLARS_CV_ENGINE_WARN_SECONDS",
        "CR-32: the engine warning it tuned is gone",
        # The tombstone sets the old trigger threshold to prove nothing prints.
        allowed_in=frozenset({"polars-cv/tests/test_removed_surfaces.py"}),
    ),
    # Typed-op P1: expression parameters cross the boundary as positional
    # `{"$slot": n}` indices assigned by `SlotTable`; no text, no names.
    Removed("expr_key", "P1: display-text expression identity (CR-31)"),
    Removed("_EXPR_KEYS", "P1: the process-wide expr_key registry"),
    Removed(
        "expr_column_names",
        "P1: the name list that bound expression params to input columns",
        # The tombstone sends it to prove the kwargs struct rejects it.
        allowed_in=frozenset({"polars-cv/tests/test_removed_surfaces.py"}),
    ),
    Removed("name_to_slot", "P1: Rust name->slot binding"),
    Removed("shape_pipeline", "P1: replaced by the `shape_node` id"),
    Removed("op_probe_json", "P1: probe re-serialization; `op_json` is serde"),
    Removed("param_probe_json", "P1: probe re-serialization; `op_json` is serde"),
    Removed("_build_column_bindings", "P1: bindings come from the SlotTable"),
    Removed("_get_ordered_columns", "P1: input columns come from the SlotTable"),
    # Typed-op P2: the typed catalogue (`src/ops/`) plus the shrinking
    # `LEGACY_OPS` replace the one name registry; ops migrate out of it.
    Removed("KNOWN_OPS", "P2: split into the typed catalogue and LEGACY_OPS"),
    Removed("is_all_literal", "P2: OpSpec::is_static (typed ops visit slots)"),
    Removed("as_f64_vec", "P2: histogram edges are a typed Bins::Edges"),
    Removed("range_min", "P2: histogram `range` is one two-element field"),
    Removed("range_max", "P2: histogram `range` is one two-element field"),
    # Typed-op P3 (view family): axis lists are typed `Literal<u32>` lists and
    # checked by the engine op's own `validate` at plan time.
    Removed("as_int_list", "P3: axes are Vec<Literal<u32>>"),
    Removed("_literal_axes", "P3: a Literal field refuses an expression"),
    Removed("_require_axes_within_rank", "P3: ViewOp::validate at plan time"),
    # Typed-op P3 (compute family): `NormalizeMethod` is a fieldless
    # `named_variants!` enum, so no enum is registered without a table.
    Removed("REGISTERED_WITHOUT_A_TABLE", "P3: every registered enum has a table"),
    # Typed-op P3 (image family): every resize variant reads a typed
    # `Param<FilterType>`, whose only spellings are `NAMED`.
    Removed("resolve_filter", "P3: Param<FilterType>"),
    # Typed-op P3 (colour/filter/reductions/phash/channel): the per-parameter
    # legacy readers these ops used are gone with them.
    Removed("from_str_name", "P3: Literal<ColorSpace> reads NAMED only"),
    Removed("resolve_interpolation", "P3: Param<InterpolationType>"),
    Removed("resolve_border_value", "P3: Param<f64> border_value"),
    Removed("maybe_usize_literal", "P3: Option<Literal<u32>> axis"),
    Removed("opt_u32_literal", "P3: Literal<u32> hash_size"),
    Removed("resolve_f32_list", "P3: Vec<Param<f32>>"),
    Removed("resolve_usize_list", "P3: Vec<Param<u32>>"),
    Removed("as_param_slice", "P3: typed list fields"),
    Removed("_param_list", "P3: _encode_field encodes typed list fields"),
    Removed("is_wire_param", "P3: no nested param lists to hoist"),
    # Typed-op P3 (geometry family): `extract_contours(min_area)` is an
    # `Option<Param<f64>>`.
    Removed("maybe_f64", "P3: Option<Param<f64>> min_area"),
    Removed("resolve_f64", "P3: Param<f64> reads its column directly"),
    # Typed-op P3 (binary family): operands are typed `NodeRef` fields and
    # flags are `Param<bool>`.
    Removed("opt_bool_dyn", "P3: Param<bool> invert"),
    Removed("resolve_bool", "P3: Param<bool> reads its column directly"),
    Removed("_add_binary_op", "P3: Pipeline._add_node_op encodes by catalogue"),
    Removed("_add_channel_merge", "P3: Pipeline._add_node_op encodes by catalogue"),
    # Typed-op P3 exit: every op is typed, so the name-keyed legacy resolution
    # and everything that served it are gone. `LEGACY_OPS`/`LegacyOpSpec` stay
    # (empty / unreachable from the wire) until P6 deletes the protocol.
    Removed("resolve_op_inner", "P3: every op resolves through OpDef"),
    Removed("OpParams", "P3: serde rejects an unknown field on a typed op"),
    Removed("req_enum", "P3: Param<Enum> fields"),
    Removed("opt_enum", "P3: Param<Enum> fields"),
    Removed("resolve_str", "P3: Param<Enum> reads its column directly"),
    Removed("resolve_string", "P3: Literal<T> structural fields"),
    Removed("legacy_probe_spec", "P3: typed ops probe through their slots"),
    Removed("resolve_rasterize_style", "P3: Rasterize::with_size"),
    Removed("_enum_param", "P3: _encode_field encodes Param<Enum> fields"),
    Removed("strict_param_tests", "P3: typed rejection table in ops::tests"),
    Removed("unread_param_tests", "P3: deny_unknown_fields on every op"),
    Removed("resolve_op_arms_are_all_known_ops", "P3: no name-keyed arms"),
    Removed("known_ops_all_resolve", "P3: no name-keyed arms"),
    # Typed-op P4 (sinks): each sink format is a typed struct
    # (`src/formats/sink.rs`) and `SinkFormat` is generated from it.
    Removed("SINK_PARAM_APPLIES", "P4: the typed sink formats"),
    Removed("from_sink_format", "P4: Sink::image_codec"),
    Removed("image_codec_format", "P4: SinkKind::image_codec"),
)


@dataclass(frozen=True)
class Hit:
    symbol: str
    path: str
    line: int
    reason: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: `{self.symbol}` ({self.reason})"


def find_hits(removed: tuple[Removed, ...], files: dict[str, str]) -> list[Hit]:
    """Every whole-word occurrence of a removed symbol outside its allowances.

    ``files`` maps repo-relative paths to their text. History files are skipped.
    """
    if not files:
        msg = "no files to scan: an empty scan would pass vacuously"
        raise ValueError(msg)
    patterns = [(r, re.compile(rf"\b{re.escape(r.symbol)}\b")) for r in removed]
    hits: list[Hit] = []
    for path, text in sorted(files.items()):
        if path in HISTORY_FILES:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            for entry, pattern in patterns:
                if path not in entry.allowed_in and pattern.search(line):
                    hits.append(Hit(entry.symbol, path, number, entry.reason))
    return hits


def tracked_text_files(root: Path = REPO_ROOT) -> dict[str, str]:
    """Repo-relative path -> text for every tracked, UTF-8 decodable file."""
    listed = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True
    ).stdout.split(b"\0")
    files: dict[str, str] = {}
    for raw in listed:
        if not raw:
            continue
        rel = raw.decode()
        try:
            files[rel] = (root / rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue  # binary, or deleted in the working tree
    # The scan must reach both languages, or a broken listing reads as clean.
    for suffix in (".rs", ".py", ".md"):
        if not any(p.endswith(suffix) for p in files):
            msg = f"file listing found no {suffix} files under {root}"
            raise ValueError(msg)
    return files


def main() -> int:
    hits = find_hits(REMOVED, tracked_text_files())
    for hit in hits:
        print(hit)
    if hits:
        print(f"\n{len(hits)} reference(s) to removed symbols.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
