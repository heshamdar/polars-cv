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
        "TYPED_OPS_CONSOLIDATION_PLAN.md",
        "polars-cv/docs/changelog.md",
        "polars-cv/docs/user-guide/migration.md",
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
    # Typed-op P4 (sources): each source format is a typed struct
    # (`src/formats/source.rs`) and `SourceFormat` is generated from it.
    Removed("SOURCE_PARAM_APPLIES", "P4: the typed source formats"),
    Removed("PARAM_HINTS", "P4: the typed formats name where a field applies"),
    Removed("reject_inapplicable_params", "P4: io_check"),
    Removed("KNOWN_SOURCE_FORMATS", "P4: the typed source formats"),
    Removed("resolve_fill", "P4: ContourSource::fill"),
    Removed("opt_u8_value", "P4: Param<u8> fill_value/background"),
    Removed("resolve_usize", "P4: Param<u32> contour size"),
    # Typed-op P5: the geometry namespaces' kwargs are typed (`Param<T>`,
    # `ColumnRef`) and each expression kwarg carries its own `{"$slot": n}`.
    Removed("InputSlots", "P5: slots ride in the typed kwargs"),
    Removed("parse_named", "P5: Param<Enum> kwargs"),
    Removed("require_named", "P5: Param<Enum> kwargs"),
    Removed("required_f64", "P5: GeomParams::required"),
    # Typed-op P6: every op is typed, so the untyped legacy protocol is gone.
    Removed("LegacyOpSpec", "P6: every op deserializes as a TypedOp"),
    Removed("LEGACY_OPS", "P6: the catalogue is the op set"),
    Removed("OP_NAMES", "P6: TYPED_OPS is generated from the catalogue"),
    Removed("known_ops", "P6: TYPED_OPS is generated from the catalogue"),
    Removed("_encode_literal", "P6: ParamValue.to_wire"),
    Removed("_UNIFORM_PARITY_ENUMS", "P6: the Python enums are generated"),
    Removed("test_every_rust_enum_is_parity_checked", "P6: generated enums"),
    Removed("enum_variants", "P6: tests read the enum_catalog FFI"),
    Removed("enum_names", "P6: tests read the enum_catalog FFI"),
    Removed("registered_variants", "P6: only the enum catalogue reads REGISTRY"),
    Removed("registered_names", "P6: only the enum catalogue reads REGISTRY"),
    Removed(
        "row_error_policy_names_match_serde",
        "P6: the graph policies parse through NAMED (literal_field)",
    ),
    Removed(
        "null_param_policy_names_match_serde",
        "P6: the graph policies parse through NAMED (literal_field)",
    ),
    Removed("op_output_channels", "P7a: plan_step applies the channel rule"),
    Removed("binary_output_dtype", "P7a: plan_step(other_dtype=)"),
    Removed("parse_binary_op", "P7a: plan_step reads the resolved GraphStep"),
    Removed("_update_output_dtype", "P7a: one plan_step per append"),
    Removed("_update_shape_hints", "P7a: one plan_step per append"),
    Removed("_apply_shape_contract", "P7a: one plan_step per append"),
    Removed("_drop_hints_below_rank", "P7a: plan_step clips hints to the rank"),
    Removed("_update_channels_from_rule", "P7a: plan_step applies the channel rule"),
    Removed("_current_input_dims", "P7a: plan_step builds infer_shape's input"),
    Removed("_input_dims_for", "P7a: plan_step builds infer_shape's input"),
    Removed("_update_hw_from_infer_shape", "P7a: plan_step infers H/W"),
    Removed("_require_input_domain", "P7a: plan_step checks the input domain"),
    Removed("update_dtype", "P7a: a binary op passes other_dtype instead"),
    Removed("op_schema", "P7b: per-op entering states replace the batch folds"),
    Removed(
        "_compute_output_domain_dtype_ndim",
        "P7b: a slice replays from its entering state",
    ),
    Removed("_hint_snapshots", "P7b: _entering holds each op's whole state"),
    Removed("_initial_output_dtype", "P7b: _state_at(0)"),
    Removed("_initial_expected_ndim", "P7b: _state_at(0)"),
    Removed("_rewrite_ops", "P7b: _replay recomputes rather than re-keys"),
    Removed("_POSITION_KEYED_FIELDS", "P7b: _replay recomputes rather than re-keys"),
    Removed("_set_ops_slice", "P7b: _replay"),
    Removed("_commit_reordered_ops", "P7b: _replay"),
    Removed("_commit_eliminated_ops", "P7b: _replay"),
    Removed("_entering_dims_at", "P7b: _entering_dims reads a PlanState"),
    Removed("_entering_dims", "P7c: the passes run in Rust (passes.rs)"),
    Removed("_eliminate_identities_inplace", "P7c: node_pass(identity_elimination)"),
    Removed("_op_is_identity_at", "P7c: passes::is_identity"),
    Removed("_output_shape_equals_input", "P7c: passes::shape_preserved"),
    Removed(
        "_hoist_spatial_windows_inplace", "P7c: node_pass(spatial_window_pushdown)"
    ),
    Removed("_compute_spatial_pushdown", "P7c: passes::hoist_spatial_windows"),
    Removed("_spatial_transfer", "P7c: passes::hoist_spatial_windows"),
    Removed("_is_spatial_window", "P7c: GraphStep::is_spatial_window"),
    Removed("_SPATIAL_BARRIER", "P7c: passes::hoist_spatial_windows"),
    Removed("_op_reads_sibling_nodes", "P7c: GraphStep::reads_other_nodes"),
    Removed("_names_nodes", "P7c: GraphStep::reads_other_nodes"),
    Removed("_op_contract_for", "P7c: the passes read contracts in Rust"),
    Removed("_pass_handlers", "P7c: dispatch on the generated LogicalPass"),
    Removed("op_contract", "P7c: plan_step and the Rust passes read contracts"),
    Removed("op_identity_rule", "P7c: passes::is_identity"),
    Removed("op_infer_shape", "P7c: lib.rs infer_shape, called in Rust only"),
    Removed("rank_rule_name", "P7c: no contract strings cross the FFI"),
    Removed("channel_rule_name", "P7c: no contract strings cross the FFI"),
    Removed("spatial_rule_name", "P7c: no contract strings cross the FFI"),
    Removed("identity_rule_name", "P7c: no contract strings cross the FFI"),
    Removed("dtype_rule_name", "P7c: no contract strings cross the FFI"),
    Removed("_CONTRACT_KEYS", "P7c: op_contract deleted"),
    Removed(
        "bit_exact",
        "P7c: nothing read it; the equivalence guard byte-compares every pass",
    ),
    Removed(
        "test_flags_match_registry_both_directions",
        "P7c: OptFlags and the pass list are generated from one catalogue",
    ),
    Removed(
        "test_every_logical_pass_is_a_rust_pass",
        "P7c: OptFlags and the pass list are generated from one catalogue",
    ),
    Removed("io_check", "P7d: plan_source (validates and plans) and sink_check"),
    Removed("_seed_from_contour_rasterize", "P7d: plan::source_state"),
    Removed(
        "expected_encoding",
        "P7d: Rust reads histogram buckets off the ops",
        # The test that the wire key is now refused sends it.
        allowed_in=frozenset({"polars-cv/src/graph/types.rs"}),
    ),
    Removed(
        "unknown_expected_encoding_is_a_compile_error",
        "P7d: expected_encoding is no longer a wire field",
    ),
    Removed("_shape_hints", "P7e: Pipeline._state (a Rust-computed PlanState)"),
    Removed("_asserted_dims", "P7e: PlanState.asserted"),
    Removed("_shape_declared", "P7e: PlanState.declared"),
    Removed("_current_domain", "P7e: PlanState.domain"),
    Removed("_expected_ndim", "P7e: PlanState.ndim"),
    Removed("ShapeHints", "P7e: PlanState.dims"),
    Removed("is_supplied", "P8: source() keywords default to None"),
    Removed("infer_shape", "P9: Op::shape returns an OpShape, evaluated symbolically"),
    Removed("infer_shape_probe", "P9: shapes are symbolic (OpDef::shape)"),
    Removed("resolve_op_from_json_probe", "P9: planning_step, one ParamCtx::planning"),
    Removed("probe_step", "P9: planning_step"),
    Removed("unknown_dim_probe", "P9: an unknown input size is Dim::Input(k)"),
    Removed("PRESERVED_DIM", "P9: an unknown input size is Dim::Input(k)"),
    Removed("deciding_params", "P9: identity rules are value-independent; OpShape::preserves"),
    Removed("is_probe", "P9: ParamCtx::is_planning"),
    Removed("probe_value", "P9: WireScalar::planning_value"),
    Removed("output_hw", "P9: ImageOpKind::shape"),
    Removed("_matrix_param_from_floats", "P10: dead (no caller since warp_affine is generated)"),
    Removed("get_output_nodes", "P10: dead (no caller)"),
    Removed("is_multi_output", "P10: dead (no caller)"),
    Removed("_field_names", "P10: dead (OptFlags fields come from PASS_CATALOG)"),
    Removed(
        "output_encoding",
        "P8: Rust reads histogram buckets off the ops (OutputSpec.histogram_buckets)",
    ),
    Removed("_source_param_defaults", "P8: source() keywords default to None"),
    Removed(
        "shape_asserted",
        "P7e: an output carries its planned state; Rust reads the facts off it",
        # The Rust OutputSpec field it is read into, and the graph's reporting.
        allowed_in=frozenset(
            {
                "polars-cv/src/graph/types.rs",
                "polars-cv/src/graph/compiled.rs",
                "polars-cv/src/graph/encode.rs",
                "polars-cv/src/graph/sink_kind.rs",
            }
        ),
    ),
    Removed("ShapeAssertion", "P7e: an assertion is plan_assert's wire dict"),
    Removed("_plan_step", "P7e: _push_op calls plan_step and keeps its state"),
    Removed("_apply_step", "P7e: _push_op calls plan_step and keeps its state"),
    Removed("_restore", "P7e: _replay assigns the start state"),
    Removed("_require_ndim_is_consistent", "P7e: plan::assert_shape"),
    Removed("_require_dim_is_assertable", "P7e: plan::assert_shape"),
    Removed("_shape_ref_dims", "P7e: Pipeline._canvas_of"),
    Removed("_is_known", "P7e: PlanState.dim"),
    Removed("sink_check", "P7e: plan_sink validates the sink and plans it"),
    Removed("SOURCES_RESOLVED_FROM_COLUMN", "P7e: Source::resolves_from_column"),
    Removed("SINKS_WITH_TYPED_ELEMENTS", "P7e: Sink::has_typed_elements"),
    Removed("_require_concrete_sink_dtype", "P7e: plan::check_sink"),
    Removed("_array_sink_needs_shape", "P7e: plan::check_sink"),
    Removed("_validate_sink_params", "P7e: lazy._check_sink -> plan_sink"),
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
