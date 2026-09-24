//! polars-cv: A Polars plugin for vision/array operations.
//!
//! This crate provides expression functions for applying image and array
//! processing pipelines to Polars DataFrame columns, powered by view-buffer.

mod cloud;
mod cloud_auth;
mod contour;
mod execute;
mod ext_types;
mod fetch;
mod formats;
mod geom_arity;
mod geom_params;
mod geom_schema;
mod graph;
mod image_metadata;
mod naming;
mod ops;
mod output;
mod params;
mod plan;
mod point;
mod read_bytes;

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::derive::polars_expr;

use crate::plan::plan_step;
use serde::Deserialize;

/// Python module entry point for maturin.
/// The module name `_lib` must match pyproject.toml's `module-name = "polars_cv._lib"`.
#[pymodule]
#[pyo3(name = "_lib")]
fn polars_cv_lib(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Baked in at compile time so a stale extension is detectable. The install is
    // editable, so the Python sources are always the working tree's while this
    // extension stays at its last `maturin develop`; `polars_cv.build_info()`
    // compares the two.
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    // A content hash of both crates' sources, from `build.rs`. The version
    // above cannot detect staleness *within* a release cycle -- it is the same
    // literal until the next bump, which is the whole window the check exists
    // for. This moves whenever the built artifact could differ.
    m.add("__source_hash__", env!("POLARS_CV_SOURCE_HASH"))?;
    m.add_function(wrap_pyfunction!(op_contract, m)?)?;
    m.add_function(wrap_pyfunction!(op_identity_rule, m)?)?;
    m.add_function(wrap_pyfunction!(op_infer_shape, m)?)?;
    m.add_function(wrap_pyfunction!(plan_step, m)?)?;
    m.add_function(wrap_pyfunction!(op_catalog, m)?)?;
    m.add_function(wrap_pyfunction!(io_catalog, m)?)?;
    m.add_function(wrap_pyfunction!(enum_catalog, m)?)?;
    m.add_function(wrap_pyfunction!(io_check, m)?)?;
    m.add_function(wrap_pyfunction!(point_schema, m)?)?;
    m.add_function(wrap_pyfunction!(contour_schema, m)?)?;
    m.add_function(wrap_pyfunction!(bbox_schema, m)?)?;
    m.add_function(wrap_pyfunction!(extension_types, m)?)?;
    m.add_function(wrap_pyfunction!(rotation_matrix_2d, m)?)?;
    Ok(())
}

// ============================================================================
// Contract introspection (single-authority bridge for the Python schema layer)
// ============================================================================

/// Canonical short name for a view-buffer `DType`.
///
/// Delegates to `DType::NAMED` — the same table the Python `DType` enum
/// mirrors — so the two vocabularies line up by construction.
fn dtype_short_name(dt: view_buffer::DType) -> &'static str {
    dt.short_name()
}

/// Parse a short dtype name back into a view-buffer `DType`.
///
/// Inverse of [`dtype_short_name`]. Used to turn the Python schema layer's
/// dtype strings into the `DType` the canonical [`OutputDTypeRule::resolve`]
/// authority operates on.
pub(crate) fn parse_dtype(s: &str) -> Result<view_buffer::DType, String> {
    view_buffer::DType::from_short_name(s).ok_or_else(|| format!("unknown dtype {s:?}"))
}

/// A planner error (a plain message, so the planning core needs no
/// interpreter) as the `ValueError` Python sees.
pub(crate) fn py_value_error(msg: String) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(msg)
}

/// Canonical string for an output-dtype rule.
///
/// This is the shared vocabulary the Python planner reads (via `op_contract`)
/// to infer output dtypes: `preserve`, `promote`, `fixed:<dtype>`.
fn dtype_rule_name(rule: view_buffer::OutputDTypeRule) -> String {
    use view_buffer::OutputDTypeRule as R;
    match rule {
        R::PreserveInput => "preserve".to_string(),
        R::PromoteToFloat => "promote".to_string(),
        R::Fixed(d) => format!("fixed:{}", dtype_short_name(d)),
        R::ForceF64 => "fixed:f64".to_string(),
        R::ForceI64 => "fixed:i64".to_string(),
        R::ForceU64 => "fixed:u64".to_string(),
        R::ForceU32 => "fixed:u32".to_string(),
    }
}

/// Canonical string for an output-rank rule.
///
/// The plan-time vocabulary the Python schema layer reads (it no longer
/// re-declares the effect): `preserve`, `reduce_one`, `fixed:<n>`, `unknown`.
fn rank_rule_name(rule: view_buffer::OutputRankRule) -> String {
    use view_buffer::OutputRankRule as R;
    match rule {
        R::PreserveRank => "preserve".to_string(),
        R::ReduceByOne => "reduce_one".to_string(),
        R::Fixed(n) => format!("fixed:{n}"),
        R::Unknown => "unknown".to_string(),
    }
}

/// Canonical string for an output-channel rule.
///
/// The vocabulary the Python planner reads for channel inference: `preserve`,
/// `fixed:<n>`, `strip_restore:<color_channels>`, `n/a`, `unknown`.
fn channel_rule_name(rule: view_buffer::OutputChannelRule) -> String {
    use view_buffer::OutputChannelRule as R;
    match rule {
        R::PreserveChannels => "preserve".to_string(),
        R::Fixed(n) => format!("fixed:{n}"),
        R::StripProcessRestore { color_channels } => format!("strip_restore:{color_channels}"),
        R::NotApplicable => "n/a".to_string(),
        R::Unknown => "unknown".to_string(),
    }
}

/// Canonical string for a spatial-dependency rule.
///
/// The plan-time vocabulary the Python planner reads for spatial-window
/// (crop/ROI) commutation: `pointwise`, `neighborhood:<radius>`, `global`,
/// `geometric`. The neighborhood radius is emitted so the planner can size a
/// halo without re-deriving it.
fn spatial_rule_name(rule: view_buffer::SpatialDependency) -> String {
    use view_buffer::SpatialDependency as S;
    match rule {
        S::Pointwise => "pointwise".to_string(),
        S::Neighborhood(support) => format!("neighborhood:{}", support.radius),
        S::Global => "global".to_string(),
        S::Geometric(_) => "geometric".to_string(),
    }
}

/// String spelling of an [`IdentityRule`](view_buffer::IdentityRule) crossing
/// the FFI — a bespoke `_name` helper like `spatial_rule_name`, not the
/// `named_variants!` registry, so nothing is owed to the parity machinery.
fn identity_rule_name(rule: view_buffer::IdentityRule) -> String {
    use view_buffer::IdentityRule as I;
    match rule {
        I::Never => "never".to_string(),
        // The deciding-param gate (an expression-bound deciding param forces
        // `never`) is applied in `op_identity_rule`, which has the original spec;
        // the bare variant spells as `always`.
        I::Always { .. } => "always".to_string(),
        I::WhenShapePreserved { .. } => "when_shape_preserved".to_string(),
        I::WhenDtypePreserved => "when_dtype_preserved".to_string(),
    }
}

/// Resolve one serialized op spec to its `ViewDto`, mapping errors to Python.
///
/// Shared by `plan_step` and `op_contract` so neither re-implements the
/// deserialize → resolve path.
///
/// Expression parameters (dynamic, per-row values like a column-driven resize
/// height) are *neutralized* with a placeholder before resolution: each
/// referenced column is bound to a one-element `Int64` series. The schema
/// knowledge these functions expose — output dtype rule, domain, and the
/// dimensionality rule — never depends on the concrete numeric value of a
/// dimensional parameter, so the placeholder is sound and lets introspection
/// work on the same live op specs the planner sees (which routinely carry
/// expression params) rather than only literal-only ops.
pub(crate) fn resolve_op_from_json(op_json: &str) -> Result<crate::graph::step::GraphStep, String> {
    // Structural schema (domain/dtype/rank/channel rules) never depends on the
    // concrete value of a dimensional param, so any placeholder works here.
    resolve_op_from_json_probe(op_json, 1)
}

/// Like [`resolve_op_from_json`] but binds each expression param to a specific
/// `probe` value instead of `1`. Used by `op_infer_shape` to detect which
/// output dimensions depend on a per-row expression (they vary across probes)
/// versus which are fixed by literal params (identical across probes).
pub(crate) fn resolve_op_from_json_probe(
    op_json: &str,
    probe: i64,
) -> Result<crate::graph::step::GraphStep, String> {
    use crate::params::ParamCtx;

    let op: crate::ops::TypedOp = serde_json::from_str(op_json).map_err(|e| e.to_string())?;
    // Every slot reads a placeholder column holding `probe`.
    let placeholders = vec![Series::new("".into(), &[probe]); op.min_inputs()];
    // A *probe* context: placeholders are integers, so a dynamic enum or flag
    // param cannot be read from one. `ParamCtx::probe` tells the enum/bool
    // accessors to substitute their default instead. Sound because only params
    // with no shape/rank/dtype effect are allowed to be dynamic, so the variant
    // probing picks cannot change the inferred schema.
    let ctx = ParamCtx::probe(&placeholders, probe);
    op.resolve(0, &ctx).map_err(|e| format!("resolve_op: {e}"))
}

/// Plan-time output shape for a single-buffer op — the single authority for
/// per-dimension H/W geometry the Python planner reads instead of re-deriving.
///
/// `input_dims` carries the current per-dimension sizes, each `None` when the
/// dimension is unknown at plan time. The returned dims propagate unknowns: a
/// dimension is `Some(n)` only when it is identical across every probe (fixed by
/// literal params and known input dims) and `None` when it varies (it depends on
/// an unknown input dim or a per-row expression param).
///
/// The probe set includes 90-degree multiples so a discontinuous shape function
/// — rotate's zero-copy 90/180/270 fast path swaps H and W — is correctly seen
/// as unknown for an expression angle over a non-square image, while a literal
/// angle still resolves to its exact branch.
///
/// `None` when the step has no inferable shape (a graph-level step, a
/// data-dependent output); a `ValueError` when the op's parameters do not fit
/// the input — the two used to share `ValueError`, and the planner swallowed
/// both as "not inferable".
#[pyfunction]
fn op_infer_shape(
    op_json: &str,
    input_dims: Vec<Option<i64>>,
) -> PyResult<Option<Vec<Option<i64>>>> {
    infer_shape(op_json, &input_dims).map_err(py_value_error)
}

/// [`op_infer_shape`], for the Rust planner ([`plan::step`]).
pub(crate) fn infer_shape(
    op_json: &str,
    input_dims: &[Option<i64>],
) -> Result<Option<Vec<Option<i64>>>, String> {
    const PROBES: [i64; 4] = [7, 13, 90, 180];
    let mut runs: Vec<Vec<i64>> = Vec::with_capacity(PROBES.len());
    for &p in &PROBES {
        match infer_shape_probe(op_json, input_dims, p)? {
            Some(run) => runs.push(run),
            None => return Ok(None),
        }
    }
    let first = &runs[0];
    // Rank is structural (never data-dependent), so it must be stable across
    // probes; a variation signals a contract bug rather than an unknown.
    if runs.iter().any(|r| r.len() != first.len()) {
        return Err("op_infer_shape: output rank varied across shape probes".to_string());
    }
    Ok(Some(
        (0..first.len())
            .map(|i| {
                let v = first[i];
                if runs.iter().all(|r| r[i] == v) {
                    return Some(v);
                }
                // An unknown input axis the op carries through unchanged: every
                // probe's output equals that probe's own input. Its size is still
                // unknown, but it is provably *the input's* size, which is what
                // the identity-elimination pass needs to prove a full-frame crop is
                // a no-op. Reported as `PRESERVED_DIM`; callers that want a size
                // treat it as unknown. (This used to arrive by accident: a crop's
                // `usize::MAX` "to the end" extent, cast to i64, was -1.)
                let unknown_input = matches!(input_dims.get(i), Some(None));
                let carried = PROBES
                    .iter()
                    .zip(&runs)
                    .all(|(&probe, r)| r[i] == unknown_dim_probe(probe));
                (unknown_input && carried).then_some(PRESERVED_DIM)
            })
            .collect(),
    ))
}

/// `op_infer_shape`'s "this output axis is the unknown input axis, unchanged".
const PRESERVED_DIM: i64 = -1;

/// The value an unknown input dim takes in one probe run.
///
/// Distinct from the value expression params take in the same run (`probe`),
/// so an output that merely equals a per-row parameter — `resize(height=
/// pl.col("h"))` — cannot pass for an input axis carried through unchanged.
fn unknown_dim_probe(probe: i64) -> i64 {
    2 * probe + 1
}

/// One probe of [`op_infer_shape`]: resolve the op with expression params bound
/// to `probe`, substitute each unknown input dim with `probe`, and run the op's
/// `infer_shape`.
/// `Ok(None)` when the op has no inferable shape; `Err` when its parameters do
/// not fit the input (the op's own `validate`).
fn infer_shape_probe(
    op_json: &str,
    input_dims: &[Option<i64>],
    probe: i64,
) -> Result<Option<Vec<i64>>, String> {
    use crate::graph::step::GraphStep;

    let step = resolve_op_from_json_probe(op_json, probe)?;
    // Buffer ops and geometry steps both carry an `Op` with a real
    // `infer_shape`. Geometry has to be included or the planner has no shape
    // authority for `rasterize`, whose output canvas is fixed by its own
    // width/height params — the Python side then had to assign those hints
    // itself, a side effect the lazy continuation replay silently skipped.
    let op: &dyn view_buffer::Op = match &step {
        GraphStep::Buffer(dto) => dto.as_op(),
        GraphStep::Geometry(geo) => geo,
        // Only buffer and geometry steps carry an inferable shape.
        _ => return Ok(None),
    };
    let input_shape: Vec<usize> = input_dims
        .iter()
        .map(|d| d.unwrap_or_else(|| unknown_dim_probe(probe)).max(1) as usize)
        .collect();
    // The op's own `validate` is the authority on which parameters fit the
    // input: run it against the planned shape. Unknown sizes are placeholders
    // there, so only a failure that depends on the rank alone is a verdict.
    // (A size-level failure against fully known dims is still left to
    // execution, where it has always been a row error; moving it to build time
    // is a behaviour change for the symbolic-shape phase, P9.)
    if let Err(e) = op.validate(&[input_shape.as_slice()], &[]) {
        if e.depends_only_on_rank() {
            return Err(e.to_string());
        }
    }
    // `infer_shape` implementations index their input shape directly. The
    // rank-level mismatches that would panic there were rejected by `validate`
    // above; a size-level one that could not be judged (unknown sizes) may
    // still panic on placeholder sizes, and is "not inferable" rather than a
    // `PanicException` escaping into an ordinary builder call.
    let Ok(out) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        op.infer_shape(&[input_shape.as_slice()])
    })) else {
        return Ok(None);
    };
    // A step whose output shape is data-dependent (extract_contours) returns
    // an empty shape; report it as "not inferable" rather than as rank 0.
    if out.is_empty() {
        return Ok(None);
    }
    Ok(Some(out.iter().map(|&x| x as i64).collect()))
}

/// Shared dtype resolution for `plan_step` (and, transitively, `op_contract`).
///
/// This is the single authority the Python schema layer defers to instead of
/// re-applying a parallel dtype rule: it composes view-buffer's
/// `ViewDto::output_dtype_rule()` with `OutputDTypeRule::resolve`.
///
/// `input_dtype` is a short dtype name (`"u8"`, `"f32"`, …) or the sentinel
/// `"auto"` used for image sources whose decoded dtype is not yet known. For
/// `"auto"`, input-dependent rules (`PreserveInput`, `PromoteToFloat`)
/// propagate `"auto"`; fixed/force rules resolve to their concrete dtype. A
/// structural `out_dtype` parameter (e.g. `normalize`) is not an override here:
/// it is folded into the op's own `Fixed` rule, so it flows through
/// `output_dtype_rule()` like any other fixed dtype.
pub(crate) fn output_dtype_for(
    step: &crate::graph::step::GraphStep,
    input_dtype: &str,
) -> Result<String, String> {
    use view_buffer::OutputDTypeRule as R;
    let rule = step.output_dtype_rule();

    if input_dtype == "auto" {
        return Ok(match rule {
            // Output follows the (unknown) input: stays unknown.
            R::PreserveInput | R::PromoteToFloat => "auto".to_string(),
            // Fixed/force rules ignore the input dtype.
            _ => dtype_short_name(rule.resolve(view_buffer::DType::U8)).to_string(),
        });
    }

    let in_dt = parse_dtype(input_dtype)?;
    Ok(dtype_short_name(rule.resolve(in_dt)).to_string())
}

/// The field names of the `{x, y}` point struct the geometry surfaces publish.
///
/// A runtime accessor plus a Python parity test, rather than a generated module. That
/// keeps `polars_cv.geometry` importable with no compiled extension present,
/// which a generated file would also do but at the cost of a generator and a
/// regenerate-and-diff guard for two field names.
///
/// Read by `test_point_schema_matches_the_rust_declaration`, which holds
/// `geometry.schemas.POINT_SCHEMA` to this in both directions. Without that
/// test this accessor is decoration — Python would still carry its own
/// spelling and the two could drift apart unnoticed.
#[pyfunction]
fn point_schema() -> Vec<String> {
    crate::geom_schema::POINT_FIELD_NAMES
        .iter()
        .map(|s| (*s).to_string())
        .collect()
}

/// The contour `{exterior, holes, is_closed}` field names, in wire order.
///
/// The sibling of [`point_schema`] for contours: read by
/// `test_contour_schema_matches_the_rust_declaration`, which holds
/// `geometry.schemas.CONTOUR_SCHEMA` to `geom_schema::CONTOUR_FIELD_NAMES` in
/// both directions.
#[pyfunction]
fn contour_schema() -> Vec<String> {
    crate::geom_schema::CONTOUR_FIELD_NAMES
        .iter()
        .map(|s| (*s).to_string())
        .collect()
}

/// The bbox `{x, y, width, height}` field names, in wire order.
///
/// The sibling of [`point_schema`] for bounding boxes: read by
/// `test_bbox_schema_matches_the_rust_declaration`, which holds
/// `geometry.schemas.BBOX_SCHEMA` to `geom_schema::BBOX_FIELD_NAMES` in both
/// directions.
#[pyfunction]
fn bbox_schema() -> Vec<String> {
    crate::geom_schema::BBOX_FIELD_NAMES
        .iter()
        .map(|s| (*s).to_string())
        .collect()
}

/// Every polars-cv extension type as `(name, empty storage Series)`, in
/// [`ext_types::ExtType::ALL`] order.
///
/// Read by `test_python_types_match_the_rust_declaration`, which holds
/// `polars_cv.extension_types.EXTENSION_TYPES` to this in both directions. The
/// storage crosses as a zero-length Series rather than field names so the whole
/// dtype — nesting, element types, field order — is compared, not just the
/// top-level names `point_schema` and its siblings publish.
#[pyfunction]
fn extension_types() -> Vec<(&'static str, pyo3_polars::PySeries)> {
    ext_types::ExtType::ALL
        .iter()
        .map(|t| {
            let empty = Series::new_empty(PlSmallStr::from_static(t.name()), &t.storage());
            (t.name(), pyo3_polars::PySeries(empty))
        })
        .collect()
}

/// The 2x3 rotation+scale matrix about `(cx, cy)` — the same authority
/// (`AffineParams::rotation_matrix_2d`) that `from_rotation` builds on.
///
/// It exists so the Python planner's literal `rotate_and_scale` reads this
/// matrix instead of transliterating the trig: `_rotation_matrix`'s all-literal
/// path calls it, keeping the rotation formula in one place. The per-row
/// `pl.Expr` path stays in Python — the engine cannot evaluate an expression at
/// plan time — which is the one remaining, guard-sanctioned copy.
///
/// `angle_deg` is `f64` so the returned matrix matches Python's f64 arithmetic
/// exactly (the planner feeds these straight into a literal `warp_affine`).
#[pyfunction]
fn rotation_matrix_2d(angle_deg: f64, cx: f64, cy: f64, scale: f64) -> Vec<f64> {
    view_buffer::ops::affine::AffineParams::rotation_matrix_2d(angle_deg, cx, cy, scale).to_vec()
}

/// The typed op catalogue as JSON: every typed op's name, Python method name,
/// visibility, docs and fields. The same text is committed as
/// `tests/golden/op_catalog.json`, which `scripts/gen_ops.py` reads; Python
/// tests compare the two so a stale commit cannot pass against a newer build.
#[pyfunction]
fn op_catalog() -> String {
    crate::ops::catalog_json()
}

/// The source/sink catalogue as JSON (`tests/golden/io_catalog.json`): every
/// format and the fields it reads, which `scripts/gen_ops.py` generates
/// `SourceFormat`/`SinkFormat` from.
#[pyfunction]
fn io_catalog() -> String {
    crate::formats::io_catalog_json()
}

/// The enum catalogue as JSON (`tests/golden/enum_catalog.json`): every
/// registered enum's name, doc and spellings, which `scripts/gen_ops.py`
/// generates the Python enum classes from.
#[pyfunction]
fn enum_catalog() -> String {
    crate::naming::enum_catalog_json()
}

/// Validate one serialized source or sink (`kind`) against its typed format, so
/// the builder refuses it when it is written rather than at `collect()`.
///
/// The same deserializer the graph uses; there is no second validator.
#[pyfunction]
fn io_check(kind: &str, spec_json: &str) -> PyResult<()> {
    let result = match kind {
        "sink" => serde_json::from_str::<crate::formats::sink::Sink>(spec_json).map(|_| ()),
        "source" => serde_json::from_str::<crate::formats::source::Source>(spec_json).map(|_| ()),
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "io_check: unknown kind '{other}'"
            )))
        }
    };
    result.map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Return the full contract for a single serialized op spec.
///
/// Returns a dict with the canonical `dtype_rule`, `rank_rule`, `channel_rule`,
/// `spatial_rule` and `is_spatial_window` plus `input_domains` and
/// `output_domain` (from
/// view-buffer's `Domain::name()`). This is the single authority the Python schema layer reads
/// instead of re-declaring, covering the dtype, dimensionality/channel and
/// domain knowledge that previously lived in parallel Python tables
/// (`OPERATION_CONTRACTS` and `_OPERATION_OUTPUT_DOMAIN`).
///
/// Input domain is published only as the *set* `input_domains`. A singular
/// `input_domain` key was published alongside it and read by nothing: two
/// spellings of one fact across an FFI boundary, free to disagree the moment a
/// step's accepted set stopped being a single domain — which is exactly what
/// binary ops and reductions did. `GraphStep::input_domain` remains internal to
/// the executor, where the primary domain is what geometry encoding needs.
#[pyfunction]
fn op_contract(py: Python<'_>, op_json: &str) -> PyResult<Py<PyAny>> {
    let dto = resolve_op_from_json(op_json).map_err(py_value_error)?;
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("dtype_rule", dtype_rule_name(dto.output_dtype_rule()))?;
    dict.set_item("rank_rule", rank_rule_name(dto.output_rank_rule()))?;
    dict.set_item("channel_rule", channel_rule_name(dto.output_channel_rule()))?;
    dict.set_item("spatial_rule", spatial_rule_name(dto.spatial_dependency()))?;
    dict.set_item("is_spatial_window", dto.is_spatial_window())?;
    dict.set_item(
        "input_domains",
        dto.input_domains()
            .iter()
            .map(|d| d.name())
            .collect::<Vec<_>>(),
    )?;
    dict.set_item("output_domain", dto.output_domain().name())?;
    Ok(dict.into())
}

/// The op's identity rule — under what condition an identity-elimination pass
/// may delete it. See [`IdentityRule`](view_buffer::IdentityRule).
///
/// Deliberately **not** part of [`op_contract`], whose published contract is
/// "structural, independent of any parameter value". The `Always` verdict *does*
/// depend on literal parameter values (`pad(0, 0, 0, 0)`), so it lives in its
/// own function.
///
/// Expression-parameter safety is *structural*, resting on the op's declaration
/// rather than on which placeholder value [`resolve_op_from_json`] happens to
/// bind. A value-sensitive verdict — `Always`, or a `WhenShapePreserved` whose
/// candidacy rests on a literal (a crop's `(0, 0)` origin) — names the parameters
/// whose literal values it read
/// ([`IdentityRule::deciding_params`](view_buffer::IdentityRule::deciding_params)):
/// if any deciding param was expression-bound in the *original* spec, the op
/// cannot be proven a no-op at plan time and comes back `never`, regardless of
/// what the neutralized placeholder resolved to. An expression on an *irrelevant*
/// param (a `pad` fill value behind zero amounts) is not a deciding param, so it
/// correctly leaves the op `always`.
///
/// This removes the coupling to the placeholder value: an op that is a no-op at
/// the placeholder can no longer be spoofed, because the gate keys on whether
/// the deciding param was per-row, not on its value.
#[pyfunction]
fn op_identity_rule(op_json: &str) -> PyResult<String> {
    // The names of parameters that are per-row (slots) in the *original* spec,
    // before `resolve_op_from_json` neutralizes them to a placeholder.
    let op: crate::ops::TypedOp = serde_json::from_str(op_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let mut expr_params: std::collections::HashSet<&str> = std::collections::HashSet::new();
    op.visit_slots(&mut |name, _| {
        expr_params.insert(name);
    });

    let rule = resolve_op_from_json(op_json)
        .map_err(py_value_error)?
        .identity_rule();
    // A per-row deciding param means the no-op condition cannot be proven at
    // plan time: the op keeps computing on rows where the value is not the
    // identity value, so it is not removable. Read through the one accessor so
    // every variant that names deciding params is gated, not just `Always`.
    if rule
        .deciding_params()
        .iter()
        .any(|p| expr_params.contains(p))
    {
        return Ok("never".to_string());
    }
    Ok(identity_rule_name(rule))
}

// ============================================================================
// Graph Execution
// ============================================================================

/// Kwargs for the graph-based pipeline function.
///
/// Closed: a kwarg Python emits and Rust does not declare is a drift bug, and
/// this is the outermost struct of the plugin boundary.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphKwargs {
    /// JSON-serialized pipeline graph specification. Expression parameters
    /// in it are positional slots into the call's input series.
    pub graph_json: String,
}

/// Shared implementation for graph execution.
///
/// Handles both single-output and multi-output graphs uniformly. The compiled
/// form of the graph (parsed spec, topological order, slot-bound params,
/// pre-resolved static ops) is fetched from the process-wide cache, so under
/// the streaming engine repeated per-morsel invocations skip re-compilation.
/// Everything data-dependent ("auto" dtype resolution, per-row decode/params)
/// happens inside `CompiledGraph::execute` per call.
fn execute_graph(inputs: &[Series], kwargs: &GraphKwargs) -> PolarsResult<Series> {
    let compiled = crate::graph::get_or_compile(&kwargs.graph_json)?;
    compiled.execute(inputs)
}

/// Unified pipeline graph execution for single output.
///
/// This function handles single-output graph execution using the unified
/// graph format. Returns appropriately typed column based on domain/dtype.
///
/// Use this when you know the graph has only one output ("_output" key).
#[polars_expr(output_type_func_with_kwargs=unified_output_dtype)]
fn vb_graph(inputs: &[Series], kwargs: GraphKwargs) -> PolarsResult<Series> {
    execute_graph(inputs, &kwargs)
}

/// Compute the output dtype for unified graph (single or multi-output).
///
/// This function receives kwargs and parses the graph JSON to determine
/// the exact output type based on domain and dtype information:
/// - Single output: Returns appropriate typed column (Binary, Float64, List, etc.)
/// - Multi-output: Returns Struct with appropriately typed fields
fn unified_output_dtype(input_fields: &[Field], kwargs: GraphKwargs) -> PolarsResult<Field> {
    let name = if !input_fields.is_empty() {
        input_fields[0].name().clone()
    } else {
        PlSmallStr::from_static("output")
    };

    // The compiled graph is fetched from the same cache the execution path
    // uses, and `"auto"` sentinels are resolved by the same
    // `resolved_output_specs` — the planned and executed schema are computed
    // by exactly one piece of logic and cannot diverge.
    let compiled = crate::graph::get_or_compile(&kwargs.graph_json)?;
    let graph = compiled.graph();
    let resolved = crate::graph::resolved_output_specs(
        graph,
        &input_fields
            .iter()
            .map(|f| f.dtype().clone())
            .collect::<Vec<_>>(),
    );

    // The null_with_message error policy appends a reserved `_error` field,
    // which forces struct output even for single-output graphs. This mirrors
    // the execution path exactly (same compiled graph, same resolved specs).
    let with_message = graph.on_error == crate::graph::RowErrorPolicy::NullWithMessage;
    if graph.is_single_output() && !with_message {
        // Single output mode - return typed field based on domain/sink/dtype
        let (_, spec) = resolved
            .first()
            .ok_or_else(|| polars_err!(ComputeError: "Single output graph missing _output key"))?;
        let dtype = crate::graph::dtype_for_output(spec)?;
        Ok(Field::new(name, dtype))
    } else {
        // Multi-output mode - build Struct with typed fields (alias-sorted)
        let mut fields: Vec<Field> = Vec::with_capacity(resolved.len() + 1);
        for (alias, spec) in &resolved {
            let dtype = crate::graph::dtype_for_output(spec)?;
            fields.push(Field::new(PlSmallStr::from(alias.as_str()), dtype));
        }
        if with_message {
            fields.push(Field::new(
                PlSmallStr::from_static("_error"),
                DataType::String,
            ));
        }

        Ok(Field::new(name, DataType::Struct(fields)))
    }
}
