//! polars-cv: A Polars plugin for vision/array operations.
//!
//! This crate provides expression functions for applying image and array
//! processing pipelines to Polars DataFrame columns, powered by view-buffer.

mod cloud;
mod cloud_auth;
mod contour;
mod engine_warning;
mod execute;
mod fetch;
mod geom_arity;
mod geom_params;
mod geom_schema;
mod graph;
mod image_metadata;
mod naming;
mod output;
mod params;
mod pipeline;
mod point;
mod read_bytes;

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::derive::polars_expr;
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
    m.add_function(wrap_pyfunction!(binary_output_dtype, m)?)?;
    m.add_function(wrap_pyfunction!(op_contract, m)?)?;
    m.add_function(wrap_pyfunction!(op_schema, m)?)?;
    m.add_function(wrap_pyfunction!(op_infer_shape, m)?)?;
    m.add_function(wrap_pyfunction!(op_output_channels, m)?)?;
    m.add_function(wrap_pyfunction!(enum_variants, m)?)?;
    m.add_function(wrap_pyfunction!(enum_names, m)?)?;
    m.add_function(wrap_pyfunction!(known_ops, m)?)?;
    m.add_function(wrap_pyfunction!(point_schema, m)?)?;
    m.add_function(wrap_pyfunction!(rotate_affine_params, m)?)?;
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
fn parse_dtype(s: &str) -> PyResult<view_buffer::DType> {
    view_buffer::DType::from_short_name(s)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("unknown dtype {s:?}")))
}

/// Canonical string for an output-dtype rule.
///
/// This is the shared vocabulary the Python planner reads (via `op_contract`)
/// to infer output dtypes: `preserve`, `promote`, `fixed:<dtype>`,
/// `config:<dtype>`.
fn dtype_rule_name(rule: view_buffer::OutputDTypeRule) -> String {
    use view_buffer::OutputDTypeRule as R;
    match rule {
        R::PreserveInput => "preserve".to_string(),
        R::PromoteToFloat => "promote".to_string(),
        R::Fixed(d) => format!("fixed:{}", dtype_short_name(d)),
        R::Configurable(d) => format!("config:{}", dtype_short_name(d)),
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

/// Resolve one serialized op spec to its `ViewDto`, mapping errors to Python.
///
/// Shared by `op_schema` and `op_contract` so neither re-implements the
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
pub(crate) fn resolve_op_from_json(op_json: &str) -> PyResult<crate::graph::step::GraphStep> {
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
) -> PyResult<crate::graph::step::GraphStep> {
    use crate::params::{ParamCtx, ParamValue};

    let mut op_spec: crate::pipeline::OpSpec = serde_json::from_str(op_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("invalid op json: {e}")))?;
    // Bind each expression param to a placeholder slot holding `1_i64`,
    // mirroring what graph compilation does with the real input columns.
    // `label_reduce.contours` carries the column *name* through the step and
    // stays unbound, exactly as in `graph::compiled::bind_graph_params`.
    let keep_named = op_spec.op == "label_reduce";
    // rasterize-by-shape-reference carries no width/height (they come from
    // another node's buffer at execution, via the RasterizeShapeRef
    // resolver). Give introspection placeholder dims so the op resolves; the
    // structural schema never depends on their values.
    //
    // They are *expression* placeholders, not literals, because `op_infer_shape`
    // does read their values: it reports a dimension as known only when it is
    // identical across probes, and a literal placeholder would publish a 1x1
    // canvas as fact for a mask sized by another node.
    if op_spec.op == "rasterize" && op_spec.params.contains_key("shape_ref") {
        for dim in ["width", "height"] {
            op_spec
                .params
                .entry(dim.to_string())
                .or_insert(ParamValue::Expr {
                    col: Some("__shape_ref__".to_string()),
                });
        }
    }
    let mut placeholders: Vec<Series> = Vec::new();
    for (pname, p) in op_spec.params.iter_mut() {
        if keep_named && pname == "contours" {
            continue;
        }
        if matches!(p, ParamValue::Expr { .. }) {
            *p = ParamValue::Slot {
                idx: placeholders.len(),
            };
            placeholders.push(Series::new("".into(), &[probe]));
        } else if let ParamValue::Literal { value } = p {
            // A literal may itself be a list of ParamValue dicts (reshape's
            // shape). Neutralize any expression entries the same way so the
            // op's structural schema (here: the target rank = entry count)
            // is introspectable regardless of per-row dims.
            if let Some(arr) = value.as_array_mut() {
                for entry in arr.iter_mut() {
                    if entry.get("type").and_then(|t| t.as_str()) == Some("expr") {
                        *entry = serde_json::json!({"type": "literal", "value": probe});
                    }
                }
            }
        }
    }
    // A *probe* context: placeholders are integers, so a dynamic enum or flag
    // param cannot be read from one. `ParamCtx::probe` tells the enum/bool
    // accessors to substitute their default instead. Sound because only params
    // with no shape/rank/dtype effect are allowed to be dynamic, so the variant
    // probing picks cannot change the inferred schema.
    let ctx = ParamCtx::probe(&placeholders);
    crate::execute::resolve_op(&op_spec, 0, &ctx)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("resolve_op: {e}")))
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
#[pyfunction]
fn op_infer_shape(op_json: &str, input_dims: Vec<Option<i64>>) -> PyResult<Vec<Option<i64>>> {
    const PROBES: [i64; 4] = [7, 13, 90, 180];
    let runs: Vec<Vec<i64>> = PROBES
        .iter()
        .map(|&p| infer_shape_probe(op_json, &input_dims, p))
        .collect::<PyResult<_>>()?;
    let first = &runs[0];
    // Rank is structural (never data-dependent), so it must be stable across
    // probes; a variation signals a contract bug rather than an unknown.
    if runs.iter().any(|r| r.len() != first.len()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "op_infer_shape: output rank varied across shape probes",
        ));
    }
    Ok((0..first.len())
        .map(|i| {
            let v = first[i];
            runs.iter().all(|r| r[i] == v).then_some(v)
        })
        .collect())
}

/// Plan-time output channel count for a single op — the single authority the
/// Python planner reads instead of re-deriving the rule's arithmetic.
///
/// `input_channels` is the current channel hint, `None` when unknown at plan
/// time. The result is `None` whenever the count is not determinable: the op
/// produces no `[H, W, C]` image (`NotApplicable`), its effect is not knowable
/// from the rule alone (`Unknown`), or a channel-dependent rule was given an
/// unknown input.
///
/// This exists because the Python side used to re-implement
/// `OutputChannelRule::apply` by parsing the stringified rule, and the two
/// readings disagreed: `apply` returns `None` for `NotApplicable` while Python
/// left the hint unchanged. That divergence was invisible only because every
/// `NotApplicable` op also dropped below rank 3 — where the planner clears the
/// channel hint anyway — except `histogram(output="quantized")`, which was
/// mislabelled and happens to preserve channels. Two errors cancelling is not
/// a contract, so the arithmetic now lives in one place.
#[pyfunction]
#[pyo3(signature = (op_json, input_channels=None))]
fn op_output_channels(op_json: &str, input_channels: Option<usize>) -> PyResult<Option<usize>> {
    let step = resolve_op_from_json(op_json)?;
    Ok(step.output_channel_rule().apply(input_channels))
}

/// One probe of [`op_infer_shape`]: resolve the op with expression params bound
/// to `probe`, substitute each unknown input dim with `probe`, and run the op's
/// `infer_shape`.
fn infer_shape_probe(op_json: &str, input_dims: &[Option<i64>], probe: i64) -> PyResult<Vec<i64>> {
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
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "op_infer_shape: only buffer and geometry ops have an inferable shape",
            ))
        }
    };
    let input_shape: Vec<usize> = input_dims
        .iter()
        .map(|d| d.unwrap_or(probe).max(1) as usize)
        .collect();
    // `infer_shape` implementations index their input shape directly, so an
    // op whose parameters disagree with the input rank (a transpose carrying
    // three axes over rank-2 data) panics rather than returning an error.
    // This is a *planning* call reached from an ordinary Python builder, so a
    // panic here would escape as a `PanicException` with a Rust backtrace
    // instead of the ValueError the builder contract promises. Catch it and
    // report "not inferable"; the builder validates the parameters itself and
    // raises the actionable message.
    let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        op.infer_shape(&[input_shape.as_slice()])
    }))
    .map_err(|_| {
        pyo3::exceptions::PyValueError::new_err(
            "op_infer_shape: operation parameters are inconsistent with the input rank",
        )
    })?;
    // A step whose output shape is data-dependent (extract_contours) returns
    // an empty shape; report it as "not inferable" rather than as rank 0.
    if out.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "op_infer_shape: output shape is not knowable at plan time",
        ));
    }
    Ok(out.iter().map(|&x| x as i64).collect())
}

/// Shared dtype resolution for `op_schema` (and, transitively, `op_contract`).
///
/// This is the single authority the Python schema layer defers to instead of
/// re-applying a parallel dtype rule: it composes view-buffer's
/// `ViewDto::output_dtype_rule()` with `OutputDTypeRule::resolve`.
///
/// `input_dtype` is a short dtype name (`"u8"`, `"f32"`, …) or the sentinel
/// `"auto"` used for image sources whose decoded dtype is not yet known. For
/// `"auto"`, input-dependent rules (`PreserveInput`, `PromoteToFloat`)
/// propagate `"auto"`; fixed/configurable rules resolve to their concrete
/// dtype. An `out_dtype` literal parameter overrides the result only for the
/// `Configurable` rule, mirroring the configurable-output contract.
fn output_dtype_for(
    step: &crate::graph::step::GraphStep,
    op_json: &str,
    input_dtype: &str,
) -> PyResult<String> {
    use view_buffer::OutputDTypeRule as R;
    let dto = step;
    let rule = dto.output_dtype_rule();

    // The out_dtype override is honored only for the configurable rule
    // (other rules ignore it).
    let override_dt = if matches!(rule, R::Configurable(_)) {
        out_dtype_override(op_json)?
    } else {
        None
    };

    if input_dtype == "auto" {
        if let Some(d) = override_dt {
            return Ok(dtype_short_name(d).to_string());
        }
        return Ok(match rule {
            // Output follows the (unknown) input: stays unknown.
            R::PreserveInput | R::PromoteToFloat => "auto".to_string(),
            // Fixed/configurable/force rules ignore the input dtype.
            _ => dtype_short_name(rule.resolve(view_buffer::DType::U8, None)).to_string(),
        });
    }

    let in_dt = parse_dtype(input_dtype)?;
    Ok(dtype_short_name(rule.resolve(in_dt, override_dt)).to_string())
}

/// Resolve one op's full schema effect: `(domain, dtype, ndim)`.
///
/// The single planning-time authority the Python `Pipeline` consults per
/// appended op — including the param-dependent cases (`cast` target,
/// `histogram` output mode, reduction `axis` presence) that previously lived
/// as Python-side special cases. Inputs are the pipeline's current state;
/// `"auto"` dtype and `None` ndim propagate where a rule cannot resolve them.
///
/// One deliberate special case: `histogram(output="buckets")` reports dtype
/// `"auto"` — buckets are struct-encoded by the sink, so an element dtype is
/// an encoding concern, not a schema one.
#[pyfunction]
#[pyo3(signature = (op_json, input_domain, input_dtype, input_ndim=None))]
fn op_schema(
    op_json: &str,
    input_domain: &str,
    input_dtype: &str,
    input_ndim: Option<usize>,
) -> PyResult<(String, String, Option<usize>)> {
    use crate::graph::step::GraphStep;
    use view_buffer::ops::{Domain, HistogramOutput, OutputRankRule};

    let step = resolve_op_from_json(op_json)?;

    let out_domain = step.output_domain();
    let domain = if out_domain == Domain::Any {
        input_domain.to_string()
    } else {
        out_domain.name().to_string()
    };

    let dtype = if matches!(&step, GraphStep::Histogram(op) if op.output == HistogramOutput::Buckets)
    {
        "auto".to_string()
    } else {
        output_dtype_for(&step, op_json, input_dtype)?
    };

    let ndim = match step.output_rank_rule() {
        OutputRankRule::Fixed(n) => Some(n),
        OutputRankRule::PreserveRank => input_ndim,
        OutputRankRule::ReduceByOne => input_ndim.map(|n| n.saturating_sub(1).max(1)),
        OutputRankRule::Unknown => None,
    };
    // Scalar/vector domains pin the dimensionality regardless of the rule.
    let ndim = match domain.as_str() {
        "scalar" => Some(0),
        "vector" => Some(1),
        _ => ndim,
    };

    Ok((domain, dtype, ndim))
}

/// Map a Python-facing binary op name to its view-buffer `BinaryOp`.
///
/// Reads `BinaryOp::NAMED` — the same table `resolve_op` dispatches on and the
/// registry surfaces — so the planner's two-input dtype query, the executor and
/// Python cannot drift.
fn parse_binary_op(name: &str) -> PyResult<view_buffer::BinaryOp> {
    view_buffer::naming::lookup(view_buffer::BinaryOp::NAMED, name).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("unknown binary op {name:?}"))
    })
}

/// Resolve the output dtype of a binary op given *both* operand dtypes.
///
/// This is the two-input analogue of [`output_dtype_for`]. Binary ops promote
/// across both operands (and Divide/Ratio further promote to float for true
/// division), so the planner cannot reuse the single-input rule — it defers to
/// view-buffer's [`BinaryOp::output_dtype`] authority, the same one execution
/// uses, so plan and exec dtypes are computed once.
///
/// Either operand may be the `"auto"` sentinel (an image source whose decoded
/// dtype is not yet known); the result is then `"auto"`, and a downstream typed
/// list/array sink requires the user to supply an explicit dtype.
#[pyfunction]
fn binary_output_dtype(op_name: &str, left: &str, right: &str) -> PyResult<String> {
    if left == "auto" || right == "auto" {
        return Ok("auto".to_string());
    }
    let op = parse_binary_op(op_name)?;
    let l = parse_dtype(left)?;
    let r = parse_dtype(right)?;
    Ok(dtype_short_name(op.output_dtype(l, r)).to_string())
}

/// Extract the literal `out_dtype` parameter from a serialized op spec, if any.
///
/// Returns `None` when the parameter is absent or is an expression (dynamic),
/// in which case the rule's default applies.
fn out_dtype_override(op_json: &str) -> PyResult<Option<view_buffer::DType>> {
    let op_spec: crate::pipeline::OpSpec = serde_json::from_str(op_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("invalid op json: {e}")))?;
    match op_spec.params.get("out_dtype") {
        Some(crate::params::ParamValue::Literal { value }) => match value.as_str() {
            Some(s) => Ok(Some(parse_dtype(s)?)),
            None => Ok(None),
        },
        _ => Ok(None),
    }
}

/// Return the string variants of a Rust enum, for Python<->Rust parity checks.
///
/// Reads `view_buffer::naming::REGISTRY`, so registering an enum there is what
/// makes it queryable from Python — one act, not two. Its names come from the
/// same canonical `NAMED` table the executor's parameter parser consumes, so
/// the names surfaced to Python and the names the executor accepts cannot
/// drift.
///
/// The graph's source/sink formats are not here because they have no Rust
/// enum: the boundary carries them as plain strings and Python's
/// `SourceFormat`/`SinkFormat` are their single definition. view-buffer's
/// shadowing copies were deleted with its unreachable composition layer, so
/// there is no longer a format vocabulary to reconcile.
///
/// Two registries are consulted, not one: most vocabularies are the engine's,
/// but a few describe things the engine has no concept of (how a graph handles
/// a failing row, what a null parameter means, how a path read reports an
/// unreadable file) and live in [`crate::naming::PLUGIN_REGISTRY`]. Both are
/// read the same way, and there is no hand-written arm for either — the arm
/// `BinaryOp` used to need is gone, its table having moved next to the enum.
#[pyfunction]
fn enum_variants(name: &str) -> PyResult<Vec<String>> {
    let variants: Vec<&str> = view_buffer::naming::registered_variants(name)
        .or_else(|| crate::naming::registered_variants(name))
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "no canonical Rust enum named {name}; known: {:?}",
                enum_names()
            ))
        })?;
    Ok(variants.into_iter().map(str::to_string).collect())
}

/// The name of every enum `enum_variants` can answer for.
///
/// Exists so the Python parity tests can iterate the vocabularies rather than
/// hand-listing them. A hand-written list is what let `LabelReduction` and
/// `LabelRegionMode` sit unchecked: they had `NAMED` tables, and no test named
/// them, so nothing noticed. A test that reads this cannot miss a new enum.
#[pyfunction]
fn enum_names() -> Vec<String> {
    view_buffer::naming::registered_names()
        .into_iter()
        .chain(crate::naming::registered_names())
        .map(str::to_string)
        .collect()
}

/// The field names of the `{x, y}` point struct the geometry surfaces publish.
///
/// Surfaced the way [`enum_variants`] surfaces the naming registry: a runtime
/// accessor plus a Python parity test, rather than a generated module. That
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

/// The affine parameters a `rotate` executes as, for a known input shape.
///
/// Returns `(matrix, output_height, output_width)` straight out of
/// `AffineParams::from_rotation` — **the** rotation-matrix authority, and the
/// one an unfused `rotate` actually runs through (`ComputeOp::RotateAffine`).
///
/// It exists so the Python planner's affine fusion can *read* that matrix
/// instead of recomputing it. It used to transliterate `from_rotation` line for
/// line, which meant a `rotate()` produced its matrix from Rust when it stood
/// alone and from Python when a neighbouring op made it fusible — two
/// implementations of one formula, differing already in angle normalisation
/// (`angle % 360` in Python, raw in Rust) and in rounding (Python's `round` is
/// half-to-even, Rust's is half-away-from-zero). Nothing compared them; the
/// test that looked like it did compared Python against a third copy of itself.
#[pyfunction]
fn rotate_affine_params(
    angle_deg: f32,
    input_height: u32,
    input_width: u32,
    expand: bool,
) -> PyResult<(Vec<f64>, u32, u32)> {
    let params = view_buffer::ops::affine::AffineParams::from_rotation(
        angle_deg,
        input_height,
        input_width,
        expand,
        // Neither affects the matrix or the output size; the caller keeps the
        // op's own values for these and only wants the geometry.
        view_buffer::ops::affine::InterpolationType::Bilinear,
        0.0,
    );
    Ok((
        params.matrix.to_vec(),
        params.output_height,
        params.output_width,
    ))
}

/// Return the names of every operation the executor can resolve.
///
/// This is the registry surfaced from [`crate::execute::KNOWN_OPS`] so Python
/// can assert that every op a `Pipeline` emits is executable (B1) without
/// hand-syncing a second list.
#[pyfunction]
fn known_ops() -> Vec<String> {
    crate::execute::KNOWN_OPS
        .iter()
        .map(|s| s.to_string())
        .collect()
}

/// Return the full contract for a single serialized op spec.
///
/// Returns a dict with the canonical `dtype_rule`, `rank_rule` and
/// `channel_rule` plus `input_domains` and `output_domain` (from view-buffer's
/// `Domain::name()`). This is the single authority the Python schema layer reads
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
    let dto = resolve_op_from_json(op_json)?;
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("dtype_rule", dtype_rule_name(dto.output_dtype_rule()))?;
    dict.set_item("rank_rule", rank_rule_name(dto.output_rank_rule()))?;
    dict.set_item("channel_rule", channel_rule_name(dto.output_channel_rule()))?;
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
    /// JSON-serialized pipeline graph specification.
    pub graph_json: String,
    /// Names of expression columns (for resolving dynamic parameters).
    #[serde(default)]
    pub expr_column_names: Vec<String>,
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
    let compiled = crate::graph::get_or_compile(&kwargs.graph_json, &kwargs.expr_column_names)?;
    // Held for the duration of the call so concurrent morsel invocations are
    // observed; warns once if a large batch runs single-threaded (in-memory
    // engine). See `engine_warning`.
    let _call_guard =
        crate::engine_warning::CallGuard::enter(inputs.first().map(|s| s.len()).unwrap_or(0));
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
    let compiled = crate::graph::get_or_compile(&kwargs.graph_json, &kwargs.expr_column_names)?;
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
