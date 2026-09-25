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
mod passes;
mod plan;
mod point;
mod read_bytes;

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::derive::polars_expr;

use crate::passes::{node_pass, pass_catalog};
use crate::plan::{_plan_state_from_json, plan_assert, plan_sink, plan_source, plan_step};
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
    m.add_class::<plan::State>()?;
    m.add_function(wrap_pyfunction!(_plan_state_from_json, m)?)?;
    m.add_function(wrap_pyfunction!(plan_step, m)?)?;
    m.add_function(wrap_pyfunction!(plan_source, m)?)?;
    m.add_function(wrap_pyfunction!(plan_assert, m)?)?;
    m.add_function(wrap_pyfunction!(node_pass, m)?)?;
    m.add_function(wrap_pyfunction!(pass_catalog, m)?)?;
    m.add_function(wrap_pyfunction!(op_catalog, m)?)?;
    m.add_function(wrap_pyfunction!(io_catalog, m)?)?;
    m.add_function(wrap_pyfunction!(enum_catalog, m)?)?;
    m.add_function(wrap_pyfunction!(plan_sink, m)?)?;
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

/// A planner error (a plain message, so the planning core needs no
/// interpreter) as the `ValueError` Python sees.
pub(crate) fn py_value_error(msg: String) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(msg)
}

/// Resolve one serialized op spec at plan time, for its rules (domain,
/// dtype, rank, channels, identity), which no per-row value can change.
///
/// Shared by `plan_step` and the passes so neither re-implements the
/// deserialize → resolve path.
pub(crate) fn resolve_op_from_json(op_json: &str) -> Result<crate::graph::step::GraphStep, String> {
    let op: crate::ops::TypedOp = serde_json::from_str(op_json).map_err(|e| e.to_string())?;
    planning_step(&op)
}

/// Resolve a typed op at plan time, for its rules (see
/// [`ParamCtx::planning`](crate::params::ParamCtx::planning)).
pub(crate) fn planning_step(
    op: &crate::ops::TypedOp,
) -> Result<crate::graph::step::GraphStep, String> {
    op.resolve(0, &crate::params::ParamCtx::planning())
        .map_err(|e| format!("resolve_op: {e}"))
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
