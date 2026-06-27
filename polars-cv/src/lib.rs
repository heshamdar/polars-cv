//! polars-cv: A Polars plugin for vision/array operations.
//!
//! This crate provides expression functions for applying image and array
//! processing pipelines to Polars DataFrame columns, powered by view-buffer.

mod cloud;
mod contour;
mod execute;
mod graph;
mod image_metadata;
mod output;
mod params;
mod pipeline;
mod point;

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

/// Python module entry point for maturin.
/// The module name `_lib` must match pyproject.toml's `module-name = "polars_cv._lib"`.
#[pymodule]
#[pyo3(name = "_lib")]
fn polars_cv_lib(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(op_output_dtype, m)?)?;
    m.add_function(wrap_pyfunction!(binary_output_dtype, m)?)?;
    m.add_function(wrap_pyfunction!(op_contract, m)?)?;
    m.add_function(wrap_pyfunction!(enum_variants, m)?)?;
    m.add_function(wrap_pyfunction!(known_ops, m)?)?;
    Ok(())
}

// ============================================================================
// Contract introspection (single-authority bridge for the Python schema layer)
// ============================================================================

/// Canonical short name for a view-buffer `DType`.
///
/// Matches the values of the Python `DType` enum so the two vocabularies line up.
fn dtype_short_name(dt: view_buffer::DType) -> &'static str {
    use view_buffer::DType;
    match dt {
        DType::U8 => "u8",
        DType::I8 => "i8",
        DType::U16 => "u16",
        DType::I16 => "i16",
        DType::U32 => "u32",
        DType::I32 => "i32",
        DType::U64 => "u64",
        DType::I64 => "i64",
        DType::F32 => "f32",
        DType::F64 => "f64",
    }
}

/// Parse a short dtype name back into a view-buffer `DType`.
///
/// Inverse of [`dtype_short_name`]. Used to turn the Python schema layer's
/// dtype strings into the `DType` the canonical [`OutputDTypeRule::resolve`]
/// authority operates on.
fn parse_dtype(s: &str) -> PyResult<view_buffer::DType> {
    use view_buffer::DType;
    Ok(match s {
        "u8" => DType::U8,
        "i8" => DType::I8,
        "u16" => DType::U16,
        "i16" => DType::I16,
        "u32" => DType::U32,
        "i32" => DType::I32,
        "u64" => DType::U64,
        "i64" => DType::I64,
        "f32" => DType::F32,
        "f64" => DType::F64,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown dtype {other:?}"
            )))
        }
    })
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
/// Shared by `op_output_dtype` and `op_contract` so neither re-implements the
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
fn resolve_op_from_json(op_json: &str) -> PyResult<view_buffer::ViewDto> {
    use crate::params::{ParamCtx, ParamValue};

    let mut op_spec: crate::pipeline::OpSpec = serde_json::from_str(op_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("invalid op json: {e}")))?;
    // Bind each expression param to a placeholder slot holding `1_i64`,
    // mirroring what graph compilation does with the real input columns.
    // `label_reduce.contours` carries the column *name* through the DTO and
    // stays unbound, exactly as in `graph::compiled::bind_graph_params`.
    let keep_named = op_spec.op == "label_reduce";
    let mut placeholders: Vec<Series> = Vec::new();
    for (pname, p) in op_spec.params.iter_mut() {
        if keep_named && pname == "contours" {
            continue;
        }
        if matches!(p, ParamValue::Expr { .. }) {
            *p = ParamValue::Slot {
                idx: placeholders.len(),
            };
            placeholders.push(Series::new("".into(), &[1_i64]));
        }
    }
    let ctx = ParamCtx::from_inputs(&placeholders);
    crate::execute::resolve_op(&op_spec, 0, &ctx)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("resolve_op: {e}")))
}

/// Resolve the concrete output dtype of one op given its input dtype.
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
#[pyfunction]
fn op_output_dtype(op_json: &str, input_dtype: &str) -> PyResult<String> {
    use view_buffer::OutputDTypeRule as R;
    let dto = resolve_op_from_json(op_json)?;
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

/// Map a Python-facing binary op name to its view-buffer `BinaryOp`.
///
/// These are the same op strings the `Pipeline` emits (and `resolve_op`
/// consumes); kept here so the planner's two-input dtype query does not need a
/// full serialized op spec.
fn parse_binary_op(name: &str) -> PyResult<view_buffer::BinaryOp> {
    use view_buffer::BinaryOp as B;
    Ok(match name {
        "add" => B::Add,
        "subtract" => B::Subtract,
        "multiply" => B::Multiply,
        "blend" => B::Blend,
        "divide" => B::Divide,
        "ratio" => B::Ratio,
        "maximum" => B::Maximum,
        "minimum" => B::Minimum,
        "bitwise_and" => B::BitwiseAnd,
        "bitwise_or" => B::BitwiseOr,
        "bitwise_xor" => B::BitwiseXor,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown binary op {other:?}"
            )))
        }
    })
}

/// Resolve the output dtype of a binary op given *both* operand dtypes.
///
/// This is the two-input analogue of [`op_output_dtype`]. Binary ops promote
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
/// Supports the enums that have a single canonical Rust definition the Python
/// user-facing enum must mirror. `DType` and `Domain` are sourced from
/// view-buffer (their `dtype_short_name` / `Domain::name`). Format enums are
/// intentionally not exposed here: view-buffer's `SourceFormat`/`SinkFormat`
/// use a different vocabulary than the graph's string formats, a divergence
/// slated for consolidation rather than enforcement.
#[pyfunction]
fn enum_variants(name: &str) -> PyResult<Vec<String>> {
    use view_buffer::ops::Domain;
    use view_buffer::DType;
    let variants: Vec<String> = match name {
        "DType" => [
            DType::U8,
            DType::I8,
            DType::U16,
            DType::I16,
            DType::U32,
            DType::I32,
            DType::U64,
            DType::I64,
            DType::F32,
            DType::F64,
        ]
        .iter()
        .map(|d| dtype_short_name(*d).to_string())
        .collect(),
        "Domain" => [
            Domain::Buffer,
            Domain::Contour,
            Domain::Scalar,
            Domain::Vector,
            Domain::Any,
        ]
        .iter()
        .map(|d| d.name().to_string())
        .collect(),
        // User-facing API enums. Each maps its variants through an exhaustive
        // `match`, so adding a variant to the view-buffer enum is a compile error
        // here until the canonical string is supplied — the Rust enum is the
        // authority and the Python parity tests assert the surfaced set matches.
        "NormalizeMethod" => {
            use view_buffer::ops::NormalizeMethod as M;
            [
                M::MinMax,
                M::ZScore,
                M::Preset {
                    mean: vec![],
                    std: vec![],
                },
            ]
            .iter()
            .map(|m| {
                match m {
                    M::MinMax => "minmax",
                    M::ZScore => "zscore",
                    M::Preset { .. } => "preset",
                }
                .to_string()
            })
            .collect()
        }
        "ColorSpace" => {
            use view_buffer::ops::ColorSpace as C;
            [C::Rgb, C::Bgr, C::Hsv, C::Lab, C::YCbCr, C::Gray]
                .iter()
                .map(|c| {
                    match c {
                        C::Rgb => "rgb",
                        C::Bgr => "bgr",
                        C::Hsv => "hsv",
                        C::Lab => "lab",
                        C::YCbCr => "ycbcr",
                        C::Gray => "gray",
                    }
                    .to_string()
                })
                .collect()
        }
        "HashAlgorithm" => {
            use view_buffer::ops::HashAlgorithm as H;
            [H::Average, H::Difference, H::Perceptual, H::Blockhash]
                .iter()
                .map(|h| {
                    match h {
                        H::Average => "average",
                        H::Difference => "difference",
                        H::Perceptual => "perceptual",
                        H::Blockhash => "blockhash",
                    }
                    .to_string()
                })
                .collect()
        }
        "HistogramOutput" => {
            use view_buffer::ops::HistogramOutput as H;
            [H::Counts, H::Normalized, H::Quantized, H::Edges, H::Buckets]
                .iter()
                .map(|h| {
                    match h {
                        H::Counts => "counts",
                        H::Normalized => "normalized",
                        H::Quantized => "quantized",
                        H::Edges => "edges",
                        H::Buckets => "buckets",
                    }
                    .to_string()
                })
                .collect()
        }
        "PadMode" => {
            use view_buffer::ops::dto::PadMode as P;
            [P::Constant, P::Edge, P::Reflect, P::Symmetric]
                .iter()
                .map(|p| {
                    match p {
                        P::Constant => "constant",
                        P::Edge => "edge",
                        P::Reflect => "reflect",
                        P::Symmetric => "symmetric",
                    }
                    .to_string()
                })
                .collect()
        }
        "PadPosition" => {
            use view_buffer::ops::dto::PadPosition as P;
            [P::Center, P::TopLeft, P::BottomRight]
                .iter()
                .map(|p| {
                    match p {
                        P::Center => "center",
                        P::TopLeft => "top-left",
                        P::BottomRight => "bottom-right",
                    }
                    .to_string()
                })
                .collect()
        }
        // FilterType: Rust's `Triangle` is surfaced as "bilinear" (its API name).
        // Python deliberately exposes only a subset (nearest/bilinear/lanczos3);
        // the parity test asserts Python ⊆ this set, not equality.
        "FilterType" => {
            use view_buffer::ops::FilterType as F;
            [
                F::Nearest,
                F::Triangle,
                F::CatmullRom,
                F::Gaussian,
                F::Lanczos3,
            ]
            .iter()
            .map(|f| {
                match f {
                    F::Nearest => "nearest",
                    F::Triangle => "bilinear",
                    F::CatmullRom => "catmullrom",
                    F::Gaussian => "gaussian",
                    F::Lanczos3 => "lanczos3",
                }
                .to_string()
            })
            .collect()
        }
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "no canonical Rust enum named {other}"
            )))
        }
    };
    Ok(variants)
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
/// `channel_rule` plus `input_domain` and `output_domain` (from view-buffer's
/// `Domain::name()`). This is the single authority the Python schema layer reads
/// instead of re-declaring, covering the dtype, dimensionality/channel and
/// domain knowledge that previously lived in parallel Python tables
/// (`OPERATION_CONTRACTS` and `_OPERATION_OUTPUT_DOMAIN`).
#[pyfunction]
fn op_contract(py: Python<'_>, op_json: &str) -> PyResult<Py<PyAny>> {
    let dto = resolve_op_from_json(op_json)?;
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("dtype_rule", dtype_rule_name(dto.output_dtype_rule()))?;
    dict.set_item("rank_rule", rank_rule_name(dto.output_rank_rule()))?;
    dict.set_item("channel_rule", channel_rule_name(dto.output_channel_rule()))?;
    dict.set_item("input_domain", dto.input_domain().name())?;
    dict.set_item("output_domain", dto.output_domain().name())?;
    Ok(dict.into())
}

// ============================================================================
// Graph Execution
// ============================================================================

/// Kwargs for the graph-based pipeline function.
#[derive(Debug, Deserialize)]
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
    let resolved =
        crate::graph::resolved_output_specs(graph, input_fields.first().map(|f| f.dtype()));

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
