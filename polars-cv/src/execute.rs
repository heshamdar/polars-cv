//! Pipeline execution engine.
//!
//! This module handles the execution of vision pipelines on Polars Series,
//! including parameter resolution and view-buffer integration.

use polars::prelude::*;

use view_buffer::{
    geometry::rasterize::rasterize, DType, GeometryOp, ImageAdapter, ImageCodec, PlannedDType,
    ViewBuffer,
};

use crate::graph::step::GraphStep;
use crate::params::{get, OpParams, ParamCtx, ParamValue};
use crate::pipeline::{OpSpec, SinkSpec, SourceSpec};
use view_buffer::geometry::label::{LabelReduction, LabelRegionMode};
use view_buffer::naming;

/// Parse rasterize's optional style parameters `(fill_value, background)` —
/// shared with the graph executor's rasterize-by-shape-ref path so the two
/// sites cannot diverge.
pub(crate) fn resolve_rasterize_style(
    params: &OpParams<'_>,
    row_idx: usize,
    ctx: &ParamCtx,
) -> PolarsResult<(u8, u8)> {
    Ok((
        get::opt_u8(params, "fill_value", 255, row_idx, ctx)?,
        get::opt_u8(params, "background", 0, row_idx, ctx)?,
    ))
}

/// Decode a contour source by parsing the geometry and rasterizing to ViewBuffer.
///
/// The column may hold one contour per row or a whole set (`List[Contour]`) —
/// `parse_contour_set` accepts both, and the set is painted as a union, exactly
/// as the `rasterize` op paints the set `extract_contours` produces.
pub fn decode_contour_source(
    value: &AnyValue,
    row_idx: usize,
    source: &SourceSpec,
    ctx: &ParamCtx,
) -> PolarsResult<ViewBuffer> {
    // Parse via the plugin's single contour parser (contour.rs).
    let contours = crate::contour::parse_contour_set(value)?;

    // Resolve dimensions
    let (width, height) = resolve_contour_dimensions(row_idx, source, ctx)?;

    // Get fill and background values (both per-row capable)
    let (fill_value, background) = source.resolve_fill(row_idx, ctx)?;

    // Rasterize the contours to a ViewBuffer
    Ok(rasterize(&contours, width, height, fill_value, background))
}

/// Decode a contour source with explicit dimensions (for graph execution with shape inference).
///
/// This variant is used when dimensions are resolved from a shape reference (another node's buffer)
/// rather than from explicit width/height parameters.
pub fn decode_contour_source_with_dims(
    value: &AnyValue,
    width: u32,
    height: u32,
    fill_value: u8,
    background: u8,
) -> PolarsResult<ViewBuffer> {
    // Parse via the plugin's single contour parser (contour.rs).
    let contours = crate::contour::parse_contour_set(value)?;

    // Rasterize the contours to a ViewBuffer
    Ok(rasterize(&contours, width, height, fill_value, background))
}

/// Resolve contour dimensions from pipeline source spec.
fn resolve_contour_dimensions(
    row_idx: usize,
    source: &SourceSpec,
    ctx: &ParamCtx,
) -> PolarsResult<(u32, u32)> {
    // shape_node sources never reach this function: the graph executor
    // resolves the referenced node's dimensions and calls
    // `decode_contour_source_with_dims` instead (see compiled.rs).

    // Get explicit width and height
    let width = source
        .width
        .as_ref()
        .ok_or_else(|| polars_err!(ComputeError: "Contour source requires 'width' parameter"))?
        .resolve_usize(row_idx, ctx)? as u32;

    let height = source
        .height
        .as_ref()
        .ok_or_else(|| polars_err!(ComputeError: "Contour source requires 'height' parameter"))?
        .resolve_usize(row_idx, ctx)? as u32;

    Ok((width, height))
}

/// Decode a JPEG at a reduced IDCT scale sufficient for `max_size` pixels on
/// the long side.
///
/// Picks the smallest of the decoder's supported scale factors (1/8, 1/4,
/// 1/2, 1) whose output is >= `max_size` on at least one axis, so the long
/// side never drops below `min(max_size, original)` — downstream resizes
/// down to `max_size` never upscale. Returns `None` for non-JPEG bytes or
/// pixel formats the scaled path does not cover (16-bit, CMYK); the caller
/// falls back to the full decoder.
fn decode_jpeg_scaled(bytes: &[u8], max_size: u32) -> Option<ViewBuffer> {
    // JPEG SOI marker; anything else takes the regular decode path.
    if bytes.len() < 2 || bytes[0] != 0xFF || bytes[1] != 0xD8 {
        return None;
    }
    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(bytes));
    let requested = max_size.min(u16::MAX as u32) as u16;
    let (width, height) = decoder.scale(requested, requested).ok()?;
    let pixels = decoder.decode().ok()?;
    let info = decoder.info()?;
    let (h, w) = (height as usize, width as usize);
    // Shapes mirror ImageAdapter::decode: grayscale is [H, W, 1].
    match info.pixel_format {
        jpeg_decoder::PixelFormat::L8 => {
            Some(ViewBuffer::from_vec_with_shape(pixels, vec![h, w, 1]))
        }
        jpeg_decoder::PixelFormat::RGB24 => {
            Some(ViewBuffer::from_vec_with_shape(pixels, vec![h, w, 3]))
        }
        _ => None,
    }
}

/// Decode encoded image bytes (PNG/JPEG/TIFF/…) into a ViewBuffer, honouring
/// the source's decode-scale and dtype settings.
///
/// Reached for `image_bytes` sources, `file_path` sources once their bytes are
/// read, and `auto` sources that resolved to image bytes. The executor
/// dispatches on its `SourceFormat` before calling, so this takes no format.
/// It used to take one, which made `file_path` and `auto` rows clone their
/// whole `SourceSpec` to overwrite the format string first (CR-37). `blob`/`raw`
/// sources never reach it: they decode zero-copy via
/// `graph::decode::decode_binary_zero_copy`.
pub fn decode_image_bytes(bytes: &[u8], source: &SourceSpec) -> PolarsResult<ViewBuffer> {
    // An explicit decode-scale assertion lets JPEG decode skip work via IDCT
    // scaling; other formats fall through to a full decode.
    let scaled = source
        .decode_max_size
        .and_then(|max_size| decode_jpeg_scaled(bytes, max_size));
    let buf = match scaled {
        Some(buf) => buf,
        None => ImageAdapter::decode(bytes)
            .map_err(|e| polars_err!(ComputeError: "Failed to decode image: {:?}", e))?,
    };
    // If source spec declares an expected dtype, cast to it.
    // This is a no-op when the decoded dtype already matches.
    if let Some(ref dtype_str) = source.dtype {
        let target = parse_dtype(dtype_str)?;
        if buf.dtype() != target {
            return Ok(buf.cast(target));
        }
    }
    Ok(buf)
}

/// Encode the result buffer to a binary sink format.
///
/// Handles the byte-producing sinks only: `png`/`jpeg`/`webp`/`tiff`/`blob`.
/// The other sink formats never reach this function — `numpy`/`torch` are
/// encoded as zero-copy structs (`crate::output`) and `list`/`array` as typed
/// nested values, both directly in `graph::encode::encode_node_output` (the
/// sole caller).
pub fn encode_sink(buffer: &ViewBuffer, sink: &SinkSpec) -> PolarsResult<Vec<u8>> {
    if sink.format.as_str() == "blob" {
        // VIEW protocol: self-describing, so no codec precondition applies.
        return Ok(buffer.to_blob());
    }

    let Some(codec) = ImageCodec::from_sink_format(sink.format.as_str()) else {
        return Err(polars_err!(ComputeError: "Unknown sink format: {}", sink.format));
    };

    // The same check the planner ran before publishing this query's schema
    // (`dtype_for_output`). Reaching a failure here means the planner had less
    // information than we do now — a source whose dtype was still "auto", or a
    // shape only the data could settle — not that the two disagree.
    codec
        .check_shape(
            PlannedDType::Known(buffer.dtype()),
            Some(buffer.shape()),
            None,
        )
        .map_err(|msg| polars_err!(ComputeError: "{}", msg))?;

    match codec {
        ImageCodec::Png => ImageAdapter::encode(buffer, image::ImageFormat::Png)
            .map_err(|e| polars_err!(ComputeError: "Failed to encode PNG: {:?}", e)),
        ImageCodec::Jpeg => ImageAdapter::encode_jpeg(buffer, sink.quality)
            .map_err(|e| polars_err!(ComputeError: "Failed to encode JPEG: {:?}", e)),
        ImageCodec::WebP => ImageAdapter::encode(buffer, image::ImageFormat::WebP)
            .map_err(|e| polars_err!(ComputeError: "Failed to encode WebP: {:?}", e)),
        ImageCodec::Tiff => ImageAdapter::encode_tiff(buffer)
            .map_err(|e| polars_err!(ComputeError: "Failed to encode TIFF: {:?}", e)),
    }
}

/// The operations still resolved by name through [`resolve_op_inner`].
///
/// The typed-op migration (`TYPED_OPS_PLAN.md`) moves ops from here into the
/// typed catalogue (`crate::ops::TypedOp`) family by family; the two sets are
/// disjoint and together are exactly the executable ops
/// (`typed_and_legacy_ops_partition_the_op_set`). It must list exactly the
/// top-level match arms in [`resolve_op_inner`]: `known_ops_all_resolve`
/// guards the forward direction and `resolve_op_arms_are_all_known_ops` the
/// reverse, so a migrated op cannot leave its arm behind.
pub const LEGACY_OPS: &[&str] = &["extract_shape", "label_reduce", "rasterize"];

/// Resolve an operation specification to a [`GraphStep`].
///
/// Single-buffer ops become `GraphStep::Buffer(ViewDto)` (executed via the
/// engine's `ViewExpr`); multi-input and domain-changing ops become typed
/// graph-level steps. Node references and expression column names enter the
/// step here — they never reach the engine's `ViewDto`.
///
/// Every parameter on the spec must be read by the arm that handles it. See
/// [`resolve_op_inner`] for why, and [`OpParams`] for how.
pub fn resolve_op(op_spec: &OpSpec, row_idx: usize, ctx: &ParamCtx) -> PolarsResult<GraphStep> {
    let op_spec = match op_spec {
        // Typed: serde has already rejected any field the op does not declare,
        // and the op's `OpDef` destructures every one it does.
        OpSpec::Typed(op) => return op.resolve(row_idx, ctx),
        OpSpec::Legacy(spec) => spec,
    };
    let params = OpParams::new(&op_spec.params);
    let step = resolve_op_inner(&op_spec.op, &params, row_idx, ctx)?;

    // A parameter the arm never looked at is a parameter that did nothing. It
    // still rode in the op's identity — so two pipelines that behave
    // identically hash differently for CSE and compile to separate graph-cache
    // entries — and it read, to whoever passed it, as a request that was
    // honoured. `scale`/`clamp` carried an `out_dtype` like that.
    //
    // This runs at both ends of the boundary with no extra wiring:
    // `resolve_op_from_json` backs the `op_schema` FFI, so a stray parameter
    // fails in Python while the `Pipeline` is being built, and
    // `CompiledGraph::compile` resolves every spec before any row executes.
    let unread = params.unread()?;
    if !unread.is_empty() {
        let mut names: Vec<&str> = unread;
        names.sort_unstable();
        polars_bail!(ComputeError:
            "operation '{}': parameter(s) {:?} are not read by this operation. \
             A parameter that reaches no code path is not a no-op — it enters \
             the op's identity and silently discards what the caller asked for.",
            op_spec.op, names);
    }
    Ok(step)
}

/// The per-operation dispatch behind [`resolve_op`].
///
/// Split out so the parameter-use check has exactly one place to run and the
/// arms have no way to return past it.
///
/// It takes the op *name* and an [`OpParams`], never the `OpSpec` — an arm that
/// could still reach `op_spec.params` could read a parameter without recording
/// it, which is not a tracker so much as a suggestion. Two of `crop`'s reads
/// were exactly that shape before the signature was narrowed, and the compiler
/// is what found them.
fn resolve_op_inner(
    op_name: &str,
    params: &OpParams<'_>,
    row_idx: usize,
    ctx: &ParamCtx,
) -> PolarsResult<GraphStep> {
    match op_name {
        // Geometry operations
        "rasterize" => {
            // `rasterize(shape=<node>)` names another graph node to take the
            // mask dimensions from. `CompiledGraph::compile` resolves it before
            // this arm is reached and substitutes the concrete width/height, so
            // the parameter belongs to the op but is consumed a layer up.
            params.acknowledge("shape_ref");
            let width = get_param(params, "width")?.resolve_usize(row_idx, ctx)? as u32;
            let height = get_param(params, "height")?.resolve_usize(row_idx, ctx)? as u32;
            let (fill_value, background) = resolve_rasterize_style(params, row_idx, ctx)?;
            Ok(GraphStep::Geometry(GeometryOp::Rasterize {
                width,
                height,
                fill_value,
                background,
            }))
        }
        // Reduction operations
        "extract_shape" => {
            // Extract shape returns buffer dimensions as a vector
            Ok(GraphStep::ExtractShape)
        }
        "label_reduce" => {
            // The contour set is an operand column, not a value: the step
            // keeps its input position and reads the whole row's list itself.
            let contours_slot = match get_param(params, "contours")? {
                ParamValue::Slot { idx } => *idx,
                ParamValue::Literal { .. } => {
                    return Err(polars_err!(
                        ComputeError: "label_reduce contours parameter must be a Polars expression"
                    ))
                }
            };
            let reduction = get::opt_enum(
                params,
                "reduction",
                LabelReduction::NAMED,
                &[],
                LabelReduction::Max,
                row_idx,
                ctx,
            )?;
            let region_mode = get::opt_enum(
                params,
                "region_mode",
                LabelRegionMode::NAMED,
                &[],
                LabelRegionMode::Interior,
                row_idx,
                ctx,
            )?;
            Ok(GraphStep::LabelReduce {
                contours_slot,
                reduction,
                region_mode,
            })
        }

        other => Err(polars_err!(ComputeError: "Unknown operation: {}", other)),
    }
}

/// Get a required parameter, recording the read.
fn get_param<'a>(params: &OpParams<'a>, name: &str) -> PolarsResult<&'a ParamValue> {
    params
        .get(name)
        .ok_or_else(|| polars_err!(ComputeError: "Missing required parameter: {}", name))
}

/// Parse a dtype string to DType (canonical short names from `DType::NAMED`).
fn parse_dtype(s: &str) -> PolarsResult<DType> {
    DType::from_short_name(s).ok_or_else(|| {
        polars_err!(ComputeError:
            "Unknown dtype: {}, expected one of {:?}", s, naming::names(DType::NAMED))
    })
}

#[cfg(test)]
mod strict_param_tests {
    //! One failure policy for operation parameters: an *absent* optional
    //! parameter takes its documented default, but a parameter that is
    //! *present and invalid* (unknown enum string, wrong type, out of range)
    //! must be an error — never silently coerced to a default. These tests
    //! pin that policy for every parameter that historically swallowed
    //! errors.
    //!
    //! A **null** per-row value is a separate axis and is deliberately not
    //! covered here: it is neither absent nor invalid, and what it means is
    //! chosen by `NullParamPolicy` (`params.rs`) — raise, or null the affected
    //! rows. It is never coerced to a default under either policy, so the rule
    //! above still holds; see `params::tests::test_null_policy_*`.

    use super::*;
    use crate::params::ParamValue;
    use serde_json::json;
    use std::collections::HashMap;

    fn op_with(name: &str, params: &[(&str, serde_json::Value)]) -> OpSpec {
        OpSpec::Legacy(crate::pipeline::LegacyOpSpec {
            op: name.to_string(),
            params: params
                .iter()
                .map(|(k, v)| (k.to_string(), ParamValue::Literal { value: v.clone() }))
                .collect::<HashMap<_, _>>(),
        })
    }

    fn resolve_err(spec: &OpSpec) -> String {
        resolve_op(spec, 0, &ParamCtx::empty())
            .expect_err("invalid parameter must be rejected")
            .to_string()
    }

    #[test]
    fn rasterize_invalid_fill_value_errors() {
        let base = [("width", json!(8)), ("height", json!(8))];
        let mut params = base.to_vec();
        params.push(("fill_value", json!("red")));
        let err = resolve_err(&op_with("rasterize", &params));
        assert!(err.contains("fill_value"), "{err}");

        let mut params = base.to_vec();
        params.push(("fill_value", json!(300)));
        let err = resolve_err(&op_with("rasterize", &params));
        assert!(
            err.contains("fill_value"),
            "out-of-range u8 must error: {err}"
        );

        let mut params = base.to_vec();
        params.push(("background", json!(-1)));
        let err = resolve_err(&op_with("rasterize", &params));
        assert!(err.contains("background"), "{err}");
    }
}

#[cfg(test)]
mod known_ops_tests {
    use super::*;
    use std::collections::HashMap;

    /// Build an OpSpec with no params (enough to exercise the name dispatch).
    fn op(name: &str) -> OpSpec {
        OpSpec::Legacy(crate::pipeline::LegacyOpSpec {
            op: name.to_string(),
            params: HashMap::new(),
        })
    }

    /// Every name in LEGACY_OPS must be a real resolve_op arm: with empty params
    /// most arms fail with a missing-param error, but none may fall through to
    /// the "Unknown operation" catch-all.
    #[test]
    fn known_ops_all_resolve() {
        let ctx = ParamCtx::empty();
        for name in LEGACY_OPS {
            if let Err(e) = resolve_op(&op(name), 0, &ctx) {
                let msg = e.to_string();
                assert!(
                    !msg.contains("Unknown operation"),
                    "LEGACY_OPS lists '{name}' but resolve_op has no arm for it: {msg}"
                );
            }
        }
    }

    /// A name that is not an arm must be rejected by the catch-all, so the
    /// registry can't silently accept bogus ops.
    #[test]
    fn unknown_op_is_rejected() {
        let ctx = ParamCtx::empty();
        let err = resolve_op(&op("definitely_not_a_real_op"), 0, &ctx)
            .expect_err("bogus op must not resolve");
        assert!(err.to_string().contains("Unknown operation"));
    }

    /// Reverse guard: every top-level match arm in `resolve_op_inner` must be
    /// listed in LEGACY_OPS, so a new arm cannot silently bypass the registry
    /// (the forward direction is covered by `known_ops_all_resolve`).
    ///
    /// The scan reads this file's source between the `resolve_op_inner` header
    /// and its "Unknown operation" catch-all. That is where the arms live —
    /// `resolve_op` is the wrapper that runs the parameter-use check around
    /// them, and anchoring here rather than on it keeps the scanned region to
    /// the match itself. Top-level arm patterns sit at one match-nesting level
    /// (8-space indent under rustfmt, which CI enforces); deeper string arms
    /// (e.g. normalize's method match) are excluded by the indent check.
    #[test]
    fn resolve_op_arms_are_all_known_ops() {
        let src = include_str!("execute.rs");
        let start = src
            .find("fn resolve_op_inner")
            .expect("resolve_op_inner not found");
        let end = start
            + src[start..]
                .find("Unknown operation")
                .expect("resolve_op catch-all not found");
        let mut arm_names: Vec<&str> = Vec::new();
        let mut guard_arms: Vec<&str> = Vec::new();
        for line in src[start..end].lines() {
            let trimmed = line.trim_start();
            let indent = line.len() - trimmed.len();
            if indent != 8 {
                continue;
            }
            // Continuation lines of a wrapped arm, and the closing brace of a
            // block-bodied one, carry no pattern. Everything else at this
            // indent starts an arm and must be classified.
            if trimmed.is_empty()
                || trimmed.starts_with("//")
                || trimmed.starts_with('}')
                || trimmed.starts_with("=>")
                || trimmed.starts_with("&&")
                || trimmed.starts_with("||")
                || trimmed.starts_with('|')
            {
                continue;
            }
            if trimmed.starts_with('"') {
                // Fall through to the string-literal handling below.
            } else {
                // Anything else registers ops without naming them in a form
                // this scan can read: a guard arm, an `@` binding, a bare
                // binder. Record the whole pattern and require it to be
                // explicitly known below.
                //
                // The previous version only recognised a guard arm when the
                // pattern and its `=>` shared a line, and silently skipped
                // every other shape. rustfmt moves `=>` to the next line once
                // the condition is long enough, and an `@` binding never had
                // one -- both let a whole op family become executable with no
                // LEGACY_OPS entry. "Anything I do not recognise is ignored"
                // was the bug; "anything I do not recognise fails" is the
                // guard.
                let pattern = trimmed
                    .split("=>")
                    .next()
                    .unwrap_or(trimmed)
                    .trim()
                    .trim_end_matches('{')
                    .trim();
                guard_arms.push(pattern);
                continue;
            }
            // Arm patterns look like `"name" => {` or `"a" | "b" => ...`;
            // collect every string literal before the `=>`.
            let pattern = trimmed.split("=>").next().unwrap_or(trimmed);
            for (i, piece) in pattern.split('"').enumerate() {
                if i % 2 == 1 {
                    arm_names.push(piece);
                }
            }
        }
        // Every op in LEGACY_OPS is a string arm found above, so the scan
        // cannot rot to a subset without this failing. A count floor was used
        // here before; it was both too weak (10 arms could drop out of indent
        // 8 unnoticed) and too brittle (deprecating an op tripped it), so the
        // relationship is pinned instead of a magic number.
        let unaccounted: Vec<&&str> = LEGACY_OPS
            .iter()
            .filter(|n| !arm_names.contains(n))
            .collect();
        assert!(
            unaccounted.is_empty(),
            "these LEGACY_OPS have no string arm: {unaccounted:?} — either \
             resolve_op changed shape or the source scan has rotted"
        );
        // A guard arm registers ops without naming them. The one that did —
        // the binary family, dispatched through `BinaryOp::NAMED` — is typed
        // now, so the only non-string arm left is the catch-all that produces
        // the "Unknown operation" error this scan terminates on.
        for arm in &guard_arms {
            assert_eq!(
                *arm, "other",
                "resolve_op has a guard arm '{arm}' that registers ops without \
                 naming them; list them as string arms instead"
            );
        }
        for name in &arm_names {
            assert!(
                LEGACY_OPS.contains(name),
                "resolve_op has an arm for '{name}' that is missing from LEGACY_OPS"
            );
        }
    }

    /// LEGACY_OPS must be sorted and unique so the registry is easy to scan and
    /// diff against the Python OP_NAMES set.
    #[test]
    fn known_ops_sorted_and_unique() {
        for pair in LEGACY_OPS.windows(2) {
            assert!(
                pair[0] < pair[1],
                "LEGACY_OPS must be sorted/unique; '{}' !< '{}'",
                pair[0],
                pair[1]
            );
        }
    }
}

#[cfg(test)]
mod unread_param_tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashMap;

    /// Build an `OpSpec` from a name and literal params.
    fn op_with(name: &str, params: &[(&str, serde_json::Value)]) -> OpSpec {
        OpSpec::Legacy(crate::pipeline::LegacyOpSpec {
            op: name.to_string(),
            params: params
                .iter()
                .map(|(k, v)| (k.to_string(), ParamValue::Literal { value: v.clone() }))
                .collect::<HashMap<_, _>>(),
        })
    }

    /// `(op name, base params, the parameter under test)`.
    type UnreadCase<'a> = (&'a str, &'a [(&'a str, serde_json::Value)], &'a str);

    /// `(op name, params that must all be consumed)`.
    type AcceptedCase<'a> = (&'a str, &'a [(&'a str, serde_json::Value)]);

    fn resolve_err(spec: &OpSpec) -> String {
        resolve_op(spec, 0, &ParamCtx::empty())
            .expect_err("expected resolve_op to reject the spec")
            .to_string()
    }

    /// The known-bad half: a parameter no arm reads must be rejected, by name.
    ///
    /// `scale`/`clamp`, the two ops that actually shipped this (an accepted,
    /// unread `out_dtype`), are typed now and refuse the key at
    /// deserialization (`ops::tests::scale_and_clamp_refuse_an_out_dtype`).
    /// This keeps the legacy tracker honest until its last op migrates.
    #[test]
    fn a_parameter_no_arm_reads_is_rejected() {
        let cases: &[UnreadCase<'_>] = &[("extract_shape", &[], "sigma")];
        for (op, base, stray) in cases {
            let mut params = base.to_vec();
            params.push((stray, json!("u8")));
            let err = resolve_err(&op_with(op, &params));
            assert!(
                err.contains(stray) && err.contains(op),
                "{op}: error must name the operation and the unread parameter \
                 '{stray}', got: {err}"
            );
        }
    }

    /// The known-good half: a checker that rejects everything proves nothing.
    ///
    /// These specs carry parameters read through *helpers* rather than a
    /// literal `get_param` call in the arm (`resolve_rasterize_style`) plus `rasterize`'s
    /// `shape_ref`, which a layer above the arm consumes. All must resolve.
    #[test]
    fn parameters_read_through_helpers_are_accepted() {
        let cases: &[AcceptedCase<'_>] = &[(
            "rasterize",
            &[
                ("width", json!(8)),
                ("height", json!(8)),
                ("fill_value", json!(255)),
                ("background", json!(0)),
                ("shape_ref", json!("other_node")),
            ],
        )];
        for (op, params) in cases {
            let spec = op_with(op, params);
            assert!(
                resolve_op(&spec, 0, &ParamCtx::empty()).is_ok(),
                "{op}: every parameter here is consumed, so it must resolve; \
                 got: {}",
                resolve_err(&spec)
            );
        }
    }
}
