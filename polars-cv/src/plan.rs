//! The plan-time effect of appending one op: the Python planner's one call.
//!
//! [`plan_step`] takes the pipeline's tracked state and one serialized op and
//! returns the state after it — the input-domain check, the schema fold
//! (domain, dtype, rank), the H/W the op's `infer_shape` gives, its channel
//! rule, and the clipping of every hint to the output rank. These used to be
//! four FFI calls sequenced by eight Python helpers, any of which a caller
//! could skip; one call cannot be half-applied.
//!
//! A two-input op's dtype depends on both operands, so a binary op takes the
//! other operand's dtype, and only a binary op may: passing one to any other
//! op, or omitting it for a binary op, is an error rather than a fallback to
//! the one-input rule.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::py_value_error;
use view_buffer::ops::{Domain, HistogramOutput, OutputRankRule};

use crate::graph::step::GraphStep;

/// The tracked plan-time state an op is appended to.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct State {
    pub domain: String,
    pub dtype: String,
    pub ndim: Option<usize>,
    /// Known H/W/C sizes; `None` is unknown (or per-row, which is unknown at
    /// plan time).
    pub dims: [Option<i64>; 3],
}

/// The state after one op. A hint is replaced where `dims[i]` is `Some`, and
/// left as it was where it is `None`: an op whose input rank is unknown says
/// nothing about H/W, which may hold a user's per-row assertion.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Step {
    pub domain: String,
    pub dtype: String,
    pub ndim: Option<usize>,
    pub dims: [Option<Option<i64>>; 3],
}

/// Apply one op to `state`. See the module docs.
pub(crate) fn step(
    op_json: &str,
    state: &State,
    other_dtype: Option<&str>,
) -> Result<Step, String> {
    let op: crate::ops::TypedOp = serde_json::from_str(op_json).map_err(|e| e.to_string())?;
    let step = crate::resolve_op_from_json(op_json)?;

    // Input domain, from the step's own contract.
    let current = view_buffer::naming::lookup(Domain::NAMED, &state.domain)
        .ok_or_else(|| format!("unknown domain {:?}", state.domain))?;
    let accepted = step.input_domains();
    if !accepted.iter().any(|d| *d == Domain::Any || *d == current) {
        let expected: Vec<&str> = accepted.iter().map(|d| d.name()).collect();
        return Err(format!(
            "{}() expects {} input but pipeline is currently in {} domain. Add a \
             domain-converting operation (e.g., rasterize() for contour→buffer, \
             extract_contours() for buffer→contour).",
            op.name(),
            expected.join(" or "),
            state.domain
        ));
    }

    let (out_domain, ndim) = fold(&step, &state.domain, state.ndim);
    let dtype = match (&step, other_dtype) {
        (GraphStep::Binary { op, .. }, Some(other)) => binary_dtype(*op, &state.dtype, other)?,
        (GraphStep::Binary { .. }, None) => {
            return Err(format!(
                "{}() combines two operands: its dtype needs the other operand's",
                op.name()
            ))
        }
        (_, Some(_)) => {
            return Err(format!(
                "{}() has one operand; only a binary op takes another's dtype",
                op.name()
            ))
        }
        (_, None) => single_input_dtype(&step, &state.dtype)?,
    };

    // H/W: from the op's `infer_shape` over the shape it consumes.
    let mut dims: [Option<Option<i64>>; 3] = [None; 3];
    if let Some(input) = input_dims(&step, state) {
        match crate::infer_shape(op_json, &input)? {
            // No inferable shape (an axis reduction, a histogram, a binary
            // op): unknown, never the stale pre-op values.
            None => {
                dims[0] = Some(None);
                dims[1] = Some(None);
            }
            Some(out) => {
                // A negative dim is "the unknown input axis, unchanged".
                let size = |i: usize| out.get(i).copied().flatten().filter(|d| *d >= 0);
                dims[0] = Some(size(0));
                dims[1] = Some(size(1));
            }
        }
    }
    // Channels: the op's channel rule over the incoming count.
    let channels_in = state.dims[2].and_then(|c| usize::try_from(c).ok());
    dims[2] = Some(
        step.output_channel_rule()
            .apply(channels_in)
            .map(|c| c as i64),
    );
    // A dimension the output rank does not have has no size.
    if let Some(n) = ndim {
        for (axis, dim) in dims.iter_mut().enumerate() {
            if axis >= n {
                *dim = Some(None);
            }
        }
    }

    Ok(Step {
        domain: out_domain,
        dtype,
        ndim,
        dims,
    })
}

/// An op's output domain and rank over the incoming ones.
fn fold(step: &GraphStep, domain: &str, ndim: Option<usize>) -> (String, Option<usize>) {
    let out_domain = match step.output_domain() {
        Domain::Any => domain.to_string(),
        d => d.name().to_string(),
    };
    let ndim = match step.output_rank_rule() {
        OutputRankRule::Fixed(n) => Some(n),
        OutputRankRule::PreserveRank => ndim,
        OutputRankRule::ReduceByOne => ndim.map(|n| n.saturating_sub(1).max(1)),
        OutputRankRule::Unknown => None,
    };
    // Scalar and vector domains pin the rank whatever the rule says.
    let ndim = match out_domain.as_str() {
        "scalar" => Some(0),
        "vector" => Some(1),
        _ => ndim,
    };
    (out_domain, ndim)
}

/// An op's output dtype from its own one-input rule.
///
/// Histogram buckets are struct-encoded by the sink, so their element dtype is
/// an encoding concern, not a schema one: `"auto"`.
fn single_input_dtype(step: &GraphStep, dtype: &str) -> Result<String, String> {
    match step {
        GraphStep::Histogram(h) if h.output == HistogramOutput::Buckets => Ok("auto".to_string()),
        _ => crate::output_dtype_for(step, dtype),
    }
}

/// The input shape to hand `infer_shape`, or `None` to not ask.
///
/// Unknown input rank normally means "do not ask" — `infer_shape` indexes its
/// input, so a fabricated shape would publish a fabricated result. A step that
/// *builds* a buffer from another domain (`rasterize`) is the exception: it
/// consumes no buffer, so there is no input shape to be unknown about.
fn input_dims(step: &GraphStep, state: &State) -> Option<Vec<Option<i64>>> {
    match state.ndim {
        Some(n) if n >= 1 => Some(
            (0..n)
                .map(|i| state.dims.get(i).copied().flatten())
                .collect(),
        ),
        _ if !step.input_domains().contains(&Domain::Buffer)
            && step.output_domain() == Domain::Buffer =>
        {
            Some(Vec::new())
        }
        _ => None,
    }
}

/// A binary op's output dtype over both operands; `"auto"` if either is.
fn binary_dtype(op: view_buffer::BinaryOp, left: &str, right: &str) -> Result<String, String> {
    if left == "auto" || right == "auto" {
        return Ok("auto".to_string());
    }
    let dtype = op.output_dtype(crate::parse_dtype(left)?, crate::parse_dtype(right)?);
    Ok(dtype.short_name().to_string())
}

/// Python entry point for [`step`].
///
/// Returns `{"domain", "dtype", "ndim", "dims"}`, where `dims` lists
/// `(axis, size)` for exactly the hints the op replaces (`size` `None` for
/// unknown); Python names the axes (`HINT_DIMS`).
#[pyfunction]
#[pyo3(signature = (op_json, domain, dtype, ndim, dims, other_dtype=None))]
pub(crate) fn plan_step<'py>(
    py: Python<'py>,
    op_json: &str,
    domain: String,
    dtype: String,
    ndim: Option<usize>,
    dims: [Option<i64>; 3],
    other_dtype: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let state = State {
        domain,
        dtype,
        ndim,
        dims,
    };
    let out = step(op_json, &state, other_dtype).map_err(py_value_error)?;
    let result = PyDict::new(py);
    result.set_item("domain", out.domain)?;
    result.set_item("dtype", out.dtype)?;
    result.set_item("ndim", out.ndim)?;
    let replaced: Vec<(usize, Option<i64>)> = out
        .dims
        .iter()
        .enumerate()
        .filter_map(|(axis, dim)| dim.map(|size| (axis, size)))
        .collect();
    result.set_item("dims", replaced)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn state(domain: &str, dtype: &str, ndim: Option<usize>, dims: [Option<i64>; 3]) -> State {
        State {
            domain: domain.into(),
            dtype: dtype.into(),
            ndim,
            dims,
        }
    }

    fn image() -> State {
        state("buffer", "u8", Some(3), [Some(100), Some(50), Some(3)])
    }

    fn run(op: serde_json::Value, s: &State, other: Option<&str>) -> Result<Step, String> {
        step(&op.to_string(), s, other)
    }

    #[test]
    fn a_resize_replaces_hw_and_keeps_channels() {
        let out = run(
            json!({"op": "resize", "height": 4, "width": 6, "filter": "bilinear"}),
            &image(),
            None,
        )
        .unwrap();
        assert_eq!(out.dims, [Some(Some(4)), Some(Some(6)), Some(Some(3))]);
        assert_eq!(
            (out.domain.as_str(), out.dtype.as_str(), out.ndim),
            ("buffer", "u8", Some(3))
        );
    }

    #[test]
    fn an_unknown_input_rank_leaves_hw_untouched() {
        let s = state("buffer", "u8", None, [None, None, None]);
        let out = run(json!({"op": "grayscale"}), &s, None).unwrap();
        assert_eq!(out.dims[..2], [None, None], "H/W must be kept, not cleared");
    }

    #[test]
    fn hints_beyond_the_output_rank_are_cleared() {
        let out = run(json!({"op": "channel_select", "index": 0}), &image(), None).unwrap();
        assert_eq!(out.ndim, Some(2));
        assert_eq!(out.dims[2], Some(None));
    }

    #[test]
    fn a_wrong_input_domain_is_refused_naming_the_op() {
        let s = state("contour", "f64", None, [None; 3]);
        let err = run(json!({"op": "grayscale"}), &s, None).unwrap_err();
        assert!(err.contains("grayscale() expects buffer input"), "{err}");
    }

    #[test]
    fn a_binary_op_needs_and_uses_the_other_dtype() {
        let add = json!({"op": "divide", "other": "n0"});
        assert!(run(add.clone(), &image(), None).is_err());
        assert_eq!(run(add, &image(), Some("u8")).unwrap().dtype, "f32");
        let err = run(json!({"op": "grayscale"}), &image(), Some("u8")).unwrap_err();
        assert!(err.contains("only a binary op"), "{err}");
    }

    #[test]
    fn a_binary_op_keeps_a_vector_operand_a_vector() {
        let hash = state("vector", "u8", Some(1), [Some(8), None, None]);
        let out = run(
            json!({"op": "bitwise_xor", "other": "n0"}),
            &hash,
            Some("u8"),
        )
        .unwrap();
        assert_eq!((out.domain.as_str(), out.ndim), ("vector", Some(1)));
    }

    #[test]
    fn rasterize_builds_its_canvas_from_its_own_size() {
        let s = state("contour", "f64", None, [None; 3]);
        let out = run(
            json!({"op": "rasterize", "size": [8, 6], "fill_value": 255, "background": 0}),
            &s,
            None,
        )
        .unwrap();
        assert_eq!(out.domain, "buffer");
        assert_eq!(out.dims, [Some(Some(8)), Some(Some(6)), Some(Some(1))]);
    }
}
