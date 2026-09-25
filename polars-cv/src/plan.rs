//! The plan-time effect of appending one op: the Python planner's one call.
//!
//! [`plan_step`] takes the pipeline's tracked state and one serialized op and
//! returns the state after it — the input-domain check, the schema fold
//! (domain, dtype, rank), the H/W the op's symbolic `shape` gives, its channel
//! rule, and the clipping of every hint to the output rank. These used to be
//! four FFI calls sequenced by eight Python helpers, any of which a caller
//! could skip; one call cannot be half-applied.
//!
//! A two-input op plans over both operands, so a binary op takes the other
//! operand's state, and only a binary op may: passing one to any other op, or
//! omitting it for a binary op, is an error rather than a fallback to the
//! one-input rule.

use pyo3::prelude::*;

use crate::py_value_error;
use view_buffer::ops::{Dim, Domain, HistogramOutput, OutputRankRule};
use view_buffer::PlannedDType;

use crate::graph::step::GraphStep;

/// The planner's state at one op boundary — the one representation of it.
///
/// Typed throughout: a [`Domain`], a [`PlannedDType`] (the dtype lattice the
/// execution side resolves with too), and sizes. Python holds these objects
/// as they are (`polars_cv._lib.PlanState`) and reads the wire spellings
/// through its getters; it never builds or edits one. Nothing about it crosses
/// the graph wire: the graph loader plans the graph itself
/// ([`resolved_output_specs`](crate::graph::resolved_output_specs)).
#[pyclass(frozen, from_py_object, module = "polars_cv._lib", name = "PlanState")]
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct State {
    #[serde(deserialize_with = "wire_domain")]
    pub domain: Domain,
    #[serde(deserialize_with = "wire_dtype")]
    pub dtype: PlannedDType,
    #[serde(default)]
    pub ndim: Option<usize>,
    /// Known sizes of dimensions 0..3 (`[H, W, C]` for an image); `None` is
    /// unknown (a per-row size is unknown at plan time).
    #[serde(default)]
    pub dims: [Option<usize>; 3],
}

fn wire_domain<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Domain, D::Error> {
    crate::ops::param::literal_field(d)
}

fn wire_dtype<'de, D: serde::Deserializer<'de>>(d: D) -> Result<PlannedDType, D::Error> {
    use serde::de::Error;
    let name = <String as serde::Deserialize>::deserialize(d)?;
    PlannedDType::parse(&name).ok_or_else(|| D::Error::custom(format!("unknown dtype {name:?}")))
}

#[pymethods]
impl State {
    /// The state of a pipeline with no source yet: a buffer, nothing known.
    #[new]
    fn unsourced() -> Self {
        State::new(Domain::Buffer, PlannedDType::Unknown, None)
    }

    /// The names `assert_shape` gives dimensions 0, 1 and 2.
    #[classattr]
    #[allow(non_snake_case)]
    fn DIM_NAMES() -> (&'static str, &'static str, &'static str) {
        let [h, w, c] = DIM_NAMES;
        (h, w, c)
    }

    /// ``buffer``, ``contour``, ``scalar`` or ``vector``.
    #[getter(domain)]
    fn py_domain(&self) -> &'static str {
        self.domain.name()
    }

    /// The element dtype (``u8``, ``f32``, …), ``auto_float`` when only known
    /// to be a float, or ``auto`` when not known until decode.
    #[getter(dtype)]
    fn py_dtype(&self) -> &'static str {
        self.dtype.as_str()
    }

    /// The rank, or ``None`` when not known at plan time.
    #[getter(ndim)]
    fn py_ndim(&self) -> Option<usize> {
        self.ndim
    }

    /// Known sizes of dimensions 0, 1, 2; ``None`` is unknown or per-row.
    #[getter(dims)]
    fn py_dims(&self) -> (Option<usize>, Option<usize>, Option<usize>) {
        let [h, w, c] = self.dims;
        (h, w, c)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }

    fn __repr__(&self) -> String {
        format!(
            "PlanState(domain={:?}, dtype={:?}, ndim={:?}, dims={:?})",
            self.domain.name(),
            self.dtype.as_str(),
            self.ndim,
            self.dims,
        )
    }

    fn __copy__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __deepcopy__(slf: Py<Self>, _memo: &Bound<'_, PyAny>) -> Py<Self> {
        slf
    }

    /// The pickled form (JSON).
    fn _wire(&self) -> String {
        serde_json::json!({
            "domain": self.domain.name(),
            "dtype": self.dtype.as_str(),
            "ndim": self.ndim,
            "dims": self.dims,
        })
        .to_string()
    }

    /// Pickled by value, through the wire form.
    fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<(Py<PyAny>, (String,))> {
        let restore = slf
            .py()
            .import("polars_cv._lib")?
            .getattr("_plan_state_from_json")?;
        Ok((restore.unbind(), (slf.get()._wire(),)))
    }
}

/// Unpickle a [`State`] (see `State::__reduce__`).
#[pyfunction]
pub(crate) fn _plan_state_from_json(wire: &str) -> PyResult<State> {
    serde_json::from_str(wire).map_err(|e| py_value_error(e.to_string()))
}

/// The name of dimension `axis` in the `[H, W, C]` spelling `assert_shape`'s
/// keywords use.
pub(crate) const DIM_NAMES: [&str; 3] = ["height", "width", "channels"];

/// The planned states of the other graph nodes an op may read by id (a binary
/// op's `other`, a canvas taken from another node). The builder passes the
/// referenced lazy expressions' states; the graph loader passes every node
/// planned so far. An op naming a node with no planned state is refused.
pub(crate) type Refs = std::collections::HashMap<String, State>;

fn referenced<'a>(refs: &'a Refs, op: &str, node: &str) -> Result<&'a State, String> {
    refs.get(node)
        .ok_or_else(|| format!("{op}() reads node '{node}', which has no planned state"))
}

/// Apply an `assert_shape` declaration to `state`, refusing one the state
/// contradicts: a rank that is already known differently, a dimension the
/// rank does not have, or a size that disagrees with a known one. A per-row
/// size is declared but is no plan-time fact.
fn declare(mut state: State, op: &crate::ops::declare::AssertShape) -> Result<State, String> {
    use crate::ops::Param;

    if let Some(rank) = op.rank.map(|r| r.get() as usize) {
        if !(1..=DIM_NAMES.len()).contains(&rank) {
            return Err(format!(
                "assert_shape(dims=...) supports 1 to {} dimensions ({}), got {rank}. \
                 Higher-rank shapes are not tracked by the planner; pass the shape to the \
                 sink instead (.sink('array', shape=[...])).",
                DIM_NAMES.len(),
                DIM_NAMES.join(", ")
            ));
        }
        if let Some(current) = state.ndim.filter(|c| *c != rank) {
            return Err(format!(
                "assert_shape(dims=...) declares a rank-{rank} output, but this pipeline is \
                 already known to produce rank {current}. Drop the assertion, or correct its \
                 length."
            ));
        }
        state.ndim = Some(rank);
    }
    for (axis, declared) in op.dims.iter().enumerate() {
        let Some(declared) = declared else {
            continue;
        };
        let name = DIM_NAMES[axis];
        if let Some(ndim) = state.ndim.filter(|n| axis >= *n) {
            return Err(format!(
                "assert_shape({name}=...) names dimension {axis}, which a rank-{ndim} output \
                 does not have. The shape hints are positional — {} are dimensions 0, 1 and \
                 2 — so use assert_shape(dims=[...]) for anything that is not an [H, W, C] \
                 image.",
                DIM_NAMES.join(", ")
            ));
        }
        let size = match declared {
            Param::Lit(0) => {
                return Err(format!(
                    "assert_shape({name}=0): each size must be a positive int or None"
                ))
            }
            Param::Lit(size) => Some(*size as usize),
            Param::Slot(_) => None,
        };
        if let (Some(known), Some(size)) = (state.dims[axis], size) {
            if known != size {
                return Err(format!(
                    "assert_shape({name}={size}) contradicts the {name} {known} the pipeline \
                     already establishes at this point. An assertion cannot change what the \
                     data is — remove it, or fix the value."
                ));
            }
        }
        state.dims[axis] = size;
    }
    Ok(state)
}

/// The state after appending `op` to `state`. See the module docs.
pub(crate) fn step(op: &crate::ops::TypedOp, state: &State, refs: &Refs) -> Result<State, String> {
    use crate::ops::geometry::RasterSize;
    use crate::ops::{NodeRef, TypedOp};

    let step = crate::planning_step(op)?;

    // Input domain, from the step's own contract.
    let accepted = step.input_domains();
    if !accepted.contains(&state.domain) {
        let expected: Vec<&str> = accepted.iter().map(|d| d.name()).collect();
        return Err(format!(
            "{}() expects {} input but pipeline is currently in {} domain. Add a \
             domain-converting operation (e.g., rasterize() for contour→buffer, \
             extract_contours() for buffer→contour).",
            op.name(),
            expected.join(" or "),
            state.domain.name()
        ));
    }
    // A declaration's whole effect is on the plan.
    if let TypedOp::AssertShape(declared) = op {
        return declare(state.clone(), declared);
    }

    let (out_domain, ndim) = fold(&step, state.domain, state.ndim);
    let other = match &step {
        GraphStep::Binary { other, .. } => Some(referenced(refs, op.name(), other)?),
        _ => None,
    };
    let dtype = match (&step, other) {
        (GraphStep::Binary { op, .. }, Some(other)) => binary_dtype(*op, state.dtype, other.dtype),
        _ => single_input_dtype(&step, state.dtype),
    };

    // H/W: the op's own shape over the shapes it consumes, symbolically — a
    // per-row parameter or an unknown input size leaves its axis unknown.
    // `None` leaves a size as it was: an op whose input rank is unknown says
    // nothing about H/W.
    let mut dims: [Option<Option<usize>>; 3] = [None; 3];
    if let Some(input) = input_dims(&step, state) {
        check_rank(&step, &input)?;
        let other_input = other.and_then(|o| input_dims(&step, o));
        let inputs: Vec<&[Dim]> = std::iter::once(input.as_slice())
            .chain(other_input.as_deref())
            .collect();
        // A binary op whose other operand has no known rank has no shape to
        // broadcast against: unknown, never the left operand's shape alone.
        let out = match (other, &other_input) {
            (Some(_), None) => None,
            _ => op.shape().and_then(|shape| shape.dims(&inputs)),
        };
        let size = |axis: usize| {
            let dim = out.as_ref().and_then(|out| out.get(axis).copied());
            dim.and_then(Dim::known)
        };
        dims[0] = Some(size(0));
        dims[1] = Some(size(1));
    }
    // A canvas taken from another node has that node's planned H/W.
    if let TypedOp::Rasterize(r) = op {
        if let RasterSize::FromNode(NodeRef(node)) = &r.size {
            let canvas = referenced(refs, op.name(), node)?;
            dims[0] = Some(canvas.dims[0]);
            dims[1] = Some(canvas.dims[1]);
        }
    }
    // Channels: the op's channel rule over the incoming count.
    dims[2] = Some(step.output_channel_rule().apply(state.dims[2]));
    // A dimension the output rank does not have has no size.
    if let Some(n) = ndim {
        for (axis, dim) in dims.iter_mut().enumerate() {
            if axis >= n {
                *dim = Some(None);
            }
        }
    }

    let mut next = state.clone();
    next.domain = out_domain;
    next.dtype = dtype;
    next.ndim = ndim;
    for (size, replaced) in next.dims.iter_mut().zip(dims) {
        if let Some(replaced) = replaced {
            *size = replaced;
        }
    }
    Ok(next)
}

/// An op's output domain and rank over the incoming ones.
fn fold(step: &GraphStep, domain: Domain, ndim: Option<usize>) -> (Domain, Option<usize>) {
    let out_domain = step.output_domain(domain);
    let ndim = match step.output_rank_rule() {
        OutputRankRule::Fixed(n) => Some(n),
        OutputRankRule::PreserveRank => ndim,
        OutputRankRule::ReduceByOne => ndim.map(|n| n.saturating_sub(1).max(1)),
        OutputRankRule::Unknown => None,
    };
    // Scalar and vector domains pin the rank whatever the rule says.
    let ndim = match out_domain {
        Domain::Scalar => Some(0),
        Domain::Vector => Some(1),
        Domain::Buffer | Domain::Contour => ndim,
    };
    (out_domain, ndim)
}

/// An op's output dtype from its own one-input rule, over the planned input
/// dtype — the one dtype lattice ([`OutputDTypeRule::resolve_planned`]).
///
/// Histogram buckets are struct-encoded by the sink, so their element dtype is
/// an encoding concern, not a schema one: unknown.
///
/// [`OutputDTypeRule::resolve_planned`]: view_buffer::OutputDTypeRule::resolve_planned
fn single_input_dtype(step: &GraphStep, dtype: PlannedDType) -> PlannedDType {
    match step {
        GraphStep::Histogram(h) if h.output == HistogramOutput::Buckets => PlannedDType::Unknown,
        _ => step.output_dtype_rule().resolve_planned(dtype),
    }
}

impl State {
    /// A fresh state: nothing known about the sizes.
    pub(crate) fn new(domain: Domain, dtype: PlannedDType, ndim: Option<usize>) -> State {
        State {
            domain,
            dtype,
            ndim,
            dims: [None; 3],
        }
    }
}

/// The state a source hands the first op.
///
/// Exhaustive over the formats: what each decodes to is a fact about the
/// format. A contour source decodes by rasterizing, so its state is the
/// `rasterize` op's over the contour domain — the source and the op cannot
/// publish different masks.
pub(crate) fn source_state(
    source: &crate::formats::source::Source,
    refs: &Refs,
) -> Result<State, String> {
    use crate::formats::source::Source;

    let buffer = |dtype: Option<view_buffer::DType>, ndim: Option<usize>| {
        let dtype = dtype.map_or(PlannedDType::Unknown, PlannedDType::Known);
        State::new(Domain::Buffer, dtype, ndim)
    };
    let dtype = source.dtype();
    Ok(match source {
        // Raw bytes decode to a flat 1-D buffer of the declared dtype.
        Source::Raw(_) => buffer(dtype, Some(1)),
        // Decoded images are always `[H, W, C]`; the dtype is the caller's
        // assertion or unknown until decode (PNG u8, 16-bit PNG u16, TIFF ...).
        Source::ImageBytes(_) | Source::FilePath(_) => buffer(dtype, Some(3)),
        // Rank follows the column (nesting depth, blob header, or the path the
        // column's dtype routes to), known only with the input.
        Source::Auto(_) | Source::Blob(_) | Source::List(_) | Source::Array(_) => {
            buffer(dtype, None)
        }
        Source::Contour(s) => {
            let rasterize = crate::ops::TypedOp::Rasterize(crate::ops::geometry::Rasterize {
                size: s.size.clone(),
                fill_value: s.fill_value.unwrap_or(crate::ops::Param::Lit(255)),
                background: s.background.unwrap_or(crate::ops::Param::Lit(0)),
            });
            let contours = State::new(Domain::Contour, PlannedDType::Unknown, None);
            step(&rasterize, &contours, refs)?
        }
    })
}

/// Python entry point for [`source_state`]: validate a serialized source
/// against its typed format, and return the state it starts a pipeline in.
/// `refs` are the states of the nodes it reads (a contour canvas's).
#[pyfunction]
#[pyo3(signature = (source_json, refs=None))]
pub(crate) fn plan_source(source_json: &str, refs: Option<Refs>) -> PyResult<State> {
    let source: crate::formats::source::Source =
        serde_json::from_str(source_json).map_err(|e| py_value_error(e.to_string()))?;
    source_state(&source, &refs.unwrap_or_default()).map_err(py_value_error)
}

/// The shape an op consumes, symbolically: each known size, and `Input(k)`
/// for an unknown one. `None` when the rank is unknown, so there is no shape
/// to reason about — except for a step that *builds* a buffer from another
/// domain (`rasterize`), which consumes no buffer at all.
pub(crate) fn input_dims(step: &GraphStep, state: &State) -> Option<Vec<Dim>> {
    match state.ndim {
        Some(n) if n >= 1 => Some(
            (0..n)
                .map(|axis| match state.dims.get(axis).copied().flatten() {
                    Some(size) => Dim::Known(size),
                    None => Dim::Input(axis),
                })
                .collect(),
        ),
        _ if !step.input_domains().contains(&Domain::Buffer)
            && step.output_domain(state.domain) == Domain::Buffer =>
        {
            Some(Vec::new())
        }
        _ => None,
    }
}

/// The op's own `validate`, refused while the pipeline is built wherever the
/// plan knows enough: every verdict when the whole input shape is known, and
/// only the verdicts that depend on the rank alone (a channel op on a rank-2
/// buffer) otherwise. Sizes the plan does not know are passed as 1, which no
/// rank-level verdict reads; a size-level failure then stays a row error.
fn check_rank(step: &GraphStep, input: &[Dim]) -> Result<(), String> {
    let op: &dyn view_buffer::Op = match step {
        GraphStep::Buffer(dto) => dto.as_op(),
        GraphStep::Geometry(geo) => geo,
        _ => return Ok(()),
    };
    if input.is_empty() {
        return Ok(());
    }
    let known: Option<Vec<usize>> = input.iter().map(|d| d.known()).collect();
    let fully_known = known.is_some();
    let shape = known.unwrap_or_else(|| input.iter().map(|d| d.known().unwrap_or(1)).collect());
    match op.validate(&[shape.as_slice()], &[]) {
        Err(e) if fully_known || e.depends_only_on_rank() => Err(e.to_string()),
        _ => Ok(()),
    }
}

/// A binary op's output dtype over both operands; unknown unless both are
/// known.
fn binary_dtype(
    op: view_buffer::BinaryOp,
    left: PlannedDType,
    right: PlannedDType,
) -> PlannedDType {
    match (left, right) {
        (PlannedDType::Known(l), PlannedDType::Known(r)) => {
            PlannedDType::Known(op.output_dtype(l, r))
        }
        _ => PlannedDType::Unknown,
    }
}

/// Python entry point for [`step`]: the state after appending `op_json`.
/// `refs` are the states of the nodes the op reads by id.
#[pyfunction]
#[pyo3(signature = (op_json, state, refs=None))]
pub(crate) fn plan_step(op_json: &str, state: State, refs: Option<Refs>) -> PyResult<State> {
    let op: crate::ops::TypedOp =
        serde_json::from_str(op_json).map_err(|e| py_value_error(e.to_string()))?;
    step(&op, &state, &refs.unwrap_or_default()).map_err(py_value_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn state(domain: &str, dtype: &str, ndim: Option<usize>, dims: [Option<usize>; 3]) -> State {
        State {
            dims,
            ..State::new(
                view_buffer::naming::lookup(Domain::NAMED, domain).unwrap(),
                PlannedDType::parse(dtype).unwrap(),
                ndim,
            )
        }
    }

    fn image() -> State {
        state("buffer", "u8", Some(3), [Some(100), Some(50), Some(3)])
    }

    fn op(v: serde_json::Value) -> crate::ops::TypedOp {
        serde_json::from_value(v).unwrap()
    }

    fn run(v: serde_json::Value, s: &State) -> Result<State, String> {
        step(&op(v), s, &Refs::new())
    }

    fn run_with(v: serde_json::Value, s: &State, refs: &[(&str, State)]) -> Result<State, String> {
        let refs: Refs = refs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect();
        step(&op(v), s, &refs)
    }

    #[test]
    fn a_resize_replaces_hw_and_keeps_channels() {
        let out = run(
            json!({"op": "resize", "height": 4, "width": 6, "filter": "bilinear"}),
            &image(),
        )
        .unwrap();
        assert_eq!(out.dims, [Some(4), Some(6), Some(3)]);
        assert_eq!(
            (out.domain.name(), out.dtype.as_str(), out.ndim),
            ("buffer", "u8", Some(3))
        );
    }

    #[test]
    fn an_unknown_input_rank_leaves_hw_untouched() {
        // H is a declared size over an unknown rank (assert_shape on a list).
        let s = state("buffer", "u8", None, [Some(7), None, None]);
        let out = run(json!({"op": "grayscale"}), &s).unwrap();
        assert_eq!(
            out.dims[..2],
            [Some(7), None],
            "H/W must be kept, not cleared"
        );
    }

    #[test]
    fn hints_beyond_the_output_rank_are_cleared() {
        let out = run(json!({"op": "channel_select", "index": 0}), &image()).unwrap();
        assert_eq!(out.ndim, Some(2));
        assert_eq!(out.dims[2], None);
    }

    #[test]
    fn a_wrong_input_domain_is_refused_naming_the_op() {
        let s = state("contour", "f64", None, [None; 3]);
        let err = run(json!({"op": "grayscale"}), &s).unwrap_err();
        assert!(err.contains("grayscale() expects buffer input"), "{err}");
    }

    #[test]
    fn a_binary_op_reads_its_other_operand_by_id() {
        let add = json!({"op": "divide", "other": "n0"});
        let err = run(add.clone(), &image()).unwrap_err();
        assert!(
            err.contains("reads node 'n0', which has no planned state"),
            "{err}"
        );
        let out = run_with(add.clone(), &image(), &[("n0", image())]).unwrap();
        assert_eq!(out.dtype.as_str(), "f32");
        // An operand of unknown rank leaves the broadcast shape unknown.
        let unranked = state("buffer", "u8", None, [None; 3]);
        let out = run_with(add, &image(), &[("n0", unranked)]).unwrap();
        assert_eq!(out.dims[..2], [None, None]);
    }

    #[test]
    fn a_binary_op_keeps_a_vector_operand_a_vector() {
        let hash = state("vector", "u8", Some(1), [Some(8), None, None]);
        let out = run_with(
            json!({"op": "bitwise_xor", "other": "n0"}),
            &hash,
            &[("n0", hash.clone())],
        )
        .unwrap();
        assert_eq!((out.domain, out.ndim), (Domain::Vector, Some(1)));
    }

    #[test]
    fn rasterize_takes_its_canvas_from_its_size_or_the_node_it_names() {
        let s = state("contour", "f64", None, [None; 3]);
        let raster =
            |size| json!({"op": "rasterize", "size": size, "fill_value": 255, "background": 0});
        let out = run(raster(json!([8, 6])), &s).unwrap();
        assert_eq!(out.domain, Domain::Buffer);
        assert_eq!(out.dims, [Some(8), Some(6), Some(1)]);
        let out = run_with(raster(json!("n0")), &s, &[("n0", image())]).unwrap();
        assert_eq!(out.dims, [Some(100), Some(50), Some(1)]);
        let err = run(raster(json!("n0")), &s).unwrap_err();
        assert!(err.contains("reads node 'n0'"), "{err}");
    }

    fn declare_op(rank: Option<u32>, dims: serde_json::Value) -> serde_json::Value {
        json!({"op": "assert_shape", "rank": rank, "dims": dims})
    }

    #[test]
    fn a_declaration_sets_what_is_unknown() {
        let s = state("buffer", "u8", None, [None; 3]);
        let out = run(declare_op(Some(3), json!([8, null, {"$slot": 1}])), &s).unwrap();
        assert_eq!(out.ndim, Some(3));
        // A per-row size is declared, but no plan-time fact.
        assert_eq!(out.dims, [Some(8), None, None]);
    }

    #[test]
    fn a_declaration_the_state_contradicts_is_refused() {
        let image = state("buffer", "u8", Some(3), [Some(10), Some(20), Some(3)]);
        let err = run(declare_op(None, json!([11, null, null])), &image).unwrap_err();
        assert!(
            err.contains("assert_shape(height=11) contradicts the height 10"),
            "{err}"
        );
        let err = run(declare_op(Some(2), json!([null, null, null])), &image).unwrap_err();
        assert!(
            err.contains(
                "declares a rank-2 output, but this pipeline is already known to produce rank 3"
            ),
            "{err}"
        );
        let flat = state("buffer", "u8", Some(2), [None; 3]);
        let err = run(declare_op(None, json!([null, null, 3])), &flat).unwrap_err();
        assert!(
            err.contains("assert_shape(channels=...) names dimension 2, which a rank-2 output"),
            "{err}"
        );
        let err = run(declare_op(None, json!([0, null, null])), &flat).unwrap_err();
        assert!(err.contains("positive int"), "{err}");
        let err = run(declare_op(Some(4), json!([null, null, null])), &flat).unwrap_err();
        assert!(err.contains("supports 1 to 3 dimensions"), "{err}");
        // Agreeing is fine.
        let ok = run(declare_op(None, json!([10, null, null])), &image).unwrap();
        assert_eq!(ok.dims, [Some(10), Some(20), Some(3)]);
    }

    #[test]
    fn each_source_format_plans_its_own_state() {
        let plan = |v: serde_json::Value, refs: &Refs| {
            let source = serde_json::from_value(v).unwrap();
            let s = source_state(&source, refs).unwrap();
            (s.domain.name(), s.dtype.as_str(), s.ndim, s.dims)
        };
        let none = Refs::new();
        let buffer = |dtype, ndim| ("buffer", dtype, ndim, [None; 3]);
        assert_eq!(
            plan(json!({"format": "raw", "dtype": "u16"}), &none),
            buffer("u16", Some(1))
        );
        assert_eq!(
            plan(json!({"format": "image_bytes"}), &none),
            buffer("auto", Some(3))
        );
        assert_eq!(
            plan(json!({"format": "file_path", "dtype": "f32"}), &none),
            buffer("f32", Some(3))
        );
        assert_eq!(plan(json!({"format": "auto"}), &none), buffer("auto", None));
        assert_eq!(
            plan(json!({"format": "list", "dtype": "f32"}), &none),
            buffer("f32", None)
        );
        let contour = |size: serde_json::Value| json!({"format": "contour", "size": size, "fill_value": 255, "background": 0});
        assert_eq!(
            plan(contour(json!([10, 12])), &none),
            ("buffer", "u8", Some(3), [Some(10), Some(12), Some(1)])
        );
        // A canvas from another node is that node's planned size.
        let refs: Refs = [("n0".to_string(), image())].into_iter().collect();
        assert_eq!(
            plan(contour(json!("n0")), &refs).3,
            [Some(100), Some(50), Some(1)]
        );
    }

    /// The H/W `step` plans from the op's symbolic shape: literal params and
    /// known sizes give exact dims, and a per-row param or unknown size leaves
    /// only the axes it decides unknown — no placeholder value is involved.
    #[test]
    fn shapes_are_planned_symbolically() {
        let hw = |v: serde_json::Value, s: &State| {
            let out = run(v, s).unwrap();
            (out.dims[0], out.dims[1])
        };
        let unknown = state("buffer", "u8", Some(3), [None, None, Some(3)]);
        let resize = |h: serde_json::Value| json!({"op": "resize", "height": h, "width": 100, "filter": "bilinear"});
        assert_eq!(hw(resize(json!(224)), &unknown), (Some(224), Some(100)));
        assert_eq!(hw(resize(json!({"$slot": 0})), &unknown), (None, Some(100)));
        let pad = json!({"op": "pad", "top": 1, "bottom": 2, "left": 3, "right": 4, "value": 0.0, "mode": "constant"});
        let square = state("buffer", "u8", Some(3), [Some(10), Some(10), Some(3)]);
        assert_eq!(hw(pad, &square), (Some(13), Some(17)));
        let rotate = |angle: serde_json::Value| json!({"op": "rotate", "angle": angle, "expand": false, "interpolation": "nearest", "border_value": 0.0});
        // A literal 90 swaps H/W; a per-row angle might, so H/W are unknown —
        // unless the image is square, where a swap changes nothing.
        assert_eq!(hw(rotate(json!(90.0)), &image()), (Some(50), Some(100)));
        assert_eq!(hw(rotate(json!({"$slot": 0})), &image()), (None, None));
        assert_eq!(
            hw(rotate(json!({"$slot": 0})), &square),
            (Some(10), Some(10))
        );
    }
}
