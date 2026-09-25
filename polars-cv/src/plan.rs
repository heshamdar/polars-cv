//! The planner: a pipeline's [`Plan`] and the step that extends it.
//!
//! [`step`] takes the state at one op boundary and one typed op and returns
//! the state after it — the input-domain check, the schema fold (domain,
//! dtype, rank), the sizes the op's symbolic `shape` gives, its channel rule,
//! and the clipping of every size to the output rank. One call cannot be
//! half-applied.
//!
//! [`Plan`] is a pipeline's source, its typed ops and the state at every op
//! boundary, and the only record of them: Python holds the plan object, and
//! every change to it (an append, a slice, a reorder, a pass, a rebase onto an
//! upstream node) is a method here that plans each op it keeps with [`step`].
//!
//! A two-input op plans over both operands, so a binary op takes the other
//! operand's state, and only a binary op may: passing one to any other op, or
//! omitting it for a binary op, is an error rather than a fallback to the
//! one-input rule.

use pyo3::prelude::*;

use crate::py_value_error;
use view_buffer::ops::{Dim, Domain, HistogramOutput, OpShape};
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
fn declare(
    mut state: State,
    rank: &Option<crate::ops::Literal<u32>>,
    dims: &[Option<crate::ops::Param<u32>>; 3],
) -> Result<State, String> {
    use crate::ops::Param;

    if let Some(rank) = rank.map(|r| r.get() as usize) {
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
    for (axis, declared) in dims.iter().enumerate() {
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
    use crate::ops::{NodeRef, TypedOp};
    use view_buffer::geometry::ops::RasterSize;

    // The op is read as it is: every rule below is the step's own, over its
    // literal values, and a per-row value is unknown — never stood in for.
    op.check().map_err(|e| format!("{}: {e}", op.name()))?;
    let step = op;

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
    if let TypedOp::Graph(crate::ops::graph::GraphOp::AssertShape { rank, dims }) = op {
        return declare(state.clone(), rank, dims);
    }

    let binary = match step {
        GraphStep::Graph(graph) => graph.binary(),
        _ => None,
    };
    let other = match binary {
        Some((_, other)) => Some(referenced(refs, op.name(), &other.0)?),
        None => None,
    };
    let dtype = match (binary, other) {
        (Some((op, _)), Some(other)) => binary_dtype(op, state.dtype, other.dtype),
        _ => single_input_dtype(step, state.dtype),
    };

    // The shape, symbolic over the op's per-row fields.
    let shape = step.shape();
    let input = input_dims(step, state);
    let other_input = other.and_then(|o| input_dims(step, o));
    if let Some(input) = &input {
        check_rank(step, input)?;
    }
    let mut ranks = vec![input.as_ref().map(Vec::len)];
    if other.is_some() {
        ranks.push(other_input.as_ref().map(Vec::len));
    }
    let out_domain = step.output_domain(state.domain);
    // Scalar and vector domains pin the rank whatever the shape says.
    let ndim = match out_domain {
        Domain::Scalar => Some(0),
        Domain::Vector => Some(1),
        Domain::Buffer | Domain::Contour => shape.rank(&ranks),
    };
    let mut dims: [Option<usize>; 3] = match (&input, other.is_some(), &other_input) {
        (Some(input), false, _) => known_sizes(shape.dims(&[input])),
        (Some(input), true, Some(other_input)) => known_sizes(shape.dims(&[input, other_input])),
        // A binary operand of unknown rank: nothing to broadcast against.
        (_, true, _) => [None; 3],
        // An input of unknown rank: a size is known after the op only where
        // the shape gives it whatever the rank (a grayscale keeps a declared
        // H; a resize replaces it).
        (None, false, _) => sizes_over_any_rank(&shape, &state.dims),
    };
    // A dimension the output rank does not have has no size (a scalar's
    // single slot, a vector's pinned rank).
    if let Some(n) = ndim {
        dims.iter_mut().skip(n).for_each(|d| *d = None);
    }
    // A canvas taken from another node has that node's planned H/W.
    if let TypedOp::Geometry(view_buffer::GeometryOp::Rasterize {
        size: RasterSize::FromNode(NodeRef(node)),
        ..
    }) = op
    {
        let canvas = referenced(refs, op.name(), node)?;
        dims[0] = canvas.dims[0];
        dims[1] = canvas.dims[1];
    }

    Ok(State {
        domain: out_domain,
        dtype,
        ndim,
        dims,
    })
}

/// The known sizes of dimensions 0..3 of a planned output shape.
fn known_sizes(out: Option<Vec<Dim>>) -> [Option<usize>; 3] {
    std::array::from_fn(|axis| {
        out.as_ref()
            .and_then(|out| out.get(axis).copied())
            .and_then(Dim::known)
    })
}

/// The sizes `shape` gives over an input of unknown rank whose known sizes
/// are `sizes`: evaluated over each rank the planner tracks, a size is known
/// only where every rank that has the axis agrees on it.
fn sizes_over_any_rank(shape: &OpShape, sizes: &[Option<usize>; 3]) -> [Option<usize>; 3] {
    let outs: Vec<Vec<Dim>> = (1..=sizes.len())
        .filter_map(|rank| {
            let input: Vec<Dim> = (0..rank)
                .map(|axis| sizes[axis].map_or(Dim::Input(axis), Dim::Known))
                .collect();
            shape.dims(&[&input])
        })
        .collect();
    std::array::from_fn(|axis| {
        let mut claims = outs.iter().filter_map(|out| out.get(axis).copied());
        let first = claims.next()?.known()?;
        claims.all(|d| d.known() == Some(first)).then_some(first)
    })
}

/// An op's output dtype from its own one-input rule, over the planned input
/// dtype — the one dtype lattice ([`OutputDTypeRule::resolve_planned`]).
///
/// Histogram buckets are struct-encoded by the sink, so their element dtype is
/// an encoding concern, not a schema one: unknown.
///
/// [`OutputDTypeRule::resolve_planned`]: view_buffer::OutputDTypeRule::resolve_planned
fn single_input_dtype(step: &crate::ops::TypedOp, dtype: PlannedDType) -> PlannedDType {
    match step {
        GraphStep::Histogram(h) if h.output.get() == HistogramOutput::Buckets => {
            PlannedDType::Unknown
        }
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
            let rasterize = crate::ops::TypedOp::Geometry(view_buffer::GeometryOp::Rasterize {
                size: s.size.clone(),
                fill_value: s.fill_value.unwrap_or(crate::ops::Param::Lit(255)),
                background: s.background.unwrap_or(crate::ops::Param::Lit(0)),
            });
            let contours = State::new(Domain::Contour, PlannedDType::Unknown, None);
            step(&rasterize, &contours, refs)?
        }
    })
}

/// The shape an op consumes, symbolically: each known size, and `Input(k)`
/// for an unknown one. `None` when the rank is unknown, so there is no shape
/// to reason about — except for a step that *builds* a buffer from another
/// domain (`rasterize`), which consumes no buffer at all.
pub(crate) fn input_dims(step: &crate::ops::TypedOp, state: &State) -> Option<Vec<Dim>> {
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
fn check_rank(step: &crate::ops::TypedOp, input: &[Dim]) -> Result<(), String> {
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

/// One appended op with what it was planned against: the state entering it
/// and the states of the nodes it reads by id. A rewrite plans it again from
/// these, so no per-position fact is ever carried across by hand.
#[derive(Debug, Clone)]
struct Planned {
    op: crate::ops::TypedOp,
    entering: State,
    refs: Refs,
}

/// A pipeline's plan: its source, its typed ops and the state at every op
/// boundary — the one record of a `Pipeline`'s ops (Python holds only this,
/// its expression table and its node references).
///
/// Immutable: every rewrite (an append, a slice, a reorder, a deletion, a
/// rebase onto an upstream) returns a new plan whose every state was planned
/// by [`step`], so an op cannot be appended or moved with part of its
/// plan-time effect skipped. Expression parameters are `{"$slot": i}` over
/// the pipeline's own expression table; [`Plan::to_spec`] maps them onto a
/// graph's inputs.
#[pyclass(frozen, skip_from_py_object, module = "polars_cv._lib", name = "Plan")]
#[derive(Debug, Clone)]
pub(crate) struct Plan {
    source: Option<crate::formats::source::Source>,
    start: State,
    ops: Vec<Planned>,
    state: State,
}

impl Plan {
    /// `ops` planned again from `start`, in order.
    fn replanned(
        source: Option<crate::formats::source::Source>,
        start: State,
        ops: impl IntoIterator<Item = (crate::ops::TypedOp, Refs)>,
    ) -> Result<Plan, String> {
        let mut plan = Plan {
            source,
            start: start.clone(),
            ops: Vec::new(),
            state: start,
        };
        for (op, refs) in ops {
            plan = plan.pushed(op, refs)?;
        }
        Ok(plan)
    }

    fn pushed(mut self, op: crate::ops::TypedOp, refs: Refs) -> Result<Plan, String> {
        let next = step(&op, &self.state, &refs)?;
        let entering = std::mem::replace(&mut self.state, next);
        self.ops.push(Planned { op, entering, refs });
        Ok(self)
    }

    fn states(&self) -> Vec<State> {
        self.ops
            .iter()
            .map(|p| p.entering.clone())
            .chain(std::iter::once(self.state.clone()))
            .collect()
    }

    fn typed_ops(&self) -> Vec<crate::ops::TypedOp> {
        self.ops.iter().map(|p| p.op.clone()).collect()
    }

    /// The state at op boundary `position`: entering that op, or the final
    /// state at the end.
    fn state_before(&self, position: usize) -> Result<State, String> {
        match position.cmp(&self.ops.len()) {
            std::cmp::Ordering::Less => Ok(self.ops[position].entering.clone()),
            std::cmp::Ordering::Equal => Ok(self.state.clone()),
            std::cmp::Ordering::Greater => Err(format!(
                "position {position} is past the plan's {} ops",
                self.ops.len()
            )),
        }
    }

    /// The ops at `positions`, in that order, planned again from the state at
    /// boundary `start` (default: entering the first of them).
    fn selected(&self, positions: &[usize], start: Option<usize>) -> Result<Plan, String> {
        let at = start.or(positions.first().copied()).unwrap_or(0);
        let start = self.state_before(at)?;
        let ops = positions
            .iter()
            .map(|&i| {
                self.ops
                    .get(i)
                    .map(|p| (p.op.clone(), p.refs.clone()))
                    .ok_or_else(|| format!("no op at position {i}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Plan::replanned(self.source.clone(), start, ops)
    }

    /// A plan from its pickle wire (see `Plan::__reduce__`).
    fn from_wire(wire: &str) -> Result<Plan, String> {
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct WireOp {
            op: crate::ops::TypedOp,
            refs: Refs,
        }
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct WirePlan {
            source: Option<crate::formats::source::Source>,
            start: State,
            ops: Vec<WireOp>,
        }
        let plan: WirePlan = serde_json::from_str(wire).map_err(|e| e.to_string())?;
        Plan::replanned(
            plan.source,
            plan.start,
            plan.ops.into_iter().map(|o| (o.op, o.refs)),
        )
    }
}

/// `value` with every `{"$slot": i}` replaced by `{"$slot": map[i]}`.
fn remap_slots(value: &mut serde_json::Value, map: &[usize]) -> Result<(), String> {
    use crate::ops::param::SLOT_KEY;
    use serde_json::Value;
    match value {
        Value::Object(obj) if obj.len() == 1 && obj.contains_key(SLOT_KEY) => {
            let local = obj[SLOT_KEY]
                .as_u64()
                .and_then(|i| usize::try_from(i).ok())
                .ok_or_else(|| format!("a slot is a non-negative int, got {value}"))?;
            let global = *map
                .get(local)
                .ok_or_else(|| format!("slot {local} has no graph input (map of {})", map.len()))?;
            *value = serde_json::json!({ SLOT_KEY: global });
        }
        Value::Object(obj) => {
            for v in obj.values_mut() {
                remap_slots(v, map)?;
            }
        }
        Value::Array(items) => {
            for v in items {
                remap_slots(v, map)?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// `value` on the wire: an object field that is `null` (an absent optional
/// setting) is left out, so an unset setting and an omitted one serialize
/// alike, and the graph JSON (the compiled-graph cache key) carries only what
/// was set.
fn wire<T: serde::Serialize>(value: &T) -> serde_json::Value {
    let mut value = serde_json::to_value(value).expect("a plan value always serializes");
    if let serde_json::Value::Object(obj) = &mut value {
        obj.retain(|_, v| !v.is_null());
    }
    value
}

fn parse_op(op_json: &str) -> PyResult<crate::ops::TypedOp> {
    serde_json::from_str(op_json).map_err(|e| py_value_error(e.to_string()))
}

#[pymethods]
impl Plan {
    /// The plan of a pipeline with no source and no ops.
    #[new]
    fn empty() -> Self {
        let start = State::new(Domain::Buffer, PlannedDType::Unknown, None);
        Plan {
            source: None,
            start: start.clone(),
            ops: Vec::new(),
            state: start,
        }
    }

    /// The plan of a sourceless pipeline that continues from `start`, the
    /// output state of the node it will follow (`LazyPipelineExpr`'s
    /// builders validate against it).
    #[staticmethod]
    fn continuing(start: State) -> Self {
        Plan {
            source: None,
            start: start.clone(),
            ops: Vec::new(),
            state: start,
        }
    }

    /// This plan with `source_json` as its source (validated against the
    /// format's typed definition) and its ops planned again from the state the
    /// source starts them in. `refs` are the states of the nodes the source
    /// reads (a contour canvas's).
    #[pyo3(signature = (source_json, refs=None))]
    fn with_source(&self, source_json: &str, refs: Option<Refs>) -> PyResult<Plan> {
        let source: crate::formats::source::Source =
            serde_json::from_str(source_json).map_err(|e| py_value_error(e.to_string()))?;
        let start = source_state(&source, &refs.unwrap_or_default()).map_err(py_value_error)?;
        let ops = self.ops.iter().map(|p| (p.op.clone(), p.refs.clone()));
        Plan::replanned(Some(source), start, ops).map_err(py_value_error)
    }

    /// This plan's ops planned again from `start`, the output state of the
    /// node they now follow, with `source_json` (the graph's "receive from
    /// upstream" source) as the source.
    fn rebased(&self, source_json: &str, start: State) -> PyResult<Plan> {
        let source =
            serde_json::from_str(source_json).map_err(|e| py_value_error(e.to_string()))?;
        let ops = self.ops.iter().map(|p| (p.op.clone(), p.refs.clone()));
        Plan::replanned(Some(source), start, ops).map_err(py_value_error)
    }

    /// This plan with `op_json` appended. `refs` are the states of the nodes
    /// it reads by id.
    #[pyo3(signature = (op_json, refs=None))]
    fn push(&self, op_json: &str, refs: Option<Refs>) -> PyResult<Plan> {
        self.clone()
            .pushed(parse_op(op_json)?, refs.unwrap_or_default())
            .map_err(py_value_error)
    }

    /// The ops at `positions`, in that order, planned again from the state
    /// entering the first of them (`positions` empty: the state at `start`).
    /// A slice, a reorder and a deletion are each one call.
    #[pyo3(signature = (positions, start=None))]
    fn select(&self, positions: Vec<usize>, start: Option<usize>) -> PyResult<Plan> {
        self.selected(&positions, start).map_err(py_value_error)
    }

    /// The node-scope pass `pass_name` applied: the new plan, or `None` when
    /// it changes nothing.
    fn run_pass(&self, pass_name: &str) -> PyResult<Option<Plan>> {
        use crate::passes::{run, LogicalPass, Node};
        let pass = view_buffer::naming::lookup(LogicalPass::NAMED, pass_name)
            .ok_or_else(|| py_value_error(format!("unknown pass {pass_name:?}")))?;
        let (ops, states) = (self.typed_ops(), self.states());
        let order = run(
            pass,
            &Node {
                ops: &ops,
                states: &states,
            },
        )
        .map_err(py_value_error)?;
        order
            .map(|order| {
                let ops = order
                    .iter()
                    .map(|&i| (self.ops[i].op.clone(), self.ops[i].refs.clone()));
                Plan::replanned(self.source.clone(), self.start.clone(), ops)
            })
            .transpose()
            .map_err(py_value_error)
    }

    /// The state at op boundary `position`: entering that op, or the final
    /// state at the end.
    fn state_at(&self, position: usize) -> PyResult<State> {
        self.state_before(position).map_err(py_value_error)
    }

    /// The state after the last op.
    #[getter]
    fn state(&self) -> State {
        self.state.clone()
    }

    /// Whether the plan has a source.
    #[getter]
    fn has_source(&self) -> bool {
        self.source.is_some()
    }

    /// The source's format name, if any.
    #[getter]
    fn source_format(&self) -> Option<&'static str> {
        self.source.as_ref().map(|s| s.name())
    }

    fn __len__(&self) -> usize {
        self.ops.len()
    }

    /// Each op's wire JSON, with the pipeline's own slot numbers.
    fn ops_json(&self) -> Vec<String> {
        self.ops.iter().map(|p| wire(&p.op).to_string()).collect()
    }

    /// The source's wire JSON, with the pipeline's own slot numbers.
    fn source_json(&self) -> Option<String> {
        self.source.as_ref().map(|s| wire(s).to_string())
    }

    /// The node spec a graph serializes (`{"source": ..., "ops": [...]}`) with
    /// slot `i` renumbered to graph input `slot_map[i]`.
    fn to_spec(&self, slot_map: Vec<usize>) -> PyResult<String> {
        let mut spec = serde_json::json!({
            "source": self.source.as_ref().map(wire),
            "ops": self.ops.iter().map(|p| wire(&p.op)).collect::<Vec<_>>(),
        });
        remap_slots(&mut spec, &slot_map).map_err(py_value_error)?;
        Ok(spec.to_string())
    }

    fn __copy__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __deepcopy__(slf: Py<Self>, _memo: &Bound<'_, PyAny>) -> Py<Self> {
        slf
    }

    /// Pickled as its source and ops (with the states they read), and planned
    /// again when loaded.
    fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<(Py<PyAny>, (String,))> {
        let plan = slf.get();
        let refs = |r: &Refs| -> serde_json::Value {
            r.iter()
                .map(|(k, v)| {
                    (
                        k.clone(),
                        serde_json::from_str(&v._wire()).expect("wire state"),
                    )
                })
                .collect::<serde_json::Map<_, _>>()
                .into()
        };
        let wire = serde_json::json!({
            "source": plan.source,
            "start": serde_json::from_str::<serde_json::Value>(&plan.start._wire()).expect("wire state"),
            "ops": plan.ops.iter().map(|p| serde_json::json!({"op": p.op, "refs": refs(&p.refs)})).collect::<Vec<_>>(),
        });
        let restore = slf
            .py()
            .import("polars_cv._lib")?
            .getattr("_plan_from_json")?;
        Ok((restore.unbind(), (wire.to_string(),)))
    }
}

/// Unpickle a [`Plan`] (see `Plan::__reduce__`).
#[pyfunction]
pub(crate) fn _plan_from_json(wire: &str) -> PyResult<Plan> {
    Plan::from_wire(wire).map_err(py_value_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn remap_slots_renumbers_every_slot_and_refuses_an_unmapped_one() {
        let mut v = json!({"op": "x", "a": {"$slot": 0}, "b": [1, {"$slot": 1}]});
        remap_slots(&mut v, &[5, 7]).unwrap();
        assert_eq!(
            v,
            json!({"op": "x", "a": {"$slot": 5}, "b": [1, {"$slot": 7}]})
        );
        let mut v = json!({"a": {"$slot": 2}});
        assert!(remap_slots(&mut v, &[0]).unwrap_err().contains("slot 2"));
    }

    #[test]
    fn wire_leaves_out_unset_fields_but_keeps_nulls_inside_values() {
        let v = wire(&json!({"a": null, "b": [null, 1], "c": {"d": null}}));
        assert_eq!(v, json!({"b": [null, 1], "c": {"d": null}}));
    }

    fn pushed(plan: Plan, op: serde_json::Value) -> Plan {
        plan.pushed(serde_json::from_value(op).unwrap(), Refs::new())
            .unwrap()
    }

    #[test]
    fn select_plans_the_kept_ops_again_from_the_state_entering_them() {
        let start = state("buffer", "u8", Some(3), [Some(10), Some(20), Some(3)]);
        let mut plan = Plan::continuing(start.clone());
        plan = pushed(
            plan,
            json!({"op": "resize", "height": 4, "width": 6, "filter": "bilinear"}),
        );
        plan = pushed(plan, json!({"op": "grayscale"}));
        plan = pushed(plan, json!({"op": "cast", "dtype": "f32"}));
        let whole = plan.selected(&[0, 1, 2], None).unwrap();
        assert_eq!(whole.states(), plan.states());
        // A suffix starts from the state entering it, even when it is empty.
        let tail = plan.selected(&[2], None).unwrap();
        assert_eq!(tail.state_before(0).unwrap(), plan.state_before(2).unwrap());
        assert_eq!(tail.state, plan.state);
        let empty = plan.selected(&[], Some(3)).unwrap();
        assert_eq!(empty.state, plan.state);
        // A reorder is planned, not carried: cast then grayscale.
        let reordered = plan.selected(&[0, 2, 1], None).unwrap();
        assert_eq!(reordered.state.dtype, plan.state.dtype);
        assert!(plan.selected(&[3], None).is_err());
    }

    #[test]
    fn a_plan_survives_its_pickle_wire() {
        let plan = pushed(
            Plan::continuing(state("buffer", "u8", Some(3), [None; 3])),
            json!({"op": "resize", "height": 4, "width": 6, "filter": "bilinear"}),
        );
        let wire = json!({
            "source": null,
            "start": serde_json::from_str::<serde_json::Value>(&plan.start._wire()).unwrap(),
            "ops": [{"op": serde_json::from_str::<serde_json::Value>(&plan.ops_json()[0]).unwrap(), "refs": {}}],
        });
        let back = Plan::from_wire(&wire.to_string()).unwrap();
        assert_eq!(back.states(), plan.states());
        assert_eq!(back.ops_json(), plan.ops_json());
    }

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
    fn an_op_that_keeps_hw_keeps_a_size_over_an_unknown_rank() {
        // H is a declared size over an unknown rank (assert_shape on a list).
        let s = state("buffer", "u8", None, [Some(7), None, None]);
        let out = run(json!({"op": "grayscale"}), &s).unwrap();
        assert_eq!(
            out.dims[..2],
            [Some(7), None],
            "grayscale keeps H/W, so a declared H survives"
        );
    }

    #[test]
    fn a_size_an_op_changes_is_not_carried_over_an_unknown_rank() {
        // The declared H cannot survive a resize to 4, whatever the rank.
        let s = state("buffer", "u8", None, [Some(7), None, None]);
        let out = run(
            json!({"op": "resize", "height": 4, "width": 6, "filter": "bilinear"}),
            &s,
        )
        .unwrap();
        assert_ne!(
            out.dims[0],
            Some(7),
            "a stale size was carried across a resize"
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
