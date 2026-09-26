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

use crate::formats::Format as _;
use crate::py_value_error;
use view_buffer::ops::{Dim, Domain, HistogramOutput};
use view_buffer::PlannedDType;

use crate::graph::step::GraphStep;

/// What the planner knows about a buffer's shape: the one representation of
/// its rank and sizes, so a size past the rank cannot be held.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum PlannedShape {
    /// The rank is known: one entry per axis, `None` where the size is not
    /// (a per-row size is unknown at plan time).
    Ranked(Vec<Option<usize>>),
    /// The rank is not known. `leading[i]` is the size axis `i` has *if the
    /// data has that axis* (an `assert_shape(height=...)` over a list column);
    /// trailing unknowns are trimmed, so equal knowledge compares equal.
    Unranked { leading: Vec<Option<usize>> },
}

impl PlannedShape {
    /// Nothing known, not even the rank.
    pub(crate) fn unknown() -> Self {
        PlannedShape::Unranked {
            leading: Vec::new(),
        }
    }

    /// A known rank (or none) with no sizes known.
    pub(crate) fn of_rank(rank: Option<usize>) -> Self {
        match rank {
            Some(n) => PlannedShape::Ranked(vec![None; n]),
            None => PlannedShape::unknown(),
        }
    }

    /// Leading sizes over an unknown rank, trailing unknowns trimmed.
    fn unranked(mut leading: Vec<Option<usize>>) -> Self {
        while leading.last() == Some(&None) {
            leading.pop();
        }
        PlannedShape::Unranked { leading }
    }

    pub(crate) fn rank(&self) -> Option<usize> {
        match self {
            PlannedShape::Ranked(sizes) => Some(sizes.len()),
            PlannedShape::Unranked { .. } => None,
        }
    }

    /// The sizes held: one per axis when ranked, the leading ones otherwise.
    pub(crate) fn sizes(&self) -> &[Option<usize>] {
        match self {
            PlannedShape::Ranked(sizes) | PlannedShape::Unranked { leading: sizes } => sizes,
        }
    }

    /// The size of `axis`, when known.
    pub(crate) fn size(&self, axis: usize) -> Option<usize> {
        self.sizes().get(axis).copied().flatten()
    }

    /// The whole shape, when the rank and every size are known.
    pub(crate) fn concrete(&self) -> Option<Vec<usize>> {
        match self {
            PlannedShape::Ranked(sizes) => sizes.iter().copied().collect(),
            PlannedShape::Unranked { .. } => None,
        }
    }

    /// The shape an op consumes, symbolically: each known size, and
    /// `Input(k)` for an unknown one. `None` when the rank is unknown.
    fn symbolic(&self) -> Option<Vec<Dim>> {
        match self {
            PlannedShape::Ranked(sizes) => Some(
                sizes
                    .iter()
                    .enumerate()
                    .map(|(axis, size)| size.map_or(Dim::Input(axis), Dim::Known))
                    .collect(),
            ),
            PlannedShape::Unranked { .. } => None,
        }
    }

    /// This shape with `axis` set to `size` where the shape has that axis.
    fn with_size(mut self, axis: usize, size: Option<usize>) -> Self {
        match &mut self {
            PlannedShape::Ranked(sizes) => {
                if let Some(slot) = sizes.get_mut(axis) {
                    *slot = size;
                }
                self
            }
            PlannedShape::Unranked { leading } => {
                let mut leading = std::mem::take(leading);
                if leading.len() <= axis {
                    leading.resize(axis + 1, None);
                }
                leading[axis] = size;
                PlannedShape::unranked(leading)
            }
        }
    }

    /// The wire form: `{"ndim": n | null, "dims": [...]}` (the pickle's).
    fn wire(&self) -> serde_json::Value {
        serde_json::json!({"ndim": self.rank(), "dims": self.sizes()})
    }
}

/// The planner's state at one op boundary — the one representation of it.
///
/// Typed throughout: a [`Domain`], a [`PlannedDType`] (the dtype lattice the
/// execution side resolves with too), and a [`PlannedShape`]. Python holds
/// these objects as they are (`polars_cv._lib.PlanState`) and reads the wire
/// spellings through its getters; it never builds or edits one. Nothing about
/// it crosses the graph wire: the graph loader plans the graph itself
/// ([`resolved_output_specs`](crate::graph::resolved_output_specs)).
#[pyclass(frozen, from_py_object, module = "polars_cv._lib", name = "PlanState")]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct State {
    pub domain: Domain,
    pub dtype: PlannedDType,
    pub shape: PlannedShape,
}

/// A [`State`]'s pickle form, read strictly (an unknown field is refused).
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct WireState {
    #[serde(deserialize_with = "wire_domain")]
    domain: Domain,
    #[serde(deserialize_with = "wire_dtype")]
    dtype: PlannedDType,
    #[serde(default)]
    ndim: Option<usize>,
    #[serde(default)]
    dims: Vec<Option<usize>>,
}

impl<'de> serde::Deserialize<'de> for State {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        State::from_wire(WireState::deserialize(d)?).map_err(serde::de::Error::custom)
    }
}

impl State {
    fn from_wire(w: WireState) -> Result<State, String> {
        let shape = match w.ndim {
            Some(n) if w.dims.len() == n => PlannedShape::Ranked(w.dims),
            Some(n) => {
                return Err(format!(
                    "a rank-{n} state carries {} sizes: {:?}",
                    w.dims.len(),
                    w.dims
                ))
            }
            None => PlannedShape::unranked(w.dims),
        };
        Ok(State {
            domain: w.domain,
            dtype: w.dtype,
            shape,
        })
    }
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
        State::new(
            Domain::Buffer,
            PlannedDType::Unknown,
            PlannedShape::unknown(),
        )
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
        self.shape.rank()
    }

    /// Known sizes, one per dimension when the rank is known (``None`` is
    /// unknown or per-row); over an unknown rank, the sizes declared for the
    /// leading dimensions.
    #[getter(dims)]
    fn py_dims<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        pyo3::types::PyTuple::new(py, self.shape.sizes())
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }

    fn __repr__(&self) -> String {
        format!(
            "PlanState(domain={:?}, dtype={:?}, ndim={:?}, dims={:?})",
            self.domain.name(),
            self.dtype.as_str(),
            self.shape.rank(),
            self.shape.sizes(),
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
        let mut wire = self.shape.wire();
        wire["domain"] = self.domain.name().into();
        wire["dtype"] = self.dtype.as_str().into();
        wire.to_string()
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

/// The keyword names `assert_shape` gives dimensions 0, 1 and 2.
pub(crate) const DIM_NAMES: [&str; 3] = ["height", "width", "channels"];

/// How an `assert_shape` names dimension `axis`: its keyword for a leading
/// (keyword) declaration, its position in `dims=` otherwise.
pub(crate) fn declared_name(axis: usize, exact: bool) -> String {
    match DIM_NAMES.get(axis) {
        Some(name) if !exact => (*name).to_string(),
        _ => format!("dims[{axis}]"),
    }
}

/// Where an `assert_shape` declaration and a shape disagree.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Mismatch {
    /// `dims=` pins a rank the shape does not have.
    Rank { declared: usize, have: usize },
    /// A declared size names an axis the shape does not have.
    MissingAxis { axis: usize, rank: usize },
    /// A declared size differs from the shape's.
    Size {
        axis: usize,
        declared: usize,
        have: usize,
    },
}

/// `shape` with the declaration applied: `dims` are the declared sizes
/// (`None` declares nothing about that axis, a per-row size included) and
/// `exact` pins the rank to their count; otherwise they are the leading axes'.
/// The first way the two disagree is refused.
///
/// The one reading of a declaration: the planner applies it to the planned
/// shape, and execution checks each row by applying it to the row's shape.
pub(crate) fn apply_declaration(
    shape: &PlannedShape,
    dims: &[Option<usize>],
    exact: bool,
) -> Result<PlannedShape, Mismatch> {
    let mut sizes = match shape {
        PlannedShape::Ranked(have) => {
            if exact && have.len() != dims.len() {
                return Err(Mismatch::Rank {
                    declared: dims.len(),
                    have: have.len(),
                });
            }
            if let Some(axis) = (have.len()..dims.len()).find(|&a| dims[a].is_some()) {
                return Err(Mismatch::MissingAxis {
                    axis,
                    rank: have.len(),
                });
            }
            have.clone()
        }
        // An exact declaration pins the rank; the sizes held for axes past
        // it described data that does not exist, so they go.
        PlannedShape::Unranked { leading } if exact => (0..dims.len())
            .map(|axis| leading.get(axis).copied().flatten())
            .collect(),
        PlannedShape::Unranked { leading } => {
            let mut leading = leading.clone();
            leading.resize(leading.len().max(dims.len()), None);
            leading
        }
    };
    for (axis, declared) in dims.iter().enumerate() {
        let Some(declared) = *declared else { continue };
        match sizes[axis] {
            Some(have) if have != declared => {
                return Err(Mismatch::Size {
                    axis,
                    declared,
                    have,
                })
            }
            _ => sizes[axis] = Some(declared),
        }
    }
    Ok(match shape {
        PlannedShape::Ranked(_) => PlannedShape::Ranked(sizes),
        PlannedShape::Unranked { .. } if exact => PlannedShape::Ranked(sizes),
        PlannedShape::Unranked { .. } => PlannedShape::unranked(sizes),
    })
}

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
    state: &State,
    dims: &[Option<crate::ops::Param<u32>>],
    exact: bool,
) -> Result<State, String> {
    use crate::ops::Param;

    let sizes: Vec<Option<usize>> = dims
        .iter()
        .map(|d| match d {
            Some(Param::Lit(size)) => Some(*size as usize),
            Some(Param::Slot(_)) | None => None,
        })
        .collect();
    let shape = apply_declaration(&state.shape, &sizes, exact).map_err(|m| match m {
        Mismatch::Rank { declared, have } => format!(
            "assert_shape(dims=...) declares a rank-{declared} output, but this pipeline is \
             already known to produce rank {have}. Drop the assertion, or correct its length."
        ),
        Mismatch::MissingAxis { axis, rank } => format!(
            "assert_shape({}=...) names dimension {axis}, which a rank-{rank} output does not \
             have. The shape keywords are positional — {} are dimensions 0, 1 and 2 — so use \
             assert_shape(dims=[...]) for anything that is not an [H, W, C] image.",
            declared_name(axis, exact),
            DIM_NAMES.join(", ")
        ),
        Mismatch::Size {
            axis,
            declared,
            have,
        } => {
            let name = declared_name(axis, exact);
            let what = match DIM_NAMES.get(axis) {
                Some(keyword) if !exact => format!("{keyword} {have}"),
                _ => format!("size {have} of dimension {axis}"),
            };
            format!(
                "assert_shape({name}={declared}) contradicts the {what} the pipeline already \
                 establishes at this point. An assertion cannot change what the data is — \
                 remove it, or fix the value."
            )
        }
    })?;
    Ok(State {
        shape,
        ..state.clone()
    })
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
    if let TypedOp::Graph(crate::ops::graph::GraphOp::AssertShape { dims, exact }) = op {
        return declare(state, dims, exact.get());
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
    // The step's own validation over what the plan knows of its inputs: every
    // error is a verdict on a known fact, so each is raised. An input of
    // unknown rank gives it nothing to decide.
    let inputs: Option<Vec<&[Dim]>> = match (&input, other) {
        (Some(input), None) => Some(vec![input]),
        (Some(input), Some(_)) => other_input.as_deref().map(|o| vec![input.as_slice(), o]),
        (None, _) => None,
    };
    if let Some(inputs) = inputs {
        let dtypes: Vec<PlannedDType> = std::iter::once(state.dtype)
            .chain(other.map(|o| o.dtype))
            .collect();
        step.validate(&inputs, &dtypes)
            .map_err(|e| format!("{}(): {e}", op.name()))?;
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
    let known = |out: Option<Vec<Dim>>| -> Vec<Option<usize>> {
        out.unwrap_or_default()
            .into_iter()
            .map(Dim::known)
            .collect()
    };
    let sizes = match (&input, other.is_some(), &other_input) {
        (Some(input), false, _) => known(shape.dims(&[input])),
        (Some(input), true, Some(other_input)) => known(shape.dims(&[input, other_input])),
        // A binary operand of unknown rank: nothing to broadcast against.
        (_, true, _) => Vec::new(),
        // An input of unknown rank: a size is known after the op only where
        // the shape gives it whatever the rank (a grayscale keeps a declared
        // H; a resize replaces it).
        (None, false, _) => shape.dims_over_unknown_rank(state.shape.sizes()),
    };
    // The rank decides which sizes exist: a scalar has none, a vector its one.
    let mut planned = match ndim {
        Some(n) => PlannedShape::Ranked(
            (0..n)
                .map(|axis| sizes.get(axis).copied().flatten())
                .collect(),
        ),
        None => PlannedShape::unranked(sizes),
    };
    // A canvas taken from another node has that node's planned H/W.
    if let TypedOp::Geometry(view_buffer::GeometryOp::Rasterize {
        size: RasterSize::FromNode(NodeRef(node)),
        ..
    }) = op
    {
        let canvas = referenced(refs, op.name(), node)?;
        planned = planned
            .with_size(0, canvas.shape.size(0))
            .with_size(1, canvas.shape.size(1));
    }

    Ok(State {
        domain: out_domain,
        dtype,
        shape: planned,
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
    pub(crate) fn new(domain: Domain, dtype: PlannedDType, shape: PlannedShape) -> State {
        State {
            domain,
            dtype,
            shape,
        }
    }
}

/// The state a source hands the first op.
///
/// Exhaustive over the formats: what each decodes to is a fact about the
/// format.
pub(crate) fn source_state(source: &crate::formats::source::Source) -> State {
    use crate::formats::source::Source;

    let buffer = |dtype: Option<view_buffer::DType>, ndim: Option<usize>| {
        let dtype = dtype.map_or(PlannedDType::Unknown, PlannedDType::Known);
        State::new(Domain::Buffer, dtype, PlannedShape::of_rank(ndim))
    };
    let dtype = source.dtype();
    match source {
        // Raw bytes decode to a flat 1-D buffer of the declared dtype.
        Source::Raw { .. } => buffer(dtype, Some(1)),
        // Decoded images are always `[H, W, C]`; the dtype is the caller's
        // assertion or unknown until decode (PNG u8, 16-bit PNG u16, TIFF ...).
        Source::ImageBytes { .. } | Source::FilePath { .. } => buffer(dtype, Some(3)),
        // Rank follows the column (nesting depth, blob header, or the path the
        // column's dtype routes to), known only with the input.
        Source::Auto { .. } | Source::Blob { .. } | Source::List { .. } | Source::Array { .. } => {
            buffer(dtype, None)
        }
        // The column's contour set, as `extract_contours` publishes one:
        // f64 coordinates, no rank. Rasterizing is the `rasterize` op's.
        Source::Contour { .. } => State::new(
            Domain::Contour,
            PlannedDType::Known(view_buffer::DType::F64),
            PlannedShape::unknown(),
        ),
    }
}

/// The shape an op consumes, symbolically: each known size, and `Input(k)`
/// for an unknown one. `None` when the rank is unknown, so there is no shape
/// to reason about — except for a step that *builds* a buffer from another
/// domain (`rasterize`), which consumes no buffer at all.
pub(crate) fn input_dims(step: &crate::ops::TypedOp, state: &State) -> Option<Vec<Dim>> {
    match state.shape.symbolic() {
        Some(dims) if !dims.is_empty() => Some(dims),
        _ if !step.input_domains().contains(&Domain::Buffer)
            && step.output_domain(state.domain) == Domain::Buffer =>
        {
            Some(Vec::new())
        }
        _ => None,
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
        let start = State::unsourced();
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
    /// source starts them in.
    fn with_source(&self, source_json: &str) -> PyResult<Plan> {
        let source: crate::formats::source::Source =
            serde_json::from_str(source_json).map_err(|e| py_value_error(e.to_string()))?;
        let start = source_state(&source);
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

    /// A state of rank `ndim` (or unknown) whose first sizes are `dims`.
    fn state(domain: &str, dtype: &str, ndim: Option<usize>, dims: [Option<usize>; 3]) -> State {
        let shape = match ndim {
            Some(n) => PlannedShape::Ranked(
                (0..n)
                    .map(|axis| dims.get(axis).copied().flatten())
                    .collect(),
            ),
            None => PlannedShape::unranked(dims.to_vec()),
        };
        State::new(
            view_buffer::naming::lookup(Domain::NAMED, domain).unwrap(),
            PlannedDType::parse(dtype).unwrap(),
            shape,
        )
    }

    /// The sizes of dimensions 0, 1 and 2 (`None` unknown or absent).
    fn dims3(s: &State) -> [Option<usize>; 3] {
        std::array::from_fn(|axis| s.shape.size(axis))
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
        assert_eq!(dims3(&out), [Some(4), Some(6), Some(3)]);
        assert_eq!(
            (out.domain.name(), out.dtype.as_str(), out.shape.rank()),
            ("buffer", "u8", Some(3))
        );
    }

    #[test]
    fn an_op_that_keeps_hw_keeps_a_size_over_an_unknown_rank() {
        // H is a declared size over an unknown rank (assert_shape on a list).
        let s = state("buffer", "u8", None, [Some(7), None, None]);
        let out = run(json!({"op": "grayscale"}), &s).unwrap();
        assert_eq!(
            dims3(&out)[..2],
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
            dims3(&out)[0],
            Some(7),
            "a stale size was carried across a resize"
        );
    }

    #[test]
    fn hints_beyond_the_output_rank_are_cleared() {
        let out = run(json!({"op": "channel_select", "index": 0}), &image()).unwrap();
        assert_eq!(out.shape.rank(), Some(2));
        assert_eq!(out.shape.sizes().len(), 2);
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
        assert_eq!(dims3(&out)[..2], [None, None]);
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
        assert_eq!((out.domain, out.shape.rank()), (Domain::Vector, Some(1)));
    }

    #[test]
    fn rasterize_takes_its_canvas_from_its_size_or_the_node_it_names() {
        let s = state("contour", "f64", None, [None; 3]);
        let raster =
            |size| json!({"op": "rasterize", "size": size, "fill_value": 255, "background": 0});
        let out = run(raster(json!([8, 6])), &s).unwrap();
        assert_eq!(out.domain, Domain::Buffer);
        assert_eq!(dims3(&out), [Some(8), Some(6), Some(1)]);
        let out = run_with(raster(json!("n0")), &s, &[("n0", image())]).unwrap();
        assert_eq!(dims3(&out), [Some(100), Some(50), Some(1)]);
        let err = run(raster(json!("n0")), &s).unwrap_err();
        assert!(err.contains("reads node 'n0'"), "{err}");
    }

    fn declare_op(exact: bool, dims: serde_json::Value) -> serde_json::Value {
        json!({"op": "assert_shape", "dims": dims, "exact": exact})
    }

    #[test]
    fn a_declaration_sets_what_is_unknown() {
        let s = state("buffer", "u8", None, [None; 3]);
        let out = run(declare_op(true, json!([8, null, {"$slot": 1}])), &s).unwrap();
        assert_eq!(out.shape.rank(), Some(3));
        // A per-row size is declared, but no plan-time fact.
        assert_eq!(dims3(&out), [Some(8), None, None]);
    }

    #[test]
    fn a_declaration_of_any_rank_is_planned() {
        let s = state("buffer", "u8", None, [None; 3]);
        let out = run(declare_op(true, json!([2, 3, 4, 5])), &s).unwrap();
        assert_eq!(
            out.shape,
            PlannedShape::Ranked(vec![Some(2), Some(3), Some(4), Some(5)])
        );
        // A rank-4 shape carries its fourth size through an op that keeps it.
        let out = run(json!({"op": "cast", "dtype": "f32"}), &out).unwrap();
        assert_eq!(out.shape.concrete(), Some(vec![2, 3, 4, 5]));
        // Leading sizes over an unknown rank, then the rank: an exact
        // declaration keeps what it agrees with and drops what its rank lacks.
        let hinted = run(declare_op(false, json!([null, null, 3])), &s).unwrap();
        assert_eq!(
            hinted.shape,
            PlannedShape::unranked(vec![None, None, Some(3)])
        );
        let out = run(declare_op(true, json!([4, 5])), &hinted).unwrap();
        assert_eq!(out.shape, PlannedShape::Ranked(vec![Some(4), Some(5)]));
    }

    #[test]
    fn a_per_row_declaration_keeps_a_known_size() {
        let image = state("buffer", "u8", Some(3), [Some(10), Some(20), Some(3)]);
        let out = run(declare_op(false, json!([{"$slot": 0}])), &image).unwrap();
        assert_eq!(dims3(&out), [Some(10), Some(20), Some(3)]);
    }

    #[test]
    fn a_declaration_the_state_contradicts_is_refused() {
        let image = state("buffer", "u8", Some(3), [Some(10), Some(20), Some(3)]);
        let err = run(declare_op(false, json!([11])), &image).unwrap_err();
        assert!(
            err.contains("assert_shape(height=11) contradicts the height 10"),
            "{err}"
        );
        let err = run(declare_op(true, json!([10, 21, 3])), &image).unwrap_err();
        assert!(
            err.contains("assert_shape(dims[1]=21) contradicts the size 20 of dimension 1"),
            "{err}"
        );
        let err = run(declare_op(true, json!([null, null])), &image).unwrap_err();
        assert!(
            err.contains(
                "declares a rank-2 output, but this pipeline is already known to produce rank 3"
            ),
            "{err}"
        );
        let flat = state("buffer", "u8", Some(2), [None; 3]);
        let err = run(declare_op(false, json!([null, null, 3])), &flat).unwrap_err();
        assert!(
            err.contains("assert_shape(channels=...) names dimension 2, which a rank-2 output"),
            "{err}"
        );
        let err = run(declare_op(false, json!([0])), &flat).unwrap_err();
        assert!(err.contains("positive int"), "{err}");
        let err = run(declare_op(false, json!([null])), &flat).unwrap_err();
        assert!(err.contains("declares nothing"), "{err}");
        let err = run(declare_op(true, json!([null, null, null, null])), &flat).unwrap_err();
        assert!(err.contains("declares a rank-4 output"), "{err}");
        // Agreeing is fine.
        let ok = run(declare_op(false, json!([10])), &image).unwrap();
        assert_eq!(dims3(&ok), [Some(10), Some(20), Some(3)]);
    }

    #[test]
    fn each_source_format_plans_its_own_state() {
        let plan = |v: serde_json::Value| {
            let s = source_state(&serde_json::from_value(v).unwrap());
            (s.domain.name(), s.dtype.as_str(), s.shape.rank(), dims3(&s))
        };
        let buffer = |dtype, ndim| ("buffer", dtype, ndim, [None; 3]);
        assert_eq!(
            plan(json!({"format": "raw", "dtype": "u16"})),
            buffer("u16", Some(1))
        );
        assert_eq!(
            plan(json!({"format": "image_bytes"})),
            buffer("auto", Some(3))
        );
        assert_eq!(
            plan(json!({"format": "file_path", "dtype": "f32"})),
            buffer("f32", Some(3))
        );
        assert_eq!(plan(json!({"format": "auto"})), buffer("auto", None));
        assert_eq!(
            plan(json!({"format": "list", "dtype": "f32"})),
            buffer("f32", None)
        );
        // A contour source decodes the set; rasterizing is an op.
        assert_eq!(
            plan(json!({"format": "contour"})),
            ("contour", "f64", None, [None; 3])
        );
    }

    /// The H/W `step` plans from the op's symbolic shape: literal params and
    /// known sizes give exact dims, and a per-row param or unknown size leaves
    /// only the axes it decides unknown — no placeholder value is involved.
    #[test]
    fn shapes_are_planned_symbolically() {
        let hw = |v: serde_json::Value, s: &State| {
            let out = run(v, s).unwrap();
            (dims3(&out)[0], dims3(&out)[1])
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

/// Plan-time validation is sound: a refusal over what the plan knows is a
/// verdict every row would reach. For every catalogue op and every way of
/// knowing only part of an input's shape, a refusal over the partly known
/// shape implies a refusal over each whole shape it could be.
#[cfg(test)]
mod validation_soundness {
    use super::*;
    use view_buffer::ops::validation::ValidationError;

    /// Every shape of rank `1..=max_rank` whose sizes are drawn from `sizes`.
    fn shapes(max_rank: usize, sizes: &[usize]) -> Vec<Vec<usize>> {
        let mut out: Vec<Vec<usize>> = vec![Vec::new()];
        let mut all = Vec::new();
        for _ in 0..max_rank {
            out = out
                .iter()
                .flat_map(|s| {
                    sizes.iter().map(move |&n| {
                        let mut s = s.clone();
                        s.push(n);
                        s
                    })
                })
                .collect();
            all.extend(out.iter().cloned());
        }
        all
    }

    /// `shape` with the sizes `mask` selects unknown.
    fn masked(shape: &[usize], mask: u32) -> Vec<Dim> {
        shape
            .iter()
            .enumerate()
            .map(|(axis, &n)| {
                if mask & (1 << axis) != 0 {
                    Dim::Input(axis)
                } else {
                    Dim::Known(n)
                }
            })
            .collect()
    }

    type Validate<'a> = &'a dyn Fn(&[&[Dim]]) -> Result<(), ValidationError>;

    /// The partly known inputs `validate` refuses while some whole shape they
    /// could be is accepted, as `(partly known, whole)` — empty when sound.
    /// `inputs` is 1, or 2 for an op over two operands.
    fn unsound_verdicts(validate: Validate<'_>, inputs: usize) -> Vec<String> {
        let known = |shapes: &[&[usize]]| {
            let dims: Vec<Vec<Dim>> = shapes
                .iter()
                .map(|s| view_buffer::ops::shape_rule::known_dims(s))
                .collect();
            let dims: Vec<&[Dim]> = dims.iter().map(Vec::as_slice).collect();
            validate(&dims).is_ok()
        };
        let mut found = Vec::new();
        let mut check = |whole: &[&[usize]], partial: &[Vec<Dim>]| {
            let partial_refs: Vec<&[Dim]> = partial.iter().map(Vec::as_slice).collect();
            if validate(&partial_refs).is_err() && known(whole) && found.len() < 5 {
                found.push(format!("refused {partial:?}, but {whole:?} is accepted"));
            }
        };
        if inputs == 1 {
            for shape in shapes(4, &[1, 2, 3, 5]) {
                for mask in 1..(1u32 << shape.len()) {
                    check(&[&shape], &[masked(&shape, mask)]);
                }
            }
        } else {
            let all = shapes(3, &[1, 2, 3]);
            for a in &all {
                for b in &all {
                    let bits = a.len() + b.len();
                    for mask in 1..(1u32 << bits) {
                        let (ma, mb) = (mask & ((1 << a.len()) - 1), mask >> a.len());
                        check(&[a, b], &[masked(a, ma), masked(b, mb)]);
                    }
                }
            }
        }
        found
    }

    /// Ops whose verdicts turn on sizes, beyond each op's one catalogue
    /// sample.
    fn size_sensitive() -> Vec<crate::ops::TypedOp> {
        [
            serde_json::json!({"op": "channel_select", "index": 2}),
            serde_json::json!({"op": "channel_swap", "order": [1, 0]}),
            serde_json::json!({"op": "normalize", "method": "preset",
                               "mean": [0.5, 0.5, 0.5], "std": [0.2, 0.2, 0.2]}),
            serde_json::json!({"op": "crop", "top": 2, "left": 0, "height": 1}),
            serde_json::json!({"op": "reshape", "shape": [3, 5]}),
            serde_json::json!({"op": "transpose", "axes": [1, 0]}),
            serde_json::json!({"op": "reduce_max", "axis": 3}),
        ]
        .into_iter()
        .map(|v| serde_json::from_value(v).unwrap())
        .collect()
    }

    #[test]
    fn a_refusal_over_what_the_plan_knows_holds_for_every_row() {
        let mut unsound = Vec::new();
        for op in crate::ops::TypedOp::samples()
            .into_iter()
            .chain(size_sensitive())
        {
            let inputs = match &op {
                GraphStep::Graph(graph) if graph.binary().is_some() => 2,
                _ => 1,
            };
            let validate = |shapes: &[&[Dim]]| op.validate(shapes, &[]);
            let found = unsound_verdicts(&validate, inputs);
            if !found.is_empty() {
                unsound.push(format!("{}: {found:?}", op.name()));
            }
        }
        assert!(
            unsound.is_empty(),
            "unsound plan-time verdicts:\n{unsound:#?}"
        );
    }

    /// The checker, on a three-channel check that reads an unknown channel
    /// count as 1 (the placeholder the planner used to pass), and on the same
    /// check saying nothing about an unknown one.
    #[test]
    fn the_checker_catches_a_placeholder_and_passes_the_real_check() {
        let refuse = || {
            Err(ValidationError::Generic {
                message: "three channels are required".into(),
            })
        };
        let placeholder = |shapes: &[&[Dim]]| match shapes[0] {
            [_, _, c] if c.known().unwrap_or(1) == 3 => Ok(()),
            _ => refuse(),
        };
        assert!(
            !unsound_verdicts(&placeholder, 1).is_empty(),
            "the checker missed a verdict read off a placeholder size"
        );
        let real = |shapes: &[&[Dim]]| match shapes[0] {
            [_, _, c] if c.known().is_none_or(|c| c == 3) => Ok(()),
            _ => refuse(),
        };
        assert_eq!(unsound_verdicts(&real, 1), Vec::<String>::new());
    }
}
