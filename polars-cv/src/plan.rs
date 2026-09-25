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
/// through its getters; it never builds or edits one. Each graph output
/// carries its node's final state on the wire as `planned`
/// ([`OutputSpec`](crate::graph::types::OutputSpec)); a field left out there is
/// unknown, never guessed.
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
    /// Which of `dims` the user asserted (`assert_shape`) rather than an op
    /// inferred: a divergence at execution is then theirs to fix.
    #[serde(default)]
    pub asserted: [bool; 3],
    /// A shape declaration (`assert_shape`, a canvas taken from another node)
    /// reached this lineage, so the sizes may rest on a claim.
    #[serde(default)]
    pub declared: bool,
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

    /// Which of ``dims`` the user asserted rather than an op inferred.
    #[getter(asserted)]
    fn py_asserted(&self) -> (bool, bool, bool) {
        let [h, w, c] = self.asserted;
        (h, w, c)
    }

    /// Whether a shape declaration reached this lineage.
    #[getter(declared)]
    fn py_declared(&self) -> bool {
        self.declared
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }

    fn __repr__(&self) -> String {
        format!(
            "PlanState(domain={:?}, dtype={:?}, ndim={:?}, dims={:?}, asserted={:?}, declared={})",
            self.domain.name(),
            self.dtype.as_str(),
            self.ndim,
            self.dims,
            self.asserted,
            self.declared
        )
    }

    fn __copy__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __deepcopy__(slf: Py<Self>, _memo: &Bound<'_, PyAny>) -> Py<Self> {
        slf
    }

    /// The wire form (JSON) a graph output's `planned` carries.
    fn _wire(&self) -> String {
        serde_json::json!({
            "domain": self.domain.name(),
            "dtype": self.dtype.as_str(),
            "ndim": self.ndim,
            "dims": self.dims,
            "asserted": self.asserted,
            "declared": self.declared,
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

/// One declared dimension of an [`Assertion`].
#[derive(Debug, Clone, Copy, PartialEq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Declared {
    /// A literal size; zero and negative sizes are refused on the wire.
    Size(std::num::NonZeroU32),
    /// A per-row size: declared, but no plan-time fact.
    PerRow,
    /// Declared unknown (a canvas whose source node's size is itself unknown).
    Unknown,
}

/// A shape declaration at one op boundary.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Assertion {
    /// The rank `assert_shape(dims=[...])` pins, if given.
    #[serde(default)]
    pub ndim: Option<usize>,
    /// What each of dimensions 0..3 is declared as; `None` leaves it alone.
    pub dims: [Option<Declared>; 3],
    /// The user's `assert_shape` (`true`), or a canvas taken from another
    /// node's inferred size (`false`) — which decides who a divergence at
    /// execution is reported against.
    pub by_user: bool,
}

/// Apply `assertion` to `state`, refusing one the state contradicts: a rank
/// that is already known differently, a dimension the rank does not have, or
/// a size that disagrees with a known one. `after_op` names the op the
/// assertion follows, for the message.
pub(crate) fn assert_shape(
    mut state: State,
    assertion: &Assertion,
    after_op: Option<&str>,
) -> Result<State, String> {
    state.declared = true;
    if let Some(ndim) = assertion.ndim {
        if let Some(current) = state.ndim.filter(|c| *c != ndim) {
            return Err(format!(
                "assert_shape(dims=...) declares a rank-{ndim} output, but this pipeline is \
                 already known to produce rank {current}. Drop the assertion, or correct its \
                 length."
            ));
        }
        state.ndim = Some(ndim);
    }
    for (axis, declared) in assertion.dims.iter().enumerate() {
        let Some(declared) = declared else {
            continue;
        };
        let name = DIM_NAMES[axis];
        if *declared == Declared::Unknown {
            state.dims[axis] = None;
            state.asserted[axis] = false;
            continue;
        }
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
            Declared::Size(size) => Some(size.get() as usize),
            Declared::PerRow | Declared::Unknown => None,
        };
        if let (Some(known), Some(size)) = (state.dims[axis], size) {
            if known != size {
                let source = after_op.map_or("the source".to_string(), |op| {
                    format!("the {op}() before it")
                });
                return Err(format!(
                    "assert_shape({name}={size}) contradicts the {name} {known} that {source} \
                     already establishes. An assertion cannot change what the data is — \
                     remove it, or fix the value."
                ));
            }
        }
        state.dims[axis] = size;
        if assertion.by_user {
            state.asserted[axis] = true;
        }
    }
    Ok(state)
}

/// The state after one op. A hint is replaced where `dims[i]` is `Some`, and
/// left as it was where it is `None`: an op whose input rank is unknown says
/// nothing about H/W, which may hold a user's per-row assertion.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Step {
    pub domain: Domain,
    pub dtype: PlannedDType,
    pub ndim: Option<usize>,
    pub dims: [Option<Option<usize>>; 3],
}

/// Apply one op to `state`. See the module docs. `other` is a binary op's
/// other operand.
pub(crate) fn step(op_json: &str, state: &State, other: Option<&State>) -> Result<Step, String> {
    let op: crate::ops::TypedOp = serde_json::from_str(op_json).map_err(|e| e.to_string())?;
    let step = crate::resolve_op_from_json(op_json)?;

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

    let (out_domain, ndim) = fold(&step, state.domain, state.ndim);
    let dtype = match (&step, other) {
        (GraphStep::Binary { op, .. }, Some(other)) => binary_dtype(*op, state.dtype, other.dtype),
        (GraphStep::Binary { .. }, None) => {
            return Err(format!(
                "{}() combines two operands: it needs the other operand's state",
                op.name()
            ))
        }
        (_, Some(_)) => {
            return Err(format!(
                "{}() has one operand; only a binary op takes another's state",
                op.name()
            ))
        }
        (_, None) => single_input_dtype(&step, state.dtype),
    };

    // H/W: the op's own shape over the shapes it consumes, symbolically — a
    // per-row parameter or an unknown input size leaves its axis unknown.
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

    Ok(Step {
        domain: out_domain,
        dtype,
        ndim,
        dims,
    })
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
    /// This state with `step` applied. Every size is now the ops' inference,
    /// so none is the user's any more; whether a declaration reached the
    /// lineage stays.
    fn after(mut self, step: Step) -> State {
        self.domain = step.domain;
        self.dtype = step.dtype;
        self.ndim = step.ndim;
        for (dim, replaced) in self.dims.iter_mut().zip(step.dims) {
            if let Some(size) = replaced {
                *dim = size;
            }
        }
        self.asserted = [false; 3];
        self
    }

    /// A fresh state: nothing known about the sizes, nothing declared.
    pub(crate) fn new(domain: Domain, dtype: PlannedDType, ndim: Option<usize>) -> State {
        State {
            domain,
            dtype,
            ndim,
            dims: [None; 3],
            asserted: [false; 3],
            declared: false,
        }
    }
}

/// The state a source hands the first op.
///
/// Exhaustive over the formats: what each decodes to is a fact about the
/// format. A contour source decodes by rasterizing, so its state is the
/// `rasterize` op's over the contour domain — the source and the op cannot
/// publish different masks.
pub(crate) fn source_state(source: &crate::formats::source::Source) -> Result<State, String> {
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
            let op_json = serde_json::to_string(&rasterize).map_err(|e| e.to_string())?;
            let contours = State::new(Domain::Contour, PlannedDType::Unknown, None);
            let planned = step(&op_json, &contours, None)?;
            contours.after(planned)
        }
    })
}

/// Python entry point for [`source_state`]: validate a serialized source
/// against its typed format, and return the state it starts a pipeline in.
#[pyfunction]
pub(crate) fn plan_source(source_json: &str) -> PyResult<State> {
    let source: crate::formats::source::Source =
        serde_json::from_str(source_json).map_err(|e| py_value_error(e.to_string()))?;
    source_state(&source).map_err(py_value_error)
}

/// Python entry point for [`assert_shape`]: `assertion_json` is an
/// [`Assertion`]; `after_op` names the op it follows, if any.
#[pyfunction]
#[pyo3(signature = (state, assertion_json, after_op=None))]
pub(crate) fn plan_assert(
    state: State,
    assertion_json: &str,
    after_op: Option<&str>,
) -> PyResult<State> {
    let assertion: Assertion = serde_json::from_str(assertion_json).map_err(|e| {
        py_value_error(format!(
            "assert_shape(): each size must be a positive int or None ({e})"
        ))
    })?;
    assert_shape(state, &assertion, after_op).map_err(py_value_error)
}

/// Refuse a sink the planned `state` cannot give a Polars schema: a typed
/// `list`/`array` element with no known dtype, an `array` with no shape, a
/// `list` with no rank — unless the `source` resolves them from the input
/// column when the query is planned. `alias` names the output in a message.
pub(crate) fn check_sink(
    sink: &crate::formats::sink::Sink,
    state: &State,
    source: Option<&crate::formats::source::Source>,
    alias: Option<&str>,
) -> Result<(), String> {
    use crate::formats::sink::Sink;

    let from_column = source.is_some_and(|s| s.resolves_from_column());
    let where_ = alias.map_or(String::new(), |a| format!(" (alias '{a}')"));
    let name = sink.name();
    if sink.has_typed_elements() && !state.dtype.is_concrete() && !from_column {
        return Err(format!(
            "Element dtype is unknown for the '{name}' sink{where_}: the decoded dtype of an \
             image/blob source is only known at runtime, so a typed Polars '{name}' output \
             cannot be planned. Supply an explicit dtype — e.g. source(..., dtype=\"u16\") or \
             a .cast(\"u16\") before the sink."
        ));
    }
    match sink {
        Sink::Array(a) if a.shape.is_none() && state.dims.iter().any(Option::is_none) => {
            let missing: Vec<&str> = DIM_NAMES
                .iter()
                .zip(state.dims)
                .filter(|(_, d)| d.is_none())
                .map(|(n, _)| *n)
                .collect();
            let unknown = if missing.is_empty() {
                "the output rank".to_string()
            } else {
                missing.join(", ")
            };
            Err(format!(
                "an 'array' sink{where_} needs the full output shape at planning time, and this \
                 pipeline's is not known: {unknown}. Three ways to supply it:\n  \
                 .sink('array', shape=[8, 8, 3])   — always works; the shape belongs to the \
                 sink\n  .assert_shape(dims=[8, 8, 3])     — when you know it and the source \
                 does not (a list/array column's shape is only settled during execution)\n  \
                 .resize(height=8, width=8)        — supplies height and width only"
            ))
        }
        Sink::List(_) if state.ndim.is_none() && !from_column => Err(
            "Number of dimensions (ndim) is unknown for 'list' sink. This should not happen for \
             standard sources."
                .to_string(),
        ),
        _ => Ok(()),
    }
}

/// Python entry point for [`check_sink`]: validate a serialized sink against
/// its typed format, then against the output node's planned `state` and its
/// source (`None` for a source-less pipeline).
#[pyfunction]
#[pyo3(signature = (sink_json, state, source_json=None, alias=None))]
pub(crate) fn plan_sink(
    sink_json: &str,
    state: State,
    source_json: Option<&str>,
    alias: Option<&str>,
) -> PyResult<()> {
    let sink: crate::formats::sink::Sink =
        serde_json::from_str(sink_json).map_err(|e| py_value_error(e.to_string()))?;
    let source: Option<crate::formats::source::Source> = source_json
        .map(serde_json::from_str)
        .transpose()
        .map_err(|e| py_value_error(e.to_string()))?;
    check_sink(&sink, &state, source.as_ref(), alias).map_err(py_value_error)
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

/// Python entry point for [`step`]: the state after appending `op_json`;
/// `other` is a binary op's other operand's state.
#[pyfunction]
#[pyo3(signature = (op_json, state, other=None))]
pub(crate) fn plan_step(op_json: &str, state: State, other: Option<State>) -> PyResult<State> {
    let planned = step(op_json, &state, other.as_ref()).map_err(py_value_error)?;
    Ok(state.after(planned))
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

    fn run(op: serde_json::Value, s: &State, other: Option<&State>) -> Result<Step, String> {
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
            (out.domain.name(), out.dtype.as_str(), out.ndim),
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
    fn a_binary_op_needs_and_uses_the_other_operand() {
        let add = json!({"op": "divide", "other": "n0"});
        assert!(run(add.clone(), &image(), None).is_err());
        let out = run(add.clone(), &image(), Some(&image())).unwrap();
        assert_eq!(out.dtype.as_str(), "f32");
        // An operand of unknown rank leaves the broadcast shape unknown.
        let unranked = state("buffer", "u8", None, [None; 3]);
        let out = run(add, &image(), Some(&unranked)).unwrap();
        assert_eq!(out.dims[..2], [Some(None), Some(None)]);
        let err = run(json!({"op": "grayscale"}), &image(), Some(&image())).unwrap_err();
        assert!(err.contains("only a binary op"), "{err}");
    }

    #[test]
    fn a_binary_op_keeps_a_vector_operand_a_vector() {
        let hash = state("vector", "u8", Some(1), [Some(8), None, None]);
        let out = run(
            json!({"op": "bitwise_xor", "other": "n0"}),
            &hash,
            Some(&hash),
        )
        .unwrap();
        assert_eq!((out.domain, out.ndim), (Domain::Vector, Some(1)));
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
        assert_eq!(out.domain, Domain::Buffer);
        assert_eq!(out.dims, [Some(Some(8)), Some(Some(6)), Some(Some(1))]);
    }

    fn assertion(v: serde_json::Value) -> Assertion {
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn an_assertion_declares_what_is_unknown_and_marks_it_the_users() {
        let s = state("buffer", "u8", None, [None; 3]);
        let a =
            assertion(json!({"ndim": 3, "dims": [{"size": 8}, null, "per_row"], "by_user": true}));
        let out = assert_shape(s, &a, None).unwrap();
        assert_eq!(out.ndim, Some(3));
        assert_eq!(out.dims, [Some(8), None, None]);
        assert_eq!(out.asserted, [true, false, true]);
        assert!(out.declared);
        // The next op's inference is not the user's.
        let op = json!({"op": "invert"}).to_string();
        let planned = step(&op, &out, None).unwrap();
        assert_eq!(out.after(planned).asserted, [false; 3]);
    }

    #[test]
    fn an_assertion_the_state_contradicts_is_refused() {
        let image = state("buffer", "u8", Some(3), [Some(10), Some(20), Some(3)]);
        let a = |v| assertion(v);
        let err = assert_shape(
            image.clone(),
            &a(json!({"dims": [{"size": 11}, null, null], "by_user": true})),
            Some("resize"),
        )
        .unwrap_err();
        assert!(
            err.contains(
                "assert_shape(height=11) contradicts the height 10 that the resize() before it"
            ),
            "{err}"
        );
        let err = assert_shape(
            image.clone(),
            &a(json!({"ndim": 2, "dims": [null, null, null], "by_user": true})),
            None,
        )
        .unwrap_err();
        assert!(
            err.contains(
                "declares a rank-2 output, but this pipeline is already known to produce rank 3"
            ),
            "{err}"
        );
        let flat = state("buffer", "u8", Some(2), [None; 3]);
        let err = assert_shape(
            flat,
            &a(json!({"dims": [null, null, {"size": 3}], "by_user": true})),
            None,
        )
        .unwrap_err();
        assert!(
            err.contains("assert_shape(channels=...) names dimension 2, which a rank-2 output"),
            "{err}"
        );
        // Agreeing, or declaring a canvas unknown, is fine; a canvas is not the user's.
        let ok = assert_shape(
            image,
            &a(json!({"dims": [{"size": 10}, "unknown", null], "by_user": false})),
            None,
        )
        .unwrap();
        assert_eq!(
            (ok.dims, ok.asserted),
            ([Some(10), None, Some(3)], [false; 3])
        );
    }

    #[test]
    fn a_sink_the_plan_cannot_type_is_refused_unless_the_column_will() {
        let sink = |v: serde_json::Value| -> crate::formats::sink::Sink {
            serde_json::from_value(v).unwrap()
        };
        let source = |v: serde_json::Value| -> crate::formats::source::Source {
            serde_json::from_value(v).unwrap()
        };
        let image = state("buffer", "auto", Some(3), [None; 3]);
        let bytes = source(json!({"format": "image_bytes"}));
        let column = source(json!({"format": "list"}));
        let check = |k, s: &State, src| check_sink(&sink(k), s, Some(src), None);

        let err = check(json!({"format": "list"}), &image, &bytes).unwrap_err();
        assert!(
            err.contains("Element dtype is unknown for the 'list' sink"),
            "{err}"
        );
        assert!(check(json!({"format": "list"}), &image, &column).is_ok());
        assert!(check(json!({"format": "png"}), &image, &bytes).is_ok());

        let typed = state("buffer", "u8", Some(3), [Some(4), None, Some(3)]);
        let err = check(json!({"format": "array"}), &typed, &bytes).unwrap_err();
        assert!(err.contains("not known: width."), "{err}");
        assert!(check(
            json!({"format": "array", "shape": [4, 4, 3]}),
            &typed,
            &bytes
        )
        .is_ok());

        let unranked = state("buffer", "u8", None, [None; 3]);
        let err = check(json!({"format": "list"}), &unranked, &bytes).unwrap_err();
        assert!(err.contains("ndim) is unknown for 'list' sink"), "{err}");
        assert!(check(json!({"format": "list"}), &unranked, &column).is_ok());
    }

    #[test]
    fn each_source_format_plans_its_own_state() {
        let plan = |v: serde_json::Value| {
            let source = serde_json::from_value(v).unwrap();
            let s = source_state(&source).unwrap();
            (s.domain.name(), s.dtype.as_str(), s.ndim, s.dims)
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
        let contour = |size: serde_json::Value| json!({"format": "contour", "size": size, "fill_value": 255, "background": 0});
        assert_eq!(
            plan(contour(json!([10, 12]))),
            ("buffer", "u8", Some(3), [Some(10), Some(12), Some(1)])
        );
        // A canvas from another node is not a fact about this source.
        assert_eq!(plan(contour(json!("n0"))).3, [None, None, Some(1)]);
    }

    /// The H/W `step` plans from the op's symbolic shape: literal params and
    /// known sizes give exact dims, and a per-row param or unknown size leaves
    /// only the axes it decides unknown — no placeholder value is involved.
    #[test]
    fn shapes_are_planned_symbolically() {
        let hw = |op: serde_json::Value, s: &State| {
            let out = run(op, s, None).unwrap();
            (out.dims[0].unwrap(), out.dims[1].unwrap())
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
        // A contour input has no shape: the canvas is rasterize's own, or
        // unknown when it comes from another node.
        let contours = state("contour", "auto", None, [None; 3]);
        let rasterize = |size: serde_json::Value| json!({"op": "rasterize", "size": size, "fill_value": 255, "background": 0});
        assert_eq!(
            hw(rasterize(json!([10, 12])), &contours),
            (Some(10), Some(12))
        );
        assert_eq!(hw(rasterize(json!("n0")), &contours), (None, None));
    }
}
