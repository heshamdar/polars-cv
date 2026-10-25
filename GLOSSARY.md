# Glossary

Several words carry more than one meaning across the two Rust crates and the
Python package. This file fixes one meaning per word for prose (docs, comments,
commit messages, reviews), and lists the type names that use a word some other
way, so a reader can tell which meaning applies.

When you write prose, use the word in its glossary meaning. When you mean one of
the other senses, name the type (`ViewDto`, `ExecutionPlan`), not the bare word.

## The words

| Word | Means | Not |
|------|-------|-----|
| **view** | A zero-copy layout change: a new shape, strides or offset over the same bytes (`ViewOp`: transpose, flip, crop, reshape, …). | Any buffer, an op in general, or the wire format |
| **buffer** | A strided N-D array of one dtype, owned or borrowed (`ViewBuffer`). The `buffer` domain is a buffer. | A view (a buffer may be the *result* of one) |
| **op** | One operation a user appends to a `Pipeline`, as the typed catalogue names it (`resize`, `grayscale`, `extract_contours`). | A graph node, or an engine plan step |
| **step** | One of a graph node's ops after resolution, in the plugin's vocabulary (`GraphStep`). `TypedOp` is the step as it arrives on the wire (`GraphStep<Wire>`). | A step of an engine plan (that is a `PlanStep`) or a plan boundary (`plan::step`, a function) |
| **node** | A node of the pipeline graph: one input (a column decode or another node) and a linear run of ops (`GraphNode` in both Python and Rust). | A node of a `ViewExpr` (`ExprNode`) |
| **graph** | The DAG of nodes and named outputs one `.sink()` serializes (`PipelineGraph` in Python, `UnifiedGraph` once Rust parses it, `CompiledGraph` once cached). | A `ViewExpr` chain |
| **plan** | What the planner knows about a pipeline at every op boundary: domain, dtype and shape (`Plan`, its `PlanState`s, extended by `plan::step`). | How the engine will execute a buffer's ops (that is an `ExecutionPlan`) |
| **kernel** | The loop that computes an op's output over a buffer, in `view-buffer/src/kernels/<kind>.rs` (an op's `execute` entry in `ops/<family>.rs` calls it). | A convolution kernel's weights, a `FusedKernel` (a lowered chain of scalar ops) or an `ExecutionPlan` step |
| **domain** | The kind of value at one point of a pipeline: `buffer`, `contour`, `scalar` or `vector` (`Domain`). | A dtype |
| **sink** | How an output leaves the graph: its format, its planned Polars dtype and its encoding (`Sink` in `formats/sink.rs`; `SinkKind` in `graph/sink_kind.rs`; the dtype and per-row encode in `graph/sink.rs`). | |
| **source** | How a node's input column becomes its first buffer (`Source` in `formats/source.rs`; the per-row decode in `graph/source.rs`). | |

## Names that use a word another way

These are established names. Read them by their definition here, not by the word
in them.

| Name | Where | What it actually is |
|------|-------|---------------------|
| `ViewBuffer` | `view-buffer/src/core/buffer.rs` | Any strided buffer, not only a view. The crate is named after it. |
| `ViewExpr` | `view-buffer/src/expr.rs` | The engine's lazy builder: a buffer plus a chain of ops, which `.plan()` lowers to an `ExecutionPlan` and `.execute()` runs. |
| `ViewDto` | `view-buffer/src/ops/dto.rs` | Every single-buffer engine op (all five families), not only views. |
| `ExprNode` | `view-buffer/src/expr.rs` | A link of a `ViewExpr` chain, not a graph node. |
| `ExecutionPlan`, `PlanStep` | `view-buffer/src/execution/plan.rs` | The engine's lowered op sequence for one buffer, after fusion. Not the planner's plan. |
| `NodeOutput` | `polars-cv/src/graph/node_output.rs` | A graph node's value for one row, in any domain. |
| `GraphOp` | `polars-cv/src/ops/graph.rs` | The ops that exist only at graph level (they read other nodes or input columns), one family among the steps. |
| `LazyPipelineExpr` | `polars-cv/python/polars_cv/lazy.py` | A Python handle on a graph under construction (a node plus the graph it belongs to). Not a Polars expression until `.sink()`. |
| VIEW protocol, VIEW wire code | `view-buffer/src/protocol.rs`, `dtype_table!` | The binary blob format (`MAGIC_BYTES = b"VIEW"`) and its dtype codes. A serialized format: its name and codes are frozen. |

## The engine's scope

`view-buffer` is the kernel library the plugin executes with: the strided
buffer and its layout and element-wise ops, and the image and CV kernels
(codecs, filters, contour extraction, rasterization, histograms and perceptual
hashing). "Engine" means that library. It is not a general tensor framework, and
nothing outside this repository consumes it (`publish = false`).
