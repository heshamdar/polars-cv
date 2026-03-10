# AGENTS.md — polars-cv

A **Polars plugin for high-performance vision and array operations**. Users define pipelines that operate directly on DataFrame columns.

```python
import polars as pl
from polars_cv import Pipeline

pipe = Pipeline().source("image_bytes").resize(height=224, width=224).grayscale()
df.with_columns(processed=pl.col("image").cv.pipe(pipe).sink("numpy"))
```

This is a **pre-release, largely AI-developed project**. Fix inconsistencies when you encounter them.

## Architecture

```
Python: polars_cv          — Pipeline builder, expression namespaces, graph construction, validation
Rust: polars-cv (plugin)   — Graph execution, source/sink encoding, op dispatch, cloud I/O
Rust: view-buffer (engine) — ViewBuffer, ViewExpr, stride-aware ops, kernel fusion, zero-copy interop
```

Data flow: Python Pipeline spec -> JSON graph -> `register_plugin_function("vb_graph", ...)` -> Rust topological execution -> per-row: decode source -> apply ops -> encode sink -> Polars Series.

## Guiding Principles

1. **Zero-copy by default** — prefer view operations over allocation
2. **Lazy evaluation** — Python builds the plan, Rust executes it
3. **Explicit over implicit** — no hidden assumptions about dtype, shape, or domain
4. **Strong planning-time contracts** — `collect_schema()` must match `collect()` output
5. **Composition with low coupling** — operations are independent, composable units
6. **Leverage existing implementations** — leverage existing implementations for image, array, and geometry operations where possible.

## Build and Development

```bash
cd polars-cv
maturin develop --release        # Build Rust plugin into .venv
uv run pytest tests/             # Run tests
uv run ruff check python/ tests/ # Lint Python
cargo clippy --workspace         # Lint Rust
```

## Skills (On-Demand Context)

Detailed architecture, conventions, and workflows are available as Cursor Skills in `.cursor/skills/`. The agent loads these automatically when relevant. Available skills:

| Skill | When to use |
|-------|-------------|
| `adding-new-operation` | Adding a new pipeline operation (full cross-layer workflow) |
| `python-api-architecture` | Working on Python API code (pipeline, lazy, graph, types) |
| `rust-plugin-architecture` | Working on Rust plugin code (polars-cv/src/) |
| `view-buffer-engine` | Working on the tensor engine (view-buffer/src/) |
| `geometry-subsystem` | Working with geometry types, contours, bboxes, points |
| `detection-metrics` | Working with detection metrics, matchers, curves |
| `testing-polars-cv` | Writing or modifying tests |
| `running-benchmarks` | Running or extending benchmarks |

## Known Issues

- **Tiling (no-op):** `configure_tiling` / `get_tiling_config` are exposed but non-functional.
- **`cloud_options` not serialized:** `SourceSpec.to_dict()` omits `cloud_options`.
- **`ANNOTATED_POINT_SCHEMA` not re-exported:** Missing from `polars_cv/__init__.py` `__all__`.
- **`PipelineSpec` consolidation:** Serde types in `pipeline.rs` could be consolidated. `PipelineSpec` wrapper may be removable.
- **Inconsistent test fixtures:** Some test files redefine shared fixtures instead of importing from `conftest.py`.
