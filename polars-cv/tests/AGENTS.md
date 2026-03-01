# AGENTS.md — Tests (`polars-cv/tests/`)

> Read the [root AGENTS.md](../../AGENTS.md) first for project-wide context.
> Update this file when you add new test patterns, fixtures, or change testing conventions.

## Purpose

All Python tests for polars-cv. Tests use **pytest** exclusively. Coverage includes the Python API, pipeline builder, schema inference, plugin execution, and reference correctness against NumPy/OpenCV.

## Running Tests

```bash
cd polars-cv
uv run pytest tests/                              # all tests (plugin must be built)
uv run pytest tests/ -k "not plugin_required"      # builder/schema tests only
uv run pytest tests/reference/ -v                  # reference tests
python scripts/test_multiple_python.py --all       # multi-Python (3.10-3.13)
```

The compiled plugin (`.so`/`.pyd`) must exist at `python/polars_cv/_lib.abi3.so`. Build with `maturin develop --release`.

## Test Categories

### Unit Tests (no plugin required)

| File | What it tests |
|------|--------------|
| `test_pipeline_builder.py` | Pipeline construction, domain tracking, operation chaining |
| `test_lazy_schema.py` | LazyPipelineExpr schema inference |
| `test_dtype_contracts.py` | Operation dtype/ndim contracts in `OPERATION_CONTRACTS` |
| `test_serialization.py` | Pipeline JSON serialization |
| `test_geometry_schemas.py` | Geometry schema definitions and validation |

### Integration Tests (plugin required)

| File | What it tests |
|------|--------------|
| `test_integration.py` | Basic pipeline execution through Polars |
| `test_expression_params.py` | Dynamic parameters via `pl.Expr` |
| `test_zero_copy_sources.py` | List/array source decoding, zero-copy paths |
| `test_zero_copy_output.py` | Numpy/torch sink output, zero-copy extraction |
| `test_sink_typing.py` | Sink format dtype inference and validation |
| `test_multi_output.py` | Multi-output pipelines with aliases |
| `test_multi_upstream.py` | Multi-source graph execution |
| `test_source_types.py` | Various source formats (image_bytes, blob, raw, file_path) |
| `test_contour_plugin.py` | Contour namespace plugin operations |
| `test_contour_source.py` | Contour-to-mask rasterization |
| `test_error_handling.py` | Error messages and edge cases |
| `test_feature_extraction.py` | Extract shape, histogram, perceptual hash |
| `test_lazy_composition.py` | LazyPipelineExpr composition patterns |
| `test_schema_inference.py` | Planning-time vs execution-time schema consistency |
| `test_unified_graph.py` | Graph construction and execution |
| `test_cse_optimization.py` | Common subexpression elimination |
| `test_precompile.py` | Pipeline precompilation |
| `test_mask_metrics.py` | mask_iou, mask_dice |
| `test_hash_comparison.py` | hamming_distance, hash_similarity |
| `test_perceptual_hash.py` | Perceptual hashing |
| `test_http_sources.py` | HTTP/HTTPS file sources |
| `test_statistical_reductions.py` | reduce_sum, reduce_mean, etc. |
| `test_typed_nodes.py` | Typed list/array node outputs |
| `test_numpy_helpers.py` | `numpy_from_struct` utility |
| `test_detection_table.py` | DetectionTable construction, schema validation, accessors |
| `test_matchers.py` | PreMatchedAdapter protocol compliance, class/n_gts support |
| `test_precision_recall.py` | PR curve, AP, mAP, precision/recall/f1 at threshold, confusion |
| `test_bbox_matching.py` | Rust bbox_pairwise_iou and bbox_match_detections |
| `test_bootstrap_vectorized.py` | Sequential and vectorized bootstrap, BootstrapResult |
| `test_image_metadata.py` | Header-only metadata extraction |
| `test_on_error.py` | `on_error="null"` graceful error handling |
| `test_display.py` | `show_images()` display utility |
| `test_tiff_integration.py` | TIFF format integration |
| `test_alpha_channel.py` | Alpha channel decode/encode, AlphaMode contracts, planning-time inference |

### Reference Tests (`tests/reference/`)

Compare polars-cv output against NumPy/OpenCV ground truth — these are the **correctness guarantees**.

| File | What it tests |
|------|--------------|
| `test_binary_ops_ref.py` | Binary ops vs NumPy |
| `test_contour_ops_ref.py` | Contour operations vs OpenCV/Shapely |
| `test_extract_ref.py` | Contour extraction vs OpenCV |
| `test_histogram_ref.py` | Histogram computation vs NumPy |
| `test_pairwise_ref.py` | Pairwise contour ops |
| `test_perceptual_hash_ref.py` | Perceptual hashing vs imagehash |
| `test_rasterize_ref.py` | Rasterization vs OpenCV/PIL |
| `test_reductions_ref.py` | Reductions vs NumPy |
| `test_phase1_ref.py` | Channel, intensity, padding ops vs NumPy/PIL |
| `test_color_ref.py` | Color space conversions vs OpenCV/NumPy |
| `test_morphology_ref.py` | Morphological ops (erode, dilate, open, close, gradient) vs OpenCV |

Files with `_gaps` in the name (e.g., `test_binary_ops_gaps.py`) are regular tests despite the name — the gaps they tested have been filled.

## Conventions

### Test Structure

- **Class-based**: `class TestSomething:` with methods
- **Docstrings**: All test classes and methods should have docstrings
- **Type annotations**: All fixtures and test methods should have return type annotations
- **Contour native sink**: `extract_contours().sink("native")` returns `List[Struct]`, not a single struct. Access via `.list.get(0).struct.field("exterior")`.

### Plugin Requirement

```python
from tests.conftest import plugin_required

@plugin_required
class TestMyFeature:
    """Tests that need the compiled Rust plugin."""
    ...
```

Tests that only exercise the Python layer do NOT need `@plugin_required`.

### Shared Fixtures (`conftest.py`)

| Fixture | Purpose |
|---------|---------|
| `create_test_png` | Factory: create PNG bytes for given width, height, color |
| `encode_png` | Encode a numpy array as PNG bytes |
| `sample_image_bytes` | Minimal 1x1 red PNG (no PIL dependency) |
| `plugin_required` | Skip if compiled plugin not available |

Reference tests have additional fixtures in `reference/conftest.py` (session-scoped images, contour data).

### Writing a New Test

```python
"""Tests for my plugin feature."""
from __future__ import annotations
from typing import TYPE_CHECKING, Callable
import polars as pl
import pytest
from polars_cv import Pipeline
from tests.conftest import plugin_required

if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest

@plugin_required
class TestMyFeature:
    """Tests that execute through the Rust plugin."""

    def test_end_to_end(self, create_test_png: Callable) -> None:
        """Verify end-to-end execution."""
        png_bytes = create_test_png(100, 100)
        df = pl.DataFrame({"image": [png_bytes]})
        pipe = Pipeline().source("image_bytes").my_op(param=42)
        result = df.with_columns(
            output=pl.col("image").cv.pipe(pipe).sink("numpy")
        )
        assert result["output"].dtype == pl.Struct(...)
```

### Known Inconsistencies (Fix When Touching)

1. **Duplicate `_plugin_available()` / `plugin_required`**: Some test files redefine these instead of importing from `conftest.py`. Use the shared one.
2. **Duplicate fixture patterns**: Many files define their own PNG creation fixtures instead of using `create_test_png` / `encode_png` from conftest.
3. **Inconsistent PIL/no-PIL approaches**: The conftest approach (PIL with graceful skip) is preferred.
