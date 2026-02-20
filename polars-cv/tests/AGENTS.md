# AGENTS.md — Tests (`polars-cv/tests/`)

> Read the [root AGENTS.md](../../AGENTS.md) first for project-wide context.
> Update this file when you add new test patterns, fixtures, or change testing conventions.

## Purpose

All Python tests for polars-cv live here. Tests use **pytest** exclusively (no unittest). Tests cover the Python API, pipeline builder, schema inference, plugin execution, and reference correctness against NumPy/OpenCV.

## Running Tests

```bash
cd polars-cv

# Run all tests (requires compiled plugin for @plugin_required tests)
uv run pytest tests/

# Run without plugin (builder/schema tests only)
uv run pytest tests/ -k "not plugin_required"

# Run specific test file
uv run pytest tests/test_pipeline_builder.py -v

# Run reference tests
uv run pytest tests/reference/ -v

# Multi-Python testing (3.10-3.13)
python scripts/test_multiple_python.py --all
```

The compiled plugin (`.so`/`.pyd`) must exist at `python/polars_cv/_lib.abi3.so` for plugin tests. Build with `maturin develop --release`.

## Test Categories

### Unit Tests (no plugin required)

Test the Python layer in isolation — pipeline building, type tracking, schema inference, validation.

| File | What it tests |
|------|--------------|
| `test_pipeline_builder.py` | Pipeline construction, domain tracking, operation chaining |
| `test_lazy_schema.py` | LazyPipelineExpr schema inference |
| `test_dtype_contracts.py` | Operation dtype/ndim contracts in `OPERATION_CONTRACTS` |
| `test_serialization.py` | Pipeline JSON serialization |
| `test_geometry_schemas.py` | Geometry schema definitions and validation |

### Integration Tests (plugin required)

Test end-to-end execution through the Rust plugin.

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

### Reference Tests (`tests/reference/`)

Compare polars-cv output against NumPy/OpenCV ground truth. These are the **correctness guarantees**.

| File | What it tests |
|------|--------------|
| `test_binary_ops_ref.py` | Binary ops (add, blend, bitwise) vs NumPy |
| `test_contour_ops_ref.py` | Contour operations vs OpenCV/Shapely |
| `test_extract_ref.py` | Contour extraction vs OpenCV |
| `test_histogram_ref.py` | Histogram computation vs NumPy |
| `test_pairwise_ref.py` | Pairwise contour ops |
| `test_perceptual_hash_ref.py` | Perceptual hashing vs imagehash |
| `test_rasterize_ref.py` | Rasterization vs OpenCV/PIL |
| `test_reductions_ref.py` | Reductions vs NumPy |

### Gap Tests (Naming Artifact)

Files with `_gaps` in the name (e.g., `test_binary_ops_gaps.py`, `test_resize_gaps.py`) were created to test missing functionality. **The gaps have been filled** — the naming is now misleading. These are regular tests despite their names. Consider renaming them when touching these files.

## Conventions

### Test Structure

- **Class-based**: Most tests use `class TestSomething:` with methods
- **Docstrings**: All test classes and methods should have docstrings
- **Type annotations**: All fixtures and test methods should have return type annotations
- **Section comments**: Files use `# ==== Section Name ====` separators
- **Contour native sink shape**: `extract_contours().sink("native")` returns `List[Struct]` (a list of contours), not a single struct. Assertions should access contour members via list operations (for example, `.list.get(0).struct.field("exterior")`).

### Plugin Requirement

```python
from tests.conftest import plugin_required

@plugin_required
class TestMyFeature:
    """Tests that need the compiled Rust plugin."""
    ...
```

Tests that only exercise the Python layer (pipeline building, schema inference) do NOT need `@plugin_required`.

### Fixtures

#### Shared Fixtures (`conftest.py`)

| Fixture | Type | Purpose |
|---------|------|---------|
| `create_test_png` | `Callable` | Factory: create PNG bytes for given width, height, color |
| `encode_png` | `Callable` | Encode a numpy array as PNG bytes |
| `sample_image_bytes` | `bytes` | Minimal 1x1 red PNG (no PIL dependency) |
| `plugin_required` | marker | Skip if compiled plugin not available |

#### Reference Fixtures (`reference/conftest.py`)

| Fixture | Scope | Purpose |
|---------|-------|---------|
| `test_image_rgb` | session | 256x256x3 RGB numpy array |
| `test_image_gray` | session | 256x256 grayscale numpy array |
| `sample_images` | function | Two 100x100 RGB images for binary ops |
| `binary_mask` | function | 100x100 mask with center square |
| `standard_contours` | function | Square, triangle, circle contours |
| `simple_contour` | function | Single square contour |
| Various mask/contour fixtures | function | Specialized test data |

### Known Inconsistencies (Fix When Touching)

1. **Duplicate `_plugin_available()` / `plugin_required`**: Some test files (e.g., `test_error_handling.py`, `reference/test_binary_ops_ref.py`) redefine these instead of importing from `conftest.py`. Use the shared one.

2. **Duplicate fixture patterns**: Many files define their own PNG creation fixtures (`simple_rgb_bytes`, `rgb_image_bytes`, inline PNG byte arrays) instead of using `create_test_png` / `encode_png` from conftest. Consolidate when touching these files.

3. **Inconsistent PIL/no-PIL approaches**: Some tests use PIL to create test images, others use hand-crafted minimal PNG bytes. The conftest approach (PIL with graceful skip) is preferred.

## Writing New Tests

### For a new Python-only feature:

```python
"""Tests for my new feature."""
from __future__ import annotations
from typing import TYPE_CHECKING
import pytest
from polars_cv import Pipeline

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture

class TestMyFeature:
    """Tests for my feature."""

    def test_basic_behavior(self) -> None:
        """Verify basic behavior."""
        pipe = Pipeline().source("image_bytes").my_op(param=42)
        assert pipe.current_domain() == "buffer"
```

### For a new plugin feature:

```python
"""Tests for my plugin feature."""
from __future__ import annotations
from typing import TYPE_CHECKING, Callable
import numpy as np
import polars as pl
import pytest
from polars_cv import Pipeline
from tests.conftest import plugin_required

if TYPE_CHECKING:
    pass

@plugin_required
class TestMyPluginFeature:
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

### For a reference test:

```python
"""Reference tests for my_op against NumPy/OpenCV."""
# Place in tests/reference/test_my_op_ref.py
```

## Outer tests/ Directory

There is also a `tests/` directory at the **workspace root** (outside `polars-cv/`) containing `test_tiff_integration.py`. This is an inconsistency — it should ideally be consolidated into `polars-cv/tests/`. Be aware of its existence when running tests from different working directories.
