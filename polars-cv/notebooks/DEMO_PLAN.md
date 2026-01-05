# polars-cv Comprehensive Demo Notebook

## Status: ✅ Complete

The demo is implemented in `demo_comprehensive.py` using Jupytext percent format.

## How to Use

### Option 1: Convert to Jupyter Notebook (Recommended)

```bash
# Install jupytext if not already installed
pip install jupytext

# Convert to notebook
jupytext --to notebook demo_comprehensive.py

# Open the notebook
jupyter notebook demo_comprehensive.ipynb
```

### Option 2: Run directly as Python script

```bash
cd polars-cv/notebooks
uv run python demo_comprehensive.py
```

### Option 3: Use with VS Code

VS Code with the Jupyter extension can open `.py` files with percent markers directly.

---

## Demo Contents

### 1. Setup & Imports
- Import polars-cv and dependencies
- Create helper functions for displaying images
- Explain the plugin's core value proposition

### 2. Basic Pipeline Operations
- Decoding images from bytes
- Resize with different filter types (nearest, bilinear, lanczos3)
- Grayscale conversion, threshold, blur
- Cropping and flipping
- Chained operations with ML preprocessing example

### 3. DType Promotion & Normalization
- MinMax and ZScore normalization
- Automatic dtype promotion (u8 → f32)
- Scale and clamp operations

### 4. Dynamic Parameters with Expressions
- Using `pl.col()` for per-row parameters
- Dynamic resize based on metadata columns
- Dynamic cropping based on bounding boxes

### 5. Geometry Operations
- Contour schemas (CONTOUR_SCHEMA, POINT_SCHEMA, BBOX_SCHEMA)
- Geometric measures (area, perimeter, centroid, bbox, convexity)
- Rasterizing contours to masks

### 6. Lazy Pipeline Composition
- `cv.pipe()` vs `cv.pipeline()` modes
- Composing multiple pipelines
- Apply mask operations
- Fused execution

### 7. Multi-Output Pipelines
- Using `.alias()` for checkpoints
- Multi-output sink with dict format
- Extracting outputs from Struct column

### 8. ML Workflow: IoU Calculation
- Generate fake heatmap predictions
- Process ground truth contours
- Compute IoU and Dice coefficients
- Visualize prediction vs ground truth overlays

### 9. Lazy Scalability Demo
- Generate synthetic dataset on disk
- Lazy scan and processing with Polars
- Sink to parquet with processed results

### 10. PyTorch Integration
- Sink to torch format
- Convert to PyTorch tensors
- Create DataLoader-compatible dataset

### 11. Conclusion
- Summary of all capabilities

---

## Issue Log

Issues discovered during notebook creation that need to be addressed:

### 🟢 Fixed Issues

1. **Contour geometry operations now fully implemented in Rust** (Fixed 2025-12-30)
   - All 18 contour operations are now implemented: area, perimeter, centroid, bounding_box, winding, is_convex, convex_hull, flip, translate, scale, simplify, normalize, to_absolute, ensure_winding, contains_point, iou, dice, hausdorff_distance
   - Added 33 comprehensive integration tests in `tests/test_contour_plugin.py`

2. **source("contour") pipeline source now implemented** (Fixed 2025-12-30)
   - Contours can be used as pipeline sources with automatic rasterization
   - Usage: `Pipeline().source("contour", width=200, height=100).sink("numpy")`
   - Supports explicit dimensions or dynamic dimensions via column expressions
   - 11 tests added in `tests/test_contour_source.py`

### 🔴 Confirmed Bugs / Missing Implementations

1. **Multi-output pipeline not implemented in Rust** (BLOCKER for Section 7)
   - `Pipeline.sink({"alias": "format", ...})` generates JSON with `multi_sink` field but Rust expects `sink`
   - Error: `Failed to parse pipeline: missing field 'sink'`
   - **Workaround**: Run separate pipelines for each output (less efficient)

2. **NumPy sink includes metadata header** (Helper added)
   - The `numpy` sink includes a metadata header with dtype and shape information
   - **Fixed**: Use `numpy_from_bytes()` helper from `polars_cv` to convert output to numpy array

### 🟡 Features Not Tested (Need Rust Implementation Verification)

1. **Binary operations between LazyPipelineExpr**: The `add`, `subtract`, `multiply`, `divide` operations are defined in Python but need Rust backend verification.

2. **`extract_contours` operation**: Defined in Pipeline but not tested.

3. **Rasterize operation**: Now working via `source("contour", width=..., height=...)` which rasterizes contours in-pipeline.

### 🟢 Confirmed Working Features

- ✅ Basic pipeline (source → resize → sink)
- ✅ Grayscale conversion
- ✅ Threshold (binary)
- ✅ Blur (Gaussian)
- ✅ Crop
- ✅ Flip (horizontal/vertical)
- ✅ Chained operations
- ✅ Dynamic parameters with `pl.col()`
- ✅ Lazy composition with `.cv.pipe()` and `.sink()`
- ✅ Normalization (minmax, zscore) - with header skip
- ✅ Filter types (nearest, bilinear, lanczos3)

### Feature Requests

1. **Native IoU computation on masks**: Having a pipeline operation for pixel-wise IoU between two binary masks would simplify the ML workflow.

2. **Direct NumPy array output**: Currently arrays come as bytes with header. A direct-to-array sink or documented header format would be convenient.

3. **Shape information in output**: The numpy/torch sinks output raw bytes without shape metadata in a documented format.


