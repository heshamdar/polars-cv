# Codebase Review: polars-cv

**Date:** 2026-02-06
**Scope:** Full codebase review of polars-cv (v0.6.0) — Rust core, Polars plugin, Python API, CI/CD

---

## Executive Summary

polars-cv is a well-structured, performance-focused Polars plugin for computer vision with ~30k lines of code across Rust and Python. The architecture is sound: a low-level tensor library (`view-buffer`) with zero-copy semantics, a Polars plugin layer for expression-based execution, and a Python API with lazy pipeline composition.

This review identified **8 high-severity**, **15 medium-severity**, and **12 low-severity** issues across security, correctness, performance, and configuration categories.

---

## 1. Security Issues

### 1.1 [HIGH] File Path Source Allows Arbitrary Local File Reads / SSRF

**Location:** `polars-cv/src/graph/types.rs:294-314`

The `file_path` source format reads from paths provided in the data column without any sanitization:

```rust
let bytes = if path.starts_with("s3://") || ... || path.starts_with("https://") {
    crate::cloud::read_file(path, None)
} else {
    std::fs::read(path)
};
```

If data comes from untrusted input, an attacker can read arbitrary local files (`/etc/shadow`) or trigger SSRF via internal network URLs. **Recommendation:** Add path allowlisting/sandboxing, or at minimum document this as a trusted-input-only feature.

### 1.2 [MEDIUM] Shape Product Overflow in Blob Deserialization

**Location:** `polars-cv/src/graph/decode.rs:156`

```rust
let num_elements: usize = shape.iter().product();
```

A crafted blob with shape `[1_000_000, 1_000_000, 1_000_000]` would overflow `usize` silently in release mode, creating a buffer with invalid metadata. Subsequent reads could access out-of-bounds memory. **Recommendation:** Use `checked_mul` and validate against a maximum total size.

### 1.3 [MEDIUM] Cloud Credentials Serialized in Plain Text

**Location:** `polars-cv/python/polars_cv/_types.py:379-403`

`CloudOptions.to_dict()` serializes AWS secret keys, GCS service account paths, and Azure access keys as plain strings. These could end up in logs or error messages. **Recommendation:** Mask sensitive fields in `__repr__` and `to_dict()`.

### 1.4 [LOW] Unvalidated dtype String from Data

**Location:** `polars-cv/python/polars_cv/__init__.py:133`

`np.dtype(dtype_str)` is called with a string extracted from a Polars struct field. If struct contents come from an untrusted source, arbitrary dtype strings can be passed. **Recommendation:** Validate against an allowlist of expected dtypes.

---

## 2. Correctness Issues

### 2.1 [HIGH] Shallow Clone Shares Mutable ShapeHints

**Location:** `polars-cv/python/polars_cv/pipeline.py:247-248`

```python
new._shape_hints = self._shape_hints
```

`ShapeHints` is a mutable dataclass. After cloning, both old and new Pipeline share the same instance. Branching a pipeline into two paths with different shape-modifying operations will cause cross-mutation. **Recommendation:** Deep-copy `_shape_hints` in `_clone()`.

### 2.2 [HIGH] `SourceSpec.__eq__` Omits `cloud_options`

**Location:** `polars-cv/python/polars_cv/_types.py:427-440`

Two `SourceSpec` instances with different cloud credentials are considered equal. This breaks CSE (Common Subexpression Elimination) optimization — nodes that should fetch from different cloud accounts could be incorrectly merged. **Recommendation:** Include `cloud_options` in `__eq__` and `__hash__`.

### 2.3 [MEDIUM] `OutputSpec.to_dict` Drops WebP Quality

**Location:** `polars-cv/python/polars_cv/_types.py:583`

```python
if self.format == SinkFormat.JPEG:  # Missing SinkFormat.WEBP
    result["quality"] = self.quality
```

Multi-output WebP sinks silently drop the quality parameter. **Recommendation:** Add `SinkFormat.WEBP` to the condition.

### 2.4 [MEDIUM] Blob Deserialization Discards Stride Information

**Location:** `polars-cv/src/graph/decode.rs:123-172`

The blob decoder reads the VIEW protocol header but discards stride data, always assuming contiguous layout. Round-tripping a non-contiguous buffer through blob serialization silently corrupts its layout. **Recommendation:** Either preserve strides in the decoder or document that blob format always materializes to contiguous.

### 2.5 [MEDIUM] Topological Sort Does Not Detect Cycles

**Location:** `polars-cv/src/graph/types.rs:114-139`

`compute_topological_order` uses a simple DFS with a `visited` set but no "processing" state. Cyclic graphs produce silently incorrect execution order rather than a clear error. **Recommendation:** Add proper cycle detection with a tri-color DFS.

### 2.6 [MEDIUM] `sink()` Return Type Annotation Lies

**Location:** `polars-cv/python/polars_cv/lazy.py:146-226`

Annotated as `-> pl.Expr` but can return `PipelineGraph` when `return_expr=False`. **Recommendation:** Use `pl.Expr | PipelineGraph` or `@overload`.

### 2.7 [MEDIUM] Expression Arguments Silently Dropped in Geometry Namespaces

**Location:** `polars-cv/python/polars_cv/geometry/contours.py`, `points.py`

```python
kwargs = {
    "ref_width": ref_width if isinstance(ref_width, int) else None,
}
```

If a user passes `pl.col("width")`, the kwarg becomes `None` with no error or warning. **Recommendation:** Either support expressions or narrow the type hint and raise on non-int input.

### 2.8 [MEDIUM] Unchecked Slicing in `slice_typed_data`

**Location:** `polars-cv/src/graph/encode.rs:681-694`

```rust
TypedBufferData::U8(vals) => TypedBufferData::U8(vals[start..end].to_vec()),
```

No bounds validation before slicing. If shape doesn't match data length, this panics instead of returning an error. **Recommendation:** Add a bounds check before slicing.

### 2.9 [LOW] Triplicated Return in `Pipeline.__repr__`

**Location:** `polars-cv/python/polars_cv/pipeline.py:2456-2458`

The return statement is copy-pasted three times. Only the first executes. **Recommendation:** Remove the duplicate lines.

### 2.10 [LOW] Deprecated `pl.Utf8` Usage

**Location:** `polars-cv/python/polars_cv/geometry/schemas.py:48`

`pl.Utf8` is deprecated in favor of `pl.String`. **Recommendation:** Replace with `pl.String`.

### 2.11 [LOW] Inconsistent Default Dtype Fallbacks

**Location:** `polars-cv/src/graph/decode.rs:546` vs `encode.rs:513`

Unknown dtype strings default to `Float64` in the decoder but `u8` in the encoder. **Recommendation:** Make these consistent or raise errors for unknown dtypes.

### 2.12 [LOW] `ViewBuffer::cast` Has Incomplete Type Pairs

**Location:** `view-buffer/src/core/buffer.rs:1577-1588`

```rust
_ => unimplemented!("Cast pair {:?} -> {:?} not implemented", ...)
```

Only 4 of 90 possible cast pairs are implemented. The more complete `cast_to` method exists but they are separate APIs. **Recommendation:** Unify these or clearly document which to use.

---

## 3. Performance Issues

### 3.1 [MEDIUM] "Zero-Copy" Paths Always Copy

**Location:** `polars-cv/src/graph/decode.rs:69-80, 101`

`get_binary_row_buffer` calls `bytes.to_vec()` (copy #1), then `decode_blob_zero_copy` calls `buffer.as_slice()[..].to_vec()` (copy #2). Despite the function names, the "zero-copy" blob path performs **two full copies**. **Recommendation:** Rename the functions honestly, or implement actual zero-copy using buffer slicing.

### 3.2 [MEDIUM] `is_row_null` Allocates Full Boolean Array Per Row

**Location:** `polars-cv/src/graph/decode.rs:51`

```rust
series.is_null().get(row_idx).unwrap_or(true)
```

Creates a full `BooleanChunked` for the entire series to check one row. Called per-row in a loop. For N rows, this allocates N boolean arrays. **Recommendation:** Access the validity bitmap directly via `series.get(row_idx)` or arrow's null bitmap.

### 3.3 [MEDIUM] `numpy_from_struct(copy=False)` Always Copies

**Location:** `polars-cv/python/polars_cv/__init__.py:140-155`

```python
buffer=bytes(data),  # <-- always copies
```

The `copy=False` path calls `bytes(data)` which always copies. The API promise of zero-copy is misleading. **Recommendation:** Use `np.frombuffer` on a `memoryview` or buffer-protocol object, or document the limitation.

### 3.4 [LOW] Per-Row HashMap Allocation in Execute Loop

**Location:** `polars-cv/src/graph/types.rs:193`

Each row allocates a new `HashMap<String, NodeOutput>`. For millions of rows, the overhead is significant. **Recommendation:** Allocate once outside the loop and `clear()` per row.

### 3.5 [LOW] Redundant `to_contiguous()` Calls in Encode Path

**Location:** `polars-cv/src/graph/encode.rs:728-738`

`buf.to_contiguous()` is called, then `TypedBufferData::from_buffer` calls it again internally. The second call is likely a no-op (it checks `is_contiguous()` first), but it's unnecessary overhead. **Recommendation:** Pass the already-contiguous buffer directly.

---

## 4. API Design Issues

### 4.1 [MEDIUM] Domain Validation Inconsistency

**Location:** `polars-cv/python/polars_cv/pipeline.py`

Image operations validate domain (e.g., `self._validate_domain("buffer")`), but compute operations like `cast`, `scale`, `clamp`, and `relu` do not. They can be applied to any domain (including contour or scalar) without a build-time error. **Recommendation:** Add domain validation to all operations.

### 4.2 [MEDIUM] `normalize` Docstring Incomplete

**Location:** `polars-cv/python/polars_cv/pipeline.py:826-842`

Describes only `"minmax"` and `"zscore"` methods, omitting `"preset"` (ImageNet-style normalization). The `mean` and `std` parameters are undocumented. **Recommendation:** Document all three methods and their parameters.

### 4.3 [LOW] Hard Dependency on Optional Visualization Libraries

**Location:** `polars-cv/python/polars_cv/_graph_viz.py:7-9`

`networkx` and `graphviz` are imported unconditionally at module level. If not installed, importing `polars_cv` will fail even if the user never calls `show_graph()`. **Recommendation:** Defer imports to the `show_graph` call site.

### 4.4 [LOW] `assert` Used for Runtime Validation

**Location:** `polars-cv/python/polars_cv/expressions.py:78`

```python
assert pipe._sink is not None
```

Stripped by `-O` Python runs. **Recommendation:** Use `ValueError` or `RuntimeError`.

---

## 5. CI/CD and Configuration Issues

### 5.1 [HIGH] No Wheel Smoke Test Before PyPI Publish

**Location:** `.github/workflows/publish.yml`

Wheels are built and uploaded directly to PyPI without any validation step. A corrupt wheel could be published. **Recommendation:** Add `pip install dist/*.whl && python -c "import polars_cv"` before upload.

### 5.2 [HIGH] Python Version / ABI3 / requires-python Mismatch

**Location:** `polars-cv/pyproject.toml`, `polars-cv/Cargo.toml`

- ABI3 flag targets Python >= 3.9
- `requires-python` declares >= 3.10
- Classifiers list 3.9-3.12 (missing 3.13)
- CI only tests 3.13

This is a four-way disagreement. **Recommendation:** Align all declarations and test the minimum supported version.

### 5.3 [HIGH] No Security Auditing in CI

**Location:** `.github/workflows/ci.yml`

No `cargo audit`, `cargo deny`, or equivalent. For a library pulling in `reqwest`, `object_store`, `tokio`, and `image`, vulnerability scanning is important. **Recommendation:** Add `cargo audit` step.

### 5.4 [HIGH] Missing Platform Wheels

**Location:** `.github/workflows/publish.yml`

No wheels for `x86_64-apple-darwin` (Intel Macs) or any Windows target. Users on these platforms must compile from source. **Recommendation:** Add build targets for Intel Mac and Windows.

### 5.5 [MEDIUM] Pre-commit Config in Wrong Directory

**Location:** `polars-cv/.pre-commit-config.yaml`

Lives inside the sub-package, not at the repository root. Pre-commit won't find it automatically. **Recommendation:** Move to repository root or add a symlink.

### 5.6 [MEDIUM] No Version-Sync Automation

Version `0.6.0` appears in three files that must stay in sync manually:
- `polars-cv/Cargo.toml`
- `polars-cv/pyproject.toml`
- `view-buffer/Cargo.toml`

**Recommendation:** Use `cargo-release`, `bumpversion`, or a CI check.

### 5.7 [MEDIUM] Docs Only Deploy on Manual Trigger

**Location:** `.github/workflows/docs.yml`

Documentation is never automatically deployed on push to `main` or on release. Docs easily drift from code. **Recommendation:** Trigger on push to `main` and/or release events.

### 5.8 [LOW] `numpy>=2.2.6` Excludes All of numpy 1.x

**Location:** `polars-cv/pyproject.toml`

This very specific minimum excludes users on numpy 1.x, which is still widely used. **Recommendation:** Evaluate whether numpy 1.x support is intentionally dropped and document accordingly.

### 5.9 [LOW] Placeholder Author in view-buffer

**Location:** `view-buffer/Cargo.toml:8`

```toml
authors = ["Your Name <your.email@example.com>"]
```

**Recommendation:** Update to the actual author.

### 5.10 [LOW] Ruff Version Drift

Pre-commit pins `ruff-pre-commit` at `v0.9.2`, `pyproject.toml` requires `ruff>=0.9.0`, and CI uses `uvx ruff` (latest). These can produce different formatting results. **Recommendation:** Pin consistently across all three.

---

## 6. Rust Code Quality Notes

### 6.1 `from_slice_aligned` Allocation Safety

**Location:** `view-buffer/src/core/buffer.rs:221-254`

The function uses `std::alloc::alloc` + `Vec::from_raw_parts`. The `Vec` is constructed with `len == capacity`, which is correct, but the allocation uses a custom layout (`std::alloc::Layout`). When this `Vec` is dropped, Rust's allocator will deallocate using the default layout, not the custom alignment. This is undefined behavior if the global allocator doesn't handle arbitrary alignments. In practice, most allocators handle this correctly, but it's technically UB per the Rust documentation.

**Recommendation:** Use an aligned allocation crate (e.g., `aligned-vec`) or document the allocator requirements.

### 6.2 `to_contiguous` Element-by-Element Copy

**Location:** `view-buffer/src/core/buffer.rs:1340-1407`

The `to_contiguous` method copies elements byte-by-byte in a nested loop. For large strided buffers, this is significantly slower than memcpy-based approaches. The ndarray path in `apply_scalar_op` handles this more efficiently. **Recommendation:** Consider using ndarray's contiguous conversion or at least copy row-by-row when possible.

### 6.3 Binary Operations Don't Leverage SIMD

**Location:** `view-buffer/src/ops/binary.rs`

Binary operations (add, multiply, etc.) use scalar element-wise loops. The threshold and fused kernel operations have SIMD-friendly chunked processing, but binary ops do not. For large arrays, this is a missed optimization. **Recommendation:** Add chunked processing similar to `threshold_simd` and `apply_fused_kernel_contiguous`.

### 6.4 Reduction Axis Operations Allocate Per-Element

**Location:** `view-buffer/src/ops/reduction.rs:302-317`

`reduce_axis` allocates a `Vec` per output element (`Vec::with_capacity(axis_size)`) to gather values along the axis. This is O(output_size * axis_size) allocations. **Recommendation:** Allocate the gather buffer once and reuse it.

---

## 7. Improvement Suggestions

### Architecture
1. **Error handling strategy:** The codebase mixes panics (Rust) with Result types. The `catch_unwind` pattern in `execute.rs` and `types.rs` is a workaround. Consider migrating view-buffer operations to return `Result` types instead of panicking.
2. **Unified type promotion:** The `cast` and `cast_to` methods on `ViewBuffer` are separate APIs with different type pair coverage. Unify them.

### Testing
3. **Property-based testing:** The buffer operations (slice, flip, permute, to_contiguous) are ideal candidates for property-based testing with `proptest` to catch edge cases.
4. **Fuzzing:** The blob deserialization path should be fuzzed to catch the overflow and bounds issues identified above.

### Documentation
5. **Architecture decision records:** Document the rationale for key design decisions (zero-copy strategy, SIMD alignment, tiling approach).
6. **Contributor guide:** The `CONTRIBUTING.md` exists but could benefit from architecture diagrams showing the data flow through the layers.

### Performance
7. **Batch execution:** The row-by-row execution model is inherently limiting. Consider a batch mode where operations are applied to all rows simultaneously, leveraging SIMD across the batch dimension.
8. **Async cloud reads:** The cloud module creates a new tokio runtime per read. For batch operations, a shared runtime would avoid the overhead.

---

## Summary

| Severity | Count | Categories |
|----------|-------|------------|
| High | 8 | Security (1), Correctness (2), CI/CD (4), Performance (1 - misleading API) |
| Medium | 15 | Security (2), Correctness (5), Performance (3), API (2), CI/CD (3) |
| Low | 12 | Correctness (4), Performance (2), API (2), CI/CD (4) |
