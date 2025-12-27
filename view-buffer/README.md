# view-buffer

A zero-copy, stride-aware tensor orchestration framework for Rust.

[![Crates.io](https://img.shields.io/crates/v/view-buffer.svg)](https://crates.io/crates/view-buffer)
[![Documentation](https://docs.rs/view-buffer/badge.svg)](https://docs.rs/view-buffer)
[![License](https://img.shields.io/crates/l/view-buffer.svg)](LICENSE)

## Overview

`view-buffer` provides a unified interface for working with multi-dimensional arrays (tensors) with:

- **Zero-copy view operations**: Transpose, reshape, flip, and crop without allocating new memory
- **Lazy expression graphs**: Build computation pipelines that are optimized before execution
- **Automatic kernel fusion**: Consecutive scalar operations are fused into single passes
- **Multiple backend interop**: Seamless integration with ndarray, Arrow, and image crates
- **Cost tracking**: Introspection tools to understand allocation patterns in your pipelines

## Features

| Feature | Description |
|---------|-------------|
| `ndarray_interop` | Zero-copy views to/from `ndarray::ArrayViewD` |
| `image_interop` | Integration with the `image` crate for image processing |
| `arrow_interop` | Zero-copy interop with Apache Arrow buffers |
| `numpy_interop` | Export format compatible with NumPy |
| `torch_interop` | Export format compatible with PyTorch |
| `serde` | Serialization support for operations and plans |
| `python` | Python bindings via PyO3 |

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
view-buffer = "0.3"

# Enable features as needed
view-buffer = { version = "0.3", features = ["image_interop", "ndarray_interop"] }
```

### Basic Usage

```rust
use view_buffer::{ViewBuffer, ViewExpr, DType};

// Create a buffer from a vector
let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
let buffer = ViewBuffer::from_vec(data);

// Build a lazy expression graph
let expr = ViewExpr::new_source(buffer)
    .reshape(vec![10, 10])       // Zero-copy reshape
    .transpose(vec![1, 0])       // Zero-copy transpose
    .scale(2.0)                  // Scalar multiply
    .relu()                      // ReLU activation
    .cast(DType::U8);            // Type cast

// Execute the pipeline
let result = expr.plan().execute();

println!("Result shape: {:?}", result.shape());
println!("Result dtype: {:?}", result.dtype());
```

### Image Processing Pipeline

```rust
use view_buffer::{ViewBuffer, ViewExpr, FilterType};
use view_buffer::interop::image::ImageAdapter;

// Load an image
let buffer = ImageAdapter::open("input.png")?;

// Build processing pipeline
let processed = ViewExpr::new_source(buffer)
    .resize(400, 300, FilterType::Lanczos3)  // Resize
    .crop(vec![50, 50, 0], vec![250, 250, 3]) // Crop region
    .flip(vec![1])                            // Horizontal flip
    .grayscale()                              // Convert to grayscale
    .threshold(128);                          // Binary threshold

// Execute and save
let result = processed.plan().execute();
ImageAdapter::save(&result, "output.png")?;
```

### Zero-Copy Interop with ndarray

```rust
use view_buffer::{ViewBuffer, ViewExpr};
use view_buffer::interop::ndarray::{AsNdarray, FromNdarray};

// Create from ndarray
let array = ndarray::Array2::<f32>::zeros((100, 100));
let buffer = ViewBuffer::from_array(array.into_dyn());

// Get a zero-copy view back
let view = buffer.as_array_view::<f32>()?;
println!("Sum: {}", view.sum());
```

### Cost Analysis

```rust
use view_buffer::{ViewBuffer, ViewExpr, DType};

let buffer = ViewBuffer::from_vec(vec![1.0f32; 1000]);
let expr = ViewExpr::new_source(buffer)
    .reshape(vec![10, 100])
    .flip(vec![0])          // Zero-copy
    .scale(2.0)             // Allocating
    .cast(DType::U8);       // Allocating

// Analyze the pipeline
let report = expr.cost_report();
println!("Total allocations: {}", report.total_allocations);
println!("DType changes: {:?}", report.dtype_changes);

// Human-readable explanation
println!("{}", expr.explain_costs());
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      ViewExpr (Lazy Graph)                  │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐     │
│  │ Source  │ → │  View   │ → │ Compute │ → │  Sink   │     │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼ plan()
┌─────────────────────────────────────────────────────────────┐
│                    ExecutionPlan                            │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐                   │
│  │ Step 1  │ → │ Step 2  │ → │ Step 3  │                   │
│  └─────────┘   └─────────┘   └─────────┘                   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼ execute()
┌─────────────────────────────────────────────────────────────┐
│                      ViewBuffer                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ BufferStorage (Rust Vec / Arrow Buffer)              │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Layout (shape, strides, offset, dtype)              │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Module Structure

```
src/
├── core/           # Fundamental types (DType, Layout, ViewBuffer)
├── ops/            # Operations (View, Compute, Image, I/O)
├── expr.rs         # Lazy expression graph
├── execution/      # Planning and execution
├── protocol.rs     # Binary serialization format
└── interop/        # External library integrations
    ├── ndarray.rs  # ndarray interop
    ├── arrow.rs    # Arrow interop
    └── image.rs    # image crate interop
```

## Operation Types

### View Operations (Zero-Copy)
- `Transpose` - Permute dimensions
- `Reshape` - Change shape (requires contiguous input)
- `Flip` - Reverse along axes
- `Crop` - Extract a sub-region

### Compute Operations (Allocating)
- `Cast` - Change data type
- `Scale` - Multiply by constant
- `Relu` - ReLU activation
- `Normalize` - MinMax or ZScore normalization
- `Clamp` - Clamp to range
- `Fused` - Fused sequence of scalar ops

### Image Operations
- `Resize` - Resize with various filters
- `Blur` - Gaussian blur
- `Grayscale` - Convert to grayscale
- `Threshold` - Binary thresholding

## Binary Protocol

`view-buffer` includes a binary serialization format for efficient data transport:

```rust
// Serialize
let blob = buffer.to_blob();

// Deserialize
let recovered = ViewBuffer::from_blob(&blob)?;
```

The format consists of a 64-byte header followed by shape, strides, and contiguous data.

## Performance

- View operations are O(1) - they only modify metadata
- Contiguous buffers enable SIMD-friendly iteration
- Kernel fusion reduces memory traffic for scalar operations
- Zero-copy interop avoids unnecessary allocations

## License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

