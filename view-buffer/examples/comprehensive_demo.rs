//! Comprehensive demo showcasing view-buffer's features.
//!
//! This example demonstrates:
//! - Buffer creation and manipulation
//! - Zero-copy view operations
//! - Compute operations and kernel fusion
//! - Image processing pipelines
//! - Cost analysis and introspection (with fusion details)
//! - Binary serialization
//! - SIMD-aligned buffers
//!
//! Run with: cargo run --example comprehensive_demo --features "image_interop arrow_interop"

use view_buffer::core::buffer::SIMD_ALIGNMENT;
use view_buffer::interop::image::ImageAdapter;
use view_buffer::ops::image::FilterType;
use view_buffer::ops::scalar::{FusedKernel, ScalarOp};
use view_buffer::{DType, ViewBuffer, ViewExpr};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║          view-buffer Comprehensive Demo                      ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // =========================================================================
    // Part 1: Basic Buffer Operations
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Part 1: Basic Buffer Operations");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Create a buffer from a vector
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let buffer = ViewBuffer::from_vec(data);

    println!("Created buffer from Vec<f32>:");
    println!("  Shape: {:?}", buffer.shape());
    println!("  DType: {:?}", buffer.dtype());
    println!("  Contiguous: {}", buffer.layout_facts().is_contiguous());

    // Reshape to 2D
    let buffer_2d = ViewExpr::new_source(buffer)
        .reshape(vec![4, 6])
        .plan()
        .execute();

    println!("\nAfter reshape to [4, 6]:");
    println!("  Shape: {:?}", buffer_2d.shape());
    println!("  Strides (bytes): {:?}", buffer_2d.strides_bytes());

    // =========================================================================
    // Part 2: Zero-Copy View Operations
    // =========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Part 2: Zero-Copy View Operations");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let original_id = buffer_2d.storage_id();

    // Transpose (zero-copy)
    let transposed = buffer_2d.permute(&[1, 0]);
    println!("Transpose [1, 0]:");
    println!("  Shape: {:?}", transposed.shape());
    println!("  Zero-copy: {}", transposed.storage_id() == original_id);

    // Flip (zero-copy)
    let flipped = buffer_2d.flip(&[0]);
    println!("\nFlip axis 0:");
    println!("  Strides: {:?}", flipped.strides_bytes());
    println!("  Zero-copy: {}", flipped.storage_id() == original_id);

    // Crop/Slice (zero-copy)
    let cropped = buffer_2d.slice(&[1, 1], &[3, 5]);
    println!("\nCrop [1:3, 1:5]:");
    println!("  Shape: {:?}", cropped.shape());
    println!("  Zero-copy: {}", cropped.storage_id() == original_id);

    // Chain multiple view operations
    let chained = buffer_2d
        .permute(&[1, 0])
        .flip(&[0])
        .slice(&[0, 0], &[4, 2]);
    println!("\nChained: transpose -> flip -> slice:");
    println!("  Shape: {:?}", chained.shape());
    println!("  Zero-copy: {}", chained.storage_id() == original_id);

    // =========================================================================
    // Part 3: Compute Operations and Kernel Fusion
    // =========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Part 3: Compute Operations and Kernel Fusion");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Create test data
    let test_data: Vec<f32> = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0];
    let test_buf = ViewBuffer::from_vec(test_data);

    // Individual operations
    let expr = ViewExpr::new_source(test_buf.clone());
    let scaled = expr.scale(2.0).plan().execute();
    print_f32_buffer("scale(2.0)", &scaled);

    let expr = ViewExpr::new_source(test_buf.clone());
    let relued = expr.relu().plan().execute();
    print_f32_buffer("relu()", &relued);

    // Automatic fusion: scale -> scale -> relu
    println!("\nAutomatic Kernel Fusion:");
    let expr = ViewExpr::new_source(test_buf.clone());
    let pipeline = expr.scale(2.0).scale(0.5).relu();

    // Show the optimized plan
    let plan = pipeline.plan();
    println!("  Optimized plan steps: {}", plan.steps.len());
    for (i, step) in plan.steps.iter().enumerate() {
        println!("    Step {}: {:?}", i + 1, step);
    }

    let result = plan.execute();
    print_f32_buffer("  Result", &result);

    // Manual fused kernel
    println!("\nManual Fused Kernel: (x * 2) + 1 -> relu");
    let mut kernel = FusedKernel::new();
    kernel.push(ScalarOp::Mul(2.0));
    kernel.push(ScalarOp::Add(1.0));
    kernel.push(ScalarOp::Relu);

    let fused_result = test_buf.apply_fused_kernel(&kernel);
    print_f32_buffer("  Result", &fused_result);

    // =========================================================================
    // Part 4: Type Casting and Normalization
    // =========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Part 4: Type Casting and Normalization");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // U8 to F32
    let u8_data: Vec<u8> = vec![0, 64, 128, 192, 255];
    let u8_buf = ViewBuffer::from_vec(u8_data);
    println!("Original U8 buffer: {:?}", u8_buf.dtype());

    let expr = ViewExpr::new_source(u8_buf);
    let f32_buf = expr.cast(DType::F32).plan().execute();
    println!("After cast to F32: {:?}", f32_buf.dtype());
    print_f32_buffer("  Values", &f32_buf);

    // Normalization (MinMax)
    let norm_data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let norm_buf = ViewBuffer::from_vec(norm_data);
    let expr = ViewExpr::new_source(norm_buf)
        .reshape(vec![5, 1])
        .normalize(view_buffer::NormalizeMethod::MinMax);
    let normalized = expr.plan().execute();
    println!("\nMinMax Normalization [10, 20, 30, 40, 50] -> [0, 1]:");
    print_f32_buffer("  Result", &normalized);

    // =========================================================================
    // Part 5: Image Processing Pipeline
    // =========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Part 5: Image Processing Pipeline");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Generate a synthetic 200x200 RGB image (gradient)
    let (width, height) = (200, 200);
    let mut img_data = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let r = ((x as f32 / width as f32) * 255.0) as u8;
            let g = ((y as f32 / height as f32) * 255.0) as u8;
            let b = 128u8;
            img_data.extend_from_slice(&[r, g, b]);
        }
    }

    let img_buf = ViewBuffer::from_vec(img_data);
    let source = ViewExpr::new_source(img_buf).reshape(vec![height, width, 3]);

    println!("Created synthetic 200x200 RGB image");
    println!("  Shape: {:?}", source.shape);
    println!("  DType: {:?}", source.dtype);

    // Complex image pipeline
    let pipeline = source
        .resize(100, 100, FilterType::Triangle) // Downscale
        .crop(vec![10, 10, 0], vec![90, 90, 3]) // Crop center 80x80
        .flip(vec![1]) // Horizontal flip
        .grayscale() // Convert to grayscale
        .threshold(128); // Binary threshold

    println!("\nPipeline: resize(100,100) -> crop(80x80) -> flip -> grayscale -> threshold");

    // Show the plan
    let plan = pipeline.plan();
    println!("\nExecution plan ({} steps):", plan.steps.len());
    for (i, step) in plan.steps.iter().enumerate() {
        println!("  Step {}: {:?}", i + 1, step);
    }

    let result = plan.execute();
    println!("\nResult:");
    println!("  Shape: {:?}", result.shape());
    println!("  DType: {:?}", result.dtype());

    // Save to file
    let result_contig = result.to_contiguous();
    match ImageAdapter::save(&result_contig, "examples/comprehensive_output.png") {
        Ok(_) => println!("  Saved to 'examples/comprehensive_output.png'"),
        Err(e) => println!("  Could not save: {e}"),
    }

    // =========================================================================
    // Part 6: Cost Analysis and Introspection
    // =========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Part 6: Cost Analysis and Introspection");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let analysis_data: Vec<f32> = (0..100).map(|i| i as f32).collect();
    let analysis_buf = ViewBuffer::from_vec(analysis_data);
    let expr = ViewExpr::new_source(analysis_buf);

    // Clean pipeline: view ops -> scalar ops (fused) -> cast
    let analysis_pipeline = expr
        .reshape(vec![10, 10])
        .flip(vec![0]) // Zero-copy
        .transpose(vec![1, 0]) // Zero-copy
        .scale(2.0) // Allocating
        .relu() // Will be fused with scale!
        .cast(DType::U8); // Allocating + dtype change

    // Optimize to trigger fusion
    let optimized = analysis_pipeline.optimize();

    println!("Pipeline: reshape -> flip -> transpose -> scale -> relu -> cast(U8)");
    println!("\nCost Report (with fusion details):");
    println!("{}", optimized.explain_costs());

    let report = optimized.cost_report();
    println!("Summary:");
    println!("  Total operations: {}", report.operations.len());
    println!("  Zero-copy operations: {}", report.zero_copy_operations);
    println!("  Allocating operations: {}", report.total_allocations);
    println!("  DType changes: {}", report.dtype_changes.len());
    if !report.fusion_summary.is_empty() {
        println!("  Fused operations: {:?}", report.fusion_summary);
    }
    if !report.dtype_flow.is_empty() {
        println!(
            "  DType flow: {}",
            report
                .dtype_flow
                .iter()
                .map(|d| format!("{d:?}"))
                .collect::<Vec<_>>()
                .join(" -> ")
        );
    }

    // =========================================================================
    // Part 7: Binary Serialization
    // =========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Part 7: Binary Serialization (ViewBlob Protocol)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Create a test buffer with non-contiguous layout
    let serial_data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let serial_buf = ViewExpr::new_source(ViewBuffer::from_vec(serial_data))
        .reshape(vec![4, 4])
        .plan()
        .execute();

    // Slice to create non-contiguous buffer
    let sliced = serial_buf.slice(&[0, 0], &[4, 2]);
    println!("Original buffer (sliced column):");
    println!("  Shape: {:?}", sliced.shape());
    println!("  Contiguous: {}", sliced.layout_facts().is_contiguous());

    // Serialize to blob
    let blob = sliced.to_blob();
    println!("\nSerialized to blob:");
    println!("  Blob size: {} bytes", blob.len());
    println!(
        "  Magic bytes: {:?}",
        std::str::from_utf8(&blob[0..4]).unwrap()
    );

    // Deserialize
    let recovered = ViewBuffer::from_blob(&blob).expect("Failed to deserialize");
    println!("\nDeserialized buffer:");
    println!("  Shape: {:?}", recovered.shape());
    println!("  Contiguous: {}", recovered.layout_facts().is_contiguous());
    println!("  DType: {:?}", recovered.dtype());

    // Verify data integrity
    let (ptr, _, _, _) = recovered.as_raw_parts();
    let recovered_data = unsafe { std::slice::from_raw_parts(ptr as *const f32, 8) };
    println!("  Data: {recovered_data:?}");

    // =========================================================================
    // Part 8: Layout Compatibility
    // =========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Part 8: Layout Compatibility Checking");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let layout_data: Vec<u8> = vec![0; 100 * 100 * 3];
    let layout_buf = ViewExpr::new_source(ViewBuffer::from_vec(layout_data))
        .reshape(vec![100, 100, 3])
        .plan()
        .execute();

    let report = layout_buf.layout_report();
    println!("Layout report for [100, 100, 3] U8 buffer:");
    println!("  Shape: {:?}", report.shape);
    println!("  Strides: {:?}", report.strides);
    println!("  Contiguous: {}", report.contiguous);
    println!("  ndarray compatible: {}", report.ndarray_compatible);
    println!("  image crate compatible: {}", report.image_compatible);

    // After transpose
    let transposed = layout_buf.permute(&[1, 0, 2]);
    let report = transposed.layout_report();
    println!("\nAfter transpose [1, 0, 2]:");
    println!("  Contiguous: {}", report.contiguous);
    println!("  ndarray compatible: {}", report.ndarray_compatible);
    println!("  image crate compatible: {}", report.image_compatible);

    // =========================================================================
    // Part 9: SIMD-Aligned Buffers
    // =========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Part 9: SIMD-Aligned Buffers");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Regular buffer (may not be aligned)
    let regular_data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let regular_buf = ViewBuffer::from_vec(regular_data.clone());
    println!("Regular buffer:");
    println!(
        "  SIMD aligned ({}B): {}",
        SIMD_ALIGNMENT,
        regular_buf.is_simd_aligned()
    );

    // SIMD-aligned buffer
    let aligned_buf = ViewBuffer::from_slice_simd_aligned(&regular_data);
    println!("\nSIMD-aligned buffer:");
    println!(
        "  SIMD aligned ({}B): {}",
        SIMD_ALIGNMENT,
        aligned_buf.is_simd_aligned()
    );
    println!("  Aligned to 32B: {}", aligned_buf.is_aligned(32));
    println!("  Aligned to 64B: {}", aligned_buf.is_aligned(64));

    // Fused kernel on aligned buffer (uses SIMD fast path)
    let mut simd_kernel = FusedKernel::new();
    simd_kernel.push(ScalarOp::Mul(2.0));
    simd_kernel.push(ScalarOp::Add(1.0));
    simd_kernel.push(ScalarOp::Relu);

    println!("\nApplying fused kernel on aligned buffer...");
    println!("  Kernel: {}", simd_kernel.describe());

    let result = aligned_buf.apply_fused_kernel(&simd_kernel);
    println!("  Result shape: {:?}", result.shape());
    println!("  Result SIMD aligned: {}", result.is_simd_aligned());

    // Show first few values
    let (ptr, _, _, _) = result.as_raw_parts();
    let result_slice = unsafe { std::slice::from_raw_parts(ptr as *const f32, 8) };
    println!("  First 8 values: {result_slice:?}");

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    Demo Complete!                            ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    Ok(())
}

/// Helper function to print F32 buffer contents
fn print_f32_buffer(label: &str, buffer: &ViewBuffer) {
    let count = buffer.shape().iter().product::<usize>().min(10);
    let (ptr, _, _, _) = buffer.as_raw_parts();
    let slice = unsafe { std::slice::from_raw_parts(ptr as *const f32, count) };
    println!("{label}: {slice:?}");
}
