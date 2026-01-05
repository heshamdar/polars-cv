use view_buffer::interop::image::ImageAdapter;
use view_buffer::ops::image::FilterType;
use view_buffer::{DType, NormalizeMethod, ViewBuffer, ViewExpr};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- ViewBuffer Framework Demo ---");

    // 1. Generate Synthetic Image (800x600 RGB Gradient)
    println!("1. Generating synthetic 800x600 RGB image...");
    let width = 800;
    let height = 600;
    let mut data = Vec::with_capacity(width * height * 3);

    for y in 0..height {
        for x in 0..width {
            let r = ((x as f32 / width as f32) * 255.0) as u8;
            let g = ((y as f32 / height as f32) * 255.0) as u8;
            let b = (((x + y) as f32 / (width + height) as f32) * 255.0) as u8;
            data.extend_from_slice(&[r, g, b]);
        }
    }

    // Load into ViewBuffer (Raw 1D vector -> Reshape)
    let raw_buf = ViewBuffer::from_vec(data);

    let source = ViewExpr::new_source(raw_buf).reshape(vec![height, width, 3]);

    println!(
        "   Source Layout: {:?}",
        source.plan().source.layout_report()
    );

    let first_plan = source.plan();
    let first_result = first_plan.execute();
    match ImageAdapter::save(&first_result, "examples/demo_input.png") {
        Ok(_) => println!("   Success saving input image to 'examples/demo_input.png'"),
        Err(e) => println!("   Error saving image: {e}"),
    }

    // 2. Define Processing Pipeline (Lazy)
    // Pipeline: Resize -> Crop -> Flip -> Threshold
    println!("\n2. Building lazy execution graph...");

    let pipeline = source
        // A. Resize: Downscale to 400x300 (delegates to image crate)
        .resize(400, 300, FilterType::Triangle)
        // B. Crop: Extract a region (View Operation - Zero Copy)
        //    Y: 50..250 (height 200)
        //    X: 100..300 (width 200)
        //    C: 0..3 (all channels)
        .crop(vec![50, 100, 0], vec![250, 300, 3])
        // C. Flip: Horizontal Mirror (View Operation - Zero Copy)
        //    Axis 1 is width
        .flip(vec![1])
        .cast(DType::F32)
        .scale(2.0)
        .scale(0.5)
        .scale(-1.0)
        .scale(-1.0)
        .relu()
        .cast(DType::U8)
        .cast(DType::F32)
        .normalize(NormalizeMethod::MinMax)
        .grayscale()
        .threshold(128);

    // Note: We could add .threshold(128) here if we wanted a binary mask,
    // but let's keep it RGB for the visual demo.

    // 3. Introspection
    println!("\n3. Inspecting Plan:");
    let plan = pipeline.plan();
    for (i, step) in plan.steps.iter().enumerate() {
        println!("   Step {}: {:?}", i + 1, step);
    }

    // 4. Execution
    println!("\n4. Executing...");
    let result = plan.execute();
    let result = result.to_contiguous();
    println!("   Result Shape: {:?}", result.shape());
    println!("   Result Strides: {:?}", result.strides_bytes());
    println!("   Result Dtype: {:?}", result.dtype());
    println!(
        "   Result Layout Compatible with Image Crate? {}",
        result.layout_report().image_compatible
    );

    // 5. Output
    println!("\n5. Saving output to 'examples/demo_output.png'...");

    // In a real run, this writes to disk.
    // ImageAdapter::save handles the final conversion to contiguous bytes if the
    // resulting view (from the flip/crop) is strided.
    match ImageAdapter::save(&result, "examples/demo_output.png") {
        Ok(_) => println!("   Success!"),
        Err(e) => println!("   Error saving image: {e}"),
    }

    Ok(())
}
