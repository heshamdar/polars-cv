//! PNG-decoder evaluation harness (WS5 of the performance plan).
//!
//! Compares `image::load_from_memory` (the production decoder) against
//! `zune-png` on a synthesized corpus: gradient (high compression ratio,
//! like the benchmark suite's images) and photographic noise (low ratio),
//! at 256²/512²/1024², RGB8/RGBA8/Gray8.
//!
//! Run with:
//!   cargo test -p view-buffer --release --test png_decode_eval -- --ignored --nocapture
//!
//! Decision gate (from the approved plan): adopt zune-png behind a feature
//! only if its geomean speedup on the 8-bit RGB/RGBA cells is ≥ 1.3×.
//! The correctness check (pixel parity vs the image crate) runs as a normal
//! (non-ignored) test so it always gates.

use image::ImageEncoder;

#[derive(Clone, Copy)]
enum Pattern {
    Gradient,
    Noise,
}

#[derive(Clone, Copy)]
enum Color {
    Rgb8,
    Rgba8,
    Gray8,
}

fn synth_png(size: usize, pattern: Pattern, color: Color) -> Vec<u8> {
    let channels = match color {
        Color::Rgb8 => 3,
        Color::Rgba8 => 4,
        Color::Gray8 => 1,
    };
    let mut state = 0x9E3779B97F4A7C15u64;
    let mut data = Vec::with_capacity(size * size * channels);
    for y in 0..size {
        for x in 0..size {
            for c in 0..channels {
                let v = match pattern {
                    Pattern::Gradient => (((x + y) * 255) / (2 * size - 1)) as u8,
                    Pattern::Noise => {
                        state = state
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        ((state >> 33) % 256) as u8
                    }
                };
                let v = if c == 3 { 255 } else { v }; // opaque alpha
                data.push(v);
            }
        }
    }

    let mut out = Vec::new();
    let enc = image::codecs::png::PngEncoder::new(&mut out);
    let ct = match color {
        Color::Rgb8 => image::ExtendedColorType::Rgb8,
        Color::Rgba8 => image::ExtendedColorType::Rgba8,
        Color::Gray8 => image::ExtendedColorType::L8,
    };
    enc.write_image(&data, size as u32, size as u32, ct)
        .expect("encode");
    out
}

fn decode_image_crate(png: &[u8]) -> (Vec<u8>, u32, u32) {
    let img = image::load_from_memory(png).expect("decode");
    let (w, h) = (img.width(), img.height());
    (img.into_bytes(), w, h)
}

fn decode_zune(png: &[u8]) -> Vec<u8> {
    let mut decoder = zune_png::PngDecoder::new(png);
    decoder.decode_raw().expect("zune decode")
}

#[test]
fn zune_png_pixel_parity() {
    // Pixel-exact parity on every corpus cell zune would serve (8-bit,
    // non-interlaced). This is the correctness gate for any future adoption.
    for pattern in [Pattern::Gradient, Pattern::Noise] {
        for color in [Color::Rgb8, Color::Rgba8, Color::Gray8] {
            let png = synth_png(96, pattern, color);
            let (img_bytes, _, _) = decode_image_crate(&png);
            let zune_bytes = decode_zune(&png);
            assert_eq!(img_bytes, zune_bytes, "pixel mismatch");
        }
    }
}

#[test]
#[ignore]
fn timing_png_decoders() {
    println!(
        "{:<10} {:<6} {:>6} {:>12} {:>12} {:>8}",
        "pattern", "color", "size", "image ms", "zune ms", "speedup"
    );
    let mut rgb_speedups: Vec<f64> = Vec::new();
    for pattern in [Pattern::Gradient, Pattern::Noise] {
        for color in [Color::Rgb8, Color::Rgba8, Color::Gray8] {
            for size in [256usize, 512, 1024] {
                let png = synth_png(size, pattern, color);
                let n = if size >= 1024 { 10 } else { 40 };

                let t = std::time::Instant::now();
                for _ in 0..n {
                    std::hint::black_box(decode_image_crate(&png));
                }
                let image_ms = t.elapsed().as_secs_f64() * 1000.0 / n as f64;

                let t = std::time::Instant::now();
                for _ in 0..n {
                    std::hint::black_box(decode_zune(&png));
                }
                let zune_ms = t.elapsed().as_secs_f64() * 1000.0 / n as f64;

                let speedup = image_ms / zune_ms;
                if matches!(color, Color::Rgb8 | Color::Rgba8) {
                    rgb_speedups.push(speedup);
                }
                let (p, c) = (
                    match pattern {
                        Pattern::Gradient => "gradient",
                        Pattern::Noise => "noise",
                    },
                    match color {
                        Color::Rgb8 => "rgb8",
                        Color::Rgba8 => "rgba8",
                        Color::Gray8 => "gray8",
                    },
                );
                println!(
                    "{p:<10} {c:<6} {size:>6} {image_ms:>11.2} {zune_ms:>11.2} {speedup:>7.2}x"
                );
            }
        }
    }
    let geomean =
        (rgb_speedups.iter().map(|s| s.ln()).sum::<f64>() / rgb_speedups.len() as f64).exp();
    println!("\nGeomean speedup on RGB/RGBA cells: {geomean:.2}x (adoption gate: >= 1.30x)");
}
