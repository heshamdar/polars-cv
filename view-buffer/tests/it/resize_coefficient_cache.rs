//! The vendored `fast_image_resize` caches convolution coefficients in its
//! `Resizer` (vendor/fast_image_resize/VENDORED.md). A cache keyed on less
//! than the coefficients depend on would hand one resize another's weights.
//! So a random sequence of resizes through ONE resizer (as the engine's
//! thread-local runs them) must match, byte for byte, the same resizes each
//! through a fresh resizer (which computes everything anew).

use fast_image_resize::images::Image;
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use proptest::prelude::*;

#[derive(Debug, Clone)]
struct Step {
    src: (u32, u32),
    dst: (u32, u32),
    crop: Option<(f64, f64)>,
    alg: ResizeAlg,
    pixel: PixelType,
}

fn step() -> impl Strategy<Value = Step> {
    let size = || (1u32..40, 1u32..40);
    let alg = prop::sample::select(vec![
        ResizeAlg::Convolution(FilterType::Lanczos3),
        ResizeAlg::Convolution(FilterType::Bilinear),
        ResizeAlg::Convolution(FilterType::CatmullRom),
        ResizeAlg::Interpolation(FilterType::Lanczos3),
        ResizeAlg::SuperSampling(FilterType::Bilinear, 2),
    ]);
    let pixel = prop::sample::select(vec![PixelType::U8x3, PixelType::U8, PixelType::F32]);
    (
        size(),
        size(),
        prop::option::of((0.0f64..0.4, 0.0f64..0.4)),
        alg,
        pixel,
    )
        .prop_map(|(src, dst, crop, alg, pixel)| Step {
            src,
            dst,
            crop,
            alg,
            pixel,
        })
}

fn run(resizer: &mut Resizer, s: &Step, seed: u8) -> Vec<u8> {
    let bytes = s.src.0 as usize * s.src.1 as usize * s.pixel.size();
    let data: Vec<u8> = (0..bytes)
        .map(|i| {
            let v = (i as u32)
                .wrapping_mul(2_654_435_761)
                .wrapping_add(u32::from(seed))
                >> 24;
            // f32 pixels: keep every 4-byte group a finite value.
            if s.pixel == PixelType::F32 && i % 4 == 3 {
                0x3f
            } else {
                v as u8
            }
        })
        .collect();
    let src = Image::from_vec_u8(s.src.0, s.src.1, data, s.pixel).unwrap();
    let mut dst = Image::new(s.dst.0, s.dst.1, s.pixel);
    let mut options = ResizeOptions::new().resize_alg(s.alg);
    if let Some((l, t)) = s.crop {
        let (w, h) = (f64::from(s.src.0), f64::from(s.src.1));
        options = options.crop(l * w, t * h, w * (1.0 - l) * 0.99, h * (1.0 - t) * 0.99);
    }
    resizer.resize(&src, &mut dst, &options).unwrap();
    dst.into_vec()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn a_shared_resizer_matches_fresh_ones(steps in prop::collection::vec(step(), 1..24)) {
        let mut shared = Resizer::new();
        for (i, s) in steps.iter().enumerate() {
            let seed = i as u8;
            let got = run(&mut shared, s, seed);
            let want = run(&mut Resizer::new(), s, seed);
            prop_assert!(got == want, "step {i} ({s:?}) differs with a shared resizer");
        }
    }
}
