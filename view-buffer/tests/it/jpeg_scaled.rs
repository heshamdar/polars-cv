//! The vendored zune-jpeg's scaled (reduced-IDCT) output, held to
//! libjpeg-turbo's.
//!
//! `decode_from_plan="approximate"` decodes a JPEG at 1/2, 1/4 or 1/8 scale
//! through `JpegDecoder::set_idct_scale` (`vendor/zune-jpeg/src/scaled.rs`).
//! The fixtures in `jpeg_scaled/` (regenerate with its `generate.py`) span
//! the subsamplings (4:4:4, 4:2:2, 4:2:0, 4:1:1, 4:4:0), baseline and
//! progressive, restart intervals, grayscale and sizes that are not
//! multiples of 8 or 16; each comes with libjpeg-turbo's own scaled decode
//! (Pillow's `draft`). The scaled output must have libjpeg's size,
//! `ceil(dim * s / 8)`, stay within [`MEAN_BOUND`]/[`MAX_BOUND`] of that
//! reference and within the looser block-average bounds of zune's own full
//! decode. Scale 1 must stay the unpatched crate's output byte for byte:
//! its digests (`jpeg_scaled/full_digests.txt`) were recorded before the
//! patch, through zune directly and through `image`. Re-record only for a
//! deliberate change, with `VIEW_BUFFER_BLESS=1`.

use std::path::{Path, PathBuf};

use zune_jpeg::zune_core::bytestream::ZCursor;
use zune_jpeg::zune_core::colorspace::ColorSpace;
use zune_jpeg::zune_core::options::DecoderOptions;
use zune_jpeg::{IdctScale, JpegDecoder};

/// Every fixture, by file stem.
const CASES: &[&str] = &[
    "b444_61x37",
    "b422_61x37",
    "b420_61x37",
    "b420_130x97",
    "b420_17x9",
    "b420_rst_100x75",
    "b411_70x45",
    "b440_70x45",
    "p444_61x37",
    "p422_53x41",
    "p420_100x75",
    "p420_rst_77x50",
    "bgray_45x29",
    "pgray_45x29",
];

/// Colour fixtures whose luma-only scaled decode is pinned too.
const LUMA: &[&str] = &["b420_61x37", "p420_100x75", "b422_61x37"];

/// Fixtures zune-jpeg 0.5.8 itself decodes differently from libjpeg-turbo,
/// at full size too (progressive with restart intervals): the scaled decode
/// reads the same coefficients, so it cannot match libjpeg either. Held to
/// zune's own full decode instead, and pinned as divergent so the exclusion
/// goes once upstream decodes them right.
const ZUNE_DIVERGENT: &[&str] = &["p420_rst_77x50"];

/// `(denominator, scale)`: the output is `1/denominator` of each dimension.
const SCALES: [(usize, IdctScale); 3] = [
    (2, IdctScale::Half),
    (4, IdctScale::Quarter),
    (8, IdctScale::Eighth),
];

/// Bounds against libjpeg-turbo's scaled decode, per sample (0..=255):
/// the mean absolute difference over an image and the largest single one.
/// Measured at most 0.030 mean and 2 max over every fixture and scale, and
/// exact for luma output: the IDCTs and upsampling are libjpeg's, the colour
/// conversion is zune's own fixed point, which rounds differently by a step.
const MEAN_BOUND: f64 = 0.25;
const MAX_BOUND: u8 = 4;

/// Bounds against zune's full decode averaged over each `d x d` block, in
/// luma: a sanity check that the scaled output is the same picture, not a
/// reference (a block average is not what an IDCT computes). Measured at
/// most 1.08 mean and 6 max.
const AVG_MEAN_BOUND: f64 = 2.0;
const AVG_MAX_BOUND: u8 = 12;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/it/jpeg_scaled")
        .join(name)
}

fn is_gray(name: &str) -> bool {
    name[1..].starts_with("gray")
}

fn out_colorspace(name: &str) -> ColorSpace {
    if is_gray(name) {
        ColorSpace::Luma
    } else {
        ColorSpace::RGB
    }
}

/// A decode with `scale` set (or none, the default), as `(pixels, w, h)`.
fn decode(bytes: &[u8], scale: Option<IdctScale>, out: ColorSpace) -> (Vec<u8>, usize, usize) {
    let options = DecoderOptions::default().jpeg_set_out_colorspace(out);
    let mut decoder = JpegDecoder::new_with_options(ZCursor::new(bytes), options);
    decoder.decode_headers().unwrap();
    if let Some(scale) = scale {
        decoder.set_idct_scale(scale);
    }
    let (w, h) = decoder.dimensions().unwrap();
    assert_eq!(
        decoder.output_buffer_size().unwrap(),
        w * h * out.num_components(),
        "the buffer size follows the scaled dimensions"
    );
    let pixels = decoder.decode().unwrap();
    assert_eq!(pixels.len(), w * h * out.num_components());
    (pixels, w, h)
}

/// A committed libjpeg-turbo reference PNG, in `out`'s layout.
fn reference(file: &str, out: ColorSpace) -> (Vec<u8>, usize, usize) {
    let img = image::open(fixture(file)).unwrap();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let pixels = match out {
        ColorSpace::Luma => img.to_luma8().into_raw(),
        _ => img.to_rgb8().into_raw(),
    };
    (pixels, w, h)
}

/// `(mean, max)` absolute difference between two equal-length images.
fn diff(a: &[u8], b: &[u8]) -> (f64, u8) {
    assert_eq!(a.len(), b.len());
    let (mut sum, mut max) = (0_u64, 0_u8);
    for (&x, &y) in a.iter().zip(b) {
        let d = x.abs_diff(y);
        sum += u64::from(d);
        max = max.max(d);
    }
    (sum as f64 / a.len() as f64, max)
}

/// The libjpeg size of `len` at `1/denom`: rounded up.
fn scaled_len(len: usize, denom: usize) -> usize {
    len.div_ceil(denom)
}

/// Compare every `(case, scale)` with `check`, collecting the failures and
/// printing the measured table (`--nocapture`).
fn for_each_case(
    cases: &[&str],
    out: impl Fn(&str) -> ColorSpace,
    want: impl Fn(&str, usize, ColorSpace, &[u8], (usize, usize)) -> (Vec<u8>, usize, usize),
    bounds: (f64, u8),
) {
    let mut failures = Vec::new();
    let (mut worst_mean, mut worst_max) = (0.0_f64, 0_u8);
    for &name in cases {
        let bytes = std::fs::read(fixture(&format!("{name}.jpg"))).unwrap();
        let out = out(name);
        let (_, full_w, full_h) = decode(&bytes, None, out);
        for (denom, scale) in SCALES {
            let (got, w, h) = decode(&bytes, Some(scale), out);
            let size = (scaled_len(full_w, denom), scaled_len(full_h, denom));
            if (w, h) != size {
                failures.push(format!("{name} 1/{denom}: size {w}x{h}, want {size:?}"));
                continue;
            }
            let (expected, ew, eh) = want(name, denom, out, &bytes, size);
            assert_eq!((ew, eh), size, "{name} 1/{denom}: the reference's size");
            let (mean, max) = diff(&got, &expected);
            println!("{name:>16} 1/{denom}: mean {mean:.3} max {max}");
            worst_mean = worst_mean.max(mean);
            worst_max = worst_max.max(max);
            if mean > bounds.0 || max > bounds.1 {
                failures.push(format!("{name} 1/{denom}: mean {mean:.3}, max {max}"));
            }
        }
    }
    println!("worst: mean {worst_mean:.3} max {worst_max}");
    assert!(
        failures.is_empty(),
        "{} scaled decodes out of bounds {bounds:?}:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn scaled_decode_matches_libjpeg_turbo() {
    let cases: Vec<&str> = CASES
        .iter()
        .copied()
        .filter(|name| !ZUNE_DIVERGENT.contains(name))
        .collect();
    for_each_case(
        &cases,
        out_colorspace,
        |name, denom, out, _, _| reference(&format!("{name}.s{denom}.png"), out),
        (MEAN_BOUND, MAX_BOUND),
    );
}

/// Luma output composes with the scale: the Y channel alone, scaled, from a
/// colour file in one decode.
#[test]
fn scaled_luma_decode_matches_libjpeg_turbo() {
    for_each_case(
        LUMA,
        |_| ColorSpace::Luma,
        |name, denom, out, _, _| reference(&format!("{name}.L.s{denom}.png"), out),
        (MEAN_BOUND, MAX_BOUND),
    );
}

/// zune's full decode, averaged over each `denom x denom` block (a partial
/// block at the right or bottom edge over the pixels it has).
fn block_average(
    full: &[u8],
    (w, h): (usize, usize),
    channels: usize,
    denom: usize,
) -> (Vec<u8>, usize, usize) {
    let (ow, oh) = (scaled_len(w, denom), scaled_len(h, denom));
    let mut out = vec![0_u8; ow * oh * channels];
    for oy in 0..oh {
        for ox in 0..ow {
            for c in 0..channels {
                let (mut sum, mut n) = (0_u32, 0_u32);
                for y in oy * denom..((oy + 1) * denom).min(h) {
                    for x in ox * denom..((ox + 1) * denom).min(w) {
                        sum += u32::from(full[(y * w + x) * channels + c]);
                        n += 1;
                    }
                }
                out[(oy * ow + ox) * channels + c] = ((sum + n / 2) / n) as u8;
            }
        }
    }
    (out, ow, oh)
}

/// Luma only: libjpeg's own scaled chroma (an IDCT at the chroma plane's
/// resolution, then replicated at 1/8) is far from a block average of the
/// full image's (up to 26 mean on these fixtures, in libjpeg-turbo too).
#[test]
fn scaled_decode_is_close_to_the_full_decode_block_averaged() {
    for_each_case(
        CASES,
        |_| ColorSpace::Luma,
        |_, denom, out, bytes, _| {
            let (full, w, h) = decode(bytes, None, out);
            block_average(&full, (w, h), out.num_components(), denom)
        },
        (AVG_MEAN_BOUND, AVG_MAX_BOUND),
    );
}

/// The exclusion above is upstream's: zune's *full* decode of each
/// `ZUNE_DIVERGENT` file is already far from libjpeg-turbo's. When this
/// fails, upstream decodes it right: move the file back into the reference
/// comparison.
#[test]
fn zune_divergent_files_diverge_at_full_size() {
    for &name in ZUNE_DIVERGENT {
        let bytes = std::fs::read(fixture(&format!("{name}.jpg"))).unwrap();
        let (got, ..) = decode(&bytes, None, ColorSpace::RGB);
        let (want, ..) = reference(&format!("{name}.s1.png"), ColorSpace::RGB);
        let (mean, _) = diff(&got, &want);
        assert!(
            mean > 10.0,
            "{name}: zune's full decode now matches libjpeg-turbo ({mean:.3})"
        );
    }
}

/// FNV-1a over the bytes.
fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// Scale 1 is the unpatched decoder's output, byte for byte: through zune
/// with no scale set, with `IdctScale::Full` set, and through `image` (which
/// decodes every JPEG with this crate).
#[test]
fn full_scale_output_is_the_unpatched_crates() {
    let path = fixture("full_digests.txt");
    let mut got = String::new();
    for &name in CASES {
        let bytes = std::fs::read(fixture(&format!("{name}.jpg"))).unwrap();
        let out = out_colorspace(name);
        let (default, ..) = decode(&bytes, None, out);
        let (explicit, ..) = decode(&bytes, Some(IdctScale::Full), out);
        assert!(
            default == explicit,
            "{name}: IdctScale::Full is the default"
        );
        let via_image = image::load_from_memory(&bytes).unwrap().into_bytes();
        got += &format!("{name} {:016x} {:016x}\n", fnv(&default), fnv(&via_image));
    }
    if std::env::var_os("VIEW_BUFFER_BLESS").is_some() {
        std::fs::write(&path, &got).unwrap();
        return;
    }
    let want = std::fs::read_to_string(&path).expect("the digest fixture exists");
    assert_eq!(got, want, "a full-scale JPEG decode changed");
}

/// Decode time per image at each scale, zune's scaled IDCT against
/// jpeg-decoder's, for the JPEGs named in `JPEG_SCALED_TIMING` (paths,
/// `:`-separated). Not a correctness check: run it on an optimised build,
/// e.g. `CARGO_PROFILE_DEV_OPT_LEVEL=3 cargo test -p view-buffer
/// --all-features --test it jpeg_scaled::timing -- --ignored --nocapture`.
#[test]
#[ignore = "timing only; needs JPEG_SCALED_TIMING"]
fn timing() {
    use std::time::Instant;
    let Some(paths) = std::env::var_os("JPEG_SCALED_TIMING") else {
        return;
    };
    let jpeg_decoder_scaled = |bytes: &[u8], denom: usize| {
        let mut d = jpeg_decoder::Decoder::new(std::io::Cursor::new(bytes));
        d.read_info().unwrap();
        let info = d.info().unwrap();
        let side = |len: u16| len.div_ceil(denom as u16);
        d.scale(side(info.width), side(info.height)).unwrap();
        d.decode().unwrap()
    };
    let all: Vec<(usize, IdctScale)> = [(1, IdctScale::Full)].into_iter().chain(SCALES).collect();
    for path in std::env::split_paths(&paths) {
        let bytes = std::fs::read(&path).unwrap();
        // Rounds over every configuration in turn, keeping each one's
        // fastest: interference from anything else running lands on all of
        // them alike, and the minimum is the least disturbed run.
        let mut best = vec![[f64::INFINITY; 3]; all.len()];
        for _ in 0..25 {
            for (row, &(denom, scale)) in best.iter_mut().zip(&all) {
                let runs: [&dyn Fn(); 3] = [
                    &|| {
                        drop(std::hint::black_box(decode(
                            &bytes,
                            Some(scale),
                            ColorSpace::RGB,
                        )))
                    },
                    &|| {
                        drop(std::hint::black_box(decode(
                            &bytes,
                            Some(scale),
                            ColorSpace::Luma,
                        )))
                    },
                    &|| drop(std::hint::black_box(jpeg_decoder_scaled(&bytes, denom))),
                ];
                for (slot, run) in row.iter_mut().zip(runs) {
                    let start = Instant::now();
                    run();
                    *slot = slot.min(start.elapsed().as_secs_f64() * 1e3);
                }
            }
        }
        println!("{}", path.display());
        for (&(denom, _), [zune, luma, jd]) in all.iter().zip(&best) {
            println!(
                "  1/{denom}: zune {zune:7.2} ms (luma {luma:7.2})   jpeg-decoder {jd:7.2} ms"
            );
        }
    }
}
