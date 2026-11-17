//! `ImageAdapter::decode_png_rows` (crop pushdown: stop inflating after the
//! last row a leading crop keeps) must produce exactly the first rows of the
//! full decode, for every PNG it accepts, and decline the ones it cannot
//! reproduce, which then decode in full.

use proptest::prelude::*;
use view_buffer::{DType, ImageAdapter, ViewBuffer};

fn encode(
    w: u32,
    h: u32,
    color: png::ColorType,
    depth: png::BitDepth,
    filter: png::Filter,
    seed: u64,
) -> Vec<u8> {
    let channels = match color {
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        png::ColorType::Indexed => 1,
    };
    let bytes_per = if depth == png::BitDepth::Sixteen {
        2
    } else {
        1
    };
    let mut state = seed | 1;
    let data: Vec<u8> = (0..(w * h) as usize * channels * bytes_per)
        .map(|i| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // Smooth plus noise, so every filter type is exercised.
            ((i / 7) as u64 + (state >> 61)) as u8
        })
        .collect();
    let mut out = Vec::new();
    let mut enc = png::Encoder::new(&mut out, w, h);
    enc.set_color(color);
    enc.set_depth(depth);
    enc.set_filter(filter);
    if color == png::ColorType::Indexed {
        enc.set_palette((0..256 * 3).map(|i| i as u8).collect::<Vec<_>>());
    }
    let mut writer = enc.write_header().unwrap();
    writer.write_image_data(&data).unwrap();
    writer.finish().unwrap();
    out
}

fn first_rows(buf: &ViewBuffer, rows: usize) -> ViewBuffer {
    let s = buf.shape();
    buf.slice(&[0, 0, 0], &[rows, s[1], s[2]]).to_contiguous()
}

fn bytes_of(buf: &ViewBuffer) -> Vec<u8> {
    match buf.dtype() {
        DType::U8 => buf.as_slice::<u8>().to_vec(),
        DType::U16 => buf
            .as_slice::<u16>()
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect(),
        other => panic!("a PNG decodes to u8 or u16, not {other:?}"),
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    #[test]
    fn rows_match_the_full_decode(
        (w, h) in (1u32..70, 2u32..70),
        color in prop::sample::select(vec![
            png::ColorType::Grayscale, png::ColorType::GrayscaleAlpha,
            png::ColorType::Rgb, png::ColorType::Rgba,
        ]),
        sixteen in any::<bool>(),
        filter in prop::sample::select(vec![
            png::Filter::NoFilter, png::Filter::Sub, png::Filter::Up,
            png::Filter::Avg, png::Filter::Paeth, png::Filter::Adaptive,
        ]),
        seed in any::<u64>(),
        at in any::<prop::sample::Index>(),
    ) {
        let depth = if sixteen { png::BitDepth::Sixteen } else { png::BitDepth::Eight };
        let bytes = encode(w, h, color, depth, filter, seed);
        let rows = 1 + at.index(h as usize - 1); // 1..h, strictly inside
        let full = ImageAdapter::decode(&bytes).unwrap();
        let got = ImageAdapter::decode_png_rows(&bytes, rows)
            .expect("an 8/16-bit, non-interlaced, palette-free PNG is accepted");
        let want = first_rows(&full, rows);
        prop_assert_eq!(got.dtype(), want.dtype());
        prop_assert_eq!(got.shape(), want.shape());
        prop_assert!(bytes_of(&got) == bytes_of(&want));
    }
}

#[test]
fn declines_what_it_cannot_reproduce() {
    let rgb = encode(
        9,
        9,
        png::ColorType::Rgb,
        png::BitDepth::Eight,
        png::Filter::Paeth,
        3,
    );
    // The whole image, or none of it, is not a partial decode.
    assert!(ImageAdapter::decode_png_rows(&rgb, 9).is_none());
    assert!(ImageAdapter::decode_png_rows(&rgb, 0).is_none());
    // Palette images expand differently: full decode.
    let indexed = encode(
        9,
        9,
        png::ColorType::Indexed,
        png::BitDepth::Eight,
        png::Filter::Sub,
        3,
    );
    assert!(ImageAdapter::decode_png_rows(&indexed, 4).is_none());
    // A transparency key adds an alpha channel: full decode.
    let mut out = Vec::new();
    let mut enc = png::Encoder::new(&mut out, 4, 4);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.set_trns(vec![0u8, 0, 0, 0, 0, 0]);
    let mut writer = enc.write_header().unwrap();
    writer.write_image_data(&[7u8; 48]).unwrap();
    writer.finish().unwrap();
    assert!(ImageAdapter::decode_png_rows(&out, 2).is_none());
    // Not a PNG.
    assert!(ImageAdapter::decode_png_rows(b"\xff\xd8\xff\xe0 jpeg", 2).is_none());
}

/// A noise RGB PNG written in 1 KiB IDAT chunks (as encoders write larger
/// images), and the offset and length of its last IDAT chunk's data.
fn chunked_png() -> (Vec<u8>, usize, usize) {
    let (w, h) = (64u32, 128u32);
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let data: Vec<u8> = (0..(w * h * 3) as usize)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as u8
        })
        .collect();
    let mut bytes = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut bytes, w, h);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        enc.set_filter(png::Filter::NoFilter);
        let mut writer = enc.write_header().unwrap();
        let mut stream = writer.stream_writer_with_size(1024).unwrap();
        std::io::Write::write_all(&mut stream, &data).unwrap();
        stream.finish().unwrap();
    }
    let (mut at, mut last) = (8, None);
    while at + 8 <= bytes.len() {
        let len = u32::from_be_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
        if &bytes[at + 4..at + 8] == b"IDAT" {
            last = Some((at + 8, len));
        }
        at += 12 + len;
    }
    let (data_at, len) = last.expect("the image has IDAT chunks");
    (bytes, data_at, len)
}

#[test]
fn damage_in_a_later_chunk_is_not_read() {
    // The crop pushdown's one behaviour change, pinned: corruption in an IDAT
    // chunk after the kept rows' data (its CRC or its compressed bytes) no
    // longer fails the decode, since those chunks are never read.
    let (bytes, data_at, len) = chunked_png();
    let want = first_rows(&ImageAdapter::decode(&bytes).unwrap(), 8);
    for damaged_at in [data_at + len, data_at + len / 2] {
        let mut damaged = bytes.clone();
        damaged[damaged_at] ^= 0xff;
        assert!(ImageAdapter::decode(&damaged).is_err());
        let rows = ImageAdapter::decode_png_rows(&damaged, 8).expect("the kept rows are intact");
        assert_eq!(bytes_of(&rows), bytes_of(&want));
    }
}

#[test]
fn a_truncated_file_still_fails() {
    // `png` reads ahead of the rows it returns, so a file cut short is refused
    // by the partial decode too (the caller then reports the full decode's
    // error): truncation behaves as before.
    let (bytes, _, _) = chunked_png();
    let cut = &bytes[..bytes.len() / 2];
    assert!(ImageAdapter::decode(cut).is_err());
    assert!(ImageAdapter::decode_png_rows(cut, 8).is_none());
}
