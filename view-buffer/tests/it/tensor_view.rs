//! `ViewBuffer::from_polars_buffer_strided`: a view over untrusted
//! `(shape, strides, offset)` (a tensor struct row, CR-138). It is built only
//! if every element it can address lies inside the buffer; these are the
//! cases either side of each bound, and an exhaustive sweep of small layouts.
#![cfg(feature = "polars_interop")]

use polars_buffer::Buffer;
use view_buffer::{DType, ViewBuffer};

fn view(
    bytes: usize,
    shape: &[usize],
    strides: &[isize],
    offset: usize,
    dtype: DType,
) -> Result<ViewBuffer, String> {
    // A `Vec<u64>` backs the bytes, so they start 8-aligned.
    let words: Vec<u64> = (0..bytes.div_ceil(8) as u64).collect();
    let buffer = Buffer::from(words).try_transmute::<u8>().unwrap();
    ViewBuffer::from_polars_buffer_strided(buffer.sliced(0..bytes), shape, strides, offset, dtype)
}

#[test]
fn a_view_inside_its_data_is_built_and_reads_its_elements() {
    let bytes: Vec<u8> = (0..6).collect();
    let buf = ViewBuffer::from_polars_buffer_strided(
        Buffer::from(bytes),
        &[2, 3],
        &[-3, -1],
        5,
        DType::U8,
    )
    .unwrap();
    assert_eq!(buf.shape(), &[2, 3]);
    assert_eq!(buf.to_contiguous().as_slice::<u8>(), &[5, 4, 3, 2, 1, 0]);
}

#[test]
fn each_bound_is_exact() {
    let ok = |b, sh: &[usize], st: &[isize], o| view(b, sh, st, o, DType::U8).is_ok();
    // The last element's last byte is the data's last byte, and no further.
    assert!(ok(6, &[2, 3], &[3, 1], 0));
    assert!(!ok(5, &[2, 3], &[3, 1], 0));
    assert!(!ok(6, &[2, 3], &[3, 1], 1));
    assert!(!ok(6, &[2, 3], &[4, 1], 0));
    // A negative stride reaches the first byte, and no further.
    assert!(ok(6, &[2, 3], &[-3, 1], 3));
    assert!(!ok(6, &[2, 3], &[-3, 1], 2));
    // Rank 0 is one element.
    assert!(ok(1, &[], &[], 0));
    assert!(!ok(1, &[], &[], 1));
    assert!(!ok(0, &[], &[], 0));
    // No elements: nothing is read, but the offset is still in the data.
    assert!(ok(0, &[0, 3], &[9, -9], 0));
    assert!(ok(4, &[3, 0], &[1, 1], 4));
    assert!(!ok(4, &[3, 0], &[1, 1], 5));
    // A zero stride repeats an element.
    assert!(ok(1, &[4], &[0], 0));
}

#[test]
fn sizes_that_overflow_are_refused_not_wrapped() {
    let big = usize::MAX / 2 + 1;
    for (shape, strides) in [
        (vec![big, big], vec![0isize, 0]),
        (vec![big, 4], vec![1, 1]),
        (vec![0, big, big], vec![0, 0, 0]),
        (vec![2, 2], vec![isize::MAX, 1]),
        (vec![2, 2], vec![isize::MIN, 1]),
        (vec![3, 3], vec![isize::MAX, isize::MAX]),
    ] {
        assert!(
            view(16, &shape, &strides, 0, DType::U8).is_err(),
            "{shape:?} {strides:?}"
        );
    }
    assert!(view(16, &[2], &[1], usize::MAX, DType::U8).is_err());
}

#[test]
fn elements_must_be_whole_and_aligned() {
    // A stride that is not a whole element, an offset that misaligns them.
    assert!(view(8, &[2], &[2], 0, DType::U16).is_ok());
    assert!(view(8, &[2], &[3], 0, DType::U16).is_err());
    assert!(view(8, &[2], &[2], 1, DType::U16).is_err());
    assert!(view(16, &[2], &[4], 4, DType::F32).is_ok());
    assert!(view(16, &[2], &[4], 2, DType::F32).is_err());
    assert!(view(16, &[2], &[8], 4, DType::F64).is_err());
    // A rank mismatch has no meaning.
    assert!(view(8, &[2, 2], &[1], 0, DType::U8).is_err());
}

/// Every small layout: accepted exactly when each index it can address is
/// inside the data, by enumerating the indices.
#[test]
fn accepted_exactly_when_every_index_is_in_bounds() {
    let (mut accepted, mut refused) = (0, 0);
    for bytes in 0..7usize {
        for d0 in 0..4usize {
            for d1 in 0..4usize {
                for s0 in -4..5isize {
                    for s1 in -4..5isize {
                        for offset in 0..8usize {
                            let mut inside = offset <= bytes;
                            for i in 0..d0 as isize {
                                for j in 0..d1 as isize {
                                    let at = offset as isize + i * s0 + j * s1;
                                    inside &= at >= 0 && (at as usize) < bytes;
                                }
                            }
                            let got = view(bytes, &[d0, d1], &[s0, s1], offset, DType::U8);
                            assert_eq!(
                                got.is_ok(),
                                inside,
                                "bytes={bytes} shape=[{d0},{d1}] strides=[{s0},{s1}] offset={offset}"
                            );
                            if inside {
                                accepted += 1;
                                // Reading it touches only its own elements.
                                let n = got.unwrap().to_contiguous().as_slice::<u8>().len();
                                assert_eq!(n, d0 * d1);
                            } else {
                                refused += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(
        accepted > 1000 && refused > 1000,
        "{accepted} accepted, {refused} refused"
    );
}

#[test]
fn f16_bits_read_back_as_the_f32_they_are() {
    let values = [0.0f32, -0.0, 1.0, -2.5, 65504.0, f32::INFINITY, 6.1e-5];
    let buf = ViewBuffer::from_vec(values.to_vec());
    let back = buf.to_f16_bits().from_f16_bits();
    assert_eq!(back.dtype(), DType::F32);
    for (got, want) in back.as_slice::<f32>().iter().zip(values) {
        assert_eq!(got.to_bits(), half::f16::from_f32(want).to_f32().to_bits());
    }
    let nan = ViewBuffer::from_vec(vec![f32::NAN])
        .to_f16_bits()
        .from_f16_bits();
    assert!(nan.as_slice::<f32>()[0].is_nan());
}
