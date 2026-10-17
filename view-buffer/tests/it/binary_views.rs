//! A binary op reads its operands where they lie: on any pair of views it
//! gives exactly what it gives their packed copies.
//!
//! `zip_with` has three ways in (contiguous slices, packed rows, a strided
//! walk), so the operand pairs here reach each one: both contiguous, both
//! crops or flips, mixed layouts, a transpose, a broadcast from a view, and
//! operands of two dtypes.

use view_buffer::core::dtype::ViewType;
use view_buffer::{apply_mask, BinaryOp, ViewBuffer};

fn image<T: ViewType + num_traits::FromPrimitive>(shape: &[usize], seed: usize) -> ViewBuffer {
    let n = shape.iter().product::<usize>();
    let data: Vec<T> = (0..n)
        .map(|i| T::from_usize((i * 37 + seed * 11) % 241).unwrap())
        .collect();
    ViewBuffer::from_vec_with_shape(data, shape.to_vec())
}

/// The ways an `[h, w, c]` operand can arrive, cut from a larger base.
fn layouts(base: &ViewBuffer, h: usize, w: usize, c: usize) -> Vec<(&'static str, ViewBuffer)> {
    vec![
        (
            "contiguous",
            base.slice(&[0, 0, 0], &[h, w, c]).to_contiguous(),
        ),
        ("crop", base.slice(&[3, 2, 0], &[3 + h, 2 + w, c])),
        (
            "vflip crop",
            base.flip(&[0]).slice(&[1, 1, 0], &[1 + h, 1 + w, c]),
        ),
        (
            "hflip crop",
            base.flip(&[1]).slice(&[2, 0, 0], &[2 + h, w, c]),
        ),
        (
            "transposed crop",
            base.permute(&[1, 0, 2]).slice(&[0, 1, 0], &[h, 1 + w, c]),
        ),
    ]
}

const OPS: [BinaryOp; 6] = [
    BinaryOp::Add,
    BinaryOp::Subtract,
    BinaryOp::Multiply,
    BinaryOp::Divide,
    BinaryOp::Maximum,
    BinaryOp::Blend,
];

fn assert_same(op: &BinaryOp, a: &ViewBuffer, b: &ViewBuffer, label: &str) {
    let got = op.execute(a, b);
    let want = op.execute(&a.to_contiguous(), &b.to_contiguous());
    assert_eq!(got.shape(), want.shape(), "{op:?} {label}");
    assert_eq!(got.to_blob(), want.to_blob(), "{op:?} {label}");
}

#[test]
fn a_binary_op_reads_any_pair_of_views_as_their_packed_copies() {
    let (h, w, c) = (13, 9, 3);
    let base_a = image::<u8>(&[24, 24, c], 1);
    let base_b = image::<u8>(&[24, 24, c], 2);
    for (la, a) in layouts(&base_a, h, w, c) {
        for (lb, b) in layouts(&base_b, h, w, c) {
            for op in &OPS {
                assert_same(op, &a, &b, &format!("{la} with {lb}"));
            }
        }
    }
}

#[test]
fn a_binary_op_broadcasts_from_a_view() {
    let (h, w, c) = (13, 9, 3);
    let base = image::<f32>(&[24, 24, c], 3);
    let mask_base = image::<f32>(&[24, 24, 1], 4);
    for (la, a) in layouts(&base, h, w, c) {
        for (lm, m) in layouts(&mask_base, h, w, 1) {
            for op in &OPS {
                assert_same(op, &a, &m, &format!("{la} with a {lm} [h, w, 1] operand"));
            }
        }
    }
}

#[test]
fn a_binary_op_reads_views_of_two_dtypes() {
    let (h, w, c) = (13, 9, 3);
    let base_a = image::<u8>(&[24, 24, c], 5);
    let base_b = image::<u16>(&[24, 24, c], 6);
    for (la, a) in layouts(&base_a, h, w, c) {
        for (lb, b) in layouts(&base_b, h, w, c) {
            for op in &OPS {
                assert_same(op, &a, &b, &format!("u8 {la} with u16 {lb}"));
            }
        }
    }
}

#[test]
fn apply_mask_reads_a_view_mask_as_its_packed_copy() {
    let (h, w, c) = (13, 9, 3);
    let base = image::<u8>(&[24, 24, c], 7);
    let mask_base = image::<u8>(&[24, 24, c], 8);
    for (la, a) in layouts(&base, h, w, c) {
        for (lm, m) in layouts(&mask_base, h, w, c) {
            for invert in [false, true] {
                let got = apply_mask(&a, &m, invert);
                let want = apply_mask(&a.to_contiguous(), &m.to_contiguous(), invert);
                assert_eq!(
                    got.to_blob(),
                    want.to_blob(),
                    "{la} masked by {lm}, invert={invert}"
                );
            }
        }
    }
}

/// Views of a buffer of `shape`, cut from a larger base: the buffer packed,
/// a crop of every axis, a flip of the first and of the last axis, and the
/// first two axes swapped (when there are two).
fn views_of(shape: &[usize], seed: usize, dtype: view_buffer::DType) -> Vec<(String, ViewBuffer)> {
    let rank = shape.len();
    let base_shape: Vec<usize> = shape.iter().map(|d| d + 3).collect();
    let base = image::<u8>(&base_shape, seed).cast(dtype);
    let ones = vec![1; rank];
    let ends: Vec<usize> = shape.iter().map(|d| d + 1).collect();
    let mut out = vec![
        (
            "packed".to_string(),
            base.slice(&vec![0; rank], shape).to_contiguous(),
        ),
        ("crop".to_string(), base.slice(&ones, &ends)),
        (
            "flip first".to_string(),
            base.flip(&[0]).slice(&ones, &ends),
        ),
        (
            "flip last".to_string(),
            base.flip(&[rank - 1]).slice(&ones, &ends),
        ),
    ];
    if rank >= 2 {
        let mut swapped_shape = base_shape.clone();
        swapped_shape.swap(0, 1);
        let mut axes: Vec<usize> = (0..rank).collect();
        axes.swap(0, 1);
        let swapped = image::<u8>(&swapped_shape, seed).cast(dtype).permute(&axes);
        out.push(("swap".to_string(), swapped.slice(&ones, &ends)));
    }
    out
}

/// Every shape pairing the reader walks differently, beyond rank 3: rank 1,
/// 2 and 4, an operand of lower rank broadcast against a higher one (a row,
/// a channel vector, a scalar-like `[1]`), and empty axes. Each operand in
/// every layout, in two dtype pairs, against the packed copies.
#[test]
fn a_binary_op_reads_views_of_any_rank_and_broadcast_as_their_packed_copies() {
    let pairs: [(&[usize], &[usize]); 9] = [
        (&[37], &[37]),
        (&[11, 7], &[11, 7]),
        (&[11, 7], &[7]),
        (&[3, 5, 4, 2], &[3, 5, 4, 2]),
        (&[9, 6, 3], &[6, 3]),
        (&[9, 6, 3], &[3]),
        (&[9, 6, 3], &[1]),
        (&[0, 5, 3], &[0, 5, 3]),
        (&[4, 0, 3], &[0, 3]),
    ];
    use view_buffer::DType::{F32, F64, I16, U8};
    let mut strided = 0;
    for (sa, sb) in pairs {
        for (ta, tb) in [(U8, U8), (F32, F64), (U8, I16)] {
            for (la, a) in views_of(sa, 1, ta) {
                for (lb, b) in views_of(sb, 2, tb) {
                    strided += usize::from(!b.layout_facts().is_contiguous());
                    for op in &OPS {
                        let label = format!("{ta:?} {sa:?} {la} with {tb:?} {sb:?} {lb}");
                        assert_same(op, &a, &b, &label);
                        assert_same(op, &b, &a, &format!("{label}, swapped"));
                    }
                }
            }
        }
    }
    // Each dtype pair's views are cut in that dtype, so every pair meets
    // strided operands, not only packed casts of them.
    assert!(strided > 300, "only {strided} strided operands");
}

/// `apply_mask` with a 2-D mask over a 3-D image reads the mask where it lies,
/// in every layout, as its packed copy.
#[test]
fn apply_mask_reads_a_2d_view_mask_as_its_packed_copy() {
    let (h, w, c) = (13, 9, 3);
    let base = image::<u8>(&[24, 24, c], 9);
    for (la, a) in layouts(&base, h, w, c) {
        for (lm, m) in views_of(&[h, w], 10, view_buffer::DType::U8) {
            for invert in [false, true] {
                for image in [a.clone(), a.cast(view_buffer::DType::F32)] {
                    let got = apply_mask(&image, &m, invert);
                    let want = apply_mask(&image.to_contiguous(), &m.to_contiguous(), invert);
                    assert_eq!(
                        got.to_blob(),
                        want.to_blob(),
                        "{la} {:?} masked by a 2-D {lm} mask, invert={invert}",
                        image.dtype()
                    );
                }
            }
        }
    }
}
