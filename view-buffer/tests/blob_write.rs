//! Parity tests for `ViewBuffer::write_blob_into` / `blob_len` vs `to_blob`.
//!
//! `to_blob` is now implemented on top of `write_blob_into`; these tests pin
//! the byte-level contract (including reuse of a non-empty scratch buffer,
//! strided inputs, and the round trip through `from_blob`).

use view_buffer::{DType, ViewBuffer, ViewExpr, ViewOp};

fn cases() -> Vec<(&'static str, ViewBuffer)> {
    let base: Vec<u8> = (0..(6 * 8 * 3) as u32).map(|v| (v % 251) as u8).collect();
    let contiguous = ViewBuffer::from_vec_with_shape(base.clone(), vec![6, 8, 3]);

    // Strided: transpose produces a non-contiguous view.
    let transposed = ViewExpr::new_source(contiguous.clone())
        .apply_op(view_buffer::ViewDto::View(ViewOp::Transpose(vec![1, 0, 2])))
        .plan()
        .execute();

    let f32_buf = ViewBuffer::from_vec_with_shape(
        (0..64).map(|v| v as f32 * 0.5).collect::<Vec<f32>>(),
        vec![8, 8],
    );

    vec![
        ("contiguous_u8", contiguous),
        ("transposed_u8", transposed),
        ("f32_2d", f32_buf),
    ]
}

#[test]
fn write_blob_into_matches_to_blob() {
    for (name, buf) in cases() {
        let blob = buf.to_blob();

        let mut written = Vec::new();
        buf.write_blob_into(&mut written);
        assert_eq!(blob, written, "fresh-buffer mismatch: {name}");

        // Reused scratch with prior contents: only the appended bytes count.
        let mut scratch = vec![0xAB_u8; 17];
        buf.write_blob_into(&mut scratch);
        assert_eq!(
            &scratch[..17],
            &[0xAB; 17][..],
            "scratch prefix clobbered: {name}"
        );
        assert_eq!(&scratch[17..], &blob[..], "scratch append mismatch: {name}");
    }
}

#[test]
fn blob_len_matches_written_length() {
    for (name, buf) in cases() {
        let blob = buf.to_blob();
        assert_eq!(buf.blob_len(), blob.len(), "blob_len mismatch: {name}");
    }
}

#[test]
fn write_blob_into_round_trips() {
    for (name, buf) in cases() {
        let mut blob = Vec::new();
        buf.write_blob_into(&mut blob);
        let decoded = ViewBuffer::from_blob(&blob).expect("decode");
        assert_eq!(decoded.shape(), buf.shape(), "shape mismatch: {name}");
        assert_eq!(decoded.dtype(), buf.dtype(), "dtype mismatch: {name}");
        if buf.dtype() == DType::U8 {
            let orig = buf.to_contiguous();
            assert_eq!(
                decoded.as_slice::<u8>(),
                orig.as_slice::<u8>(),
                "payload mismatch: {name}"
            );
        }
    }
}
