//! Channel data movement: reorder and merge, in the buffer's own dtype.

use crate::core::buffer::ViewBuffer;

/// Reorder the channels of an `[H, W, C]` buffer: output channel `i` is
/// input channel `order[i]`. Pure data movement, in the input's own dtype —
/// the one channel reorder (`channel_swap`, and the colour conversions'
/// RGB <-> BGR).
pub fn apply_channel_swap(buf: &ViewBuffer, order: &[usize]) -> ViewBuffer {
    let shape = buf.shape();
    assert!(shape.len() == 3, "ChannelSwap requires 3D [H, W, C] input");
    let (h, w, c) = (shape[0], shape[1], shape[2]);
    assert!(
        order.len() == c,
        "ChannelSwap order length {} must match channel count {}",
        order.len(),
        c
    );
    // Read in place, at the view's own row stride (a crop, a vertical flip).
    let rows_buf = buf.to_dense_rows();
    crate::core::dtype::with_dtype!(buf.dtype(), T => {
        let rows = rows_buf.dense_rows::<T>().expect("a rank-2 or rank-3 buffer has packed rows after to_dense_rows");
        let mut out: Vec<T> = Vec::with_capacity(h * w * c);
        for row in rows {
            for pixel in row.chunks_exact(c) {
                out.extend(order.iter().map(|&i| pixel[i]));
            }
        }
        ViewBuffer::from_vec_with_shape(out, vec![h, w, c])
    })
}

/// Join the inputs' channel axes: each `[H, W]` input brings one channel,
/// each `[H, W, C]` input its `C`, in order, into one `[H, W, total]` buffer.
pub fn apply_channel_merge(buffers: &[&ViewBuffer]) -> ViewBuffer {
    assert!(
        !buffers.is_empty(),
        "ChannelMerge requires at least one input"
    );
    let h = buffers[0].shape()[0];
    let w = buffers[0].shape()[1];
    // Every input [H, W] or [H, W, C] of one H and W (`validate_channel_merge`).
    let counts: Vec<usize> = buffers
        .iter()
        .enumerate()
        .map(|(i, buf)| match buf.shape() {
            [bh, bw] if *bh == h && *bw == w => 1,
            [bh, bw, c] if *bh == h && *bw == w => *c,
            s => panic!("ChannelMerge input {i} has shape {s:?}, expected [{h}, {w}(, C)]"),
        })
        .collect();
    let c: usize = counts.iter().sum();

    let contigs: Vec<ViewBuffer> = buffers.iter().map(|b| b.to_contiguous()).collect();
    // Pure data movement, in the inputs' own dtype (they share one:
    // `validate_channel_merge`).
    crate::core::dtype::with_dtype!(buffers[0].dtype(), T => {
        let planes: Vec<&[T]> = contigs.iter().map(|b| b.as_slice::<T>()).collect();
        let mut out: Vec<T> = Vec::with_capacity(h * w * c);
        for pixel in 0..h * w {
            for (plane, &n) in planes.iter().zip(&counts) {
                out.extend_from_slice(&plane[pixel * n..(pixel + 1) * n]);
            }
        }
        ViewBuffer::from_vec_with_shape(out, vec![h, w, c])
    })
}
