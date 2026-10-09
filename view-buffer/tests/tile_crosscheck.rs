//! `tile` against `crop`: two paths to the same patches.
//!
//! Patch `i` of `tile(img)` must be exactly `crop(img)` at the grid's cell
//! `i` — the cell `PatchGrid::cells` lists, which is what `patch_grid` lays out
//! as rows. Swept over image sizes that do and do not divide, patch sizes,
//! strides (touching, overlapping, gapped), both edge rules, rank 2 and 3,
//! two dtypes and a strided (transposed) input, so the packing of each window
//! is checked against the crop's view of it wherever the window lies.

use view_buffer::geometry::grid::{GridEdge, PatchGrid};
use view_buffer::{ImageOp, ImageOpKind, ViewBuffer, ViewDto, ViewExpr, ViewOp};

fn run(buf: &ViewBuffer, op: ViewDto) -> ViewBuffer {
    ViewExpr::new_source(buf.clone())
        .apply_op(op)
        .plan()
        .execute()
        .to_contiguous()
}

fn tile(h: u32, w: u32, sy: u32, sx: u32, edge: GridEdge) -> ViewDto {
    ViewDto::Image(ImageOp {
        kind: ImageOpKind::Tile {
            height: h,
            width: w,
            stride_height: Some(sy),
            stride_width: Some(sx),
            edge,
        },
    })
}

fn crop(top: u32, left: u32, h: u32, w: u32) -> ViewDto {
    ViewDto::View(ViewOp::Crop {
        top,
        left,
        height: Some(h),
        width: Some(w),
    })
}

/// Every patch of `tile(buf)` against the crop of its cell.
fn check(buf: &ViewBuffer, label: &str) {
    let (ih, iw) = (buf.shape()[0] as u32, buf.shape()[1] as u32);
    for edge in [GridEdge::Drop, GridEdge::Shift] {
        for (ph, pw) in [(1, 1), (2, 3), (4, 4), (5, 2)] {
            for (sy, sx) in [(ph, pw), (1, 2), (3, 1), (ph + 2, pw + 1)] {
                let grid = PatchGrid::new([ph, pw], [sy, sx], edge).unwrap();
                let cells = grid.cells(ih, iw);
                let patches = run(buf, tile(ph, pw, sy, sx, edge));
                let what = format!("{label} {ih}x{iw} patch {ph}x{pw} stride {sy}x{sx} {edge:?}");
                let mut want_shape = vec![cells.len(), ph as usize, pw as usize];
                want_shape.extend_from_slice(&buf.shape()[2..]);
                assert_eq!(patches.shape(), want_shape.as_slice(), "{what}");
                let per: usize = want_shape[1..].iter().product();
                let bytes = patches.to_blob();
                let header = bytes.len() - per * cells.len() * buf.dtype().size_of();
                for (i, cell) in cells.iter().enumerate() {
                    let want = run(buf, crop(cell.top, cell.left, ph, pw)).to_blob();
                    let size = per * buf.dtype().size_of();
                    let got = &bytes[header + i * size..header + (i + 1) * size];
                    assert_eq!(got, &want[want.len() - size..], "{what}: patch {i}");
                }
            }
        }
    }
}

#[test]
fn every_patch_is_the_crop_of_its_cell() {
    for (h, w) in [(1, 1), (7, 9), (12, 12), (13, 6)] {
        let n = h * w;
        let gray = ViewBuffer::from_vec_with_shape(
            (0..n).map(|i| (i % 251) as u8).collect::<Vec<_>>(),
            vec![h, w],
        );
        check(&gray, "u8 [H, W]");
        let rgb = ViewBuffer::from_vec_with_shape(
            (0..n * 3).map(|i| i as f32 * 0.5).collect::<Vec<_>>(),
            vec![h, w, 3],
        );
        check(&rgb, "f32 [H, W, 3]");
        // A transposed view: patches are cut from a non-contiguous input.
        let wide = ViewBuffer::from_vec_with_shape(
            (0..n * 2).map(|i| (i * 7 % 253) as u8).collect::<Vec<_>>(),
            vec![w, h, 2],
        );
        let transposed = run(&wide, ViewDto::View(ViewOp::transpose(&[1, 0, 2])));
        let strided = wide.permute(&[1, 0, 2]);
        assert_eq!(transposed.shape(), strided.shape());
        check(&strided, "u8 [H, W, 2] transposed view");
    }
}

#[test]
fn an_image_smaller_than_a_patch_has_no_patches() {
    let buf = ViewBuffer::from_vec_with_shape(vec![1u8; 3 * 4 * 3], vec![3, 4, 3]);
    let out = run(&buf, tile(4, 4, 4, 4, GridEdge::Shift));
    assert_eq!(out.shape(), &[0, 4, 4, 3]);
}
