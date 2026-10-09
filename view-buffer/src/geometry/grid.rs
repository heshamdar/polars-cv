//! Patch grids: the cells that tile an `H × W` image.
//!
//! The one definition of "which patches cover this image". The
//! `patch_grid` expression lists them as rows, and a crop per row reads them;
//! anything else that cuts an image into patches reads [`PatchGrid::cells`]
//! rather than recomputing the arithmetic, so the two can never disagree about
//! where patch `i` is.

/// What a grid does with the remainder of an axis that a whole patch no longer
/// fits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GridEdge {
    /// Leave it uncovered: only whole patches at multiples of the stride.
    #[default]
    Drop,
    /// Cover it with one more patch aligned to the far edge, overlapping its
    /// neighbour (every patch stays whole and in bounds).
    Shift,
}

crate::naming::named_variants!(GridEdge: "What a patch grid does with the remainder of an axis a whole patch no longer fits.\n\n- DROP: leave it uncovered; only whole patches at multiples of the stride (default).\n- SHIFT: add one patch aligned to the far edge, overlapping its neighbour, so\n  every pixel is covered and every patch stays whole." {
    "drop" => Drop,
    "shift" => Shift,
});

/// One patch of a grid: its grid position and its top-left pixel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridCell {
    /// Grid row (0-based, top to bottom).
    pub row: u32,
    /// Grid column (0-based, left to right).
    pub col: u32,
    /// Top pixel row of the patch.
    pub top: u32,
    /// Left pixel column of the patch.
    pub left: u32,
}

/// A patch size, stride and edge rule; [`PatchGrid::cells`] lays it over an
/// image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PatchGrid {
    size: [u32; 2],
    stride: [u32; 2],
    edge: GridEdge,
}

impl PatchGrid {
    /// A grid of `size = [height, width]` patches every `stride` pixels.
    ///
    /// # Errors
    /// A zero size or stride, which describes no grid.
    pub fn new(size: [u32; 2], stride: [u32; 2], edge: GridEdge) -> Result<Self, String> {
        if size.contains(&0) {
            return Err(format!("patch size must be positive, got {size:?}"));
        }
        if stride.contains(&0) {
            return Err(format!("patch stride must be positive, got {stride:?}"));
        }
        Ok(Self { size, stride, edge })
    }

    /// The patch size, `[height, width]`.
    pub fn size(&self) -> [u32; 2] {
        self.size
    }

    /// The patch origins along one axis of length `extent`: every multiple of
    /// the stride where a whole patch fits, then (under [`GridEdge::Shift`])
    /// the far-edge origin if those leave a remainder. None when the axis is
    /// shorter than a patch.
    fn starts(&self, extent: u32, axis: usize) -> Vec<u32> {
        let (size, stride) = (self.size[axis], self.stride[axis]);
        let Some(last) = extent.checked_sub(size) else {
            return Vec::new();
        };
        let mut starts: Vec<u32> = (0..=last).step_by(stride as usize).collect();
        if self.edge == GridEdge::Shift && starts.last() != Some(&last) {
            starts.push(last);
        }
        starts
    }

    /// Every patch of an `height × width` image, row-major.
    pub fn cells(&self, height: u32, width: u32) -> Vec<GridCell> {
        let (tops, lefts) = (self.starts(height, 0), self.starts(width, 1));
        let mut cells = Vec::with_capacity(tops.len() * lefts.len());
        for (row, &top) in (0u32..).zip(&tops) {
            for (col, &left) in (0u32..).zip(&lefts) {
                cells.push(GridCell {
                    row,
                    col,
                    top,
                    left,
                });
            }
        }
        cells
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The naive reference: test every origin, keep whole patches on the
    /// stride lattice, then add the far-edge one per axis under `Shift`.
    fn reference(extent: u32, size: u32, stride: u32, edge: GridEdge) -> Vec<u32> {
        let mut out: Vec<u32> = (0..extent)
            .filter(|o| o % stride == 0 && o + size <= extent)
            .collect();
        if edge == GridEdge::Shift && extent >= size && out.last() != Some(&(extent - size)) {
            out.push(extent - size);
        }
        out
    }

    #[test]
    fn cells_match_the_reference_over_an_exhaustive_sweep() {
        for edge in [GridEdge::Drop, GridEdge::Shift] {
            for (h, w) in [(1, 1), (7, 13), (16, 16), (47, 31)] {
                for ph in 1..=12 {
                    for pw in [1, 5, 12] {
                        for sy in 1..=12 {
                            for sx in [1, 4, 13] {
                                let grid = PatchGrid::new([ph, pw], [sy, sx], edge).unwrap();
                                let tops = reference(h, ph, sy, edge);
                                let lefts = reference(w, pw, sx, edge);
                                let want: Vec<(u32, u32)> = tops
                                    .iter()
                                    .flat_map(|&t| lefts.iter().map(move |&l| (t, l)))
                                    .collect();
                                let got: Vec<(u32, u32)> =
                                    grid.cells(h, w).iter().map(|c| (c.top, c.left)).collect();
                                assert_eq!(
                                    got, want,
                                    "{h}x{w} size {ph}x{pw} stride {sy}x{sx} {edge:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn every_cell_is_whole_in_bounds_unique_and_row_major() {
        for edge in [GridEdge::Drop, GridEdge::Shift] {
            for h in 1..=48 {
                for size in 1..=12 {
                    for stride in 1..=12 {
                        let grid = PatchGrid::new([size, size], [stride, stride], edge).unwrap();
                        let w = 48 - h / 2;
                        let cells = grid.cells(h, w);
                        let ncols = cells.iter().map(|c| c.col + 1).max().unwrap_or(0);
                        for (i, c) in cells.iter().enumerate() {
                            assert!(c.top + size <= h && c.left + size <= w);
                            assert_eq!(i as u32, c.row * ncols + c.col, "row-major order");
                        }
                        let mut origins: Vec<_> = cells.iter().map(|c| (c.top, c.left)).collect();
                        origins.dedup();
                        assert_eq!(origins.len(), cells.len(), "no duplicate cells");
                    }
                }
            }
        }
    }

    #[test]
    fn drop_counts_whole_patches_and_shift_covers_every_pixel() {
        for h in 1..=48u32 {
            for size in 1..=12u32 {
                for stride in 1..=12u32 {
                    let drop = PatchGrid::new([size, 1], [stride, 1], GridEdge::Drop).unwrap();
                    let n = drop.cells(h, 1).len() as u32;
                    let want = if h < size { 0 } else { (h - size) / stride + 1 };
                    assert_eq!(n, want, "h {h} size {size} stride {stride}");

                    if h >= size && stride <= size {
                        let shift =
                            PatchGrid::new([size, 1], [stride, 1], GridEdge::Shift).unwrap();
                        let mut covered = vec![false; h as usize];
                        for c in shift.cells(h, 1) {
                            covered[c.top as usize..(c.top + size) as usize].fill(true);
                        }
                        assert!(
                            covered.iter().all(|&c| c),
                            "h {h} size {size} stride {stride}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn an_axis_shorter_than_a_patch_has_no_cells_under_either_edge() {
        for edge in [GridEdge::Drop, GridEdge::Shift] {
            let grid = PatchGrid::new([8, 8], [8, 8], edge).unwrap();
            assert!(grid.cells(7, 100).is_empty());
            assert!(grid.cells(100, 7).is_empty());
            assert_eq!(grid.cells(8, 8).len(), 1);
        }
    }

    #[test]
    fn zero_sizes_and_strides_are_refused() {
        assert!(PatchGrid::new([0, 4], [4, 4], GridEdge::Drop).is_err());
        assert!(PatchGrid::new([4, 4], [4, 0], GridEdge::Drop).is_err());
    }

    #[test]
    fn a_small_grid_is_exactly_as_written() {
        let grid = PatchGrid::new([4, 4], [4, 4], GridEdge::Shift).unwrap();
        let got: Vec<_> = grid
            .cells(6, 9)
            .into_iter()
            .map(|c| (c.row, c.col, c.top, c.left))
            .collect();
        assert_eq!(
            got,
            vec![
                (0, 0, 0, 0),
                (0, 1, 0, 4),
                (0, 2, 0, 5),
                (1, 0, 2, 0),
                (1, 1, 2, 4),
                (1, 2, 2, 5),
            ]
        );
    }
}
