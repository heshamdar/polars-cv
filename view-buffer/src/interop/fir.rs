//! fast_image_resize interoperability: a zero-copy [`fir::ImageView`] over a
//! buffer whose pixels are packed within each row.
//!
//! fast_image_resize reads its source one row at a time, so the rows need not
//! be adjacent, or even in ascending order: a crop (rows with a gap between
//! them) and a vertical flip (a negative row stride) are read where they lie,
//! through [`ViewBuffer::dense_rows`]. Any other layout is refused with
//! [`BufferError::IncompatibleLayout`] and has to be packed first.

use std::marker::PhantomData;

use fast_image_resize as fir;
use fir::pixels::InnerPixel;

use crate::core::buffer::{BufferError, ViewBuffer};
use crate::core::dtype::ViewType;
use crate::core::layout::ExternalLayout;
use crate::interop::{validate_layout, ExternalView};

/// A buffer's rows as fast_image_resize pixels of type `P`.
pub struct FirView<'a, P> {
    width: u32,
    height: u32,
    rows: Vec<&'a [P]>,
}

// SAFETY: every row holds exactly `width` pixels (`try_view`).
unsafe impl<P: InnerPixel> fir::ImageView for FirView<'_, P> {
    type Pixel = P;

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn iter_rows(&self, start_row: u32) -> impl Iterator<Item = &[P]> {
        // A subslice, not `skip`: fir asks for rows inside its innermost
        // loops, where `Skip`'s per-item check is measurable.
        self.rows
            .get(start_row as usize..)
            .unwrap_or_default()
            .iter()
            .copied()
    }
}

/// Builds a [`FirView`] of pixel type `P` over a buffer.
pub struct FirViewAdapter<P>(PhantomData<P>);

impl<'a, P> ExternalView<'a> for FirViewAdapter<P>
where
    P: InnerPixel,
    P::Component: ViewType,
{
    type View = FirView<'a, P>;
    const LAYOUT: ExternalLayout = ExternalLayout::FastImageResize;

    fn try_view(buf: &'a ViewBuffer) -> Result<Self::View, BufferError> {
        validate_layout(buf, Self::LAYOUT)?;
        let expected = <P::Component as ViewType>::DTYPE;
        if buf.dtype() != expected {
            return Err(BufferError::TypeMismatch {
                expected,
                got: buf.dtype(),
            });
        }
        let shape = buf.shape();
        let (h, w) = (shape[0], shape[1]);
        let channels = P::count_of_components();
        if shape.get(2).copied().unwrap_or(1) != channels {
            return Err(BufferError::ShapeMismatch {
                expected: vec![h, w, channels],
                got: shape.to_vec(),
            });
        }
        let dimension = |n: usize| u32::try_from(n).expect("fast_image_resize sizes are u32");
        let (width, height) = (dimension(w), dimension(h));
        let rows = buf
            .dense_rows::<P::Component>()
            .expect("a layout fast_image_resize accepts has packed rows")
            .into_iter()
            .map(|row| as_pixels(row, w))
            .collect();
        Ok(FirView {
            width,
            height,
            rows,
        })
    }
}

/// `components` (`count * P::count_of_components()` of them) as `count`
/// pixels.
fn as_pixels<P: InnerPixel>(components: &[P::Component], count: usize) -> &[P] {
    assert_pixel_is_its_components::<P>();
    assert_eq!(components.len(), count * P::count_of_components());
    // SAFETY: a fir pixel is `#[repr(C)]` over `[Component; N]`, which the
    // assertion above confirms by size and alignment, so `count` pixels are
    // exactly the slice's components.
    unsafe { std::slice::from_raw_parts(components.as_ptr().cast::<P>(), count) }
}

/// [`as_pixels`], mutably: a typed destination buffer as fir pixels.
pub fn as_pixels_mut<P: InnerPixel>(components: &mut [P::Component]) -> &mut [P] {
    assert_pixel_is_its_components::<P>();
    let n = P::count_of_components();
    assert_eq!(components.len() % n, 0);
    // SAFETY: as in `as_pixels`.
    unsafe {
        std::slice::from_raw_parts_mut(components.as_mut_ptr().cast::<P>(), components.len() / n)
    }
}

fn assert_pixel_is_its_components<P: InnerPixel>() {
    assert_eq!(
        (size_of::<P>(), align_of::<P>()),
        (
            size_of::<P::Component>() * P::count_of_components(),
            align_of::<P::Component>()
        ),
        "a fast_image_resize pixel is not an array of its components"
    );
}
