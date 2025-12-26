use crate::buffer::{ViewBuffer, BufferError};
use crate::dtype::{DType, ViewType};
use crate::views::{ExternalView, validate_layout}; // Import shared traits
use crate::layout::ExternalLayout;
use image::Pixel;
use std::marker::PhantomData;

/// A zero-copy view over a ViewBuffer interpreted as an image.
#[derive(Debug, Clone)]
pub struct ImageView<'a, P: Pixel> {
    pub data: &'a [P::Subpixel],
    pub width: u32,
    pub height: u32,
    pub row_stride: usize,
    _marker: PhantomData<P>,
}

impl<'a, P> ImageView<'a, P> 
where 
    P: Pixel,
    P::Subpixel: ViewType + 'static,
{
    pub fn get_pixel(&self, x: u32, y: u32) -> &[P::Subpixel] {
        let start = (y as usize * self.row_stride) + (x as usize * P::CHANNEL_COUNT as usize);
        &self.data[start..start + P::CHANNEL_COUNT as usize]
    }
}

// --- Image Adapter ---

pub struct ImageViewAdapter<P>(PhantomData<P>);

impl<'a, P> ExternalView<'a> for ImageViewAdapter<P> 
where 
    P: Pixel,
    P::Subpixel: ViewType + 'static
{
    type View = ImageView<'a, P>;
    const LAYOUT: ExternalLayout = ExternalLayout::ImageCrate;

    fn try_view(buf: &'a ViewBuffer) -> Result<Self::View, BufferError> {
        validate_layout(buf, Self::LAYOUT)?;

        if buf.dtype() != P::Subpixel::DTYPE {
            return Err(BufferError::TypeMismatch { 
                expected: P::Subpixel::DTYPE, 
                got: buf.dtype() 
            });
        }

        let shape = buf.shape();
        let (h, w) = (shape[0], shape[1]);
        let stride_bytes = buf.strides_bytes()[0];
        let elem_size = std::mem::size_of::<P::Subpixel>() as isize;
        let row_stride_elems = (stride_bytes / elem_size) as usize;

        let total_elems = row_stride_elems * h;
        let ptr = unsafe { buf.as_ptr::<P::Subpixel>() };
        
        let data = unsafe {
            std::slice::from_raw_parts(ptr, total_elems)
        };

        Ok(ImageView {
            data,
            width: w as u32,
            height: h as u32,
            row_stride: row_stride_elems,
            _marker: PhantomData,
        })
    }
}

// --- Legacy / Helper Trait ---
pub trait AsImageView {
    fn as_image_view<P>(&self) -> Result<ImageView<P>, BufferError>
    where 
        P: Pixel,
        P::Subpixel: ViewType + 'static;
}

impl AsImageView for ViewBuffer {
    fn as_image_view<P>(&self) -> Result<ImageView<P>, BufferError>
    where 
        P: Pixel,
        P::Subpixel: ViewType + 'static,
    {
        ImageViewAdapter::try_view(self)
    }
}