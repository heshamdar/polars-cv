/*
 * Local addition to the vendored zune-jpeg 0.5.8 (see VENDORED.md), under the
 * crate's own licences (MIT, Apache-2.0 or Zlib).
 */

//! Scaled output: decode at 1/2, 1/4 or 1/8 of each dimension, the libjpeg
//! way.
//!
//! Entropy decoding is unchanged (every coefficient still has to be read to
//! advance the bitstream). What shrinks is everything after it: each 8x8
//! block is inverse transformed straight to its `n x n` output (`n` = 4, 2
//! or 1) by a reduced IDCT, into component planes `n / 8` the size, and
//! upsampling and colour conversion then run on those planes.
//!
//! As in libjpeg (`jdmaster.c`), a subsampled chroma component is given a
//! *larger* IDCT than luma where that makes it land at the output size (4:2:0
//! at 1/4: luma 2x2, chroma 4x4), so most files need no upsampling at all;
//! what remains is libjpeg's ("fancy" triangle filters for 2x expansion when
//! the scale is above 1/8, replication otherwise). The reduced IDCTs are
//! libjpeg's `jidctred.c` (`jpeg_idct_4x4`, `jpeg_idct_2x2`) and its DC-only
//! `1x1`, in zune's integer arithmetic for the latter. Colour conversion is
//! zune's own. So the output is within a step or two of libjpeg-turbo's
//! scaled decode, not bit-identical to it.
//!
//! Baseline files decode into whole-image planes (the buffered path zune
//! already uses for files whose components arrive in separate scans);
//! progressive files transform their stored coefficients at the end.

use alloc::vec;
use alloc::vec::Vec;

use zune_core::bytestream::ZByteReaderTrait;
use zune_core::colorspace::ColorSpace;
use zune_core::log::{error, trace, warn};

use crate::bitstream::BitStream;
use crate::decoder::{IDCTPtr, MAX_COMPONENTS};
use crate::errors::DecodeErrors;
use crate::mcu::{McuContinuation, DCT_BLOCK};
use crate::mcu_prog::get_marker;
use crate::misc::setup_component_params;
use crate::worker::color_convert;
use crate::JpegDecoder;

/// How far to scale a decoded image down
/// ([`JpegDecoder::set_idct_scale`]).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum IdctScale {
    /// The full image: the ordinary decode.
    #[default]
    Full,
    /// 1/2 of each dimension.
    Half,
    /// 1/4 of each dimension.
    Quarter,
    /// 1/8 of each dimension: one sample per 8x8 block, its DC term.
    Eighth
}

impl IdctScale {
    /// Output samples per 8 stored ones along each dimension: the side of
    /// the block one luma IDCT writes.
    #[must_use]
    pub const fn block_size(self) -> usize {
        match self {
            IdctScale::Full => 8,
            IdctScale::Half => 4,
            IdctScale::Quarter => 2,
            IdctScale::Eighth => 1
        }
    }

    /// The decoded length of a stored dimension `len`, rounded up as libjpeg
    /// rounds: `ceil(len * block_size / 8)`.
    #[must_use]
    pub const fn scaled(self, len: usize) -> usize {
        (len * self.block_size() + 7) / 8
    }
}

/// How one component is decoded and brought to the output size.
#[derive(Copy, Clone, Default)]
pub(crate) struct ComponentPlan {
    /// Side of the block its IDCT writes (1, 2, 4 or 8).
    block:    usize,
    /// Output samples per plane sample, horizontally and vertically.
    h_expand: usize,
    v_expand: usize,
    /// Its plane's meaningful size (the rest is block padding).
    width:    usize,
    height:   usize
}

impl ComponentPlan {
    /// The IDCT writing `block x block` samples, or `None` for the
    /// decoder's own full 8x8 one.
    fn idct(self) -> Option<IDCTPtr> {
        match self.block {
            1 => Some(idct_1x1_scaled),
            2 => Some(idct_2x2_scaled),
            4 => Some(idct_4x4_scaled),
            _ => None
        }
    }
}

impl<T: ZByteReaderTrait> JpegDecoder<T> {
    /// Each component's IDCT size and expansion at the decoder's scale.
    ///
    /// libjpeg's rule: a component's IDCT doubles from the luma size while
    /// that still leaves an integral expansion on both axes, up to 8. Fails
    /// for sampling factors whose expansion is not integral (no libjpeg
    /// upsampler handles those either).
    fn scaled_plans(&self) -> Result<[ComponentPlan; MAX_COMPONENTS], DecodeErrors> {
        let scale = self.idct_scale.block_size();
        let (width, height) = (usize::from(self.info.width), usize::from(self.info.height));
        let (h_max, v_max) = (self.h_max * scale, self.v_max * scale);
        let mut plans = [ComponentPlan::default(); MAX_COMPONENTS];

        for (plan, component) in plans.iter_mut().zip(&self.components) {
            let (h, v) = (component.horizontal_sample, component.vertical_sample);
            let mut block = scale;
            while block < 8 && h_max % (h * block * 2) == 0 && v_max % (v * block * 2) == 0 {
                block *= 2;
            }
            if h_max % (h * block) != 0 || v_max % (v * block) != 0 {
                return Err(DecodeErrors::FormatStatic(
                    "Sampling factors with no integral expansion cannot be decoded scaled"
                ));
            }
            *plan = ComponentPlan {
                block,
                h_expand: h_max / (h * block),
                v_expand: v_max / (v * block),
                width: (width * h * block).div_ceil(self.h_max * 8),
                height: (height * v * block).div_ceil(self.v_max * 8)
            };
        }
        Ok(plans)
    }

    /// Whether the output never reads component `pos`'s AC coefficients, so
    /// a progressive file's AC scans of it need not be decoded: at 1/8, a
    /// component whose IDCT is the 1x1 (its DC alone; a subsampled one gets a
    /// larger IDCT, which reads AC too) or one the output does not read.
    /// Never at the other scales.
    pub(crate) fn ac_coefficients_unread(&self, pos: usize) -> bool {
        self.idct_scale == IdctScale::Eighth
            && (!self.scaled_component_needed(pos)
                || self.scaled_plans().is_ok_and(|plans| plans[pos].block == 1))
    }

    /// Whether component `pos` contributes to the output colourspace.
    fn scaled_component_needed(&self, pos: usize) -> bool {
        pos < self.options.jpeg_get_out_colorspace().num_components()
            || self.components.len() == 4
            || matches!(self.input_colorspace, ColorSpace::YCCK | ColorSpace::CMYK)
    }

    /// The components colour conversion reads: as `post_process`, a YCbCr ->
    /// Luma decode reads the Y plane alone.
    fn scaled_output_components(&self) -> usize {
        let out_colorspace = self.options.jpeg_get_out_colorspace();
        if out_colorspace == ColorSpace::Luma {
            self.components.len().min(out_colorspace.num_components())
        } else {
            self.components.len()
        }
    }

    /// Decode a baseline image at the scale set with `set_idct_scale`.
    ///
    /// The MCU loop is `decode_mcu_ycbcr_baseline`'s, with each component's
    /// IDCT writing its scaled block. When every component the output reads
    /// arrives in the first scan already at the output size, each MCU row is
    /// colour converted as soon as it is decoded, from planes one MCU row
    /// tall that stay in cache (the full decode's streaming); otherwise the
    /// planes hold the whole image and are upsampled at the end.
    pub(crate) fn decode_mcu_scaled_baseline(
        &mut self, pixels: &mut [u8]
    ) -> Result<(), DecodeErrors> {
        setup_component_params(self)?;
        self.check_tables()?;

        if self.input_colorspace == ColorSpace::Luma && self.is_interleaved {
            warn!("Grayscale image with down-sampled component, resetting component details");
            self.reset_params();
        }
        let (mcu_width, mcu_height) = if self.is_interleaved {
            (self.mcu_x, self.mcu_y)
        } else {
            (
                usize::from(self.info.width).div_ceil(8),
                usize::from(self.info.height).div_ceil(8)
            )
        };
        let plans = self.scaled_plans()?;
        let all_components_in_first_scan = usize::from(self.num_scans) == self.components.len();
        let streaming = all_components_in_first_scan
            && plans[..self.scaled_output_components()]
                .iter()
                .all(|plan| plan.h_expand == 1 && plan.v_expand == 1);
        let buffered_mcu_rows = if streaming { 1 } else { mcu_height };

        let mut planes: [Vec<i16>; MAX_COMPONENTS] = core::array::from_fn(|_| vec![]);
        let mut strides = [0; MAX_COMPONENTS];
        for pos in 0..self.components.len() {
            let needed = self.scaled_component_needed(pos);
            let plan = plans[pos];
            let component = &mut self.components[pos];
            component.needed = needed;
            component.block_size = plan.block;
            component.scaled_idct = plan.idct();
            // The plane's stride: the MCU columns' blocks at the scaled size.
            component.width_stride = component.width_stride / 8 * plan.block;
            strides[pos] = component.width_stride;
            // 128 (mid grey) where a truncated stream leaves rows undecoded.
            let rows = buffered_mcu_rows * component.vertical_sample * plan.block;
            planes[pos] = vec![128; component.width_stride * rows];
        }
        // Output rows per MCU row (the same for every component, streaming).
        let mcu_rows_out = self.components[0].vertical_sample * plans[0].block;
        let mut rows_written = 0;

        let mut stream = BitStream::new();
        let mut tmp = [0_i32; DCT_BLOCK];

        'sos: loop {
            for i in 0..mcu_height {
                if stream.overread_by > 0 {
                    if self.options.strict_mode() {
                        return Err(DecodeErrors::FormatStatic("Premature end of buffer"));
                    }
                    error!("Premature end of buffer");
                    break 'sos;
                }
                // Streaming, every MCU row decodes into the top of the planes.
                let continuation = self.decode_mcu_width::<true>(
                    mcu_width,
                    if streaming { 0 } else { i },
                    &mut tmp,
                    &mut stream,
                    &mut planes
                )?;
                if streaming {
                    rows_written +=
                        self.convert_scaled_rows(&planes, strides[0], rows_written, mcu_rows_out, pixels)?;
                }
                match continuation {
                    McuContinuation::Ok => {}
                    McuContinuation::AnotherSos if all_components_in_first_scan => {
                        warn!("More than one SOS despite already having all components");
                        break 'sos;
                    }
                    McuContinuation::AnotherSos => continue 'sos,
                    McuContinuation::Terminate => {
                        warn!("Got terminate signal, will not process further");
                        break 'sos;
                    }
                }
            }
            break;
        }

        // Read up to the EOI, as the full-size decode does.
        if !stream.seen_eoi {
            if let Ok(_marker) = get_marker(&mut self.stream, &mut stream) {
                trace!("Found marker {:?}", _marker);
            }
        }

        if streaming {
            // Rows a truncated or terminated stream never reached: mid grey,
            // as the full decode leaves them.
            let width = self.idct_scale.scaled(usize::from(self.info.width));
            let row_len = width * self.options.jpeg_get_out_colorspace().num_components();
            if let Some(rest) = pixels.get_mut(rows_written * row_len..) {
                rest.fill(128);
            }
            return Ok(());
        }
        self.finish_scaled(&planes, &strides, &plans, pixels)
    }

    /// Colour convert up to `rows` output rows starting at `first`, from
    /// planes at the output size sharing one `stride`; returns how many there
    /// were (fewer at the bottom edge).
    fn convert_scaled_rows(
        &self, planes: &[Vec<i16>; MAX_COMPONENTS], stride: usize, first: usize, rows: usize,
        pixels: &mut [u8]
    ) -> Result<usize, DecodeErrors> {
        let out_colorspace = self.options.jpeg_get_out_colorspace();
        let width = self.idct_scale.scaled(usize::from(self.info.width));
        let height = self.idct_scale.scaled(usize::from(self.info.height));
        let rows = rows.min(height.saturating_sub(first));
        let row_len = width * out_colorspace.num_components();
        let Some(out) = pixels.get_mut(first * row_len..(first + rows) * row_len) else {
            return Err(DecodeErrors::FormatStatic("Output too small for its rows"));
        };
        let mut samples: [&[i16]; MAX_COMPONENTS] = [&[]; MAX_COMPONENTS];
        for (sample, plane) in samples
            .iter_mut()
            .zip(planes)
            .take(self.scaled_output_components())
        {
            *sample = plane;
        }
        color_convert(
            &samples,
            self.color_convert_16,
            self.input_colorspace,
            out_colorspace,
            out,
            width,
            stride
        )?;
        Ok(rows)
    }

    /// Inverse transform a progressive image's coefficients at the scale set
    /// with `set_idct_scale`, then upsample and colour convert.
    pub(crate) fn finish_progressive_scaled(
        &mut self, block: &[Vec<i16>; MAX_COMPONENTS], pixels: &mut [u8]
    ) -> Result<(), DecodeErrors> {
        let plans = self.scaled_plans()?;
        let mut planes: [Vec<i16>; MAX_COMPONENTS] = core::array::from_fn(|_| vec![]);
        let mut strides = [0; MAX_COMPONENTS];
        let mut tmp = [0_i32; DCT_BLOCK];

        for (pos, component) in self.components.iter().enumerate() {
            if !self.scaled_component_needed(pos) {
                continue;
            }
            let plan = plans[pos];
            // Coefficients are stored block row by block row, each block's
            // 64 together, `width_stride / 8` blocks to a row.
            let blocks_x = component.width_stride / 8;
            let blocks_y = if self.is_interleaved {
                self.mcu_y * component.vertical_sample
            } else {
                usize::from(self.info.height).div_ceil(8)
            };
            let stride = blocks_x * plan.block;
            let mut plane = vec![128; stride * blocks_y * plan.block];
            let idct = plan.idct().unwrap_or(self.idct_func);
            let qt_table = &component.quantization_table;

            for y in 0..blocks_y {
                for x in 0..blocks_x {
                    let start = (y * blocks_x + x) * 64;
                    let Some(coefficients) = block[pos].get(start..start + 64) else {
                        return Err(DecodeErrors::FormatStatic(
                            "Invalid slice , would panic, invalid image"
                        ));
                    };
                    for ((out, &c), &q) in tmp.iter_mut().zip(coefficients).zip(qt_table) {
                        *out = i32::from(c) * q;
                    }
                    let at = (y * stride + x) * plan.block;
                    idct(&mut tmp, &mut plane[at..], stride);
                }
            }
            planes[pos] = plane;
            strides[pos] = stride;
        }
        self.finish_scaled(&planes, &strides, &plans, pixels)
    }

    /// Bring each component plane to the output size and colour convert.
    fn finish_scaled(
        &self, planes: &[Vec<i16>; MAX_COMPONENTS], strides: &[usize; MAX_COMPONENTS],
        plans: &[ComponentPlan; MAX_COMPONENTS], pixels: &mut [u8]
    ) -> Result<(), DecodeErrors> {
        let width = self.idct_scale.scaled(usize::from(self.info.width));
        let height = self.idct_scale.scaled(usize::from(self.info.height));
        let components = self.scaled_output_components();
        // libjpeg upsamples with triangle filters unless the scale is 1/8.
        let fancy = self.idct_scale.block_size() > 1;

        let direct = (0..components).all(|c| {
            plans[c].h_expand == 1 && plans[c].v_expand == 1 && strides[c] == strides[0]
        });
        if direct {
            // Every plane is at the output size already, with one stride.
            self.convert_scaled_rows(planes, strides[0], 0, height, pixels)?;
            return Ok(());
        }
        let mut expanded: [Vec<i16>; MAX_COMPONENTS] = core::array::from_fn(|_| vec![]);
        for c in 0..components {
            expanded[c] = expand(&planes[c], strides[c], plans[c], width, height, fancy);
        }
        self.convert_scaled_rows(&expanded, width, 0, height, pixels)?;
        Ok(())
    }
}

/// A component plane brought to `width x height` (libjpeg's `jdsample.c`
/// upsamplers): the 2x "fancy" triangle filters where `fancy` allows them,
/// replication otherwise. Rows and columns past the plane's meaningful
/// size repeat its last ones, as libjpeg's context rows do.
fn expand(
    plane: &[i16], stride: usize, plan: ComponentPlan, width: usize, height: usize, fancy: bool
) -> Vec<i16> {
    let (last_x, last_y) = (plan.width - 1, plan.height - 1);
    let at = |y: usize, x: usize| i32::from(plane[y.min(last_y) * stride + x.min(last_x)]);
    let mut out = vec![0_i16; width * height];
    let rows = out.chunks_exact_mut(width).enumerate();

    match (plan.h_expand, plan.v_expand) {
        // h2v1_fancy_upsample
        (2, 1) if fancy && plan.width > 2 => {
            for (y, row) in rows {
                for (x, o) in row.iter_mut().enumerate() {
                    let c = x / 2;
                    *o = match (x % 2, c) {
                        (0, 0) => at(y, 0),
                        (0, _) => (3 * at(y, c) + at(y, c - 1) + 1) >> 2,
                        (_, c) if c >= last_x => at(y, c),
                        _ => (3 * at(y, c) + at(y, c + 1) + 2) >> 2
                    } as i16;
                }
            }
        }
        // h1v2_fancy_upsample
        (1, 2) if fancy => {
            for (y, row) in rows {
                let (r, below) = (y / 2, y % 2 == 1);
                let (near, bias) = if below { (r + 1, 2) } else { (r.saturating_sub(1), 1) };
                for (x, o) in row.iter_mut().enumerate() {
                    *o = ((3 * at(r, x) + at(near, x) + bias) >> 2) as i16;
                }
            }
        }
        // h2v2_fancy_upsample
        (2, 2) if fancy && plan.width > 2 => {
            for (y, row) in rows {
                let (r, below) = (y / 2, y % 2 == 1);
                let near = if below { r + 1 } else { r.saturating_sub(1) };
                let col = |c: usize| 3 * at(r, c) + at(near, c);
                for (x, o) in row.iter_mut().enumerate() {
                    let c = x / 2;
                    *o = match (x % 2, c) {
                        (0, 0) => (col(0) * 4 + 8) >> 4,
                        (0, _) => (3 * col(c) + col(c - 1) + 8) >> 4,
                        (_, c) if c >= last_x => (col(c) * 4 + 7) >> 4,
                        _ => (3 * col(c) + col(c + 1) + 7) >> 4
                    } as i16;
                }
            }
        }
        // int_upsample, h2v1_upsample, h2v2_upsample
        (h, v) => {
            for (y, row) in rows {
                for (x, o) in row.iter_mut().enumerate() {
                    *o = at(y / v, x / h) as i16;
                }
            }
        }
    }
    out
}

// libjpeg's reduced IDCTs (`jidctred.c`): fixed point with 13 fractional
// bits, 2 more kept between the passes. Each takes the dequantized
// coefficients in natural order, as the full IDCT does, and leaves them
// unmodified (the MCU loop's partial clearing of its coefficient buffer
// relies on that).
const CONST_BITS: u32 = 13;
const PASS1_BITS: u32 = 2;

const FIX_0_211164243: i32 = 1730;
const FIX_0_509795579: i32 = 4176;
const FIX_0_601344887: i32 = 4926;
const FIX_0_720959822: i32 = 5906;
const FIX_0_765366865: i32 = 6270;
const FIX_0_850430095: i32 = 6967;
const FIX_0_899976223: i32 = 7373;
const FIX_1_061594337: i32 = 8697;
const FIX_1_272758580: i32 = 10426;
const FIX_1_451774981: i32 = 11893;
const FIX_1_847759065: i32 = 15137;
const FIX_2_172734803: i32 = 17799;
const FIX_2_562915447: i32 = 20995;
const FIX_3_624509785: i32 = 29692;

/// `x / 2^n`, rounded to nearest (libjpeg's `DESCALE`).
#[inline(always)]
fn descale(x: i32, n: u32) -> i32 {
    x.wrapping_add(1 << (n - 1)) >> n
}

/// A level-shifted sample, clamped to 0..=255.
#[inline(always)]
fn sample(x: i32) -> i16 {
    x.wrapping_add(128).clamp(0, 255) as i16
}

/// The odd part of the 8-point IDCT evaluated at the 4 outputs of
/// `jpeg_idct_4x4`, from coefficients 1, 3, 5, 7.
#[inline(always)]
fn odd_4(z4: i32, z3: i32, z2: i32, z1: i32) -> (i32, i32) {
    let tmp0 = z1
        .wrapping_mul(-FIX_0_211164243)
        .wrapping_add(z2.wrapping_mul(FIX_1_451774981))
        .wrapping_add(z3.wrapping_mul(-FIX_2_172734803))
        .wrapping_add(z4.wrapping_mul(FIX_1_061594337));
    let tmp2 = z1
        .wrapping_mul(-FIX_0_509795579)
        .wrapping_add(z2.wrapping_mul(-FIX_0_601344887))
        .wrapping_add(z3.wrapping_mul(FIX_0_899976223))
        .wrapping_add(z4.wrapping_mul(FIX_2_562915447));
    (tmp0, tmp2)
}

/// Row `k` of an 8x8 coefficient block.
#[inline(always)]
fn coefficient_row(input: &[i32; 64], k: usize) -> &[i32; 8] {
    input[k * 8..k * 8 + 8].try_into().unwrap()
}

/// `jpeg_idct_4x4`: an 8x8 block to its 4x4 output.
///
/// The column pass runs over all 8 columns without libjpeg's per-column
/// zero shortcut (exact either way), so it vectorizes; column 4 is computed
/// and never read.
fn idct_4x4_scaled(input: &mut [i32; 64], out: &mut [i16], stride: usize) {
    let input: &[i32; 64] = input;
    let [r0, r1, r2, r3, r5, r6, r7] = [0, 1, 2, 3, 5, 6, 7].map(|k| coefficient_row(input, k));
    let mut ws = [[0_i32; 8]; 4];
    let n = CONST_BITS - PASS1_BITS + 1;
    for col in 0..8 {
        let tmp0 = r0[col].wrapping_shl(CONST_BITS + 1);
        let tmp2 = r2[col]
            .wrapping_mul(FIX_1_847759065)
            .wrapping_add(r6[col].wrapping_mul(-FIX_0_765366865));
        let (tmp10, tmp12) = (tmp0.wrapping_add(tmp2), tmp0.wrapping_sub(tmp2));
        let (odd0, odd2) = odd_4(r1[col], r3[col], r5[col], r7[col]);
        ws[0][col] = descale(tmp10.wrapping_add(odd2), n);
        ws[3][col] = descale(tmp10.wrapping_sub(odd2), n);
        ws[1][col] = descale(tmp12.wrapping_add(odd0), n);
        ws[2][col] = descale(tmp12.wrapping_sub(odd0), n);
    }
    let n = CONST_BITS + PASS1_BITS + 3 + 1;
    for (row, w) in ws.iter().enumerate() {
        let tmp0 = w[0].wrapping_shl(CONST_BITS + 1);
        let tmp2 = w[2]
            .wrapping_mul(FIX_1_847759065)
            .wrapping_add(w[6].wrapping_mul(-FIX_0_765366865));
        let (tmp10, tmp12) = (tmp0.wrapping_add(tmp2), tmp0.wrapping_sub(tmp2));
        let (odd0, odd2) = odd_4(w[1], w[3], w[5], w[7]);
        let o: &mut [i16; 4] = (&mut out[row * stride..row * stride + 4]).try_into().unwrap();
        *o = [
            sample(descale(tmp10.wrapping_add(odd2), n)),
            sample(descale(tmp12.wrapping_add(odd0), n)),
            sample(descale(tmp12.wrapping_sub(odd0), n)),
            sample(descale(tmp10.wrapping_sub(odd2), n))
        ];
    }
}

/// The odd part of the 8-point IDCT evaluated at the 2 outputs of
/// `jpeg_idct_2x2`, from coefficients 1, 3, 5, 7.
#[inline(always)]
fn odd_2(z1: i32, z3: i32, z5: i32, z7: i32) -> i32 {
    z7.wrapping_mul(-FIX_0_720959822)
        .wrapping_add(z5.wrapping_mul(FIX_0_850430095))
        .wrapping_add(z3.wrapping_mul(-FIX_1_272758580))
        .wrapping_add(z1.wrapping_mul(FIX_3_624509785))
}

/// `jpeg_idct_2x2`: an 8x8 block to its 2x2 output, its column pass over
/// all 8 columns as in [`idct_4x4_scaled`].
fn idct_2x2_scaled(input: &mut [i32; 64], out: &mut [i16], stride: usize) {
    let input: &[i32; 64] = input;
    let [r0, r1, r3, r5, r7] = [0, 1, 3, 5, 7].map(|k| coefficient_row(input, k));
    let mut ws = [[0_i32; 8]; 2];
    let n = CONST_BITS - PASS1_BITS + 2;
    for col in 0..8 {
        let tmp10 = r0[col].wrapping_shl(CONST_BITS + 2);
        let tmp0 = odd_2(r1[col], r3[col], r5[col], r7[col]);
        ws[0][col] = descale(tmp10.wrapping_add(tmp0), n);
        ws[1][col] = descale(tmp10.wrapping_sub(tmp0), n);
    }
    let n = CONST_BITS + PASS1_BITS + 3 + 2;
    for (row, w) in ws.iter().enumerate() {
        let tmp10 = w[0].wrapping_shl(CONST_BITS + 2);
        let tmp0 = odd_2(w[1], w[3], w[5], w[7]);
        let o: &mut [i16; 2] = (&mut out[row * stride..row * stride + 2]).try_into().unwrap();
        *o = [
            sample(descale(tmp10.wrapping_add(tmp0), n)),
            sample(descale(tmp10.wrapping_sub(tmp0), n))
        ];
    }
}

/// An 8x8 block to its one sample: the DC term, as zune's own DC-only
/// IDCT (`idct_int_1x1`) computes it, `clamp(round(DC / 8) + 128)`.
fn idct_1x1_scaled(input: &mut [i32; 64], out: &mut [i16], _stride: usize) {
    out[0] = sample(descale(input[0], 3));
}
