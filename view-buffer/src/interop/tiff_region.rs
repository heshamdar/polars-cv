//! TIFF decoding chunk by chunk: a window of an image reads only the tiles
//! (or strips) it overlaps.
//!
//! The `tiff` crate parses the file structure (header, IFDs, tags, BigTIFF);
//! this module decodes the chunks itself, for two reasons. A window of a
//! whole-slide image must touch only its own chunks, located through the
//! offset/byte-count tables, and later fetched by byte range. And the
//! crate's own chunk reader gets two common layouts wrong: it panics on LZW
//! streams libtiff and tifffile read without complaint, and it passes a JPEG
//! tile's components through unconverted (YCbCr data comes back as if it were
//! RGB, and RGB-coded tiles fail outright).
//!
//! A layout this module does not carry — palette images, separate sample
//! planes, raw (non-JPEG) YCbCr, WhiteIsZero, signed or odd-width samples,
//! the floating-point predictor, codecs other than none/LZW/Deflate/PackBits/
//! JPEG — reports [`Layout::read`] `None`, and the caller decodes through the
//! `tiff` crate as before. A whole-image decode and a window decode of the
//! same file therefore always go through the same chunk decoder, so a window
//! is exactly the crop of the whole.

use std::io::{Cursor, Read, Seek};

use tiff::decoder::{Decoder, Limits};
use tiff::tags::Tag;

use crate::core::buffer::ViewBuffer;
use crate::core::dtype::DType;

/// The most decoded pixel data one decode may produce, in bytes: the `tiff`
/// crate's own default whole-image limit, kept so that what decoded before
/// still does and what refused still does. A window of a larger image is
/// within it; the whole image is not.
pub const DECODE_LIMIT_BYTES: usize = 256 * 1024 * 1024;

/// IFD values (the chunk offset and byte-count tables above all) may be this
/// large: a 100k × 100k slide in 256-pixel tiles has ~150k chunks, an
/// 8-byte offset each in BigTIFF — beyond the `tiff` crate's 1 MiB default.
const IFD_VALUE_LIMIT: usize = 64 * 1024 * 1024;

/// A window of the image: rows `top..bottom`, columns `left..right`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Window {
    pub top: usize,
    pub left: usize,
    pub bottom: usize,
    pub right: usize,
}

/// Open `bytes` with limits sized for large tiled images.
pub(crate) fn open<R: Read + Seek>(reader: R) -> Result<Decoder<R>, String> {
    let mut limits = Limits::default();
    limits.ifd_value_size = IFD_VALUE_LIMIT;
    Decoder::new(reader)
        .map(|d| d.with_limits(limits))
        .map_err(|e| format!("TIFF decoder creation failed: {e}"))
}

/// The element type of a TIFF's samples, as this module decodes them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Sample {
    U8,
    U16,
    F32,
    F64,
}

impl Sample {
    fn bytes(self) -> usize {
        match self {
            Sample::U8 => 1,
            Sample::U16 => 2,
            Sample::F32 => 4,
            Sample::F64 => 8,
        }
    }
}

/// How a chunk's bytes are compressed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Codec {
    None,
    Lzw,
    Deflate,
    PackBits,
    Jpeg,
}

/// One image (IFD) of a TIFF: its geometry, sample format and chunk tables.
#[derive(Debug, Clone)]
pub(crate) struct Layout {
    pub(crate) width: usize,
    pub(crate) height: usize,
    /// Tile size, or `(width, rows_per_strip)` for strips.
    chunk_width: usize,
    chunk_height: usize,
    tiled: bool,
    offsets: Vec<u64>,
    byte_counts: Vec<u64>,
    codec: Codec,
    /// Horizontal differencing (TIFF predictor 2).
    differenced: bool,
    sample: Sample,
    /// Samples stored per pixel.
    samples: usize,
    /// The data is YCbCr-coded JPEG to convert to RGB (decided per tile from
    /// the JPEG stream; see [`jpeg_is_ycbcr`]).
    jpeg_tables: Option<Vec<u8>>,
    big_endian: bool,
}

/// Read a tag's unsigned value, if present.
fn tag_u64<R: Read + Seek>(d: &mut Decoder<R>, tag: Tag) -> Result<Option<u64>, String> {
    d.find_tag_unsigned::<u64>(tag)
        .map_err(|e| format!("TIFF tag {tag:?}: {e}"))
}

impl Layout {
    /// The layout of the decoder's current image, or `None` when it is one
    /// this module does not decode (the caller then uses the `tiff` crate).
    pub(crate) fn read<R: Read + Seek>(
        d: &mut Decoder<R>,
        big_endian: bool,
    ) -> Result<Option<Layout>, String> {
        let (width, height) = d
            .dimensions()
            .map_err(|e| format!("Failed to get TIFF dimensions: {e}"))?;
        let samples = tag_u64(d, Tag::SamplesPerPixel)?.unwrap_or(1) as usize;
        let bits: Vec<u64> = d
            .find_tag_unsigned_vec::<u64>(Tag::BitsPerSample)
            .map_err(|e| format!("TIFF tag BitsPerSample: {e}"))?
            .unwrap_or_else(|| vec![1]);
        let format = d
            .find_tag_unsigned_vec::<u64>(Tag::SampleFormat)
            .map_err(|e| format!("TIFF tag SampleFormat: {e}"))?
            .unwrap_or_else(|| vec![1]);
        let photometric = tag_u64(d, Tag::PhotometricInterpretation)?;
        let compression = tag_u64(d, Tag::Compression)?.unwrap_or(1);
        let predictor = tag_u64(d, Tag::Predictor)?.unwrap_or(1);
        let planar = tag_u64(d, Tag::PlanarConfiguration)?.unwrap_or(1);

        // Every sample the same width and format.
        if bits.iter().any(|&b| b != bits[0]) || format.iter().any(|&f| f != format[0]) {
            return Ok(None);
        }
        let sample = match (bits[0], format[0]) {
            (8, 1) => Sample::U8,
            (16, 1) => Sample::U16,
            (32, 3) => Sample::F32,
            (64, 3) => Sample::F64,
            _ => return Ok(None),
        };
        let codec = match compression {
            1 => Codec::None,
            5 => Codec::Lzw,
            8 | 32946 => Codec::Deflate,
            32773 => Codec::PackBits,
            7 => Codec::Jpeg,
            _ => return Ok(None),
        };
        let integer = matches!(sample, Sample::U8 | Sample::U16);
        // The channel rules of the whole-image decoder: gray, gray + alpha,
        // RGB, RGBA for integers; gray and RGB for floats. JPEG-in-TIFF holds
        // 8-bit gray or three-component colour, tagged RGB or YCbCr.
        let supported = match (photometric, samples, codec) {
            (_, _, Codec::Jpeg) => {
                sample == Sample::U8
                    && matches!((photometric, samples), (Some(1), 1) | (Some(2 | 6), 3))
            }
            (Some(1), 1, _) | (Some(2), 3, _) => true,
            (Some(1), 2, _) | (Some(2), 4, _) => integer,
            _ => false,
        };
        let differenced = match predictor {
            1 => false,
            2 if integer && codec != Codec::Jpeg => true,
            _ => return Ok(None),
        };
        if !supported || (planar != 1 && samples > 1) {
            return Ok(None);
        }

        let tiled = d
            .find_tag(Tag::TileWidth)
            .map_err(|e| format!("TIFF tag TileWidth: {e}"))?
            .is_some();
        let (chunk_width, chunk_height, offsets, byte_counts) = if tiled {
            let tw = tag_u64(d, Tag::TileWidth)?.unwrap_or(0) as usize;
            let th = tag_u64(d, Tag::TileLength)?.unwrap_or(0) as usize;
            let offsets = d
                .get_tag_u64_vec(Tag::TileOffsets)
                .map_err(|e| format!("TIFF tag TileOffsets: {e}"))?;
            let counts = d
                .get_tag_u64_vec(Tag::TileByteCounts)
                .map_err(|e| format!("TIFF tag TileByteCounts: {e}"))?;
            (tw, th, offsets, counts)
        } else {
            let rows = tag_u64(d, Tag::RowsPerStrip)?
                .map_or(height as usize, |r| (r as usize).min(height as usize));
            let offsets = d
                .get_tag_u64_vec(Tag::StripOffsets)
                .map_err(|e| format!("TIFF tag StripOffsets: {e}"))?;
            let counts = d
                .get_tag_u64_vec(Tag::StripByteCounts)
                .map_err(|e| format!("TIFF tag StripByteCounts: {e}"))?;
            (width as usize, rows, offsets, counts)
        };
        let jpeg_tables = if codec == Codec::Jpeg {
            d.find_tag(Tag::JPEGTables)
                .map_err(|e| format!("TIFF tag JPEGTables: {e}"))?
                .map(|v| v.into_u8_vec())
                .transpose()
                .map_err(|e| format!("TIFF tag JPEGTables: {e}"))?
        } else {
            None
        };
        let layout = Layout {
            width: width as usize,
            height: height as usize,
            chunk_width,
            chunk_height,
            tiled,
            offsets,
            byte_counts,
            codec,
            differenced,
            sample,
            samples,
            jpeg_tables,
            big_endian,
        };
        if chunk_width == 0
            || chunk_height == 0
            || layout.offsets.len() != layout.byte_counts.len()
            || layout.offsets.len() < layout.across() * layout.down()
        {
            return Err("TIFF chunk tables do not cover the image".to_string());
        }
        Ok(Some(layout))
    }

    /// Chunks per row of chunks.
    fn across(&self) -> usize {
        self.width.div_ceil(self.chunk_width)
    }

    /// Rows of chunks.
    fn down(&self) -> usize {
        self.height.div_ceil(self.chunk_height)
    }

    /// The element type a decode produces.
    pub(crate) fn dtype(&self) -> DType {
        match self.sample {
            Sample::U8 => DType::U8,
            Sample::U16 => DType::U16,
            Sample::F32 => DType::F32,
            Sample::F64 => DType::F64,
        }
    }

    /// The channels a decode produces (JPEG YCbCr becomes RGB).
    pub(crate) fn channels(&self) -> usize {
        self.samples
    }

    /// The decoded size of `window`, in bytes.
    fn window_bytes(&self, w: &Window) -> usize {
        (w.bottom - w.top) * (w.right - w.left) * self.samples * self.sample.bytes()
    }

    /// The chunks `window` overlaps, as `(chunk index, chunk row, chunk col)`.
    fn chunks_in(&self, w: &Window) -> Vec<(usize, usize, usize)> {
        if w.bottom <= w.top || w.right <= w.left {
            return Vec::new();
        }
        let rows = w.top / self.chunk_height..w.bottom.div_ceil(self.chunk_height);
        let cols = w.left / self.chunk_width..w.right.div_ceil(self.chunk_width);
        rows.flat_map(|r| cols.clone().map(move |c| (r * self.across() + c, r, c)))
            .collect()
    }

    /// The byte range of chunk `index` in the file.
    pub(crate) fn chunk_range(&self, index: usize) -> std::ops::Range<u64> {
        let start = self.offsets[index];
        start..start + self.byte_counts[index]
    }

    /// Decode one chunk's compressed bytes into native-endian samples, rows of
    /// `chunk_width` pixels; `rows` of them (a strip at the bottom holds fewer
    /// than `chunk_height`).
    fn decode_chunk(&self, raw: &[u8], rows: usize) -> Result<Vec<u8>, String> {
        let row_bytes = self.chunk_width * self.samples * self.sample.bytes();
        let want = row_bytes * rows;
        let mut data = match self.codec {
            Codec::None => raw.to_vec(),
            Codec::Lzw => weezl::decode::Decoder::with_tiff_size_switch(weezl::BitOrder::Msb, 8)
                .decode(raw)
                .map_err(|e| format!("corrupt LZW data: {e}"))?,
            Codec::Deflate => {
                let mut out = Vec::with_capacity(want);
                flate2::read::ZlibDecoder::new(raw)
                    .read_to_end(&mut out)
                    .map_err(|e| format!("corrupt Deflate data: {e}"))?;
                out
            }
            Codec::PackBits => unpack_bits(raw, want)?,
            Codec::Jpeg => self.decode_jpeg(raw)?,
        };
        if data.len() < want {
            return Err(format!(
                "a TIFF chunk decoded to {} bytes, {} expected",
                data.len(),
                want
            ));
        }
        data.truncate(want);
        if self.big_endian && self.sample.bytes() > 1 {
            for v in data.chunks_exact_mut(self.sample.bytes()) {
                v.reverse();
            }
        }
        if self.differenced {
            undo_differencing(&mut data, row_bytes, self.samples, self.sample);
        }
        Ok(data)
    }

    /// A JPEG tile or strip: the file's shared tables spliced in, decoded to
    /// its stored components, then colour-converted as libjpeg would.
    fn decode_jpeg(&self, raw: &[u8]) -> Result<Vec<u8>, String> {
        let stream: std::borrow::Cow<'_, [u8]> = match &self.jpeg_tables {
            // Tables are a complete SOI..EOI stream of DQT/DHT segments: drop
            // its EOI and the chunk's SOI to make one stream.
            Some(t) if t.len() >= 4 && raw.len() >= 2 => {
                let mut s = Vec::with_capacity(t.len() + raw.len());
                s.extend_from_slice(&t[..t.len() - 2]);
                s.extend_from_slice(&raw[2..]);
                std::borrow::Cow::Owned(s)
            }
            _ => std::borrow::Cow::Borrowed(raw),
        };
        let corrupt = |e: zune_jpeg::errors::DecodeErrors| format!("corrupt JPEG data: {e:?}");
        let mut probe =
            zune_jpeg::JpegDecoder::new(zune_core::bytestream::ZCursor::new(stream.as_ref()));
        probe.decode_headers().map_err(corrupt)?;
        let stored = probe
            .input_colorspace()
            .ok_or("a JPEG tile without a colour space")?;
        let options = zune_core::options::DecoderOptions::default()
            .jpeg_set_out_colorspace(stored)
            .set_max_width(1 << 16)
            .set_max_height(1 << 16);
        let mut decoder = zune_jpeg::JpegDecoder::new_with_options(
            zune_core::bytestream::ZCursor::new(stream.as_ref()),
            options,
        );
        let mut pixels = decoder.decode().map_err(corrupt)?;
        let components = stored.num_components();
        if components != self.samples {
            return Err(format!(
                "a JPEG tile holds {components} components, the TIFF declares {}",
                self.samples
            ));
        }
        let (w, _) = decoder
            .dimensions()
            .ok_or("a JPEG tile without dimensions")?;
        if w != self.chunk_width {
            return Err(format!(
                "a JPEG tile is {w} pixels wide, the TIFF declares {}",
                self.chunk_width
            ));
        }
        if components == 3 && jpeg_is_ycbcr(stream.as_ref()) {
            ycbcr_to_rgb(&mut pixels);
        }
        Ok(pixels)
    }

    /// Decode `window` from `read_chunk`, which returns a chunk's compressed
    /// bytes by index. Only the chunks the window overlaps are read.
    pub(crate) fn decode_window(
        &self,
        window: &Window,
        mut read_chunk: impl FnMut(usize) -> Result<Vec<u8>, String>,
    ) -> Result<ViewBuffer, String> {
        let bytes = self.window_bytes(window);
        if bytes > DECODE_LIMIT_BYTES {
            return Err(format!(
                "decoding {}x{} pixels of this {}x{} TIFF needs {} MiB, over the {} MiB \
                 limit; crop it (a crop right after the source decodes only its window)",
                window.bottom - window.top,
                window.right - window.left,
                self.height,
                self.width,
                bytes >> 20,
                DECODE_LIMIT_BYTES >> 20
            ));
        }
        let px = self.samples * self.sample.bytes();
        let (out_h, out_w) = (window.bottom - window.top, window.right - window.left);
        let mut out = vec![0u8; bytes];
        for (index, r, c) in self.chunks_in(window) {
            let (y0, x0) = (r * self.chunk_height, c * self.chunk_width);
            // A tile is always stored whole (padded past the image); the last
            // strip holds only the image's remaining rows.
            let rows = if self.tiled {
                self.chunk_height
            } else {
                self.chunk_height.min(self.height - y0)
            };
            let data = self.decode_chunk(&read_chunk(index)?, rows)?;
            let (ys, ye) = (y0.max(window.top), (y0 + rows).min(window.bottom));
            let (xs, xe) = (
                x0.max(window.left),
                (x0 + self.chunk_width).min(window.right),
            );
            let chunk_row = self.chunk_width * px;
            for y in ys..ye {
                let src = (y - y0) * chunk_row + (xs - x0) * px;
                let dst = ((y - window.top) * out_w + (xs - window.left)) * px;
                let n = (xe - xs) * px;
                out[dst..dst + n].copy_from_slice(&data[src..src + n]);
            }
        }
        let shape = vec![out_h, out_w, self.samples];
        Ok(match self.sample {
            Sample::U8 => ViewBuffer::from_vec_with_shape(out, shape),
            Sample::U16 => ViewBuffer::from_vec_with_shape(from_ne::<u16, 2>(&out), shape),
            Sample::F32 => ViewBuffer::from_vec_with_shape(from_ne::<f32, 4>(&out), shape),
            Sample::F64 => ViewBuffer::from_vec_with_shape(from_ne::<f64, 8>(&out), shape),
        })
    }
}

/// Native-endian bytes as samples.
fn from_ne<T: FromNe<N>, const N: usize>(bytes: &[u8]) -> Vec<T> {
    let (samples, _) = bytes.as_chunks::<N>();
    samples.iter().map(|b| T::from_ne(*b)).collect()
}

trait FromNe<const N: usize> {
    fn from_ne(b: [u8; N]) -> Self;
}
impl FromNe<2> for u16 {
    fn from_ne(b: [u8; 2]) -> Self {
        u16::from_ne_bytes(b)
    }
}
impl FromNe<4> for f32 {
    fn from_ne(b: [u8; 4]) -> Self {
        f32::from_ne_bytes(b)
    }
}
impl FromNe<8> for f64 {
    fn from_ne(b: [u8; 8]) -> Self {
        f64::from_ne_bytes(b)
    }
}

/// Undo horizontal differencing (TIFF predictor 2) on native-endian integer
/// samples: each sample adds the same channel of the pixel to its left,
/// wrapping.
fn undo_differencing(data: &mut [u8], row_bytes: usize, samples: usize, sample: Sample) {
    for row in data.chunks_exact_mut(row_bytes) {
        match sample {
            Sample::U8 => {
                for i in samples..row.len() {
                    row[i] = row[i].wrapping_add(row[i - samples]);
                }
            }
            Sample::U16 => {
                let n = row.len() / 2;
                for i in samples..n {
                    let prev =
                        u16::from_ne_bytes([row[2 * (i - samples)], row[2 * (i - samples) + 1]]);
                    let cur = u16::from_ne_bytes([row[2 * i], row[2 * i + 1]]);
                    row[2 * i..2 * i + 2].copy_from_slice(&cur.wrapping_add(prev).to_ne_bytes());
                }
            }
            Sample::F32 | Sample::F64 => unreachable!("predictor 2 is integer-only (Layout::read)"),
        }
    }
}

/// PackBits (Apple run-length) decoding, to at least `want` bytes.
fn unpack_bits(raw: &[u8], want: usize) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(want);
    let mut i = 0;
    while i < raw.len() && out.len() < want {
        let n = raw[i] as i8;
        i += 1;
        if n >= 0 {
            let len = n as usize + 1;
            let lit = raw.get(i..i + len).ok_or("corrupt PackBits data")?;
            out.extend_from_slice(lit);
            i += len;
        } else if n != -128 {
            let byte = *raw.get(i).ok_or("corrupt PackBits data")?;
            out.extend(std::iter::repeat_n(byte, (1 - isize::from(n)) as usize));
            i += 1;
        }
    }
    Ok(out)
}

/// Whether a three-component JPEG stream holds YCbCr, by libjpeg's rules: an
/// Adobe APP14 marker's transform decides (0: RGB, else YCbCr); without one,
/// components named `R`, `G`, `B` are RGB; anything else is YCbCr (JFIF).
pub(crate) fn jpeg_is_ycbcr(stream: &[u8]) -> bool {
    let mut i = 2;
    let mut ids: Option<[u8; 3]> = None;
    while i + 4 <= stream.len() {
        if stream[i] != 0xFF {
            i += 1;
            continue;
        }
        let marker = stream[i + 1];
        if marker == 0xFF {
            i += 1;
            continue;
        }
        // Start of scan: the headers are over.
        if marker == 0xDA {
            break;
        }
        let len = u16::from_be_bytes([stream[i + 2], stream[i + 3]]) as usize;
        let body = stream.get(i + 4..i + 2 + len).unwrap_or(&[]);
        match marker {
            0xEE if body.len() >= 12 && &body[..5] == b"Adobe" => return body[11] != 0,
            0xC0..=0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF
                if body.len() >= 6 + 3 * 3 && body[5] == 3 =>
            {
                ids = Some([body[6], body[9], body[12]]);
            }
            _ => {}
        }
        i += 2 + len;
    }
    ids != Some(*b"RGB")
}

/// YCbCr → RGB in place, three bytes per pixel: libjpeg's fixed-point
/// conversion (`jdcolor.c`, ITU-R BT.601 full range as JFIF defines it).
pub(crate) fn ycbcr_to_rgb(pixels: &mut [u8]) {
    const SCALE: i32 = 16;
    const HALF: i32 = 1 << (SCALE - 1);
    let fix = |x: f64| (x * f64::from(1 << SCALE) + 0.5) as i32;
    let (cr_r, cb_b, cb_g, cr_g) = (fix(1.40200), fix(1.77200), fix(0.34414), fix(0.71414));
    let (pixels, _) = pixels.as_chunks_mut::<3>();
    for px in pixels {
        let y = i32::from(px[0]);
        let cb = i32::from(px[1]) - 128;
        let cr = i32::from(px[2]) - 128;
        let r = y + ((cr_r * cr + HALF) >> SCALE);
        let g = y + ((-cb_g * cb - cr_g * cr + HALF) >> SCALE);
        let b = y + ((cb_b * cb + HALF) >> SCALE);
        px[0] = r.clamp(0, 255) as u8;
        px[1] = g.clamp(0, 255) as u8;
        px[2] = b.clamp(0, 255) as u8;
    }
}

/// Whether `bytes` start with a TIFF or BigTIFF header.
pub fn is_tiff(bytes: &[u8]) -> bool {
    bytes.len() >= 4
        && matches!(
            &bytes[..4],
            b"II*\x00" | b"MM\x00*" | b"II+\x00" | b"MM\x00+"
        )
}

/// The first image of an in-memory TIFF, decoded over `window` (`None`: the
/// whole image) — or `Ok(None)` when its layout is one this module leaves to
/// the `tiff` crate.
///
/// `window` must lie inside the image (the caller validates it against
/// [`image_shape`]).
pub fn decode(bytes: &[u8], window: Option<Window>) -> Result<Option<ViewBuffer>, String> {
    let mut d = open(Cursor::new(bytes))?;
    let big_endian = bytes.starts_with(b"MM");
    let Some(layout) = Layout::read(&mut d, big_endian)? else {
        return Ok(None);
    };
    let window = window.unwrap_or(Window {
        top: 0,
        left: 0,
        bottom: layout.height,
        right: layout.width,
    });
    let buffer = layout.decode_window(&window, |index| {
        let range = layout.chunk_range(index);
        let (start, end) = (range.start as usize, range.end as usize);
        bytes
            .get(start..end)
            .map(<[u8]>::to_vec)
            .ok_or_else(|| format!("TIFF chunk {index} lies past the end of the file"))
    })?;
    Ok(Some(buffer))
}

/// The `[H, W, C]` shape and element type a decode of the first image of an
/// in-memory TIFF produces, or `Ok(None)` when this module does not decode it.
pub fn image_shape(bytes: &[u8]) -> Result<Option<([usize; 3], DType)>, String> {
    let mut d = open(Cursor::new(bytes))?;
    Ok(Layout::read(&mut d, bytes.starts_with(b"MM"))?
        .map(|l| ([l.height, l.width, l.channels()], l.dtype())))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packbits_decodes_literals_and_runs() {
        // 2 literals, a run of 4, the no-op -128, 1 literal.
        let raw = [1u8, 10, 11, 0xFD, 7, 0x80, 0, 9];
        assert_eq!(unpack_bits(&raw, 7).unwrap(), vec![10, 11, 7, 7, 7, 7, 9]);
        assert!(
            unpack_bits(&[5, 1], 6).is_err(),
            "a literal run past the data"
        );
    }

    #[test]
    fn differencing_is_undone_per_channel_and_wraps() {
        let mut row = vec![10u8, 200, 5, 100, 250, 1];
        undo_differencing(&mut row, 6, 2, Sample::U8);
        assert_eq!(row, vec![10, 200, 15, 44, 9, 45]);
        let mut wide: Vec<u8> = [1000u16, 65535, 2]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        undo_differencing(&mut wide, 6, 1, Sample::U16);
        assert_eq!(from_ne::<u16, 2>(&wide), vec![1000, 999, 1001]);
    }

    #[test]
    fn ycbcr_conversion_is_jfifs_within_rounding() {
        // Neutral chroma is gray exactly; everything else is the JFIF float
        // formula, to within libjpeg's fixed-point rounding.
        for y in (0..=255).step_by(5) {
            for cb in (0..=255).step_by(15) {
                for cr in (0..=255).step_by(15) {
                    let mut px = [y as u8, cb as u8, cr as u8];
                    ycbcr_to_rgb(&mut px);
                    let (yf, cbf, crf) =
                        (f64::from(y), f64::from(cb) - 128.0, f64::from(cr) - 128.0);
                    let want = [
                        yf + 1.402 * crf,
                        yf - 0.344136 * cbf - 0.714136 * crf,
                        yf + 1.772 * cbf,
                    ];
                    for (got, want) in px.iter().zip(want) {
                        let want = want.round().clamp(0.0, 255.0);
                        assert!((f64::from(*got) - want).abs() <= 1.0, "{y} {cb} {cr}");
                    }
                    if (cb, cr) == (128, 128) {
                        assert_eq!(px, [y as u8; 3]);
                    }
                }
            }
        }
    }

    #[test]
    fn jpeg_colour_follows_libjpegs_markers() {
        let sof = |ids: [u8; 3]| -> Vec<u8> {
            let mut s = vec![0xFF, 0xD8, 0xFF, 0xC0, 0, 17, 8, 0, 8, 0, 8, 3];
            for id in ids {
                s.extend_from_slice(&[id, 0x11, 0]);
            }
            s.extend_from_slice(&[0xFF, 0xDA, 0, 2]);
            s
        };
        let adobe = |transform: u8| -> Vec<u8> {
            let mut s = vec![0xFF, 0xD8, 0xFF, 0xEE, 0, 14];
            s.extend_from_slice(b"Adobe");
            s.extend_from_slice(&[0, 100, 0, 0, 0, 0, transform]);
            s.extend_from_slice(&sof([1, 2, 3])[2..]);
            s
        };
        assert!(jpeg_is_ycbcr(&sof([1, 2, 3])), "JFIF default");
        assert!(!jpeg_is_ycbcr(&sof(*b"RGB")), "components named R, G, B");
        assert!(!jpeg_is_ycbcr(&adobe(0)), "Adobe transform 0");
        assert!(jpeg_is_ycbcr(&adobe(1)), "Adobe transform 1");
    }

    #[test]
    fn a_window_reads_only_the_chunks_it_overlaps() {
        let layout = Layout {
            width: 70,
            height: 50,
            chunk_width: 16,
            chunk_height: 16,
            tiled: true,
            offsets: vec![0; 20],
            byte_counts: vec![0; 20],
            codec: Codec::None,
            differenced: false,
            sample: Sample::U8,
            samples: 1,
            jpeg_tables: None,
            big_endian: false,
        };
        let w = |top, left, bottom, right| Window {
            top,
            left,
            bottom,
            right,
        };
        let idx = |win| -> Vec<usize> { layout.chunks_in(&win).into_iter().map(|c| c.0).collect() };
        assert_eq!(idx(w(0, 0, 16, 16)), vec![0]);
        assert_eq!(idx(w(15, 15, 17, 17)), vec![0, 1, 5, 6]);
        assert_eq!(idx(w(48, 64, 50, 70)), vec![19]);
        assert!(
            idx(w(10, 10, 10, 20)).is_empty(),
            "an empty window reads nothing"
        );
    }
}
