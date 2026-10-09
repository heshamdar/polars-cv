//! TIFF decoding chunk by chunk: a window of an image reads only the tiles
//! (or strips) it overlaps, and only their entries of the chunk tables.
//!
//! The structure is read lazily by [`ifd`]: a patch of a slide costs its
//! header, its tiles' table entries and its tiles, whatever the slide's size.
//! The chunks are decoded here too, for the same reason and two more: the
//! `tiff` crate panics on LZW streams libtiff and tifffile read without
//! complaint, and passes a JPEG tile's components through unconverted
//! (YCbCr data comes back as if it were RGB, and RGB-coded tiles fail
//! outright).
//!
//! A layout this module does not carry — palette images, separate sample
//! planes, raw (non-JPEG) YCbCr, WhiteIsZero, signed or odd-width samples,
//! the floating-point predictor, codecs other than none/LZW/Deflate/PackBits/
//! JPEG — reads as [`Readable::Unsupported`] with the reason, and the caller
//! may decode it whole through the `tiff` crate. A whole-image decode and a
//! window decode of a layout this module carries therefore always go through
//! the same chunk decoder, so a window is exactly the crop of the whole.

mod ifd;

use std::io::{Cursor, Read, Seek};
use std::ops::Range;

use tiff::tags::{CompressionMethod, Tag};

use crate::core::buffer::ViewBuffer;
use crate::core::dtype::DType;

pub use ifd::TiffError;
use ifd::{malformed, Entry, Ifd, TiffFile};

/// The most decoded pixel data one decode may produce, in bytes: the `tiff`
/// crate's own default whole-image limit, kept so that what decoded before
/// still does and what refused still does. A window of a larger image is
/// within it; the whole image is not. A file this module cannot decode by
/// window is read whole only when it is no larger than this either.
pub const DECODE_LIMIT_BYTES: usize = 256 * 1024 * 1024;

/// A window of the image: rows `top..bottom`, columns `left..right`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Window {
    pub top: usize,
    pub left: usize,
    pub bottom: usize,
    pub right: usize,
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

/// One image (IFD) of a TIFF: its geometry, sample format and chunk tables
/// (their entries, not their values: [`Layout::chunk_ranges`] reads the few
/// a window needs).
#[derive(Debug, Clone)]
pub(crate) struct Layout {
    pub(crate) width: usize,
    pub(crate) height: usize,
    /// Tile size, or `(width, rows_per_strip)` for strips.
    chunk_width: usize,
    chunk_height: usize,
    tiled: bool,
    offsets: Entry,
    byte_counts: Entry,
    codec: Codec,
    /// Horizontal differencing (TIFF predictor 2).
    differenced: bool,
    sample: Sample,
    /// Samples stored per pixel.
    samples: usize,
    /// JPEG tables shared by every chunk (JPEGTables), spliced into each.
    jpeg_tables: Option<Vec<u8>>,
    big_endian: bool,
}

/// An image's layout, or why this module does not decode it.
#[derive(Debug, Clone)]
pub(crate) enum Readable {
    Layout(Layout),
    Unsupported(String),
}

impl Layout {
    /// The layout of `ifd`, or the reason this module does not decode it
    /// (the caller may then use the `tiff` crate).
    pub(crate) fn read<R: Read + Seek>(f: &mut TiffFile<R>, ifd: &Ifd) -> ifd::Result<Readable> {
        let unsupported = |reason: String| Ok(Readable::Unsupported(reason));
        let (width, height) = dimensions(f, ifd)?;
        let samples = f.tag_u64(ifd, Tag::SamplesPerPixel)?.unwrap_or(1) as usize;
        let bits = f
            .tag_u64s(ifd, Tag::BitsPerSample)?
            .unwrap_or_else(|| vec![1]);
        let format = f
            .tag_u64s(ifd, Tag::SampleFormat)?
            .unwrap_or_else(|| vec![1]);
        let photometric = f.tag_u64(ifd, Tag::PhotometricInterpretation)?;
        let compression = f.tag_u64(ifd, Tag::Compression)?.unwrap_or(1);
        let predictor = f.tag_u64(ifd, Tag::Predictor)?.unwrap_or(1);
        let planar = f.tag_u64(ifd, Tag::PlanarConfiguration)?.unwrap_or(1);

        if bits.is_empty() || format.is_empty() {
            return malformed("TIFF BitsPerSample or SampleFormat without a value");
        }
        if bits.iter().any(|&b| b != bits[0]) || format.iter().any(|&f| f != format[0]) {
            return unsupported(format!(
                "TIFF samples of mixed widths or formats ({bits:?} bits, format {format:?})"
            ));
        }
        let sample = match (bits[0], format[0]) {
            (8, 1) => Sample::U8,
            (16, 1) => Sample::U16,
            (32, 3) => Sample::F32,
            (64, 3) => Sample::F64,
            (b, f) => return unsupported(format!("{b}-bit TIFF samples of SampleFormat {f}")),
        };
        let codec = match compression {
            1 => Codec::None,
            5 => Codec::Lzw,
            8 | 32946 => Codec::Deflate,
            32773 => Codec::PackBits,
            7 => Codec::Jpeg,
            n => {
                let name = u16::try_from(n)
                    .map(|n| format!(" ({:?})", CompressionMethod::from_u16_exhaustive(n)))
                    .unwrap_or_default();
                return unsupported(format!("TIFF compression {n}{name}"));
            }
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
        if !supported {
            return unsupported(format!(
                "TIFF PhotometricInterpretation {photometric:?} with {samples} \
                 {}-bit samples under {codec:?} compression",
                bits[0]
            ));
        }
        let differenced = match predictor {
            1 => false,
            2 if integer && codec != Codec::Jpeg => true,
            p => return unsupported(format!("TIFF predictor {p} on {codec:?} {sample:?} data")),
        };
        if planar != 1 && samples > 1 {
            return unsupported(
                "planar-separate TIFF samples (PlanarConfiguration 2, one plane per sample)"
                    .to_string(),
            );
        }

        let tiled = ifd.has(Tag::TileWidth);
        let (chunk_width, chunk_height, offsets, counts) = if tiled {
            let tw = f.tag_u64(ifd, Tag::TileWidth)?.unwrap_or(0) as usize;
            let th = f.tag_u64(ifd, Tag::TileLength)?.unwrap_or(0) as usize;
            (tw, th, Tag::TileOffsets, Tag::TileByteCounts)
        } else {
            let rows = f
                .tag_u64(ifd, Tag::RowsPerStrip)?
                .map_or(height, |r| (r as usize).min(height));
            (width, rows, Tag::StripOffsets, Tag::StripByteCounts)
        };
        let (Some(&offsets), Some(&byte_counts)) = (ifd.get(offsets), ifd.get(counts)) else {
            return malformed(format!("TIFF without {offsets:?} and {counts:?}"));
        };
        let jpeg_tables = if codec == Codec::Jpeg {
            f.tag_bytes(ifd, Tag::JPEGTables)?
        } else {
            None
        };
        let layout = Layout {
            width,
            height,
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
            big_endian: f.big_endian(),
        };
        let chunks = (layout.across() * layout.down()) as u64;
        if chunk_width == 0
            || chunk_height == 0
            || offsets.count() < chunks
            || byte_counts.count() < chunks
        {
            return malformed("TIFF chunk tables do not cover the image");
        }
        Ok(Readable::Layout(layout))
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

    /// Refuse a window over [`DECODE_LIMIT_BYTES`], before anything of it
    /// is read.
    fn check_size(&self, window: &Window) -> ifd::Result<()> {
        let bytes = self.window_bytes(window);
        if bytes > DECODE_LIMIT_BYTES {
            return malformed(format!(
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
        Ok(())
    }

    /// The chunks `window` overlaps, as `(chunk index, chunk row, chunk col)`,
    /// row-major.
    fn chunks_in(&self, w: &Window) -> Vec<(usize, usize, usize)> {
        if w.bottom <= w.top || w.right <= w.left {
            return Vec::new();
        }
        let rows = w.top / self.chunk_height..w.bottom.div_ceil(self.chunk_height);
        let cols = w.left / self.chunk_width..w.right.div_ceil(self.chunk_width);
        rows.flat_map(|r| cols.clone().map(move |c| (r * self.across() + c, r, c)))
            .collect()
    }

    /// The byte ranges of `chunks` in the file: their table entries read run
    /// by run (a window's chunks in one chunk row are consecutive entries),
    /// nothing else of the tables.
    fn chunk_ranges<R: Read + Seek>(
        &self,
        f: &mut TiffFile<R>,
        chunks: &[(usize, usize, usize)],
    ) -> ifd::Result<Vec<Range<u64>>> {
        let mut ranges = Vec::with_capacity(chunks.len());
        let mut i = 0;
        while i < chunks.len() {
            let start = chunks[i].0;
            let mut end = start + 1;
            while i + (end - start) < chunks.len() && chunks[i + (end - start)].0 == end {
                end += 1;
            }
            let run = start as u64..end as u64;
            let offsets = f.u64s(&self.offsets, run.clone())?;
            let counts = f.u64s(&self.byte_counts, run)?;
            for (offset, count) in offsets.into_iter().zip(counts) {
                let Some(stop) = offset.checked_add(count) else {
                    return malformed("TIFF chunk past the end of any file");
                };
                ranges.push(offset..stop);
            }
            i += end - start;
        }
        Ok(ranges)
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

    /// Decode `window` from its `chunks` ([`Self::chunks_in`]), whose
    /// compressed bytes `read_chunk` returns by position in `chunks`.
    fn decode_window(
        &self,
        window: &Window,
        chunks: &[(usize, usize, usize)],
        mut read_chunk: impl FnMut(usize) -> ifd::Result<Vec<u8>>,
    ) -> ifd::Result<ViewBuffer> {
        self.check_size(window)?;
        let px = self.samples * self.sample.bytes();
        let (out_h, out_w) = (window.bottom - window.top, window.right - window.left);
        let mut out = vec![0u8; self.window_bytes(window)];
        for (k, &(_, r, c)) in chunks.iter().enumerate() {
            let (y0, x0) = (r * self.chunk_height, c * self.chunk_width);
            // A tile is always stored whole (padded past the image); the last
            // strip holds only the image's remaining rows.
            let rows = if self.tiled {
                self.chunk_height
            } else {
                self.chunk_height.min(self.height - y0)
            };
            let data = self
                .decode_chunk(&read_chunk(k)?, rows)
                .map_err(TiffError::Format)?;
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

/// An image's `(width, height)`: ImageWidth and ImageLength, both required.
fn dimensions<R: Read + Seek>(f: &mut TiffFile<R>, ifd: &Ifd) -> ifd::Result<(usize, usize)> {
    match (
        f.tag_u64(ifd, Tag::ImageWidth)?,
        f.tag_u64(ifd, Tag::ImageLength)?,
    ) {
        (Some(w), Some(h)) => Ok((w as usize, h as usize)),
        _ => malformed("TIFF image without ImageWidth and ImageLength"),
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

/// One level of a TIFF pyramid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LevelInfo {
    /// The image's position in the file's IFD chain.
    pub ifd: usize,
    pub width: usize,
    pub height: usize,
    /// `(width, height)` of its tiles; `None` for strips.
    pub tile: Option<(usize, usize)>,
}

/// The most IFDs a level scan walks: far beyond any pyramid, and a bound on
/// a malformed chain.
const MAX_IFDS: usize = 1024;

/// The pyramid levels of a TIFF, each with its IFD: IFD 0, then each later
/// image that is a reduced copy of the level before it — tiled or flagged
/// reduced-resolution (NewSubfileType bit 0), smaller in both axes, and of
/// level 0's aspect to within a pixel of rounding. The other images a slide
/// carries (an SVS's strip thumbnail, its label and macro photos) fail one of
/// these and are skipped. A SubIFD pyramid (OME-TIFF) is not followed: its
/// file has one level here.
///
/// The one definition of what a level is: `source(level=)` and
/// `.cv.slide_info()` both read it. Only IFD entries are read, never their
/// chunk tables.
fn levels<R: Read + Seek>(f: &mut TiffFile<R>) -> ifd::Result<Vec<(LevelInfo, Ifd)>> {
    let mut images = Vec::new();
    for (index, ifd) in f.ifds(MAX_IFDS)?.into_iter().enumerate() {
        let (width, height) = dimensions(f, &ifd)?;
        let tile = match f.tag_u64(&ifd, Tag::TileWidth)? {
            Some(tw) => Some((
                tw as usize,
                f.tag_u64(&ifd, Tag::TileLength)?.unwrap_or(0) as usize,
            )),
            None => None,
        };
        let reduced = f
            .tag_u64(&ifd, Tag::NewSubfileType)?
            .is_some_and(|t| t & 1 == 1);
        let info = LevelInfo {
            ifd: index,
            width,
            height,
            tile,
        };
        images.push((info, reduced, ifd));
    }
    let mut images = images.into_iter();
    let Some((base, _, base_ifd)) = images.next() else {
        return Ok(Vec::new());
    };
    let mut levels = vec![(base, base_ifd)];
    for (image, reduced, ifd) in images {
        let (first, last) = (&levels[0].0, &levels.last().expect("level 0").0);
        let smaller = image.width < last.width && image.height < last.height;
        let expected_height = first.height as f64 * image.width as f64 / first.width as f64;
        let same_aspect =
            (expected_height - image.height as f64).abs() <= 1.0 + 0.01 * image.height as f64;
        if (image.tile.is_some() || reduced) && smaller && same_aspect {
            levels.push((image, ifd));
        }
    }
    Ok(levels)
}

/// The pyramid levels of the TIFF `reader` holds ([`levels`]).
pub fn pyramid_levels<R: Read + Seek>(reader: R) -> Result<Vec<LevelInfo>, TiffError> {
    let mut f = TiffFile::open(reader)?;
    Ok(levels(&mut f)?.into_iter().map(|(l, _)| l).collect())
}

/// A slide's header facts: its pyramid and its scale.
#[derive(Debug, Clone, PartialEq)]
pub struct SlideInfo {
    /// [`pyramid_levels`], level 0 first.
    pub levels: Vec<LevelInfo>,
    /// Microns per pixel of level 0, `(x, y)`, when the file records it.
    pub mpp: Option<(f64, f64)>,
}

/// The pyramid and scale of a TIFF, read from its IFD entries alone.
pub fn slide_info<R: Read + Seek>(reader: R) -> Result<SlideInfo, TiffError> {
    let mut f = TiffFile::open(reader)?;
    let levels = levels(&mut f)?;
    let mpp = match levels.first() {
        Some((_, ifd0)) => microns_per_pixel(&mut f, ifd0)?,
        None => None,
    };
    Ok(SlideInfo {
        levels: levels.into_iter().map(|(l, _)| l).collect(),
        mpp,
    })
}

/// Level 0's microns per pixel: an Aperio ImageDescription's `MPP = m`, else
/// a resolution in pixels per centimetre (ResolutionUnit 3). A resolution
/// per inch is not used: writers default it to 72 dpi whatever the scale.
fn microns_per_pixel<R: Read + Seek>(
    f: &mut TiffFile<R>,
    ifd0: &Ifd,
) -> ifd::Result<Option<(f64, f64)>> {
    let description = f.tag_bytes(ifd0, Tag::ImageDescription)?.map(|b| {
        String::from_utf8_lossy(&b)
            .trim_end_matches('\0')
            .to_string()
    });
    if let Some(text) = description.filter(|t| t.starts_with("Aperio")) {
        let mpp = text.split('|').find_map(|field| {
            let (key, value) = field.split_once('=')?;
            (key.trim() == "MPP").then(|| value.trim().parse::<f64>().ok())?
        });
        if let Some(m) = mpp.filter(|m| *m > 0.0) {
            return Ok(Some((m, m)));
        }
    }
    if f.tag_u64(ifd0, Tag::ResolutionUnit)? != Some(3) {
        return Ok(None);
    }
    let x = f.tag_rational(ifd0, Tag::XResolution)?;
    let y = f.tag_rational(ifd0, Tag::YResolution)?;
    Ok(match (x, y) {
        (Some(x), Some(y)) if x > 0.0 && y > 0.0 => Some((1e4 / x, 1e4 / y)),
        _ => None,
    })
}

/// Pyramid level `level`'s position in the IFD chain, and its IFD. Level 0
/// reads the first IFD alone; a higher level walks the chain's entries.
fn level_ifd<R: Read + Seek>(f: &mut TiffFile<R>, level: u32) -> ifd::Result<(usize, Ifd)> {
    if level == 0 {
        return Ok((0, f.first_ifd()?));
    }
    let mut levels = levels(f)?;
    let n = levels.len();
    if (level as usize) >= n {
        return malformed(format!(
            "level {level} does not exist: this TIFF has {n} pyramid level{} (0..={})",
            if n == 1 { "" } else { "s" },
            n.saturating_sub(1)
        ));
    }
    let (info, ifd) = levels.swap_remove(level as usize);
    Ok((info.ifd, ifd))
}

/// Where a TIFF's bytes come from: memory, a local file, or (through the
/// plugin) a remote object read by byte range.
pub trait TiffSource: Read + Seek {
    /// Bring `ranges` in ahead of reading them. A remote source fetches them
    /// in one request; memory and local files have nothing to do.
    fn prefetch(&mut self, ranges: &[Range<u64>]) -> std::io::Result<()>;
}

impl<T: AsRef<[u8]>> TiffSource for Cursor<T> {
    fn prefetch(&mut self, _: &[Range<u64>]) -> std::io::Result<()> {
        Ok(())
    }
}

impl<R: Read + Seek> TiffSource for std::io::BufReader<R> {
    fn prefetch(&mut self, _: &[Range<u64>]) -> std::io::Result<()> {
        Ok(())
    }
}

impl<S: TiffSource + ?Sized> TiffSource for &mut S {
    fn prefetch(&mut self, ranges: &[Range<u64>]) -> std::io::Result<()> {
        (**self).prefetch(ranges)
    }
}

/// One pyramid level of a TIFF, opened: its header and IFD entries read,
/// neither its chunk tables nor its chunks.
pub struct TiffImage<S: TiffSource> {
    file: TiffFile<S>,
    layout: Readable,
    ifd: usize,
}

impl<S: TiffSource> TiffImage<S> {
    /// Open pyramid level `level` of the TIFF `source` holds.
    pub fn open(source: S, level: u32) -> Result<Self, TiffError> {
        let mut file = TiffFile::open(source)?;
        let (ifd, entries) = level_ifd(&mut file, level)?;
        let layout = Layout::read(&mut file, &entries)?;
        Ok(TiffImage { file, layout, ifd })
    }

    /// The image's position in the file's IFD chain.
    pub fn ifd(&self) -> usize {
        self.ifd
    }

    /// The `[H, W, C]` shape and element type a decode produces, or `None`
    /// for a layout this module does not decode.
    pub fn shape(&self) -> Option<([usize; 3], DType)> {
        match &self.layout {
            Readable::Layout(l) => Some(([l.height, l.width, l.channels()], l.dtype())),
            Readable::Unsupported(_) => None,
        }
    }

    /// Why this module does not decode the image, or `None` when it does.
    pub fn unsupported(&self) -> Option<&str> {
        match &self.layout {
            Readable::Layout(_) => None,
            Readable::Unsupported(reason) => Some(reason),
        }
    }

    /// The source, positioned anywhere.
    pub fn source(&mut self) -> &mut S {
        self.file.reader()
    }

    /// Decode `window` (`None`: the whole image), reading only the chunks it
    /// overlaps and their table entries — the chunks prefetched together,
    /// then read one by one. `None` for a layout this module does not
    /// decode. `window` must lie inside the image ([`Self::shape`]).
    pub fn decode(&mut self, window: Option<Window>) -> Result<Option<ViewBuffer>, TiffError> {
        let Readable::Layout(layout) = &self.layout else {
            return Ok(None);
        };
        let window = window.unwrap_or(Window {
            top: 0,
            left: 0,
            bottom: layout.height,
            right: layout.width,
        });
        layout.check_size(&window)?;
        let chunks = layout.chunks_in(&window);
        let ranges = layout.chunk_ranges(&mut self.file, &chunks)?;
        let reader = self.file.reader();
        reader.prefetch(&ranges)?;
        layout
            .decode_window(&window, &chunks, |k| {
                let range = &ranges[k];
                let len = usize::try_from(range.end - range.start)
                    .map_err(|_| TiffError::Format("TIFF chunk too large".into()))?;
                let mut chunk = vec![0u8; len];
                reader.seek(std::io::SeekFrom::Start(range.start))?;
                reader.read_exact(&mut chunk)?;
                Ok(chunk)
            })
            .map(Some)
    }
}

/// What decoding a TIFF image gave.
#[derive(Debug)]
pub enum TiffDecode {
    /// The image (or window), decoded by this module.
    Pixels(ViewBuffer),
    /// A layout this module leaves to the `tiff` crate, at this IFD, and why.
    Unsupported { ifd: usize, reason: String },
}

/// Pyramid level `level` of an in-memory TIFF, decoded over `window`
/// (`None`: the whole image).
///
/// `window` must lie inside the image (the caller validates it against
/// [`image_shape`]).
pub fn decode(bytes: &[u8], window: Option<Window>, level: u32) -> Result<TiffDecode, TiffError> {
    let mut image = TiffImage::open(Cursor::new(bytes), level)?;
    Ok(match image.decode(window)? {
        Some(buffer) => TiffDecode::Pixels(buffer),
        None => TiffDecode::Unsupported {
            ifd: image.ifd(),
            reason: image.unsupported().unwrap_or_default().to_string(),
        },
    })
}

/// The `[H, W, C]` shape and element type a decode of pyramid level `level`
/// of an in-memory TIFF produces, or `Ok(None)` when this module does not
/// decode it.
pub fn image_shape(bytes: &[u8], level: u32) -> Result<Option<([usize; 3], DType)>, TiffError> {
    Ok(TiffImage::open(Cursor::new(bytes), level)?.shape())
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
            offsets: Entry::unread(20),
            byte_counts: Entry::unread(20),
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
