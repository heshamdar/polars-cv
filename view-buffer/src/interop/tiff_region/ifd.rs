//! TIFF structure, read lazily: the header, an IFD's entries, and only the
//! tag values a caller asks for.
//!
//! Every Rust TIFF reader (the `tiff` crate, `async-tiff`) loads an IFD's
//! chunk offset and byte-count tables whole whenever it visits the IFD. For a
//! 100k × 100k slide in 256-pixel tiles that is ~2.4 MB, read again for every
//! patch, so a patch's cost grew with the slide. Here an [`Entry`] is only a
//! value's type, count and location, and [`TiffFile::u64s`] reads the
//! elements a window needs: libtiff's lazy strile loading, as GDAL reads
//! cloud-optimised GeoTIFFs.

use std::collections::HashSet;
use std::io::{ErrorKind, Read, Seek, SeekFrom};
use std::ops::Range;

use tiff::tags::Tag;

/// Why a TIFF could not be read: the bytes could not be had, or they are not
/// a TIFF this module reads. A file that ends early is the latter.
#[derive(Debug)]
pub enum TiffError {
    Io(std::io::Error),
    Format(String),
}

impl std::fmt::Display for TiffError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TiffError::Io(e) => write!(f, "reading TIFF: {e}"),
            TiffError::Format(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for TiffError {}

impl From<std::io::Error> for TiffError {
    fn from(e: std::io::Error) -> Self {
        if e.kind() == ErrorKind::UnexpectedEof {
            TiffError::Format(format!("TIFF truncated: {e}"))
        } else {
            TiffError::Io(e)
        }
    }
}

pub type Result<T> = std::result::Result<T, TiffError>;

pub(crate) fn malformed<T>(message: impl Into<String>) -> Result<T> {
    Err(TiffError::Format(message.into()))
}

/// The most entries one IFD may hold: a bound on a malformed count (real
/// IFDs hold a few dozen).
const MAX_ENTRIES: u64 = 4096;

/// The most bytes a value read whole may take (an ImageDescription,
/// JPEGTables, BitsPerSample): a bound on a malformed count. Chunk tables are
/// never read whole.
const MAX_VALUE_BYTES: u64 = 16 << 20;

/// One IFD entry: its value's type and count, and its value bytes as stored
/// in the entry — the value itself when it fits, else its file offset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Entry {
    kind: u16,
    count: u64,
    data: [u8; 8],
}

impl Entry {
    pub(crate) fn count(&self) -> u64 {
        self.count
    }

    /// An entry of `count` LONG values at offset 0: a stand-in for tests
    /// that never read it.
    #[cfg(test)]
    pub(crate) fn unread(count: u64) -> Self {
        Entry {
            kind: 4,
            count,
            data: [0; 8],
        }
    }
}

/// The byte size of one value of TIFF type `kind`, or `None` for a type
/// TIFF 6 / BigTIFF does not define.
fn type_size(kind: u16) -> Option<u64> {
    Some(match kind {
        1 | 2 | 6 | 7 => 1,   // BYTE, ASCII, SBYTE, UNDEFINED
        3 | 8 => 2,           // SHORT, SSHORT
        4 | 9 | 11 | 13 => 4, // LONG, SLONG, FLOAT, IFD
        5 | 10 | 12 => 8,     // RATIONAL, SRATIONAL, DOUBLE
        16..=18 => 8,         // LONG8, SLONG8, IFD8
        _ => return None,
    })
}

/// One image file directory: its entries, by tag, and the next IFD's offset.
#[derive(Debug, Clone)]
pub(crate) struct Ifd {
    entries: Vec<(u16, Entry)>,
    next: Option<u64>,
}

impl Ifd {
    pub(crate) fn get(&self, tag: Tag) -> Option<&Entry> {
        let tag = tag.to_u16();
        self.entries.iter().find(|(t, _)| *t == tag).map(|(_, e)| e)
    }

    pub(crate) fn has(&self, tag: Tag) -> bool {
        self.get(tag).is_some()
    }
}

/// A TIFF or BigTIFF over `R`, its header read.
pub(crate) struct TiffFile<R> {
    reader: R,
    big_endian: bool,
    bigtiff: bool,
    first: u64,
}

impl<R: Read + Seek> TiffFile<R> {
    /// Read the header of the TIFF `reader` holds.
    pub(crate) fn open(mut reader: R) -> Result<Self> {
        reader.seek(SeekFrom::Start(0))?;
        let mut head = [0u8; 8];
        reader.read_exact(&mut head)?;
        let big_endian = match &head[..2] {
            b"II" => false,
            b"MM" => true,
            _ => return malformed("not a TIFF: no II/MM byte-order mark"),
        };
        let mut file = TiffFile {
            reader,
            big_endian,
            bigtiff: false,
            first: 0,
        };
        file.first = match file.u16([head[2], head[3]]) {
            42 => u64::from(file.u32(head[4..8].try_into().expect("4 bytes"))),
            43 => {
                // BigTIFF: offset byte size (8), a zero, then an 8-byte offset.
                if file.u16([head[4], head[5]]) != 8 {
                    return malformed("BigTIFF with an offset size other than 8");
                }
                file.bigtiff = true;
                let mut first = [0u8; 8];
                file.reader.read_exact(&mut first)?;
                file.u64(first)
            }
            magic => return malformed(format!("not a TIFF: magic number {magic}")),
        };
        Ok(file)
    }

    pub(crate) fn big_endian(&self) -> bool {
        self.big_endian
    }

    pub(crate) fn reader(&mut self) -> &mut R {
        &mut self.reader
    }

    fn u16(&self, b: [u8; 2]) -> u16 {
        if self.big_endian {
            u16::from_be_bytes(b)
        } else {
            u16::from_le_bytes(b)
        }
    }

    fn u32(&self, b: [u8; 4]) -> u32 {
        if self.big_endian {
            u32::from_be_bytes(b)
        } else {
            u32::from_le_bytes(b)
        }
    }

    fn u64(&self, b: [u8; 8]) -> u64 {
        if self.big_endian {
            u64::from_be_bytes(b)
        } else {
            u64::from_le_bytes(b)
        }
    }

    /// The bytes of an offset (4 classic, 8 BigTIFF), and of an entry's
    /// value field.
    fn word(&self) -> usize {
        if self.bigtiff {
            8
        } else {
            4
        }
    }

    fn offset(&self, b: &[u8]) -> u64 {
        if self.bigtiff {
            self.u64(b[..8].try_into().expect("8 bytes"))
        } else {
            u64::from(self.u32(b[..4].try_into().expect("4 bytes")))
        }
    }

    /// Read `n` bytes at `at`.
    fn read_at(&mut self, at: u64, n: u64) -> Result<Vec<u8>> {
        let n = usize::try_from(n).map_err(|_| TiffError::Format("TIFF value too large".into()))?;
        self.reader.seek(SeekFrom::Start(at))?;
        let mut buf = vec![0u8; n];
        self.reader.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// The IFD at `offset`: its entries, not their values.
    pub(crate) fn ifd(&mut self, offset: u64) -> Result<Ifd> {
        if offset == 0 {
            return malformed("TIFF IFD offset 0");
        }
        let word = self.word() as u64;
        let (count_size, entry_size) = if self.bigtiff { (8, 20) } else { (2, 12) };
        let raw = self.read_at(offset, count_size)?;
        let count = if self.bigtiff {
            self.u64(raw[..8].try_into().expect("8 bytes"))
        } else {
            u64::from(self.u16([raw[0], raw[1]]))
        };
        if count > MAX_ENTRIES {
            return malformed(format!("TIFF IFD at {offset} claims {count} entries"));
        }
        let body = self.read_at(offset + count_size, count * entry_size + word)?;
        let (table, next) = body.split_at((count * entry_size) as usize);
        let entries = table
            .chunks_exact(entry_size as usize)
            .map(|e| {
                let tag = self.u16([e[0], e[1]]);
                let kind = self.u16([e[2], e[3]]);
                let (count, value) = if self.bigtiff {
                    (self.u64(e[4..12].try_into().expect("8 bytes")), &e[12..20])
                } else {
                    (
                        u64::from(self.u32(e[4..8].try_into().expect("4 bytes"))),
                        &e[8..12],
                    )
                };
                let mut data = [0u8; 8];
                data[..value.len()].copy_from_slice(value);
                (tag, Entry { kind, count, data })
            })
            .collect();
        let next = Some(self.offset(next)).filter(|&n| n != 0);
        Ok(Ifd { entries, next })
    }

    /// The IFD chain from the first, at most `max` long; a cycle is
    /// malformed.
    pub(crate) fn ifds(&mut self, max: usize) -> Result<Vec<Ifd>> {
        let mut seen = HashSet::new();
        let mut ifds = Vec::new();
        let mut at = Some(self.first);
        while let Some(offset) = at {
            if ifds.len() == max {
                break;
            }
            if !seen.insert(offset) {
                return malformed(format!("TIFF IFD chain loops back to {offset}"));
            }
            let ifd = self.ifd(offset)?;
            at = ifd.next;
            ifds.push(ifd);
        }
        Ok(ifds)
    }

    /// The first IFD.
    pub(crate) fn first_ifd(&mut self) -> Result<Ifd> {
        self.ifd(self.first)
    }

    /// The raw bytes of values `range` of `entry`.
    fn value_bytes(&mut self, entry: &Entry, range: Range<u64>) -> Result<Vec<u8>> {
        let Some(size) = type_size(entry.kind) else {
            return malformed(format!("TIFF value of unknown type {}", entry.kind));
        };
        if range.start > range.end || range.end > entry.count {
            return malformed(format!(
                "TIFF value {}..{} of a {}-value entry",
                range.start, range.end, entry.count
            ));
        }
        let total = entry
            .count
            .checked_mul(size)
            .ok_or_else(|| TiffError::Format("TIFF value count overflows".into()))?;
        let (start, end) = (range.start * size, range.end * size);
        if total <= self.word() as u64 {
            return Ok(entry.data[start as usize..end as usize].to_vec());
        }
        let at = self.offset(&entry.data);
        self.read_at(at + start, end - start)
    }

    /// Values `range` of an unsigned-integer entry, as `u64`s.
    pub(crate) fn u64s(&mut self, entry: &Entry, range: Range<u64>) -> Result<Vec<u64>> {
        let bytes = self.value_bytes(entry, range)?;
        Ok(match entry.kind {
            1 => bytes.into_iter().map(u64::from).collect(),
            3 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| u64::from(self.u16(*b)))
                .collect(),
            4 | 13 => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| u64::from(self.u32(*b)))
                .collect(),
            16 | 18 => bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| self.u64(*b))
                .collect(),
            kind => {
                return malformed(format!(
                    "TIFF value of type {kind} is not an unsigned integer"
                ))
            }
        })
    }

    /// A tag's first unsigned value, if the tag is present.
    pub(crate) fn tag_u64(&mut self, ifd: &Ifd, tag: Tag) -> Result<Option<u64>> {
        match ifd.get(tag) {
            None => Ok(None),
            Some(e) if e.count == 0 => malformed(format!("TIFF tag {tag:?} has no value")),
            Some(e) => Ok(Some(self.u64s(&e.clone(), 0..1)?[0])),
        }
    }

    /// A tag's unsigned values, whole (a short value such as
    /// BitsPerSample), if the tag is present.
    pub(crate) fn tag_u64s(&mut self, ifd: &Ifd, tag: Tag) -> Result<Option<Vec<u64>>> {
        let Some(e) = ifd.get(tag).copied() else {
            return Ok(None);
        };
        if e.count.saturating_mul(8) > MAX_VALUE_BYTES {
            return malformed(format!("TIFF tag {tag:?} has {} values", e.count));
        }
        self.u64s(&e, 0..e.count).map(Some)
    }

    /// A tag's bytes, whole (BYTE, ASCII or UNDEFINED), if the tag is
    /// present.
    pub(crate) fn tag_bytes(&mut self, ifd: &Ifd, tag: Tag) -> Result<Option<Vec<u8>>> {
        let Some(e) = ifd.get(tag).copied() else {
            return Ok(None);
        };
        if !matches!(e.kind, 1 | 2 | 7) {
            return malformed(format!("TIFF tag {tag:?} is not bytes (type {})", e.kind));
        }
        if e.count > MAX_VALUE_BYTES {
            return malformed(format!("TIFF tag {tag:?} has {} bytes", e.count));
        }
        self.value_bytes(&e, 0..e.count).map(Some)
    }

    /// A RATIONAL tag's first value, if the tag is present and its
    /// denominator is not zero.
    pub(crate) fn tag_rational(&mut self, ifd: &Ifd, tag: Tag) -> Result<Option<f64>> {
        let Some(e) = ifd.get(tag).copied() else {
            return Ok(None);
        };
        if e.kind != 5 || e.count == 0 {
            return Ok(None);
        }
        let b = self.value_bytes(&e, 0..1)?;
        let num = self.u32(b[..4].try_into().expect("4 bytes"));
        let den = self.u32(b[4..8].try_into().expect("4 bytes"));
        Ok((den != 0).then(|| f64::from(num) / f64::from(den)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// A file builder: header, then data blocks and IFDs appended, in
    /// either byte order and either offset size.
    struct Builder {
        bytes: Vec<u8>,
        big_endian: bool,
        bigtiff: bool,
    }

    impl Builder {
        fn new(big_endian: bool, bigtiff: bool) -> Self {
            let mut b = Builder {
                bytes: Vec::new(),
                big_endian,
                bigtiff,
            };
            b.bytes
                .extend_from_slice(if big_endian { b"MM" } else { b"II" });
            if bigtiff {
                b.put(43, 2);
                b.put(8, 2);
                b.put(0, 2);
                b.put(0, 8);
            } else {
                b.put(42, 2);
                b.put(0, 4);
            }
            b
        }

        fn put(&mut self, v: u64, n: usize) {
            let be = v.to_be_bytes();
            let le = v.to_le_bytes();
            if self.big_endian {
                self.bytes.extend_from_slice(&be[8 - n..]);
            } else {
                self.bytes.extend_from_slice(&le[..n]);
            }
        }

        fn set_first(&mut self, at: u64) {
            let mut tail = std::mem::take(&mut self.bytes);
            let (pos, n) = if self.bigtiff { (8, 8) } else { (4, 4) };
            let mut field = Builder {
                bytes: Vec::new(),
                big_endian: self.big_endian,
                bigtiff: self.bigtiff,
            };
            field.put(at, n);
            tail[pos..pos + n].copy_from_slice(&field.bytes);
            self.bytes = tail;
        }

        /// Append values of `size` bytes each; their offset.
        fn values(&mut self, values: &[u64], size: usize) -> u64 {
            let at = self.bytes.len() as u64;
            for &v in values {
                self.put(v, size);
            }
            at
        }

        /// Append an IFD of `(tag, kind, count, value-or-offset)` entries
        /// pointing at `next`; its offset.
        fn ifd(&mut self, entries: &[(u16, u16, u64, u64)], next: u64) -> u64 {
            let at = self.bytes.len() as u64;
            let (w, count) = if self.bigtiff { (8, 8) } else { (4, 2) };
            self.put(entries.len() as u64, count);
            for &(tag, kind, n, value) in entries {
                self.put(u64::from(tag), 2);
                self.put(u64::from(kind), 2);
                self.put(n, w);
                let size = type_size(kind).unwrap() as usize;
                if n as usize * size <= w {
                    // Inline: the value left-justified in the field.
                    let start = self.bytes.len();
                    self.put(value, size);
                    self.bytes.resize(start + w, 0);
                } else {
                    self.put(value, w);
                }
            }
            self.put(next, w);
            at
        }
    }

    #[test]
    fn every_byte_order_and_offset_size_reads_the_same() {
        for big_endian in [false, true] {
            for bigtiff in [false, true] {
                let mut b = Builder::new(big_endian, bigtiff);
                let table: Vec<u64> = (0..1000).map(|i| i * 1000 + 7).collect();
                let long_at = b.values(&table, 4);
                let wide_at = b.values(&table, 8);
                let text_at = b.values(&b"Aperio|MPP = 0.5\0".map(u64::from), 1);
                let rational_at = b.values(&[30, 4], 4);
                // A BigTIFF entry holds an 8-byte RATIONAL inline: numerator
                // then denominator, each in the file's byte order.
                let rational = match (bigtiff, big_endian) {
                    (false, _) => rational_at,
                    (true, false) => 30 | 4 << 32,
                    (true, true) => 30 << 32 | 4,
                };
                let second = b.ifd(&[(256, 3, 1, 640)], 0);
                let first = b.ifd(
                    &[
                        (256, 4, 1, 70_000),
                        (258, 3, 2, 0), // two SHORTs: inline in either size
                        (270, 2, 17, text_at),
                        (282, 5, 1, rational),
                        (324, 4, 1000, long_at),
                        (325, 16, 1000, wide_at),
                    ],
                    second,
                );
                b.set_first(first);
                let mut f = TiffFile::open(Cursor::new(b.bytes)).unwrap();
                assert_eq!(f.big_endian(), big_endian);
                let ifds = f.ifds(16).unwrap();
                assert_eq!(ifds.len(), 2);
                let (ifd, case) = (&ifds[0], format!("be={big_endian} big={bigtiff}"));
                assert_eq!(
                    f.tag_u64(ifd, Tag::ImageWidth).unwrap(),
                    Some(70_000),
                    "{case}"
                );
                assert_eq!(
                    f.tag_u64s(ifd, Tag::BitsPerSample).unwrap(),
                    Some(vec![0, 0])
                );
                assert_eq!(
                    f.tag_bytes(ifd, Tag::ImageDescription).unwrap().unwrap(),
                    b"Aperio|MPP = 0.5\0"
                );
                assert_eq!(f.tag_rational(ifd, Tag::XResolution).unwrap(), Some(7.5));
                // Lazily, any slice of a table, in either integer width.
                for tag in [Tag::TileOffsets, Tag::TileByteCounts] {
                    let e = *ifd.get(tag).unwrap();
                    assert_eq!(f.u64s(&e, 998..1000).unwrap(), table[998..1000], "{case}");
                    assert_eq!(f.u64s(&e, 3..6).unwrap(), table[3..6], "{case}");
                }
                assert_eq!(f.tag_u64(&ifds[1], Tag::ImageWidth).unwrap(), Some(640));
                assert_eq!(f.tag_u64(ifd, Tag::TileWidth).unwrap(), None);
            }
        }
    }

    #[test]
    fn a_table_is_read_only_as_far_as_asked() {
        /// Counts the bytes read through it.
        struct Counting(Cursor<Vec<u8>>, u64);
        impl Read for Counting {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                let n = self.0.read(buf)?;
                self.1 += n as u64;
                Ok(n)
            }
        }
        impl Seek for Counting {
            fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
                self.0.seek(pos)
            }
        }
        let mut b = Builder::new(false, true);
        let table: Vec<u64> = (0..200_000).collect();
        let at = b.values(&table, 8);
        let first = b.ifd(&[(324, 16, 200_000, at)], 0);
        b.set_first(first);
        let mut f = TiffFile::open(Counting(Cursor::new(b.bytes), 0)).unwrap();
        let ifd = f.first_ifd().unwrap();
        let e = *ifd.get(Tag::TileOffsets).unwrap();
        assert_eq!(
            f.u64s(&e, 123_456..123_458).unwrap(),
            vec![123_456, 123_457]
        );
        assert!(f.reader().1 < 200, "read {} bytes", f.reader().1);
    }

    #[test]
    fn malformed_structure_is_refused() {
        let open = |bytes: Vec<u8>| TiffFile::open(Cursor::new(bytes));
        assert!(matches!(
            open(b"PK\x03\x04....".to_vec()),
            Err(TiffError::Format(_))
        ));
        assert!(
            matches!(open(b"II*".to_vec()), Err(TiffError::Format(_))),
            "truncated"
        );
        // An IFD past the end, and a chain that loops.
        let mut b = Builder::new(false, false);
        b.set_first(1 << 20);
        assert!(matches!(
            open(b.bytes).unwrap().ifds(8),
            Err(TiffError::Format(_))
        ));
        let mut b = Builder::new(true, false);
        let at = b.bytes.len() as u64;
        b.ifd(&[(256, 3, 1, 1)], at);
        b.set_first(at);
        let err = open(b.bytes).unwrap().ifds(8).unwrap_err().to_string();
        assert!(err.contains("loops"), "{err}");
        // A value range past the entry's count.
        let mut b = Builder::new(false, false);
        let first = b.ifd(&[(256, 3, 1, 1)], 0);
        b.set_first(first);
        let mut f = open(b.bytes).unwrap();
        let ifd = f.first_ifd().unwrap();
        let e = *ifd.get(Tag::ImageWidth).unwrap();
        assert!(f.u64s(&e, 0..2).is_err());
    }
}
