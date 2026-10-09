//! Whole decodes shared by a call's rows: the patches of one image decode it
//! once.
//!
//! Cutting an image into patches makes one row per patch (`patch_grid` →
//! `explode` → a crop after the source), and every row of an image names the
//! same encoded image: the same bytes, or the same path. A format with no
//! window decode (PNG, JPEG, WebP, a TIFF layout the chunk decoder does not
//! carry) would decode the whole image for every patch. A node whose rows
//! crop their image (`roi_decode`) instead keeps one [`SharedDecodes`] per
//! call, and each row crops a view of the image's one decode.
//!
//! - **Keys** are the image's identity, not where it lies: the bytes' content
//!   (a 128-bit XXH3 and the length, so rows need not share a buffer), or the
//!   path, which also spares the file read; with the pyramid level.
//! - **One decode per key**, even across row threads: a row wanting an image
//!   another row is decoding waits for it ([`OnceLock`]). A failed decode is
//!   shared too, as its message, so every row of the image fails alike.
//! - **Bounded**: decoded images over [`SHARED_DECODE_BYTES`] are dropped
//!   least recently used first. A row holding one keeps it alive; a dropped
//!   image a later row needs is decoded again.
//! - **Safe to share**: a decoded buffer is shared by reference counting, and
//!   the engine writes in place only into a buffer it holds alone.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use view_buffer::ViewBuffer;

/// The most decoded bytes one call keeps for its rows to share: room for many
/// images per row thread.
pub(crate) const SHARED_DECODE_BYTES: usize = 1 << 30;

/// An encoded image's identity within a call.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum ImageKey {
    /// Encoded bytes, by content: their XXH3-128 and length.
    Bytes { hash: u128, len: usize },
    /// A file or object, by path.
    Path(String),
}

impl ImageKey {
    pub(crate) fn bytes(bytes: &[u8]) -> Self {
        ImageKey::Bytes {
            hash: xxhash_rust::xxh3::xxh3_128(bytes),
            len: bytes.len(),
        }
    }
}

/// What a path's one load found ([`SharedDecodes::get_or_load`]).
#[derive(Clone)]
pub(crate) enum Loaded {
    /// The image, decoded whole.
    Image(ViewBuffer),
    /// A TIFF the chunk decoder carries: each row reads its own window.
    Windowed,
}

/// One shared load: the image (or its failure's message), when it was last
/// used, and its size.
#[derive(Default)]
struct Slot {
    decoded: OnceLock<Result<Loaded, String>>,
    used: AtomicU64,
    bytes: AtomicUsize,
}

#[derive(Default)]
struct State {
    slots: HashMap<(ImageKey, u32), Arc<Slot>>,
    clock: u64,
    /// The content key of encoded bytes already hashed, by where they lie:
    /// exploded patch rows share their image's buffer, so it is hashed once.
    /// An address names the same bytes for the whole call, whose input
    /// columns outlive it.
    by_address: HashMap<(usize, usize), ImageKey>,
    #[cfg(test)]
    hashed: usize,
}

/// A call's shared whole decodes, for one node (see the module docs).
#[derive(Default)]
pub(crate) struct SharedDecodes {
    state: Mutex<State>,
    #[cfg(test)]
    budget: Option<usize>,
}

impl SharedDecodes {
    fn budget(&self) -> usize {
        #[cfg(test)]
        if let Some(b) = self.budget {
            return b;
        }
        SHARED_DECODE_BYTES
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// The slot of `key` at `level`, created if absent, marked used now.
    fn slot(&self, key: ImageKey, level: u32) -> Arc<Slot> {
        let mut state = self.lock();
        state.clock += 1;
        let now = state.clock;
        let slot = Arc::clone(state.slots.entry((key, level)).or_default());
        slot.used.store(now, Ordering::Relaxed);
        slot
    }

    /// The key of encoded `bytes`: their content's ([`ImageKey::bytes`]),
    /// hashed once per buffer the call's rows share.
    pub(crate) fn bytes_key(&self, bytes: &[u8]) -> ImageKey {
        let address = (bytes.as_ptr() as usize, bytes.len());
        if let Some(key) = self.lock().by_address.get(&address) {
            return key.clone();
        }
        let key = ImageKey::bytes(bytes);
        let mut state = self.lock();
        #[cfg(test)]
        {
            state.hashed += 1;
        }
        state.by_address.insert(address, key.clone());
        key
    }

    /// The decode of encoded bytes `key` at `level`: shared when another
    /// row made (or is making) it, else `decode`'s, kept for the rows after.
    pub(crate) fn get_or_decode(
        &self,
        key: ImageKey,
        level: u32,
        decode: impl FnOnce() -> Result<ViewBuffer, String>,
    ) -> Result<ViewBuffer, String> {
        match self.get_or_load(key, level, || decode().map(Loaded::Image))? {
            Loaded::Image(buffer) => Ok(buffer),
            Loaded::Windowed => Err("internal: encoded bytes loaded as a windowed TIFF".into()),
        }
    }

    /// The load of `key` at `level` — for a path, its probe, read and
    /// decode as one: shared when another row made (or is making) it, else
    /// `load`'s, kept for the rows after. Rows of an image starting together
    /// wait for one load rather than each reading the file.
    pub(crate) fn get_or_load(
        &self,
        key: ImageKey,
        level: u32,
        load: impl FnOnce() -> Result<Loaded, String>,
    ) -> Result<Loaded, String> {
        let slot = self.slot(key.clone(), level);
        let mut made = false;
        let loaded = slot
            .decoded
            .get_or_init(|| {
                made = true;
                load()
            })
            .clone();
        if made {
            if let Ok(Loaded::Image(buffer)) = &loaded {
                slot.bytes.store(
                    buffer.shape().iter().product::<usize>() * buffer.dtype().size_of(),
                    Ordering::Relaxed,
                );
                self.evict(&(key, level));
            }
        }
        loaded
    }

    /// Drop the least recently used decodes (but `keep`) while those kept
    /// hold more than the budget.
    fn evict(&self, keep: &(ImageKey, u32)) {
        let budget = self.budget();
        let mut state = self.lock();
        let mut held: usize = state
            .slots
            .values()
            .map(|s| s.bytes.load(Ordering::Relaxed))
            .sum();
        while held > budget {
            let oldest = state
                .slots
                .iter()
                .filter(|(k, s)| *k != keep && s.decoded.get().is_some())
                .min_by_key(|(_, s)| s.used.load(Ordering::Relaxed))
                .map(|(k, _)| k.clone());
            let Some(oldest) = oldest else { break };
            if let Some(slot) = state.slots.remove(&oldest) {
                held -= slot.bytes.load(Ordering::Relaxed);
            }
        }
    }

    #[cfg(test)]
    fn with_budget(budget: usize) -> Self {
        SharedDecodes {
            budget: Some(budget),
            ..Default::default()
        }
    }

    #[cfg(test)]
    fn kept(&self) -> usize {
        self.lock().slots.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(n: usize) -> ViewBuffer {
        ViewBuffer::from_vec_with_shape(vec![7u8; n], vec![n, 1, 1])
    }

    #[test]
    fn a_key_decodes_once_however_many_rows_ask() {
        let shared = SharedDecodes::default();
        let decodes = AtomicUsize::new(0);
        let decode = || {
            decodes.fetch_add(1, Ordering::Relaxed);
            Ok(image(10))
        };
        std::thread::scope(|s| {
            for _ in 0..8 {
                s.spawn(|| {
                    for _ in 0..50 {
                        shared
                            .get_or_decode(ImageKey::bytes(b"same image"), 0, decode)
                            .unwrap();
                    }
                });
            }
        });
        assert_eq!(decodes.load(Ordering::Relaxed), 1);
        // The level is part of the key; so is the content.
        shared
            .get_or_decode(ImageKey::bytes(b"same image"), 1, decode)
            .unwrap();
        shared
            .get_or_decode(ImageKey::bytes(b"other image"), 0, decode)
            .unwrap();
        shared
            .get_or_decode(ImageKey::Path("a.png".into()), 0, decode)
            .unwrap();
        assert_eq!(decodes.load(Ordering::Relaxed), 4);
    }

    #[test]
    fn bytes_are_hashed_once_per_buffer_and_matched_by_content() {
        let shared = SharedDecodes::default();
        let image = vec![3u8; 1000];
        let copy = image.clone();
        assert_eq!(shared.bytes_key(&image), shared.bytes_key(&image));
        assert_eq!(shared.lock().hashed, 1, "the same buffer hashes once");
        assert_eq!(
            shared.bytes_key(&copy),
            shared.bytes_key(&image),
            "equal content, one key"
        );
        assert_ne!(shared.bytes_key(&image[..999]), shared.bytes_key(&image));
    }

    #[test]
    fn a_failed_decode_is_shared_as_its_message() {
        let shared = SharedDecodes::default();
        let key = || ImageKey::Path("bad.png".into());
        let first = shared.get_or_decode(key(), 0, || Err("broken".into()));
        let second = shared.get_or_decode(key(), 0, || panic!("decoded twice"));
        assert_eq!(first.unwrap_err(), "broken");
        assert_eq!(second.unwrap_err(), "broken");
    }

    #[test]
    fn the_least_recently_used_go_over_the_budget() {
        let shared = SharedDecodes::with_budget(25);
        for name in ["a", "b", "c"] {
            shared
                .get_or_decode(ImageKey::Path(name.into()), 0, || Ok(image(10)))
                .unwrap();
        }
        assert_eq!(shared.kept(), 2, "30 bytes over a 25-byte budget");
        let decodes = AtomicUsize::new(0);
        let again = |name: &str| {
            shared
                .get_or_decode(ImageKey::Path(name.into()), 0, || {
                    decodes.fetch_add(1, Ordering::Relaxed);
                    Ok(image(10))
                })
                .unwrap()
        };
        again("c");
        assert_eq!(decodes.load(Ordering::Relaxed), 0, "the newest is kept");
        again("a");
        assert_eq!(decodes.load(Ordering::Relaxed), 1, "the oldest was dropped");
        // An image over the whole budget is still decoded and returned.
        let big = shared.get_or_decode(ImageKey::Path("big".into()), 0, || Ok(image(100)));
        assert_eq!(big.unwrap().shape(), &[100, 1, 1]);
    }
}
