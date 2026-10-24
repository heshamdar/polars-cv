//! The process-wide compiled-graph cache: one compilation per distinct graph,
//! a hit validated by full equality of the kwargs.

use polars::prelude::*;
use std::sync::{Arc, Mutex, OnceLock};

use super::compile::CompiledGraph;

// ============================================================================
// Process-wide compiled-graph cache
// ============================================================================

/// Maximum number of compiled graphs kept resident. Each entry is small
/// (the parsed spec, no data), so this mostly bounds pathological churn from
/// many distinct pipelines in one process.
pub(super) const GRAPH_CACHE_CAP: usize = 32;

pub(super) type CacheEntries = Vec<(u64, Arc<CompiledGraph>)>;

pub(super) fn graph_cache() -> &'static Mutex<CacheEntries> {
    static CACHE: OnceLock<Mutex<CacheEntries>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(Vec::new()))
}

pub(super) fn cache_key_hash(graph_json: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    graph_json.hash(&mut hasher);
    hasher.finish()
}

/// Fetch the compiled form of a graph, compiling and caching on miss.
///
/// A hit requires both the hash **and** full equality of the kwargs against
/// the stored copy — the hash alone is never trusted. Most-recently-used
/// entries are kept at the front; the lock is never held during compilation.
pub(crate) fn get_or_compile(graph_json: &str) -> PolarsResult<Arc<CompiledGraph>> {
    let hash = cache_key_hash(graph_json);

    {
        let mut cache = graph_cache().lock().unwrap();
        if let Some(pos) = cache
            .iter()
            .position(|(h, compiled)| *h == hash && compiled.key.graph_json == graph_json)
        {
            let entry = cache.remove(pos);
            let compiled = entry.1.clone();
            cache.insert(0, entry);
            return Ok(compiled);
        }
    }

    // Compile outside the lock; concurrent misses may compile the same graph
    // twice, which is harmless (the result is deterministic).
    let compiled = Arc::new(CompiledGraph::compile(graph_json)?);

    let mut cache = graph_cache().lock().unwrap();
    let already_present = cache
        .iter()
        .any(|(h, c)| *h == hash && c.key.graph_json == graph_json);
    if !already_present {
        cache.insert(0, (hash, compiled.clone()));
        cache.truncate(GRAPH_CACHE_CAP);
    }
    Ok(compiled)
}
