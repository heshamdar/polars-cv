//! The source hash: which files it covers, in what order, and the FNV-1a over them.
//!
//! The one Rust definition. `build.rs` pulls this file in as a module to bake
//! `POLARS_CV_SOURCE_HASH` into the extension, and `src/lib.rs` pulls the very
//! same file into its unit tests (`#[path = "../source_hash.rs"]`), so the
//! function the tests check is the function the build runs. Python recomputes
//! the hash in `polars_cv/_build_info.py`; both are held to the digest recorded
//! beside the fixture tree in `tests/fixtures/source_hash/`.
//!
//! This file is itself a hashed input (`INPUT_FILES`): it decides what the hash
//! covers, as `build.rs` did before it. It uses `std` only -- a build script
//! cannot use the crate's dependencies.

use std::collections::BTreeMap;
use std::path::Path;

/// The crates whose `src/**/*.rs` and manifest are hashed.
pub const CRATES: [&str; 3] = ["polars-cv", "polars-cv-macros", "view-buffer"];

/// Hashed files at fixed workspace-relative paths. The workspace manifest
/// carries `[profile.release]` and the toolchain file pins the compiler; both
/// change the artifact without touching any `.rs` file. `build.rs` and this
/// file decide what the hash covers at all.
pub const INPUT_FILES: [&str; 5] = [
    "Cargo.lock",
    "Cargo.toml",
    "rust-toolchain.toml",
    "polars-cv/build.rs",
    "polars-cv/source_hash.rs",
];

/// Every hashed input under `workspace`, keyed by its workspace-relative path.
/// An unreadable or absent file is left out, as in the Python walk.
pub fn collect(workspace: &Path) -> BTreeMap<String, Vec<u8>> {
    let mut contents = BTreeMap::new();
    for crate_dir in CRATES {
        let root = workspace.join(crate_dir);
        collect_rust_sources(&root.join("src"), &root, crate_dir, &mut contents);
        push_file(
            &root.join("Cargo.toml"),
            &format!("{crate_dir}/Cargo.toml"),
            &mut contents,
        );
    }
    for name in INPUT_FILES {
        push_file(&workspace.join(name), name, &mut contents);
    }
    contents
}

/// FNV-1a over each key's bytes then its contents, keys in sorted order.
pub fn digest(contents: &BTreeMap<String, Vec<u8>>) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325; // FNV-1a offset basis
    for (name, bytes) in contents {
        for byte in name.as_bytes().iter().chain(bytes.iter()) {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

fn collect_rust_sources(
    dir: &Path,
    crate_root: &Path,
    crate_name: &str,
    out: &mut BTreeMap<String, Vec<u8>>,
) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        // `symlink_metadata`, not `is_dir()`: the latter follows symlinks while
        // Python's `rglob` does not, so a symlinked directory under `src/`
        // would make the two hashes disagree permanently.
        let is_real_dir = std::fs::symlink_metadata(&path)
            .map(|m| m.file_type().is_dir())
            .unwrap_or(false);
        if is_real_dir {
            collect_rust_sources(&path, crate_root, crate_name, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            let rel = path
                .strip_prefix(crate_root)
                .expect("walked from crate_root")
                .to_string_lossy()
                .replace('\\', "/");
            push_file(&path, &format!("{crate_name}/{rel}"), out);
        }
    }
}

fn push_file(path: &Path, key: &str, out: &mut BTreeMap<String, Vec<u8>>) {
    if let Ok(bytes) = std::fs::read(path) {
        out.insert(key.to_string(), bytes);
    }
}
