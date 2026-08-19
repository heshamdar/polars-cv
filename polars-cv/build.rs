//! Bakes a content hash of the Rust sources into the extension.
//!
//! `CARGO_PKG_VERSION` cannot detect a stale `.so`. It is the *release*
//! version, so it changes only at a version bump — and the window the staleness
//! guard exists for is precisely the one where it does not change: editing Rust
//! mid-cycle, or pulling commits that touch Rust, without re-running
//! `maturin develop`. The install is editable, so Python sources are live while
//! the extension is not, and 53% of the test suite is gated on a `.so` merely
//! *existing*. The result was old Rust running against new Python, reporting
//! pass.
//!
//! `POLARS_CV_SOURCE_HASH` is derived from every `.rs` file in both crates plus
//! their manifests and the workspace lockfile, so it moves whenever the built
//! artifact could differ. `polars_cv.build_info()` compares it against the same
//! hash recomputed from the working tree.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn main() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest.parent().expect("crate sits in a workspace");

    let mut contents: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    for crate_dir in ["polars-cv", "view-buffer"] {
        let root = workspace.join(crate_dir);
        collect_rust_sources(&root.join("src"), &root, crate_dir, &mut contents);
        push_file(
            &root.join("Cargo.toml"),
            &format!("{crate_dir}/Cargo.toml"),
            &mut contents,
        );
    }
    push_file(&workspace.join("Cargo.lock"), "Cargo.lock", &mut contents);

    // Re-run whenever any input changes, rather than only when Cargo would
    // otherwise rebuild — a stale hash is exactly the failure being guarded.
    for key in contents.keys() {
        println!("cargo:rerun-if-changed={}", workspace.join(key).display());
    }

    let mut hash: u64 = 0xcbf2_9ce4_8422_2325; // FNV-1a offset basis
    for (name, bytes) in &contents {
        for byte in name.as_bytes().iter().chain(bytes.iter()) {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    println!("cargo:rustc-env=POLARS_CV_SOURCE_HASH={hash:016x}");
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
        if path.is_dir() {
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
