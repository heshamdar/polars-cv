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
//! `POLARS_CV_SOURCE_HASH` is derived from every `.rs` file in the three crates plus
//! their manifests and the workspace lockfile (`source_hash.rs` defines the
//! set), so it moves whenever the built artifact could differ.
//! `polars_cv.build_info()` compares it against the same hash recomputed from
//! the working tree.

// What is hashed, and how, lives in `source_hash.rs`: the same file the crate's
// unit tests include, held to the digest of a committed fixture tree.
mod source_hash;

use std::path::PathBuf;

fn main() {
    // Read at run time, not `env!`: a build script compiled in one checkout and
    // reused from a shared target directory would otherwise hash the checkout
    // it was compiled in. Cargo sets it for the script's own run.
    let manifest = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("cargo sets it"));
    let workspace = manifest.parent().expect("crate sits in a workspace");

    let contents = source_hash::collect(workspace);

    // Re-run whenever any input changes, rather than only when Cargo would
    // otherwise rebuild — a stale hash is exactly the failure being guarded.
    //
    // The `src` *directories* are watched as well as the files: watching only
    // files means adding a `.rs` that nothing already-watched references does
    // not rerun this script, so the baked hash goes stale while the recomputed
    // one moves — a mismatch `maturin develop` could not clear.
    for crate_dir in source_hash::CRATES {
        println!(
            "cargo:rerun-if-changed={}",
            workspace.join(crate_dir).join("src").display()
        );
    }
    for key in contents.keys() {
        println!("cargo:rerun-if-changed={}", workspace.join(key).display());
    }

    let hash = source_hash::digest(&contents);
    println!("cargo:rustc-env=POLARS_CV_SOURCE_HASH={hash:016x}");
    println!(
        "cargo:rustc-env=POLARS_CV_BUILD_PROFILE={}",
        build_profile()
    );
}

/// The Cargo profile this build runs under: `debug`, `release`, `benchmark`.
///
/// Cargo tells a build script only `PROFILE` (`debug` or `release`, which
/// folds `benchmark` into `release`), but every profile builds into its own
/// directory, `<target>[/<triple>]/<profile>/build/<pkg>-<hash>/out`. The
/// benchmark harness records it so two results built differently (thin vs
/// fat LTO) are never compared. Anything else is a layout this does not know,
/// so it fails the build rather than guessing.
fn build_profile() -> String {
    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));
    let parts: Vec<_> = out_dir.components().collect();
    parts
        .iter()
        .rposition(|c| c.as_os_str() == "build")
        .and_then(|i| i.checked_sub(1))
        .and_then(|i| parts[i].as_os_str().to_str())
        .map(str::to_owned)
        .unwrap_or_else(|| panic!("cannot find the profile directory in OUT_DIR {out_dir:?}"))
}
