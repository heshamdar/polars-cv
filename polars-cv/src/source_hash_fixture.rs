//! The build's source hash, held to the committed fixture tree.
//!
//! `tests/fixtures/source_hash/` is a miniature workspace; `DIGEST` beside it
//! records what its inputs hash to. `tests/test_source_hash_fixture.py` asserts
//! the same digest of the same tree through `polars_cv/_build_info.py`, so a
//! change to either implementation's inputs, order or constants fails here or
//! there without a built extension.

use std::path::Path;

use crate::source_hash::{collect, digest};

const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/source_hash");

#[test]
fn the_build_hash_of_the_fixture_tree_is_its_recorded_digest() {
    let root = Path::new(FIXTURE);
    let contents = collect(root);
    let keys: Vec<&str> = contents.keys().map(String::as_str).collect();
    // The fixture exercises every kind of input: nested sources, manifests,
    // the workspace files and a file that is not a hashed input at all.
    assert_eq!(
        keys,
        [
            "Cargo.lock",
            "Cargo.toml",
            "polars-cv-macros/Cargo.toml",
            "polars-cv/Cargo.toml",
            "polars-cv/build.rs",
            "polars-cv/source_hash.rs",
            "polars-cv/src/lib.rs",
            "polars-cv/src/nested/deep.rs",
            "view-buffer/Cargo.toml",
            "view-buffer/src/core.rs",
        ],
        "the fixture's hashed inputs changed"
    );
    let recorded = std::fs::read_to_string(root.join("DIGEST")).expect("fixture DIGEST");
    assert_eq!(format!("{:016x}", digest(&contents)), recorded.trim());
}
