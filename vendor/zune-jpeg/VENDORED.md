# Vendored `zune-jpeg` 0.5.8

Source: crates.io `zune-jpeg` 0.5.8 (MIT OR Apache-2.0 OR Zlib, all three
licence files kept), wired in by the workspace's `[patch.crates-io]`. It
replaces the 0.5.x zune-jpeg `image` decodes JPEGs with as well as the one
polars-cv calls directly; the 0.4.x zune-jpeg another dependency pulls in is
untouched.

Only `src/`, the licences, README and Changelog are vendored. The manifest
drops the (empty) dev-dependencies; tests, benches and fuzz targets were
never published. Every feature `src` names in `cfg`s is kept.

## Local changes

None yet.

Drop this directory and the patch once upstream ships an equivalent.
