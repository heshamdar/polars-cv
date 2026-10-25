"""The extension's source hash, recomputed from a checkout.

``polars-cv/build.rs`` bakes a hash of the Rust sources into the compiled
extension; ``build_info()`` recomputes it here from the working tree to tell
whether the extension was built from the sources now on disk. The two must
agree byte for byte, which makes this the second implementation of one
algorithm. Rust's is ``polars-cv/source_hash.rs`` (the file ``build.rs`` and the
crate's unit tests both include); both are held to the digest recorded beside
the committed fixture tree ``tests/fixtures/source_hash/``, asserted from each
language, so drift in either fails without a built extension.

This module imports nothing from the package root.
"""

from __future__ import annotations

from pathlib import Path

#: Files whose joint presence marks a workspace root (a source checkout). Shared
#: with ``tests/_discovery.py``'s ``CHECKOUT_MARKERS``, which adds to it.
WORKSPACE_MARKERS: tuple[str, ...] = (
    "polars-cv/Cargo.toml",
    "view-buffer/Cargo.toml",
)

#: Mirrors ``CRATES`` in ``source_hash.rs``.
_CRATES: tuple[str, ...] = ("polars-cv", "polars-cv-macros", "view-buffer")

#: Mirrors ``INPUT_FILES`` in ``source_hash.rs``.
_INPUT_FILES: tuple[str, ...] = (
    "Cargo.lock",
    "Cargo.toml",
    "rust-toolchain.toml",
    "polars-cv/build.rs",
    "polars-cv/source_hash.rs",
)


def find_workspace_root(start: Path | None = None) -> Path | None:
    """The nearest ancestor of ``start`` that is a workspace root, else ``None``.

    Searches upward from this package (or ``start``) for a directory holding
    every :data:`WORKSPACE_MARKERS` file, so the answer does not depend on how
    deep the package sits. ``None`` means an installed wheel rather than a
    source checkout.
    """
    here = (start or Path(__file__)).resolve()
    for candidate in (here, *here.parents):
        if all((candidate / marker).is_file() for marker in WORKSPACE_MARKERS):
            return candidate
    return None


def source_hash(root: Path) -> str:
    """The build's source hash of the workspace at ``root``.

    Same inputs, same order, same FNV-1a as ``source_hash.rs``: every ``.rs``
    under each crate's ``src/``, each crate's manifest, then the fixed input
    files, keyed by workspace-relative path, hashed in sorted key order. An
    absent or unreadable file is left out.
    """
    contents: dict[str, bytes] = {}

    def _push(path: Path, key: str) -> None:
        try:
            contents[key] = path.read_bytes()
        except OSError:
            pass

    for crate in _CRATES:
        crate_root = root / crate
        for rs in sorted((crate_root / "src").rglob("*.rs")):
            _push(rs, f"{crate}/{rs.relative_to(crate_root).as_posix()}")
        _push(crate_root / "Cargo.toml", f"{crate}/Cargo.toml")
    for name in _INPUT_FILES:
        _push(root / name, name)

    digest = 0xCBF29CE484222325  # FNV-1a offset basis
    for key in sorted(contents):
        for byte in key.encode() + contents[key]:
            digest ^= byte
            digest = (digest * 0x00000100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{digest:016x}"


def source_hash_from_tree() -> str | None:
    """The source hash of the checkout this package sits in.

    ``None`` when there is no checkout (an installed wheel), where staleness is
    not a question that arises.
    """
    root = find_workspace_root()
    return None if root is None else source_hash(root)
