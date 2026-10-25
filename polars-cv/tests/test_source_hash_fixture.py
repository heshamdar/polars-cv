"""The source hash is one algorithm in two languages, held to one fixture.

``polars-cv/build.rs`` hashes the Rust sources into the extension and
``polars_cv/_build_info.py`` recomputes the hash from the working tree. They
must agree byte for byte, and ``test_compiled_plugin_matches_the_rust_sources``
only notices a disagreement on a machine with a built extension.

``tests/fixtures/source_hash/`` is a miniature workspace and ``DIGEST`` beside
it records what its inputs hash to. Python asserts that digest here; Rust
asserts the same file in ``src/source_hash_fixture.rs`` -- against
``polars-cv/source_hash.rs``, the very file ``build.rs`` includes. Changing the
inputs, their order or the hash in one language fails its test with no
extension built.
"""

from __future__ import annotations

from pathlib import Path

from polars_cv._build_info import (
    WORKSPACE_MARKERS,
    find_workspace_root,
    source_hash,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "source_hash"


def test_python_hash_of_the_fixture_tree_is_its_recorded_digest() -> None:
    recorded = (FIXTURE / "DIGEST").read_text().strip()
    assert source_hash(FIXTURE) == recorded


def test_workspace_root_is_found_by_marker_not_by_depth(tmp_path: Path) -> None:
    root = tmp_path / "a" / "b"
    for marker in WORKSPACE_MARKERS:
        (root / marker).parent.mkdir(parents=True, exist_ok=True)
        (root / marker).write_text("")
    deep = root / "polars-cv" / "python" / "x" / "y" / "z" / "pkg" / "mod.py"
    deep.parent.mkdir(parents=True)
    deep.write_text("")
    assert find_workspace_root(deep) == root.resolve()


def test_no_workspace_root_is_none(tmp_path: Path) -> None:
    (tmp_path / "polars-cv").mkdir()
    (tmp_path / "polars-cv" / "Cargo.toml").write_text("")  # one marker is not enough
    assert find_workspace_root(tmp_path / "polars-cv") is None
