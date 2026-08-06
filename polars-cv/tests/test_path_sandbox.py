"""`allowed_roots`: what a path column is permitted to read.

A path column is data, and data can come from somewhere you do not control.
Without `allowed_roots` both path-reading surfaces read whatever the column
names — any local file, any reachable URL — which is right for your own paths
and wrong for anyone else's.

The mechanism lives in `fetch.rs`, the one stage both surfaces share, so these
tests exercise **both** of them against the same expectations. A sandbox
enforced on one entry point and not the other is not a sandbox; it is a
suggestion, and `.cv.read_bytes()` is the one that would have been forgotten
because it was added later.

The escape cases matter more than the happy path. A check written against the
literal string is defeated by `..`; one written against the resolved path but
compared with `str.startswith` is defeated by a sibling directory sharing the
prefix; one covering only local paths leaves `s3://` open. Each has its own
test below.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from PIL import Image

from polars_cv import Pipeline

from .conftest import plugin_required


@pytest.fixture
def tree(tmp_path: Path) -> dict[str, Path]:
    """An allowed directory, a secret one beside it, and a symlink between."""
    allowed = tmp_path / "allowed"
    secret = tmp_path / "secret"
    sibling = tmp_path / "allowed-evil"
    for d in (allowed, secret, sibling):
        d.mkdir()

    image = allowed / "ok.png"
    Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(image)
    Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(sibling / "evil.png")
    # The secret is a *valid image*, deliberately. If it were not, a leaked
    # read would still null on the `file_path` source because the decode
    # fails — and every escape test below would pass without a sandbox at
    # all. Mutation-testing a naive check is what surfaced that: three of six
    # escape cases were passing for the wrong reason.
    secret_image = secret / "secret.png"
    Image.fromarray(np.full((4, 4, 3), 255, np.uint8)).save(secret_image)
    (secret / "passwd.txt").write_bytes(b"TOPSECRET")

    escape = allowed / "escape.png"
    os.symlink(secret_image, escape)

    return {
        "allowed": allowed,
        "secret": secret,
        "sibling": sibling,
        "image": image,
        "secret_image": secret_image,
        "escape": escape,
    }


def _read_bytes(paths: list[str], **kwargs) -> pl.Series:
    """Read through the `.cv.read_bytes()` surface."""
    df = pl.DataFrame({"p": paths})
    return df.select(b=pl.col("p").cv.read_bytes(on_error="null", **kwargs))["b"]


def _source(paths: list[str], **kwargs) -> pl.Series:
    """Read through the `file_path` source surface."""
    df = pl.DataFrame({"p": paths})
    pipe = Pipeline().source("file_path", on_error="null", **kwargs).cast("u8")
    return df.select(o=pl.col("p").cv.pipe(pipe).sink("png"))["o"]


#: Both surfaces that reach `fetch.rs`. Every expectation below is asserted
#: against both, because the whole design claim is that they share one
#: mechanism — a test that checked only one would pass while the other leaked.
_SURFACES = {"read_bytes": _read_bytes, "file_path source": _source}


@plugin_required
@pytest.mark.parametrize("surface", sorted(_SURFACES))
def test_unrestricted_by_default(tree, surface) -> None:
    """No `allowed_roots` reads whatever the column names.

    Pinned deliberately: the default cannot change to deny without breaking
    every existing pipeline, so it is a compatibility promise rather than an
    oversight.
    """
    read = _SURFACES[surface]
    assert read([str(tree["image"])])[0] is not None


@plugin_required
@pytest.mark.parametrize("surface", sorted(_SURFACES))
def test_allows_inside_and_denies_outside(tree, surface) -> None:
    """The basic contract, on both surfaces."""
    read = _SURFACES[surface]
    out = read(
        [str(tree["image"]), str(tree["secret_image"])],
        allowed_roots=[str(tree["allowed"])],
    )
    assert out[0] is not None, "a path inside the root must be readable"
    assert out[1] is None, "a path outside every root must be refused"


@plugin_required
@pytest.mark.parametrize("surface", sorted(_SURFACES))
def test_traversal_out_of_the_root_is_refused(tree, surface) -> None:
    """`..` must be resolved, not compared as text.

    The escaping path literally contains the allowed root as a prefix, so any
    check that compares strings without resolving admits it.
    """
    read = _SURFACES[surface]
    escape = str(tree["allowed"] / ".." / "secret" / "secret.png")
    assert read([escape], allowed_roots=[str(tree["allowed"])])[0] is None


@plugin_required
@pytest.mark.parametrize("surface", sorted(_SURFACES))
def test_symlink_out_of_the_root_is_refused(tree, surface) -> None:
    """A symlink inside the root pointing out of it does not grant access.

    This is the case lexical normalization alone cannot catch: the path has no
    `..` and is textually inside the root. Only canonicalization sees it.
    """
    read = _SURFACES[surface]
    assert read([str(tree["escape"])], allowed_roots=[str(tree["allowed"])])[0] is None


@plugin_required
@pytest.mark.parametrize("surface", sorted(_SURFACES))
def test_sibling_directory_sharing_the_prefix_is_refused(tree, surface) -> None:
    """`/x/allowed` must not admit `/x/allowed-evil`.

    The classic prefix bug: `startswith` says yes, the filesystem says these
    are unrelated directories.
    """
    read = _SURFACES[surface]
    out = read(
        [str(tree["sibling"] / "evil.png")], allowed_roots=[str(tree["allowed"])]
    )
    assert out[0] is None


@plugin_required
@pytest.mark.parametrize("surface", sorted(_SURFACES))
def test_remote_paths_are_covered_by_the_same_list(tree, surface) -> None:
    """A local-only allowlist would leave every URL reachable.

    No network is touched: the refusal happens before the fetch, which is the
    property being asserted — a denied remote path must never become a request.
    """
    read = _SURFACES[surface]
    out = read(["s3://not-allowed/secret.png"], allowed_roots=[str(tree["allowed"])])
    assert out[0] is None


@plugin_required
def test_denial_says_what_was_refused_and_what_would_pass(tree) -> None:
    """With `on_error="raise"` the refusal is actionable.

    A sandbox that fails with "could not read" trains people to widen it
    blindly; one that names the path and the configured roots does not.
    """
    df = pl.DataFrame({"p": [str(tree["secret"] / "passwd.txt")]})
    with pytest.raises(Exception) as excinfo:
        df.select(b=pl.col("p").cv.read_bytes(allowed_roots=[str(tree["allowed"])]))
    message = str(excinfo.value)
    assert "is not permitted" in message
    assert str(tree["secret"]) in message, "the refused path must be named"
    assert "allowed_roots" in message, "the message must name the setting to change"


@plugin_required
def test_refusal_follows_on_error(tree) -> None:
    """A denial is an ordinary read failure as far as `on_error` is concerned.

    It nulls the row under `on_error="null"` and fails the query otherwise, so
    a mixed column of trusted and untrusted paths stays usable rather than
    forcing a choice between the sandbox and the query.
    """
    paths = [str(tree["image"]), str(tree["secret_image"])]
    roots = [str(tree["allowed"])]

    nulled = _read_bytes(paths, allowed_roots=roots)
    assert nulled[0] is not None and nulled[1] is None

    df = pl.DataFrame({"p": paths})
    with pytest.raises(Exception):
        df.select(b=pl.col("p").cv.read_bytes(allowed_roots=roots))


def test_unset_allowed_roots_leaves_the_graph_json_unchanged() -> None:
    """An unrestricted source must serialize exactly as it did before.

    `graph_json` is the compiled-graph cache key, so an always-emitted key —
    even `null` — would split the cache for every existing pipeline and change
    nothing else.
    """
    import json

    spec = json.loads(Pipeline().source("file_path")._to_json())["source"]
    assert "allowed_roots" not in spec

    restricted = json.loads(
        Pipeline().source("file_path", allowed_roots=["/srv/images"])._to_json()
    )["source"]
    assert restricted["allowed_roots"] == ["/srv/images"]


def test_allowed_roots_participates_in_source_identity() -> None:
    """Two sources differing only in their sandbox are not the same source.

    `SourceSpec` is hashed for CSE. If the roots were left out of `__eq__` /
    `__hash__`, a restricted and an unrestricted source would collapse into one
    node and one of them would silently get the other's policy.
    """
    a = Pipeline().source("file_path", allowed_roots=["/srv/a"])._source
    b = Pipeline().source("file_path", allowed_roots=["/srv/b"])._source
    unrestricted = Pipeline().source("file_path")._source

    assert a != b
    assert a != unrestricted
    assert hash(a) != hash(b)
