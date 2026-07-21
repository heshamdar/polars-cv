"""Tests for the ``.thumbnail(max_size)`` scaled-decode operator.

``thumbnail`` is the explicit, chainable form of
``source(..., decode_max_size=...)``; the underlying JPEG IDCT-scaled decode is
already covered by the decode-scale tests, so these focus on the new surface.
"""

from __future__ import annotations

import pytest

from polars_cv import Pipeline


class TestThumbnailBuilder:
    def test_sets_decode_max_size_on_source(self) -> None:
        pipe = Pipeline().source("image_bytes").thumbnail(64)
        assert pipe._source is not None
        assert pipe._source.decode_max_size == 64

    def test_works_on_file_path_source(self) -> None:
        pipe = Pipeline().source("file_path").thumbnail(128)
        assert pipe._source.decode_max_size == 128

    def test_is_immutable(self) -> None:
        base = Pipeline().source("image_bytes")
        thumb = base.thumbnail(32)
        assert base._source.decode_max_size is None
        assert thumb._source.decode_max_size == 32

    def test_requires_source_first(self) -> None:
        with pytest.raises(ValueError, match="requires a source"):
            Pipeline().thumbnail(64)

    def test_rejects_non_image_source(self) -> None:
        pipe = Pipeline().source("list", dtype="f32")
        with pytest.raises(ValueError, match="image_bytes.*file_path"):
            pipe.thumbnail(64)

    @pytest.mark.parametrize("bad", [0, -5, 3.5, "64", True])
    def test_rejects_bad_max_size(self, bad: object) -> None:
        pipe = Pipeline().source("image_bytes")
        with pytest.raises(ValueError, match="positive int"):
            pipe.thumbnail(bad)  # type: ignore[arg-type]

    def test_composes_with_downstream_ops(self) -> None:
        # The curation pattern: cheap thumbnail decode -> cheap feature.
        pipe = Pipeline().source("image_bytes").thumbnail(64).perceptual_hash()
        assert pipe._source.decode_max_size == 64
