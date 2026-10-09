"""Patches of one image decode it once.

Cutting an image into patches is ``patch_grid`` → ``explode`` → a crop after
the source, one patch per row. Every row of an image then names the same
encoded image (the same bytes, or the same path). A node whose rows crop their
image shares one whole decode between them for the call, so a pipeline of
per-patch ops (``crop`` → ``resize`` → ...) runs on each patch without decoding
the image once per patch.

Measured where a user would see it: ``_lib._image_decodes`` counts every
whole-image decode the plugin's sources make, and ``_fetch_bytes_read`` every
byte read from files. Every result is compared with the ``roi_decode`` pass off,
which decodes each row whole and crops it after.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

import polars_cv as cv
import polars_cv._lib as lib
from polars_cv import OptFlags, Pipeline, numpy_from_struct
from polars_cv._optimize import PASS_NAMES
from tests.conftest import plugin_required

if TYPE_CHECKING:
    from pathlib import Path

_ON = OptFlags.all()
_OFF = OptFlags(**{**{n: True for n in PASS_NAMES}, "roi_decode": False})

_SIDE, _P = 96, 32


def _encoded(seed: int, fmt: str) -> bytes:
    from PIL import Image

    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (_SIDE, _SIDE, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, fmt)
    return buf.getvalue()


def _cells(images: pl.DataFrame) -> pl.DataFrame:
    """One row per 32-pixel patch of each image (9 per image)."""
    return (
        images.with_columns(h=pl.lit(_SIDE), w=pl.lit(_SIDE))
        .with_columns(c=cv.patch_grid("h", "w", size=_P))
        .explode("c")
        .unnest("c")
    )


def _patch_pipe(source: str, **kwargs: object) -> Pipeline:
    """Crop each row's patch, then per-patch ops."""
    return (
        Pipeline()
        .source(source, **kwargs)  # type: ignore[arg-type]
        .crop(top=pl.col("top"), left=pl.col("left"), height=_P, width=_P)
        .resize(height=16, width=16)
        .grayscale()
    )


def _run(
    df: pl.DataFrame, col: str, pipe: Pipeline, flags: OptFlags
) -> tuple[list, int]:
    """The rows as arrays, and the whole decodes the query made."""
    before = lib._image_decodes()
    out = df.select(o=pl.col(col).cv.pipe(pipe).sink("numpy", opt_flags=flags))["o"]
    decodes = lib._image_decodes() - before
    return [None if v is None else numpy_from_struct(v) for v in out.to_list()], decodes


def _assert_same(on: list, off: list) -> None:
    assert len(on) == len(off)
    for a, b in zip(on, off, strict=True):
        if a is None or b is None:
            assert a is b
        else:
            np.testing.assert_array_equal(a, b)


@plugin_required
class TestPatchesShareADecode:
    @pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP"])
    def test_each_image_decodes_once(self, fmt: str) -> None:
        images = pl.DataFrame({"b": [_encoded(i, fmt) for i in range(3)]})
        cells = _cells(images)
        on, decodes = _run(cells, "b", _patch_pipe("image_bytes"), _ON)
        off, decodes_off = _run(cells, "b", _patch_pipe("image_bytes"), _OFF)
        _assert_same(on, off)
        assert decodes == 3, decodes
        assert decodes_off == cells.height, "the measurement is real"

    def test_rows_with_equal_bytes_built_apart_share_a_decode(self) -> None:
        """Rows need not share a buffer: equal bytes are one image."""
        data = _encoded(5, "PNG")
        df = pl.DataFrame(
            {
                "b": [bytes(data) for _ in range(6)],
                "top": [0, 8, 16, 32, 40, 64],
                "left": [0] * 6,
            }
        )
        on, decodes = _run(df, "b", _patch_pipe("image_bytes"), _ON)
        off, _ = _run(df, "b", _patch_pipe("image_bytes"), _OFF)
        _assert_same(on, off)
        assert decodes == 1, decodes

    def test_patches_of_files_read_and_decode_each_file_once(
        self, tmp_path: Path
    ) -> None:
        paths = []
        for i in range(2):
            p = tmp_path / f"im{i}.png"
            p.write_bytes(_encoded(10 + i, "PNG"))
            paths.append(str(p))
        cells = _cells(pl.DataFrame({"p": paths}))
        size = sum((tmp_path / f"im{i}.png").stat().st_size for i in range(2))
        read_before = lib._fetch_bytes_read()
        on, decodes = _run(cells, "p", _patch_pipe("file_path"), _ON)
        read = lib._fetch_bytes_read() - read_before
        off, _ = _run(cells, "p", _patch_pipe("file_path"), _OFF)
        _assert_same(on, off)
        assert decodes == 2, decodes
        assert read < size + 64 * 1024, (read, size)

    def test_a_declared_dtype_casts_each_patch(self) -> None:
        cells = _cells(pl.DataFrame({"b": [_encoded(1, "PNG")]}))
        pipe = (
            Pipeline()
            .source("image_bytes", dtype="f32")
            .crop(top=pl.col("top"), left=pl.col("left"), height=_P, width=_P)
        )
        on, decodes = _run(cells, "b", pipe, _ON)
        off, _ = _run(cells, "b", pipe, _OFF)
        _assert_same(on, off)
        assert on[0].dtype == np.float32
        assert decodes == 1

    def test_a_corrupt_image_fails_every_patch_alike(self) -> None:
        df = pl.DataFrame({"b": [b"not an image"] * 4, "top": [0] * 4, "left": [0] * 4})
        null_pipe = _patch_pipe("image_bytes", on_error="null")
        assert _run(df, "b", null_pipe, _ON)[0] == [None] * 4
        messages = []
        for flags in (_ON, _OFF):
            with pytest.raises(pl.exceptions.ComputeError) as excinfo:
                _run(df, "b", _patch_pipe("image_bytes"), flags)
            messages.append(str(excinfo.value))
        assert messages[0] == messages[1]

    def test_a_window_outside_the_image_is_the_crops_error(self) -> None:
        df = pl.DataFrame(
            {"b": [_encoded(2, "PNG")] * 2, "top": [0, 90], "left": [0, 0]}
        )
        messages = []
        for flags in (_ON, _OFF):
            with pytest.raises(pl.exceptions.ComputeError) as excinfo:
                _run(df, "b", _patch_pipe("image_bytes"), flags)
            messages.append(str(excinfo.value))
        assert messages[0] == messages[1]
        assert "lies outside" in messages[0]

    @pytest.mark.parametrize("op", ["invert", "add_constant", "clamp_max"])
    @pytest.mark.parametrize(
        ("size", "stride"),
        [((_P, _P), (8, 8)), ((_P, _SIDE), (8, _SIDE))],
        ids=["squares", "full-width bands"],
    )
    def test_overlapping_patches_do_not_see_each_others_writes(
        self, op: str, size: tuple[int, int], stride: tuple[int, int]
    ) -> None:
        """Patches overlap at stride 8, so they share pixels of the one decode:
        an op that may write in place must not change what the next patch
        reads. Full-width bands are contiguous views, the ones an in-place
        op could write through."""
        images = pl.DataFrame({"b": [_encoded(3, "PNG")], "h": [_SIDE], "w": [_SIDE]})
        cells = (
            images.with_columns(c=cv.patch_grid("h", "w", size=size, stride=stride))
            .explode("c")
            .unnest("c")
        )
        pipe = (
            Pipeline()
            .source("image_bytes")
            .crop(top=pl.col("top"), left=pl.col("left"), height=size[0], width=size[1])
        )
        pipe = {
            "invert": lambda p: p.invert(),
            "add_constant": lambda p: p.add_constant(7),
            "clamp_max": lambda p: p.clamp_max(100),
        }[op](pipe)
        on, decodes = _run(cells, "b", pipe, _ON)
        off, _ = _run(cells, "b", pipe, _OFF)
        _assert_same(on, off)
        assert decodes == 1
