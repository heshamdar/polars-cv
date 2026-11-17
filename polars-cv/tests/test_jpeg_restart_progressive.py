"""Progressive JPEGs with restart markers decode like libjpeg-turbo.

zune-jpeg (which ``image`` and so ``Pipeline().source("image_bytes")``
decode JPEGs with) garbled these before 0.5.15: mean absolute differences
of 25.6 and 15.3 from libjpeg-turbo at full size. Pillow cannot write a
restart interval into a progressive file, so the files are view-buffer's
committed fixtures (``view-buffer/tests/it/jpeg_scaled/generate.py``); the
reference is Pillow's own (libjpeg-turbo) decode of the same bytes.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from PIL import Image

from polars_cv import Pipeline, numpy_from_struct

from .conftest import plugin_required

_FIXTURES = (
    Path(__file__).resolve().parents[2] / "view-buffer" / "tests" / "it" / "jpeg_scaled"
)

#: Progressive (SOF2) files with a restart interval (DRI): every two MCUs,
#: and every MCU row.
_FILES = ["p420_rst_77x50.jpg", "p422_rstrow_200x150.jpg"]


def _markers(data: bytes) -> set[int]:
    """The marker codes of the segments before the first scan."""
    found, i = set(), 2
    while data[i + 1] != 0xDA:
        found.add(data[i + 1])
        i += 2 + int.from_bytes(data[i + 2 : i + 4], "big")
    return found


@plugin_required
@pytest.mark.parametrize("name", _FILES)
def test_progressive_jpeg_with_restart_markers_decodes_like_libjpeg_turbo(
    name: str,
) -> None:
    data = (_FIXTURES / name).read_bytes()
    assert {0xC2, 0xDD} <= _markers(data), "a progressive file with restarts"
    df = pl.DataFrame({"img": [data]}, schema={"img": pl.Binary})
    out = df.select(
        out=pl.col("img").cv.pipe(Pipeline().source("image_bytes")).sink("tensor")
    )
    got = numpy_from_struct(out["out"][0]).astype(np.float64)
    want = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), np.float64)
    assert got.shape == want.shape
    mean = np.abs(got - want).mean()
    assert mean <= 1.0, f"{name}: mean abs difference {mean:.3f} from libjpeg-turbo"
