from __future__ import annotations

import polars as pl
import pytest

from bench.data import DATA


@pytest.fixture(scope="session")
def manifest() -> pl.DataFrame:
    path = DATA / "coco-5k.parquet"
    if not path.exists():
        pytest.skip("run `uv run python -m bench.data` first")
    return pl.read_parquet(path)
