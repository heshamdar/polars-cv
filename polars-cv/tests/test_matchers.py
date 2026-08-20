"""Tests for detection matchers (PreMatchedAdapter and protocol compliance)."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline
from polars_cv.metrics import (
    BBoxMatcher,
    ContourMatcher,
    DetectionTable,
    PreMatchedAdapter,
    precision_recall_curve,
)
from polars_cv.metrics._matching._protocol import Matcher
from polars_cv.metrics._types import (
    COL_CLASS_ID,
    COL_IMAGE_ID,
    COL_IS_TP,
    COL_N_GTS,
)
from tests.conftest import plugin_required

if TYPE_CHECKING:
    pass


class TestPreMatchedAdapter:
    """Tests for the pre-matched pass-through adapter."""

    def test_protocol_compliance(self) -> None:
        """PreMatchedAdapter satisfies the Matcher protocol."""
        assert isinstance(PreMatchedAdapter(), Matcher)

    def test_basic_matching(self) -> None:
        """PreMatchedAdapter wraps flat data into a DetectionTable."""
        data = pl.DataFrame(
            {
                "image": ["img1", "img1", "img1", "img2", "img2"],
                "conf": [0.9, 0.7, 0.3, 0.8, 0.5],
                "tp": [True, False, True, True, False],
            }
        )
        adapter = PreMatchedAdapter()
        table = adapter.match(
            data,
            pred_col="conf",
            gt_col="tp",
            image_id_col="image",
        )

        assert isinstance(table, DetectionTable)
        det_df, meta_df = table.collect(engine="streaming")
        assert det_df.height == 5
        assert meta_df.height == 2

    def test_explicit_n_gts(self) -> None:
        """Custom n_gts_col overrides the automatic sum of TPs."""
        data = pl.DataFrame(
            {
                "image": ["img1", "img1"],
                "conf": [0.9, 0.7],
                "tp": [True, False],
                "num_gts": [5, 5],
            }
        )
        adapter = PreMatchedAdapter()
        table = adapter.match(
            data,
            pred_col="conf",
            gt_col="tp",
            image_id_col="image",
            n_gts_col="num_gts",
        )
        _, meta_df = table.collect(engine="streaming")
        n_gts_val = meta_df.filter(pl.col(COL_IMAGE_ID) == "img1")[COL_N_GTS].item()
        assert n_gts_val == 5

    def test_with_class_col(self) -> None:
        """Class column is respected."""
        data = pl.DataFrame(
            {
                "image": ["img1", "img1", "img1"],
                "conf": [0.9, 0.7, 0.5],
                "tp": [True, False, True],
                "cls": ["cat", "cat", "dog"],
            }
        )
        adapter = PreMatchedAdapter()
        table = adapter.match(
            data,
            pred_col="conf",
            gt_col="tp",
            image_id_col="image",
            class_col="cls",
        )
        det_df, _ = table.collect(engine="streaming")
        classes = det_df[COL_CLASS_ID].unique().to_list()
        assert "cat" in classes
        assert "dog" in classes

    def test_feeds_into_metric(self) -> None:
        """PreMatchedAdapter output can be consumed by precision_recall_curve."""
        data = pl.DataFrame(
            {
                "image": ["img1", "img1", "img1", "img2"],
                "conf": [0.9, 0.7, 0.3, 0.8],
                "tp": [True, False, True, True],
            }
        )
        adapter = PreMatchedAdapter()
        table = adapter.match(
            data,
            pred_col="conf",
            gt_col="tp",
            image_id_col="image",
        )
        result = precision_recall_curve(table)
        assert result.curve.height > 0

    def test_auto_image_id(self) -> None:
        """When no image_id_col is given, row index is used."""
        data = pl.DataFrame(
            {
                "conf": [0.9, 0.7],
                "tp": [True, False],
            }
        )
        adapter = PreMatchedAdapter()
        table = adapter.match(data, pred_col="conf", gt_col="tp")
        det_df, _ = table.collect(engine="streaming")
        assert det_df.height == 2


# ---------------------------------------------------------------------------
# The Matcher protocol must actually constrain its implementations
# ---------------------------------------------------------------------------


def _keyword_params(func: object) -> dict[str, inspect.Parameter]:
    """The keyword-acceptable parameters of ``func``, excluding ``self``."""
    params = inspect.signature(func).parameters  # type: ignore[arg-type]
    return {
        name: p
        for name, p in params.items()
        if name != "self"
        and p.kind
        in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    }


@pytest.mark.parametrize(
    "impl", [PreMatchedAdapter, BBoxMatcher, ContourMatcher], ids=lambda c: c.__name__
)
def test_matchers_accept_every_protocol_parameter(impl: type) -> None:
    """Each matcher must accept the whole ``Matcher.match`` keyword surface.

    ``isinstance(x, Matcher)`` — the only thing that guarded this — is an
    attribute-presence check: ``@runtime_checkable`` protocols deliberately do
    **not** compare signatures, so any class with any ``match`` attribute
    passed. That made the protocol decorative, which is worse than absent,
    because ``metrics`` dispatches through it generically.

    Widening is fine (``PreMatchedAdapter`` adds ``n_gts_col`` and friends, and
    gives ``pred_col``/``gt_col`` defaults); dropping or renaming a parameter
    the protocol promises is not, because a caller written against the
    protocol would then fail on that implementation alone.
    """
    declared = _keyword_params(Matcher.match)
    actual = _keyword_params(impl.match)
    assert declared, "probe is broken: Matcher.match declares no parameters"

    missing = sorted(set(declared) - set(actual))
    assert not missing, (
        f"{impl.__name__}.match does not accept {missing}, which "
        f"Matcher.match promises. A caller written against the protocol would "
        f"raise TypeError on this implementation."
    )


@pytest.mark.parametrize(
    "impl", [PreMatchedAdapter, BBoxMatcher, ContourMatcher], ids=lambda c: c.__name__
)
def test_matchers_require_nothing_the_protocol_omits(impl: type) -> None:
    """A matcher may add parameters, but every addition needs a default.

    Otherwise generic dispatch through ``Matcher`` — which only ever supplies
    the declared keywords — cannot construct a valid call.
    """
    declared = set(_keyword_params(Matcher.match))
    extra_required = sorted(
        name
        for name, p in _keyword_params(impl.match).items()
        if name not in declared and p.default is inspect.Parameter.empty
    )
    assert not extra_required, (
        f"{impl.__name__}.match requires {extra_required}, which Matcher.match "
        f"does not declare, so a protocol-typed caller cannot supply them."
    )


# ---------------------------------------------------------------------------
# Source-format detection belongs to Rust, not to metrics
# ---------------------------------------------------------------------------


@plugin_required
def test_contour_matcher_reads_an_image_bytes_mask(encode_png) -> None:
    """A PNG mask column must work, as `source("auto")` already makes it.

    `_detect_source_info` mapped every `pl.Binary` column to `"blob"`, but
    `resolve_auto_format` — the authority `source("auto")` uses — checks the
    VIEW magic bytes and falls back to `image_bytes`. So an image-bytes mask
    reached the blob decoder and the query failed, while the same column read
    fine through the documented default source.
    """
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 4:12] = 255
    png = encode_png(mask)

    frame = pl.DataFrame({"pred": [png], "gt": [png]})
    table = ContourMatcher().match(frame, pred_col="pred", gt_col="gt")
    detections = table.detections.collect(engine="streaming")

    assert detections.height == 1, (
        f"a PNG mask matched against itself should give one detection, got "
        f"{detections.height}"
    )
    assert detections[COL_IS_TP][0], "a mask matched against itself is a TP"


@plugin_required
def test_contour_matcher_still_reads_a_view_blob() -> None:
    """The blob path must keep working — `auto` distinguishes the two by magic."""
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 4:12] = 255
    blob = (
        pl.DataFrame({"m": [mask.tolist()]}, schema={"m": pl.List(pl.List(pl.UInt8))})
        .select(
            b=pl.col("m").cv.pipe(Pipeline().source("list", dtype="u8")).sink("blob")
        )["b"]
        .to_list()
    )
    frame = pl.DataFrame({"pred": blob, "gt": blob})
    table = ContourMatcher().match(frame, pred_col="pred", gt_col="gt")
    assert table.detections.collect(engine="streaming").height == 1
