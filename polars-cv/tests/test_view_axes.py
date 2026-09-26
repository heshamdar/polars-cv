"""Axis lists are checked against the planned rank when the builder is called.

``transpose``/``flip`` fix the output layout, so an axis list that does not fit
the input's rank is a build-time ``ValueError`` wherever the rank is known —
not an abort in plan-time shape inference, and not a row error much later.
The check belongs to the engine op's own ``validate``; these tests pin it at the
user-facing entry point.
"""

from __future__ import annotations

import pytest

from polars_cv import Pipeline

from ._plan_view import planned
from .conftest import plugin_required


def _rank3() -> Pipeline:
    pipe = Pipeline().source("image_bytes")
    assert planned(pipe).ndim == 3, "premise: an image source plans rank 3"
    return pipe


@plugin_required
class TestTranspose:
    @pytest.mark.parametrize("axes", [[1, 0], [0, 1, 2, 3], [0, 0, 1], [0, 1, 3]])
    def test_a_non_permutation_of_the_input_axes_is_rejected(
        self, axes: list[int]
    ) -> None:
        with pytest.raises(ValueError, match="axes"):
            _rank3().transpose(axes)

    def test_a_permutation_plans_the_permuted_rank(self) -> None:
        assert planned(_rank3().transpose([1, 0, 2])).ndim == 3


@plugin_required
class TestFlip:
    @pytest.mark.parametrize("axes", [[3], [0, 5]])
    def test_an_axis_outside_the_input_rank_is_rejected(self, axes: list[int]) -> None:
        with pytest.raises(ValueError, match="axes|axis"):
            _rank3().flip(axes)

    def test_in_range_axes_are_accepted(self) -> None:
        assert planned(_rank3().flip([0, 1])).ndim == 3

    def test_a_negative_axis_is_rejected_at_build_time(self) -> None:
        with pytest.raises(ValueError):
            _rank3().flip([-1])
