"""The ``spatial_rule`` op contract reaches Python through ``op_contract``.

Phase 1 of the plan-time optimizer work adds a required ``SpatialDependency``
declaration to every engine op (the Rust side is pinned by the expected-value
coverage tests in ``view-buffer/src/ops/spatial_rule.rs``). This module is the
FFI round-trip guard: it asserts the rule is surfaced on ``op_contract`` for
every op the planner emits, and that the Python-visible spelling matches the
op's true spatial dependency.

There is deliberately no consumer of the rule yet (reordering passes are a
later phase); this only pins the contract surface.
"""

import json
import re

import pytest

from polars_cv import Pipeline
from tests.conftest import plugin_required

# Known spatial-rule vocabulary: pointwise | neighborhood:<radius> | global |
# geometric. This is the exact string set ``spatial_rule_name`` emits in Rust.
_VOCAB = re.compile(r"^(pointwise|global|geometric|neighborhood:\d+)$")


def _last_spec_contract(pipe: Pipeline) -> dict:
    """``op_contract`` for a pipeline's final op."""
    from polars_cv._lib import op_contract

    spec = pipe._ops[-1]
    return op_contract(json.dumps(spec.to_dict()))


def _src() -> Pipeline:
    return Pipeline().source("image_bytes")


# (case id, pipeline, expected spatial_rule) — one representative op per class.
_CASES = [
    # Pointwise: output at (y, x) depends only on input at (y, x).
    ("cast", _src().cast("f32"), "pointwise"),
    ("invert", _src().invert(), "pointwise"),
    ("grayscale", _src().grayscale(), "pointwise"),
    ("threshold", _src().grayscale().threshold(128), "pointwise"),
    # Neighborhood: bounded radius, same coordinate system.
    ("convolve2d", _src().convolve2d([0.0] * 9, 3), "neighborhood:1"),
    ("blur_sigma1", _src().blur(1.0), "neighborhood:3"),  # ceil(3*1) = 3
    ("blur_sigma2", _src().blur(2.0), "neighborhood:6"),  # ceil(3*2) = 6
    # Global: depends on a statistic over all pixels (or is a conservative
    # barrier). Canny is global because hysteresis links edges non-locally.
    ("adjust_contrast", _src().adjust_contrast(factor=1.5), "global"),
    ("equalize_histogram", _src().grayscale().equalize_histogram(), "global"),
    ("canny", _src().grayscale().canny(), "global"),
    ("perceptual_hash", _src().perceptual_hash(), "global"),
    ("reduce_sum", _src().reduce_sum(), "global"),
    # Geometric: coordinate transform / resample.
    ("resize", _src().resize(height=8, width=8), "geometric"),
    ("rotate", _src().rotate(30.0), "geometric"),
    ("pad", _src().pad(top=1), "geometric"),
    ("crop", _src().crop(top=0, left=0, height=2, width=2), "geometric"),
    ("flip", _src().flip(axes=[0]), "geometric"),
    ("transpose", _src().transpose(axes=[1, 0, 2]), "geometric"),
]


@plugin_required
class TestSpatialRuleFFI:
    @pytest.mark.parametrize(
        "case_id, pipe, expected", _CASES, ids=[c[0] for c in _CASES]
    )
    def test_spatial_rule_matches_expected(
        self, case_id: str, pipe: Pipeline, expected: str
    ) -> None:
        """Each op's ``op_contract`` carries the spatial rule it truly has."""
        contract = _last_spec_contract(pipe)
        assert "spatial_rule" in contract, (
            f"{case_id}: op_contract did not surface 'spatial_rule' — the FFI "
            f"is not exposing the SpatialDependency declaration"
        )
        assert contract["spatial_rule"] == expected, (
            f"{case_id}: expected spatial_rule {expected!r}, "
            f"got {contract['spatial_rule']!r}"
        )

    @pytest.mark.parametrize(
        "case_id, pipe, _expected", _CASES, ids=[c[0] for c in _CASES]
    )
    def test_spatial_rule_in_vocabulary(
        self, case_id: str, pipe: Pipeline, _expected: str
    ) -> None:
        """Every emitted spelling is one the Rust stringifier can produce."""
        rule = _last_spec_contract(pipe)["spatial_rule"]
        assert _VOCAB.match(rule), (
            f"{case_id}: {rule!r} is not a known spatial_rule spelling"
        )
