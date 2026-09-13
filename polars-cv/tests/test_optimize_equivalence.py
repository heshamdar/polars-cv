"""Differential-equivalence guard for the optimization phase.

The core guarantee: toggling optimization passes changes only the *physical*
graph, never the result the user sees. This executes representative pipelines
under many flag subsets and asserts the outputs agree — byte-identical for
bit-exact passes (per ``PassSpec.bit_exact``), structurally identical for the
one pass that trades interpolation passes for speed (affine fusion), which the
matrix-composition guards in ``test_affine_builder`` / ``test_schema_parity_
chains`` already pin at the pixel level.
"""

from __future__ import annotations

import io
import itertools

import numpy as np
import polars as pl
import pytest
from PIL import Image

from polars_cv import OptFlags, Pipeline, numpy_from_struct
from polars_cv._optimize import PASS_NAMES
from tests.conftest import plugin_required


@pytest.fixture
def sample_df() -> pl.DataFrame:
    """A single RGB test image with a distinct centre square."""
    img = Image.new("RGB", (96, 96), color=(40, 90, 160))
    for x in range(24, 72):
        for y in range(24, 72):
            img.putpixel((x, y), (230, 210, 80))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return pl.DataFrame({"image": [buf.getvalue()]})


def _all_flag_subsets() -> list[OptFlags]:
    """Every combination of pass on/off."""
    return [
        OptFlags(**dict(zip(PASS_NAMES, bits)))
        for bits in itertools.product([False, True], repeat=len(PASS_NAMES))
    ]


def _arr(df: pl.DataFrame, expr: pl.Expr) -> np.ndarray:
    return numpy_from_struct(df.select(out=expr)["out"][0])


@plugin_required
class TestDifferentialEquivalence:
    def test_every_flag_subset_is_identical_on_a_bit_exact_pipeline(
        self, sample_df: pl.DataFrame
    ) -> None:
        """A pipeline with no affine run: output is byte-identical under every
        flag subset. CSE and affine fusion are both no-ops here, so this pins
        the baseline 'optimization never changes a bit-exact result'."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(width=64, height=64)
            .grayscale()
            .threshold(128)
        )
        outputs = [
            _arr(sample_df, pl.col("image").cv.pipe(pipe).sink("numpy", opt_flags=f))
            for f in _all_flag_subsets()
        ]
        for other in outputs[1:]:
            assert np.array_equal(outputs[0], other)

    def test_cse_is_byte_identical(self, sample_df: pl.DataFrame) -> None:
        """Two pipelines sharing a source prefix: CSE shares the prefix node and
        must not change a single byte of either output."""
        gray_pipe = (
            Pipeline().source("image_bytes").resize(width=50, height=50).grayscale()
        )
        thresh_pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(width=50, height=50)
            .grayscale()
            .threshold(128)
        )

        def run(flags: OptFlags) -> tuple[np.ndarray, np.ndarray]:
            gray = pl.col("image").cv.pipe(gray_pipe).alias("gray")
            thresh = pl.col("image").cv.pipe(thresh_pipe).alias("thresh")
            out = sample_df.select(
                res=gray.merge_pipe(thresh).sink(
                    {"gray": "numpy", "thresh": "numpy"}, opt_flags=flags
                )
            )["res"]
            return (
                numpy_from_struct(out.struct.field("gray")[0]),
                numpy_from_struct(out.struct.field("thresh")[0]),
            )

        on_gray, on_thresh = run(OptFlags(common_subexpression_elimination=True))
        off_gray, off_thresh = run(OptFlags(common_subexpression_elimination=False))
        assert np.array_equal(on_gray, off_gray)
        assert np.array_equal(on_thresh, off_thresh)

    def test_affine_fusion_preserves_structure_and_changes_the_graph(
        self, sample_df: pl.DataFrame
    ) -> None:
        """Affine fusion is not bit-exact (one interpolation pass instead of
        two), so the guarantee here is structural: same shape and dtype, and the
        physical graph demonstrably changed. Pixel-level correctness of the
        composed matrix is pinned by the affine-builder / schema-parity tests."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(width=64, height=64)
            .rotate(30.0)
            .rotate(15.0)
        )
        fused = _arr(
            sample_df,
            pl.col("image").cv.pipe(pipe).sink("numpy", opt_flags=OptFlags.all()),
        )
        unfused = _arr(
            sample_df,
            pl.col("image").cv.pipe(pipe).sink("numpy", opt_flags=OptFlags.none()),
        )
        assert fused.shape == unfused.shape
        assert fused.dtype == unfused.dtype
        # The physical graph must differ even though the intent is the same.
        assert "warp_affine" in pipe.explain(optimized=True)
        assert pipe.explain(optimized=True) != pipe.explain(optimized=False)

    def test_env_default_drives_sink_when_flags_omitted(
        self, sample_df: pl.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no explicit opt_flags, ``.sink()`` reads POLARS_CV_OPTIMIZATIONS.
        An all-off env yields the same bit-exact output as explicit none()."""
        from polars_cv._optimize import OPT_ENV_VAR

        pipe = Pipeline().source("image_bytes").resize(width=40, height=40).grayscale()
        monkeypatch.setenv(OPT_ENV_VAR, "none")
        from_env = _arr(sample_df, pl.col("image").cv.pipe(pipe).sink("numpy"))
        explicit = _arr(
            sample_df,
            pl.col("image").cv.pipe(pipe).sink("numpy", opt_flags=OptFlags.none()),
        )
        assert np.array_equal(from_env, explicit)
