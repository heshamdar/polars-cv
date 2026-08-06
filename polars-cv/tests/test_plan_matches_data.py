"""Plan-time schema must equal execution-time schema (the Polars invariant).

For every source × op × sink combination the schema Polars reports at planning
time (`lf.collect_schema()`) must exactly equal the schema of the collected data
(`lf.collect().schema`). This is the guard that was missing when the planner
guessed source rank (raw/list/array → 3) while the sink followed the decoded
shape — a divergence that stayed silent because no test compared the two.

The Rust `validate_output_schema` guard (graph/compiled.rs) enforces the same
invariant per row at execution time; this file pins it from the Python side
across representative shapes.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from polars_cv import Pipeline
from tests._schema_parity import assert_plan_equals_exec
from tests.conftest import plugin_required


def _assert_plan_matches_data(df: pl.DataFrame, expr: pl.Expr) -> pl.DataType:
    """Assert the planned and executed dtypes for `out` are identical.

    A thin adapter over the shared harness (``tests/_schema_parity.py``), which
    also runs the streaming engine and checks the two engines agree.
    """
    return assert_plan_equals_exec(df, expr).dtype


@plugin_required
class TestPlanMatchesDataSources:
    """Every source's declared rank/dtype must survive to execution unchanged."""

    def test_raw_1d_list_sink(self) -> None:
        # raw decodes flat 1-D: planned rank 1 must equal produced rank 1.
        data = np.array([1.0, 2.5, 100.0], dtype=np.float32)
        df = pl.DataFrame({"x": [data.tobytes()]})
        pipe = Pipeline().source("raw", dtype="f32")
        out = _assert_plan_matches_data(df, pl.col("x").cv.pipe(pipe).sink("list"))
        assert out == pl.List(pl.Float32)

    def test_raw_reshaped_2d_list_sink(self) -> None:
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        df = pl.DataFrame({"x": [data.tobytes()]})
        pipe = Pipeline().source("raw", dtype="f32").reshape([2, 2])
        out = _assert_plan_matches_data(df, pl.col("x").cv.pipe(pipe).sink("list"))
        assert out == pl.List(pl.List(pl.Float32))

    def test_list_2d_dtype_list_sink(self) -> None:
        # list source WITH explicit dtype: rank must come from the column's true
        # nesting depth (2), not the old guessed 3.
        df = pl.DataFrame({"x": [[[1, 2, 3], [4, 5, 6]]]})
        pipe = Pipeline().source("list", dtype="u8")
        out = _assert_plan_matches_data(df, pl.col("x").cv.pipe(pipe).sink("list"))
        assert out == pl.List(pl.List(pl.UInt8))

    def test_list_2d_no_dtype_list_sink(self) -> None:
        df = pl.DataFrame({"x": [[[1.0, 2.0], [3.0, 4.0]]]})
        pipe = Pipeline().source("list")
        _assert_plan_matches_data(df, pl.col("x").cv.pipe(pipe).sink("list"))

    def test_array_2d_dtype_list_sink(self) -> None:
        df = pl.DataFrame({"x": [[[1, 2], [3, 4]]]}).cast(
            {"x": pl.Array(pl.UInt8, (2, 2))}
        )
        pipe = Pipeline().source("array", dtype="u8")
        out = _assert_plan_matches_data(df, pl.col("x").cv.pipe(pipe).sink("list"))
        assert out == pl.List(pl.List(pl.UInt8))

    def test_image_grayscale_numpy_sink(self, create_test_png) -> None:
        df = pl.DataFrame({"x": [create_test_png(8, 8)]})
        pipe = Pipeline().source("image_bytes").grayscale()
        _assert_plan_matches_data(df, pl.col("x").cv.pipe(pipe).sink("numpy"))

    def test_image_reshape_2d_list_sink(self, create_test_png) -> None:
        # Image is rank-3 known; reshape to rank 2 must be reflected in the plan.
        img = create_test_png(4, 4)
        pipe = Pipeline().source("image_bytes", dtype="u8").grayscale().reshape([4, 4])
        df = pl.DataFrame({"x": [img]})
        out = _assert_plan_matches_data(df, pl.col("x").cv.pipe(pipe).sink("list"))
        assert out == pl.List(pl.List(pl.UInt8))


@plugin_required
class TestOpInferShapeAuthority:
    """op_infer_shape is the single geometry authority: literal params/known
    dims give exact output dims; expression params/unknown dims propagate to
    None. Probing includes 90-degree multiples so rotate's discontinuous fast
    path (90/180/270 swap H/W) is detected."""

    @staticmethod
    def _op_json(pipe: Pipeline) -> str:
        return json.dumps(pipe._ops[-1].to_dict())

    def test_resize_literal_dims_are_known(self) -> None:
        from polars_cv._lib import op_infer_shape

        j = self._op_json(
            Pipeline().source("image_bytes").resize(height=224, width=100)
        )
        assert op_infer_shape(j, [None, None, None]) == [224, 100, None]

    def test_resize_expression_dim_is_unknown(self) -> None:
        from polars_cv._lib import op_infer_shape

        j = self._op_json(
            Pipeline().source("image_bytes").resize(height=pl.col("h"), width=100)
        )
        assert op_infer_shape(j, [None, None, 3]) == [None, 100, 3]

    def test_pad_adds_known_borders(self) -> None:
        from polars_cv._lib import op_infer_shape

        j = self._op_json(
            Pipeline().source("image_bytes").pad(top=1, bottom=2, left=3, right=4)
        )
        assert op_infer_shape(j, [10, 10, 3]) == [13, 17, 3]

    def test_literal_90_rotate_swaps_hw(self) -> None:
        from polars_cv._lib import op_infer_shape

        j = self._op_json(Pipeline().source("image_bytes").rotate(90))
        assert op_infer_shape(j, [100, 50, 3]) == [50, 100, 3]

    def test_expression_angle_rotate_is_unknown(self) -> None:
        # A per-row angle could be a 90-multiple (H/W swap) at runtime, so the
        # spatial dims are unknown at plan time even on a known image.
        from polars_cv._lib import op_infer_shape

        j = self._op_json(Pipeline().source("image_bytes").rotate(pl.col("a")))
        assert op_infer_shape(j, [100, 50, 3]) == [None, None, 3]
