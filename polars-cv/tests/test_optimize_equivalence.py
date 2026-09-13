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


def _sink_output(
    df: pl.DataFrame, pipe: Pipeline, flags: OptFlags, fmt: str, column: str = "image"
) -> list:
    """Materialize a pipeline's output as a plain (comparable) Python list.

    Works for any sink format — for ``numpy`` the element is the zero-copy
    struct, whose ``==`` is exact — so a single helper compares buffer, scalar,
    and vector outputs for byte-identity across flag settings.
    """
    expr = pl.col(column).cv.pipe(pipe).sink(fmt, opt_flags=flags)
    return df.select(out=expr)["out"].to_list()


def _src() -> Pipeline:
    return Pipeline().source("image_bytes")


# One representative pipeline per op family, each **bit-exact** under both passes:
# none contains an adjacent affine run (a lone rotate is a run-of-one, so it does
# not fuse), and a single pipeline has no CSE sibling — so toggling any pass must
# leave the output byte-for-byte identical. A future pass that reordered or
# dropped one of these ops would break the invariant here. `fmt` is the op's
# natural sink (buffer ops → numpy; scalar/vector ops → native).
_OP_FAMILY_CASES: list[tuple[str, object, str]] = [
    ("resize", lambda p: p.resize(height=32, width=32), "numpy"),
    ("grayscale", lambda p: p.grayscale(), "numpy"),
    ("threshold", lambda p: p.grayscale().threshold(128), "numpy"),
    ("blur", lambda p: p.blur(1.0), "numpy"),
    ("convolve2d", lambda p: p.convolve2d([1.0 / 9] * 9, 3), "numpy"),
    ("convert_color", lambda p: p.convert_color("rgb", "hsv"), "numpy"),
    ("cast", lambda p: p.cast("f32"), "numpy"),
    ("scale", lambda p: p.cast("f32").scale(0.5), "numpy"),
    ("clamp", lambda p: p.cast("f32").clamp(0.0, 128.0), "numpy"),
    ("invert", lambda p: p.invert(), "numpy"),
    ("adjust_gamma", lambda p: p.adjust_gamma(gamma=2.2), "numpy"),
    ("adjust_contrast", lambda p: p.adjust_contrast(factor=1.5), "numpy"),
    ("pad", lambda p: p.pad(top=1, bottom=2, left=3, right=4), "numpy"),
    ("crop", lambda p: p.crop(top=0, left=0, height=8, width=8), "numpy"),
    ("flip", lambda p: p.flip(axes=[0]), "numpy"),
    ("transpose", lambda p: p.transpose(axes=[1, 0, 2]), "numpy"),
    ("erode", lambda p: p.grayscale().erode(ksize=3, iterations=1), "numpy"),
    ("dilate", lambda p: p.grayscale().dilate(ksize=3, iterations=1), "numpy"),
    ("canny", lambda p: p.grayscale().canny(), "numpy"),
    ("rotate_lone", lambda p: p.rotate(30.0), "numpy"),
    ("reduce_sum", lambda p: p.reduce_sum(), "native"),
    ("histogram", lambda p: p.grayscale().histogram(bins=8), "native"),
    ("perceptual_hash", lambda p: p.perceptual_hash(), "native"),
]


@plugin_required
class TestOpFamilyByteExact:
    """Optimization preserves every op family's output, byte for byte.

    Each pipeline is bit-exact under both passes (no CSE sibling, no adjacent
    affine run), so toggling any flag subset must not change a single byte. This
    is the breadth guard: a future pass that reordered or dropped one of these
    ops would fail here. (It cannot assert a pass *fired* — none does on a lone
    bit-exact pipeline; the fired guards live in the CSE and affine tests.)
    """

    @pytest.mark.parametrize(
        ("case_id", "build", "fmt"),
        _OP_FAMILY_CASES,
        ids=[c[0] for c in _OP_FAMILY_CASES],
    )
    def test_op_family_is_identical_across_all_flag_subsets(
        self, sample_df: pl.DataFrame, case_id: str, build: object, fmt: str
    ) -> None:
        pipe = build(_src())  # type: ignore[operator]
        outputs = [_sink_output(sample_df, pipe, f, fmt) for f in _all_flag_subsets()]
        # Vacuity guard: a real, non-empty result (so we are not trivially
        # comparing None == None across flags).
        assert outputs[0] and outputs[0][0] is not None, (
            f"{case_id}: produced an empty/null output"
        )
        for other in outputs[1:]:
            assert other == outputs[0], f"{case_id}: output changed under a flag subset"


def _total_ops(graph) -> int:  # type: ignore[no-untyped-def]
    """Total ops across every node of a built graph.

    CSE moves a shared prefix into one upstream node instead of repeating it in
    each sibling, so a graph where CSE fired carries strictly fewer total ops
    than the same graph with CSE off. This is the 'the pass actually fired'
    signal for the CSE tests.
    """
    return sum(len(n.pipeline._ops) for n in graph._nodes.values())


def _run_multi(
    df: pl.DataFrame, pipes_by_alias: dict[str, Pipeline], flags: OptFlags
) -> dict[str, list]:
    """Sink several aliased sibling pipelines (all reading ``image``) at once."""
    aliases = list(pipes_by_alias)
    exprs = [pl.col("image").cv.pipe(pipes_by_alias[a]).alias(a) for a in aliases]
    first, *rest = exprs
    out = df.select(
        res=first.merge_pipe(*rest).sink({a: "numpy" for a in aliases}, opt_flags=flags)
    )["res"]
    return {a: out.struct.field(a).to_list() for a in aliases}


def _built_graph(pipes_by_alias: dict[str, Pipeline], flags: OptFlags):  # type: ignore[no-untyped-def]
    """The optimized PipelineGraph for a set of aliased siblings (no execution)."""
    aliases = list(pipes_by_alias)
    exprs = [pl.col("image").cv.pipe(pipes_by_alias[a]).alias(a) for a in aliases]
    first, *rest = exprs
    return first.merge_pipe(*rest).sink(
        {a: "numpy" for a in aliases}, return_expr=False, opt_flags=flags
    )


@plugin_required
class TestCseEquivalence:
    """CSE is bit-exact: sharing a prefix must not change a byte of any output.

    Each case also asserts CSE *fired* (fewer total ops with CSE on than off), so
    a disconnected optimizer fails here rather than passing vacuously. Only the
    CSE flag is toggled; affine fusion is held on, so any affine run behaves the
    same on both sides and the comparison isolates CSE.
    """

    def _shared(self) -> dict[str, Pipeline]:
        base = Pipeline().source("image_bytes").resize(width=50, height=50).grayscale()
        return {
            "a": base,
            "b": base.threshold(128),
            "c": base.invert(),
        }

    def test_two_siblings(self, sample_df: pl.DataFrame) -> None:
        pipes = {"a": self._shared()["a"], "b": self._shared()["b"]}
        self._assert_equivalent_and_fired(sample_df, pipes)

    def test_three_siblings(self, sample_df: pl.DataFrame) -> None:
        self._assert_equivalent_and_fired(sample_df, self._shared())

    def test_per_row_expression_param_in_shared_prefix(
        self, sample_df: pl.DataFrame
    ) -> None:
        # A per-row param (pl.col) in the shared prefix must still CSE and stay
        # byte-identical.
        df = sample_df.with_columns(h=pl.lit(48))
        base = (
            Pipeline()
            .source("image_bytes")
            .resize(width=48, height=pl.col("h"))
            .grayscale()
        )
        pipes = {"a": base, "b": base.threshold(100)}
        self._assert_equivalent_and_fired(df, pipes)

    def test_cse_and_affine_run_together(self, sample_df: pl.DataFrame) -> None:
        # Shared prefix ends in an affine run: both passes act. Toggling only CSE
        # (affine held on) must stay byte-identical, and CSE must still fire.
        base = (
            Pipeline()
            .source("image_bytes")
            .resize(width=64, height=64)
            .rotate(30.0)
            .rotate(15.0)
        )
        pipes = {"a": base, "b": base.grayscale()}
        self._assert_equivalent_and_fired(sample_df, pipes, affine=True)

    def _assert_equivalent_and_fired(
        self,
        df: pl.DataFrame,
        pipes: dict[str, Pipeline],
        *,
        affine: bool = True,
    ) -> None:
        on = OptFlags(common_subexpression_elimination=True, affine_fusion=affine)
        off = OptFlags(common_subexpression_elimination=False, affine_fusion=affine)
        out_on = _run_multi(df, pipes, on)
        out_off = _run_multi(df, pipes, off)
        for alias in pipes:
            assert out_on[alias] == out_off[alias], f"CSE changed the '{alias}' output"
        # Fired: CSE collapsed the shared prefix into one node.
        assert _total_ops(_built_graph(pipes, on)) < _total_ops(
            _built_graph(pipes, off)
        ), "CSE did not fire — no shared prefix was extracted"


def _png(color: tuple[int, int, int], size: int = 64) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), color=color).save(buf, format="PNG")
    return buf.getvalue()


@plugin_required
class TestSourceRowNullVariety:
    """Equivalence holds beyond a single image_bytes row.

    Multiple rows, a null row (which rides the node-skip path CSE and fusion must
    preserve), and a non-image source. All cases use bit-exact pipelines so every
    flag subset must be byte-identical.
    """

    def test_multiple_rows(self) -> None:
        df = pl.DataFrame(
            {"image": [_png((40, 90, 160)), _png((10, 200, 30)), _png((250, 5, 90))]}
        )
        pipe = _src().resize(width=32, height=32).grayscale().threshold(128)
        outputs = [_sink_output(df, pipe, f, "numpy") for f in _all_flag_subsets()]
        assert len(outputs[0]) == 3
        for other in outputs[1:]:
            assert other == outputs[0]

    def test_null_row_is_preserved_across_flag_subsets(self) -> None:
        df = pl.DataFrame({"image": [_png((40, 90, 160)), None, _png((250, 5, 90))]})
        pipe = _src().resize(width=32, height=32).grayscale()
        outputs = [_sink_output(df, pipe, f, "numpy") for f in _all_flag_subsets()]
        # The null input row stays null in the output (numpy sink represents a
        # null row as a struct with no buffer), under every flag subset.
        assert outputs[0][1]["data"] is None
        for other in outputs[1:]:
            assert other == outputs[0]

    def test_null_row_with_cse(self) -> None:
        # Nulls ride the same node-skip path CSE reuses; a shared prefix over a
        # null row must stay byte-identical whether or not CSE fires.
        df = pl.DataFrame({"image": [_png((40, 90, 160)), None, _png((7, 7, 7))]})
        base = _src().resize(width=40, height=40).grayscale()
        pipes = {"a": base, "b": base.threshold(120)}
        on = OptFlags(common_subexpression_elimination=True)
        off = OptFlags(common_subexpression_elimination=False)
        out_on, out_off = _run_multi(df, pipes, on), _run_multi(df, pipes, off)
        for alias in pipes:
            assert out_on[alias] == out_off[alias]
            assert out_on[alias][1]["data"] is None  # null row preserved

    def test_non_image_list_source(self) -> None:
        img = [[[float(r * 4 + c)] for c in range(4)] for r in range(4)]
        df = pl.DataFrame(
            {"x": [img, img]},
            schema={"x": pl.List(pl.List(pl.List(pl.Float64)))},
        )
        pipe = Pipeline().source("list", dtype="f32").scale(0.5)
        outputs = [
            _sink_output(df, pipe, f, "numpy", column="x") for f in _all_flag_subsets()
        ]
        assert outputs[0] and outputs[0][0] is not None
        for other in outputs[1:]:
            assert other == outputs[0]


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
