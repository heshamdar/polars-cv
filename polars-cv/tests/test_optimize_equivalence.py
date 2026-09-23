"""Differential-equivalence guard for the optimization phase.

The core guarantee: toggling optimization passes changes only the *physical*
graph, never the result the user sees. This executes representative pipelines
under a representative set of flag combinations and asserts the outputs agree
byte-for-byte. Every current optimization is byte-exact (``PassSpec.bit_exact``
is ``True`` for all of them); ``TestEveryOptimizationOnOffEquivalence`` pins a
dedicated on/off differential for each registered pass so none can be added
without one.
"""

from __future__ import annotations

import io

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
    """A representative set of flag combinations.

    All-off and all-on, plus each pass individually toggled from both baselines
    (one-on-from-none and one-off-from-all). This is linear in the number of
    passes rather than the 2^N full power set — which, with both tiers now in the
    registry, would be hundreds of combinations per case — while still exercising
    every pass on its own and against the fully-optimized graph.
    """
    subsets = [OptFlags.none(), OptFlags.all()]
    for name in PASS_NAMES:
        all_on = {n: True for n in PASS_NAMES}
        all_off = {n: False for n in PASS_NAMES}
        subsets.append(OptFlags(**{**all_on, name: False}))
        subsets.append(OptFlags(**{**all_off, name: True}))
    return subsets


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


# One representative pipeline per op family, each **bit-exact**: a single
# pipeline has no CSE sibling, so toggling any pass must leave the output
# byte-for-byte identical. A future pass that reordered or dropped one of these
# ops would break the invariant here. `fmt` is the op's natural sink (buffer ops
# → numpy; scalar/vector ops → native).
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

    Each pipeline is bit-exact (no CSE sibling), so toggling any flag combination
    must not change a single byte. This is the breadth guard: a future pass that
    reordered or dropped one of these ops would fail here. (It cannot assert a
    pass *fired* — none does on a lone bit-exact pipeline; the fired guards live
    in the CSE and per-optimization tests.)
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
    CSE flag is toggled; every other pass is held on, so the comparison isolates
    CSE.
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

    def test_cse_with_other_passes_on(self, sample_df: pl.DataFrame) -> None:
        # Shared prefix, with all other passes held on: toggling only CSE must
        # stay byte-identical, and CSE must still fire.
        base = Pipeline().source("image_bytes").resize(width=64, height=64).grayscale()
        pipes = {"a": base, "b": base.threshold(120)}
        self._assert_equivalent_and_fired(sample_df, pipes)

    def _assert_equivalent_and_fired(
        self,
        df: pl.DataFrame,
        pipes: dict[str, Pipeline],
    ) -> None:
        on = OptFlags(common_subexpression_elimination=True)
        off = OptFlags(common_subexpression_elimination=False)
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

    Multiple rows, a null row (which rides the node-skip path the passes must
    preserve), and a non-image source. All cases use bit-exact pipelines so every
    flag combination must be byte-identical.
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
        assert outputs[0][1] is None
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
            assert out_on[alias][1] is None  # null row preserved

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


# Pipelines where a crop follows a run of Pointwise ops — the pushdown moves the
# crop to the front of the run. Each is bit-exact (Pointwise commutes with a
# crop exactly), so every flag subset must be byte-identical.
_POINTWISE_CROP_CASES: list[tuple[str, object]] = [
    ("grayscale", lambda p: p.grayscale().crop(top=1, left=1, height=8, width=8)),
    (
        "cast_scale_invert",
        lambda p: (
            p.cast("f32").scale(0.5).invert().crop(top=2, left=2, height=8, width=8)
        ),
    ),
    (
        "convert_color",
        lambda p: p.convert_color("rgb", "hsv").crop(top=0, left=0, height=8, width=8),
    ),
    (
        "cast_clamp",
        lambda p: (
            p.cast("f32").clamp(0.0, 200.0).crop(top=1, left=0, height=8, width=8)
        ),
    ),
    (
        "adjust_gamma",
        lambda p: p.adjust_gamma(gamma=2.2).crop(top=0, left=1, height=8, width=8),
    ),
]


@plugin_required
class TestSpatialPushdownEquivalence:
    """Hoisting a crop past a Pointwise run is byte-exact — and actually fires.

    Pointwise means output at (y, x) depends only on input at (y, x), so a crop
    commutes to the front of the run with no change to a single byte. As with the
    op-family guard this cannot assert *fired* from the output alone, so
    ``test_pass_fires`` checks the physical op order changed.
    """

    @pytest.mark.parametrize(
        ("case_id", "build"),
        _POINTWISE_CROP_CASES,
        ids=[c[0] for c in _POINTWISE_CROP_CASES],
    )
    def test_identical_across_all_flag_subsets(
        self, sample_df: pl.DataFrame, case_id: str, build: object
    ) -> None:
        pipe = build(_src())  # type: ignore[operator]
        outputs = [
            _sink_output(sample_df, pipe, f, "numpy") for f in _all_flag_subsets()
        ]
        assert outputs[0] and outputs[0][0] is not None, (
            f"{case_id}: produced an empty/null output"
        )
        for other in outputs[1:]:
            assert other == outputs[0], f"{case_id}: output changed under a flag subset"

    def test_pass_fires(self, sample_df: pl.DataFrame) -> None:
        # With the pass on the crop leads the node; with it off it trails.
        pipe = _src().grayscale().crop(top=1, left=1, height=8, width=8)

        def ops(flag: bool) -> list[str]:
            graph = (
                pl.col("image")
                .cv.pipe(pipe)
                .sink(
                    "numpy",
                    return_expr=False,
                    opt_flags=OptFlags(spatial_window_pushdown=flag),
                )
            )
            (node,) = graph._nodes.values()
            return [op.op for op in node.pipeline._ops]

        assert ops(True) == ["crop", "grayscale"]
        assert ops(False) == ["grayscale", "crop"]

    def test_per_row_crop_param_is_identical(self, sample_df: pl.DataFrame) -> None:
        # A per-row crop offset still commutes past a Pointwise op (Pointwise
        # commutes with any window) and stays byte-identical across flag subsets.
        df = sample_df.with_columns(t=pl.lit(3))
        pipe = _src().grayscale().crop(top=pl.col("t"), left=0, height=8, width=8)
        outputs = [_sink_output(df, pipe, f, "numpy") for f in _all_flag_subsets()]
        assert outputs[0] and outputs[0][0] is not None
        for other in outputs[1:]:
            assert other == outputs[0]


@plugin_required
class TestDifferentialEquivalence:
    def test_every_flag_subset_is_identical_on_a_bit_exact_pipeline(
        self, sample_df: pl.DataFrame
    ) -> None:
        """A single bit-exact pipeline: output is byte-identical under every
        flag combination, pinning the baseline 'optimization never changes a
        bit-exact result'."""
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


# Pipelines whose identity op the pass actually deletes, each byte-exact: the
# deleted op is a runtime no-op, so removing it cannot change a single output
# byte under any flag subset. This is the correctness guard for
# identity_elimination — and, for the WhenShapePreserved arm, the *only* guard
# (a wrongly-tagged flip/transpose would be deleted here and the bytes would
# diverge, failing this test).
_IDENTITY_ELIMINATION_CASES: list[tuple[str, object]] = [
    ("zero_pad", lambda p: p.pad(top=0, bottom=0, left=0, right=0).grayscale()),
    (
        "full_frame_crop",
        lambda p: p.resize(height=32, width=32).crop(
            top=0, left=0, height=32, width=32
        ),
    ),
    ("redundant_cast", lambda p: p.cast("u8").cast("u8")),
    (
        "same_shape_reshape",
        lambda p: p.resize(height=32, width=32).reshape([32, 32, 3]),
    ),
    (
        "pad_to_size_same",
        lambda p: p.resize(height=32, width=32).pad_to_size(height=32, width=32),
    ),
]


@plugin_required
class TestIdentityEliminationByteExact:
    """Deleting a no-op op changes no output byte, under every flag subset."""

    @pytest.mark.parametrize(
        ("case_id", "build"),
        _IDENTITY_ELIMINATION_CASES,
        ids=[c[0] for c in _IDENTITY_ELIMINATION_CASES],
    )
    def test_identity_elimination_is_byte_exact(
        self, sample_df: pl.DataFrame, case_id: str, build: object
    ) -> None:
        pipe = build(_src())  # type: ignore[operator]
        outputs = [
            _sink_output(sample_df, pipe, f, "numpy") for f in _all_flag_subsets()
        ]
        assert outputs[0] and outputs[0][0] is not None, (
            f"{case_id}: produced an empty/null output"
        )
        for other in outputs[1:]:
            assert other == outputs[0], f"{case_id}: output changed under a flag subset"


# One representative pipeline per optimization that toggling it exercises. Each
# is byte-exact, so enabling the pass (all others off) must not change a byte of
# the output. CSE needs sibling pipelines, so it is covered by
# ``TestCseEquivalence`` and mapped in the coverage guard below rather than here.
_PER_OPT_CASES: list[tuple[str, object, str]] = [
    (
        "identity_elimination",
        lambda p: p.resize(height=32, width=32).crop(
            top=0, left=0, height=32, width=32
        ),
        "numpy",
    ),
    (
        "spatial_window_pushdown",
        lambda p: p.grayscale().crop(top=1, left=1, height=16, width=16),
        "numpy",
    ),
    ("cast_chain_collapse", lambda p: p.cast("u16").cast("f32"), "numpy"),
    ("cast_identity", lambda p: p.cast("u8"), "numpy"),
    (
        "view_flip_involution",
        lambda p: p.flip(axes=[0]).flip(axes=[0]),
        "numpy",
    ),
    (
        "view_transpose_merge",
        lambda p: p.transpose(axes=[1, 0, 2]).transpose(axes=[1, 0, 2]),
        "numpy",
    ),
    ("scalar_fusion", lambda p: p.cast("f32").scale(0.5).invert(), "numpy"),
]

#: The one pass not expressible as a single-pipeline case; covered elsewhere.
_DIFFERENTIAL_COVERED_ELSEWHERE = {"common_subexpression_elimination"}


def test_every_optimization_has_an_on_off_differential() -> None:
    """Every registered pass has a dedicated on/off differential test.

    Pins the mandate that no optimization ships without a test proving it does
    not change output — a new pass with no case (here or, for CSE, in
    ``TestCseEquivalence``) fails this guard. Needs no plugin.
    """
    covered = {c[0] for c in _PER_OPT_CASES} | _DIFFERENTIAL_COVERED_ELSEWHERE
    assert covered == set(PASS_NAMES), (
        "optimizations without an on/off differential: "
        f"{sorted(set(PASS_NAMES) - covered)}"
    )


@plugin_required
class TestEveryOptimizationOnOffEquivalence:
    """Toggling any single optimization on preserves the output byte-for-byte.

    Baseline is all-off; enabling just the one pass must match. This is the
    literal 'output with the optimization is the same as without' guarantee, one
    parametrization per registered (single-pipeline) pass.
    """

    @pytest.mark.parametrize(
        ("name", "build", "fmt"),
        _PER_OPT_CASES,
        ids=[c[0] for c in _PER_OPT_CASES],
    )
    def test_toggling_one_optimization_preserves_output(
        self, sample_df: pl.DataFrame, name: str, build: object, fmt: str
    ) -> None:
        pipe_off = build(_src())  # type: ignore[operator]
        pipe_on = build(_src())  # type: ignore[operator]
        off = _sink_output(sample_df, pipe_off, OptFlags.none(), fmt)
        on = _sink_output(sample_df, pipe_on, OptFlags(**{name: True}), fmt)
        assert off and off[0] is not None, f"{name}: produced an empty/null output"
        assert on == off, f"{name}: output changed when the optimization was enabled"


# Regressions: inputs where an optimization once changed the result. Each case is
# asserted at the user-facing entry point (`.sink()`) across every flag subset,
# so a fix that only patched a helper would still fail here.

#: (source dtype, values) that an int -> float -> int cast chain maps differently
#: from a direct int -> int cast: the float path saturates, the int path wraps.
_INT_THROUGH_FLOAT_CHAINS: list[tuple[str, "pl.DataType", list[int]]] = [
    ("u16", pl.UInt16, [300, 1000, 40000, 7]),
    ("i16", pl.Int16, [-5, 300, -300, 7]),
]


@plugin_required
class TestOptimizationRegressions:
    """Inputs that an optimization once silently changed the output for."""

    @pytest.mark.parametrize(
        ("dtype", "pl_dtype", "values"),
        _INT_THROUGH_FLOAT_CHAINS,
        ids=[c[0] for c in _INT_THROUGH_FLOAT_CHAINS],
    )
    def test_int_through_float_cast_chain_saturates(
        self, dtype: str, pl_dtype: "pl.DataType", values: list[int]
    ) -> None:
        # f32 holds every u16/i16 exactly, so the intermediate cast is lossless —
        # but dropping it turns the saturating float -> u8 conversion into a
        # wrapping int -> u8 one (300 -> 44 instead of 255).
        df = pl.DataFrame({"a": [values]}, schema={"a": pl.List(pl_dtype)})
        pipe = Pipeline().source("list", dtype=dtype).cast("f32").cast("u8")
        outputs = [_sink_output(df, pipe, f, "list", "a") for f in _all_flag_subsets()]
        assert outputs[0] == [[min(max(v, 0), 255) for v in values]]
        for other in outputs[1:]:
            assert other == outputs[0], "a cast-chain optimization changed the output"

    def test_offset_crop_with_full_extent_is_not_eliminated(
        self, sample_df: pl.DataFrame
    ) -> None:
        # A crop whose extent equals the input's but whose origin is not (0, 0)
        # preserves the *planned* shape while running past the edge; the engine
        # clamps it to a smaller window, so it is not a no-op.
        pipe = (
            _src().resize(height=20, width=20).crop(top=5, left=5, height=20, width=20)
        )
        outputs = [
            _sink_output(sample_df, pipe, f, "numpy") for f in _all_flag_subsets()
        ]
        for other in outputs[1:]:
            assert other == outputs[0], "identity elimination deleted an offset crop"

    def test_declared_shape_reaching_a_cse_suffix_is_not_trusted(
        self, sample_df: pl.DataFrame
    ) -> None:
        # CSE moves the assert_shape into the shared prefix node; the suffix keeps
        # the H/W it implied but not the assertion. That H/W is a declaration, not
        # a fact, so it must not license deleting the crop as "full-frame".
        pipes = {
            "a": _src()
            .assert_shape(height=10, width=10)
            .grayscale()
            .crop(top=0, left=0, height=10, width=10),
            "b": _src().grayscale().threshold(128),
        }
        off = _run_multi(sample_df, pipes, OptFlags.none())
        for flags in _all_flag_subsets():
            assert _run_multi(sample_df, pipes, flags) == off, (
                f"output changed under {flags}"
            )

    def test_declared_shape_reaching_a_lazy_continuation_is_not_trusted(
        self, sample_df: pl.DataFrame
    ) -> None:
        # The continuation inherits the upstream node's asserted H/W as a hint but
        # none of its assertions — the same declaration-as-fact hole as CSE.
        upstream = pl.col("image").cv.pipe(_src().assert_shape(height=10, width=10))
        cont = upstream.pipe(
            Pipeline().grayscale().crop(top=0, left=0, height=10, width=10)
        )
        off = sample_df.select(o=cont.sink("numpy", opt_flags=OptFlags.none()))
        for flags in _all_flag_subsets():
            on = sample_df.select(o=cont.sink("numpy", opt_flags=flags))
            assert on["o"].to_list() == off["o"].to_list(), (
                f"output changed under {flags}"
            )
