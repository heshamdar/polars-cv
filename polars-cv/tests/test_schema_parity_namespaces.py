"""Plan == exec for the standalone namespace accessors.

``.cv.width()``, ``.contour.area()``, ``.point.distance()``,
``.bbox.pairwise_iou()`` and the rest do **not** go through ``vb_graph``. They
are their own ``#[polars_expr]`` functions, each declaring its output type with
``output_type=`` or ``output_type_func=`` in ``src/contour.rs``, ``point.rs``,
``image_metadata.rs`` and ``read_bytes.rs`` — a second, entirely separate
plan/runtime pair from the pipeline graph's.

Nothing tested that pair. ``polars_cv/geometry/AGENTS.md`` says so directly:
there is no generated stub and no parity test for these namespaces, so
declared types can drift from what the functions return. This file is that
test.

The case table is completeness-asserted against the namespaces' real public
methods, so a new accessor cannot be added without either a case or an
explicit exemption.
"""

from __future__ import annotations

import polars as pl
import pytest

from polars_cv.expressions import CvNamespace
from polars_cv.geometry.bbox import BBoxNamespace
from polars_cv.geometry.contours import ContourNamespace
from polars_cv.geometry.points import PointNamespace
from polars_cv.geometry.schemas import (
    BBOX_SCHEMA,
    CONTOUR_SCHEMA,
    CONTOUR_SET_SCHEMA,
    POINT_SCHEMA,
)
from tests._schema_parity import assert_plan_equals_exec
from tests.conftest import make_image_png, make_rect_png, plugin_required

# ---------------------------------------------------------------------------
# Fixtures shaped like each namespace's input column
# ---------------------------------------------------------------------------


def _square(x0: float, y0: float, size: float) -> dict:
    ring = [
        {"x": x0, "y": y0},
        {"x": x0 + size, "y": y0},
        {"x": x0 + size, "y": y0 + size},
        {"x": x0, "y": y0 + size},
    ]
    return {"exterior": ring, "holes": [], "is_closed": True}


def _contour_df() -> pl.DataFrame:
    """Contours, points and bboxes side by side, with nulls interleaved.

    Nulls are placed first as well as later: these accessors parse struct
    columns row by row, and a leading null is where a parser that reaches for
    row 0 comes apart.
    """
    return pl.DataFrame(
        {
            "a": [None, _square(0, 0, 10), _square(2, 2, 6)],
            "b": [_square(1, 1, 8), None, _square(0, 0, 10)],
            "pa": [None, {"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}],
            "pb": [{"x": 5.0, "y": 6.0}, None, {"x": 7.0, "y": 8.0}],
            "ba": [
                None,
                {"x": 0.0, "y": 0.0, "width": 4.0, "height": 4.0},
                {"x": 1.0, "y": 1.0, "width": 2.0, "height": 2.0},
            ],
            "bb": [
                {"x": 0.0, "y": 0.0, "width": 3.0, "height": 3.0},
                None,
                {"x": 2.0, "y": 2.0, "width": 5.0, "height": 5.0},
            ],
            # pairwise_iou and match_detections operate on *sets* per row, so
            # they need list-valued columns rather than a single struct.
            "aset": [None, [_square(0, 0, 10), _square(4, 4, 4)], [_square(1, 1, 3)]],
            "bset": [[_square(0, 0, 8)], None, [_square(1, 1, 3), _square(5, 5, 2)]],
            "baset": [
                None,
                [{"x": 0.0, "y": 0.0, "width": 4.0, "height": 4.0}],
                [{"x": 1.0, "y": 1.0, "width": 2.0, "height": 2.0}],
            ],
            "bbset": [
                [{"x": 0.0, "y": 0.0, "width": 3.0, "height": 3.0}],
                None,
                [{"x": 2.0, "y": 2.0, "width": 5.0, "height": 5.0}],
            ],
        },
        schema={
            "a": CONTOUR_SCHEMA,
            "b": CONTOUR_SCHEMA,
            "pa": POINT_SCHEMA,
            "pb": POINT_SCHEMA,
            "ba": BBOX_SCHEMA,
            "bb": BBOX_SCHEMA,
            "aset": CONTOUR_SET_SCHEMA,
            "bset": CONTOUR_SET_SCHEMA,
            "baset": pl.List(BBOX_SCHEMA),
            "bbset": pl.List(BBOX_SCHEMA),
        },
    )


def _image_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "img": [
                None,
                make_image_png(6, 9, 3, seed=1),
                make_image_png(4, 4, 4, seed=2),
            ]
        },
        schema={"img": pl.Binary},
    )


# ---------------------------------------------------------------------------
# Case tables, completeness-asserted below
# ---------------------------------------------------------------------------

#: ``.cv`` metadata accessors (header-only reads, no decode).
CV_CASES: dict[str, object] = {
    "width": lambda: pl.col("img").cv.width(),
    "height": lambda: pl.col("img").cv.height(),
    "channels": lambda: pl.col("img").cv.channels(),
    "image_dtype": lambda: pl.col("img").cv.image_dtype(),
}

#: ``.cv`` members that are not metadata accessors.
CV_EXEMPT = {
    "pipe": "the graph entry point, covered by every other parity file",
    "read_bytes": "path-column reader, covered separately below",
}

CONTOUR_CASES: dict[str, object] = {
    "area": lambda: pl.col("a").contour.area(),
    "perimeter": lambda: pl.col("a").contour.perimeter(),
    "centroid": lambda: pl.col("a").contour.centroid(),
    "bounding_box": lambda: pl.col("a").contour.bounding_box(),
    "is_convex": lambda: pl.col("a").contour.is_convex(),
    "winding": lambda: pl.col("a").contour.winding(),
    "contains_point": lambda: pl.col("a").contour.contains_point(pl.col("pb")),
    "iou": lambda: pl.col("a").contour.iou(pl.col("b")),
    "dice": lambda: pl.col("a").contour.dice(pl.col("b")),
    "hausdorff_distance": lambda: pl.col("a").contour.hausdorff_distance(pl.col("b")),
    "translate": lambda: pl.col("a").contour.translate(1.0, 2.0),
    "scale": lambda: pl.col("a").contour.scale(2.0, 2.0),
    "simplify": lambda: pl.col("a").contour.simplify(0.5),
    "convex_hull": lambda: pl.col("a").contour.convex_hull(),
    "normalize": lambda: pl.col("a").contour.normalize(100, 100),
    "to_absolute": lambda: pl.col("a").contour.to_absolute(100, 100),
    "flip": lambda: pl.col("a").contour.flip(),
    "ensure_winding": lambda: pl.col("a").contour.ensure_winding("ccw"),
    "pairwise_iou": lambda: pl.col("aset").contour.pairwise_iou(pl.col("bset")),
    "match_detections": lambda: pl.col("aset").contour.match_detections(
        pl.col("bset"), threshold=0.5
    ),
}
CONTOUR_EXEMPT = {
    "on_null": "a policy setter, not an expression",
    "label_reduce": "needs a paired image column; covered in test_schema_parity_ops",
}

POINT_CASES: dict[str, object] = {
    "x": lambda: pl.col("pa").point.x(),
    "y": lambda: pl.col("pa").point.y(),
    "distance": lambda: pl.col("pa").point.distance(pl.col("pb")),
    "manhattan_distance": lambda: pl.col("pa").point.manhattan_distance(pl.col("pb")),
    "angle_to": lambda: pl.col("pa").point.angle_to(pl.col("pb")),
    "midpoint": lambda: pl.col("pa").point.midpoint(pl.col("pb")),
    "interpolate": lambda: pl.col("pa").point.interpolate(pl.col("pb"), 0.5),
    "translate": lambda: pl.col("pa").point.translate(1.0, 2.0),
    "scale": lambda: pl.col("pa").point.scale(2.0, 2.0),
    "rotate": lambda: pl.col("pa").point.rotate(45.0),
    "normalize": lambda: pl.col("pa").point.normalize(100, 100),
    "to_absolute": lambda: pl.col("pa").point.to_absolute(100, 100),
    "within_bbox": lambda: pl.col("pa").point.within_bbox(pl.col("ba")),
    "distance_to_contour": lambda: pl.col("pa").point.distance_to_contour(pl.col("a")),
    "signed_distance_to_contour": lambda: pl.col("pa").point.signed_distance_to_contour(
        pl.col("a")
    ),
    "nearest_point_on_contour": lambda: pl.col("pa").point.nearest_point_on_contour(
        pl.col("a")
    ),
}
POINT_EXEMPT = {"on_null": "a policy setter, not an expression"}

BBOX_CASES: dict[str, object] = {
    "pairwise_iou": lambda: pl.col("baset").bbox.pairwise_iou(pl.col("bbset")),
    "match_detections": lambda: pl.col("baset").bbox.match_detections(
        pl.col("bbset"), threshold=0.5
    ),
}
BBOX_EXEMPT = {"on_null": "a policy setter, not an expression"}


def _public_methods(cls: type) -> set[str]:
    return {m for m in dir(cls) if not m.startswith("_")}


@pytest.mark.parametrize(
    ("cls", "cases", "exempt", "label"),
    [
        (CvNamespace, CV_CASES, CV_EXEMPT, ".cv"),
        (ContourNamespace, CONTOUR_CASES, CONTOUR_EXEMPT, ".contour"),
        (PointNamespace, POINT_CASES, POINT_EXEMPT, ".point"),
        (BBoxNamespace, BBOX_CASES, BBOX_EXEMPT, ".bbox"),
    ],
)
def test_namespace_case_table_is_complete(cls, cases, exempt, label) -> None:
    """Every public accessor needs a parity case or an explicit exemption.

    These namespaces have no generated stub and no other parity test, so this
    completeness check is the only thing standing between a new accessor and
    zero coverage of its declared output type.
    """
    methods = _public_methods(cls)
    covered = set(cases) | set(exempt)
    missing = methods - covered
    stale = covered - methods
    assert not missing, f"{label} accessors with no parity case: {sorted(missing)}"
    assert not stale, (
        f"{label} cases for accessors that no longer exist: {sorted(stale)}"
    )


# ---------------------------------------------------------------------------
# The parity sweeps
# ---------------------------------------------------------------------------


@plugin_required
@pytest.mark.parametrize("name", sorted(CV_CASES))
def test_cv_metadata_accessors(name: str) -> None:
    """Header-only metadata reads declare UInt32/String; execution must agree."""
    assert_plan_equals_exec(_image_df(), CV_CASES[name]())


@plugin_required
@pytest.mark.parametrize("name", sorted(CONTOUR_CASES))
def test_contour_accessors(name: str) -> None:
    assert_plan_equals_exec(_contour_df(), CONTOUR_CASES[name]())


@plugin_required
@pytest.mark.parametrize("name", sorted(POINT_CASES))
def test_point_accessors(name: str) -> None:
    assert_plan_equals_exec(_contour_df(), POINT_CASES[name]())


@plugin_required
@pytest.mark.parametrize("name", sorted(BBOX_CASES))
def test_bbox_accessors(name: str) -> None:
    assert_plan_equals_exec(_contour_df(), BBOX_CASES[name]())


@plugin_required
def test_read_bytes_declares_binary(tmp_path) -> None:
    """``.cv.read_bytes()`` is a path column in, Binary out."""
    paths = []
    for i in range(2):
        p = tmp_path / f"f{i}.png"
        p.write_bytes(make_rect_png(8, 8, 3))
        paths.append(str(p))

    df = pl.DataFrame({"p": [None, *paths]}, schema={"p": pl.String})
    series = assert_plan_equals_exec(df, pl.col("p").cv.read_bytes())
    assert series.dtype == pl.Binary


@plugin_required
def test_null_first_is_not_special_for_the_accessors() -> None:
    """The same accessor over a leading null and a leading value must agree.

    These functions parse a struct column row by row, and three separate
    point-struct parsers exist in the Rust plugin with different missing-field
    behaviour. A leading null is where those differ soonest.
    """
    good = _square(0, 0, 10)
    leading_null = pl.DataFrame({"a": [None, good, good]}, schema={"a": CONTOUR_SCHEMA})
    leading_value = pl.DataFrame(
        {"a": [good, None, good]}, schema={"a": CONTOUR_SCHEMA}
    )

    for expr_name, build in (
        ("area", lambda: pl.col("a").contour.area()),
        ("centroid", lambda: pl.col("a").contour.centroid()),
        ("bounding_box", lambda: pl.col("a").contour.bounding_box()),
    ):
        first = assert_plan_equals_exec(leading_null, build(), name="o").dtype
        second = assert_plan_equals_exec(leading_value, build(), name="o").dtype
        assert first == second, (
            f"{expr_name}: leading-null column planned {first}, "
            f"leading-value column planned {second}"
        )
