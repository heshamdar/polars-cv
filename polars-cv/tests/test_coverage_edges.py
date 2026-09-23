"""Edge and error branches of the pure-Python surface.

These paths are reachable from the public API but not exercised by the
behavioural suite, which drives the happy path through the plugin. They are
plain-Python validators and interop helpers, so they need no compiled extension.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from polars_cv import build_info, numpy_from_struct
from polars_cv.geometry.schemas import validate_contour
from polars_cv.metrics import BBoxMatcher
from tests.conftest import plugin_required

_BBOX_SCHEMA = pl.Struct(
    [
        pl.Field("x", pl.Float64),
        pl.Field("y", pl.Float64),
        pl.Field("width", pl.Float64),
        pl.Field("height", pl.Float64),
    ]
)


class TestValidateContour:
    """`validate_contour` rejects every malformed shape and accepts a valid one."""

    @staticmethod
    def _pt(x: float, y: float) -> dict[str, float]:
        return {"x": x, "y": y}

    def test_accepts_a_valid_contour(self) -> None:
        square = {
            "exterior": [self._pt(0, 0), self._pt(1, 0), self._pt(1, 1), self._pt(0, 1)]
        }
        assert validate_contour(square) is True

    def test_accepts_valid_holes(self) -> None:
        c = {
            "exterior": [
                self._pt(0, 0),
                self._pt(4, 0),
                self._pt(4, 4),
                self._pt(0, 4),
            ],
            "holes": [[self._pt(1, 1), self._pt(2, 1), self._pt(2, 2)]],
        }
        assert validate_contour(c) is True

    def _valid_tri(self) -> list[dict[str, float]]:
        return [self._pt(0, 0), self._pt(1, 0), self._pt(1, 1)]

    def test_rejects_malformed(self) -> None:
        tri = self._valid_tri
        cases: list[object] = [
            "not a dict",
            {"holes": []},  # missing exterior
            {"exterior": "not a list"},
            {"exterior": [self._pt(0, 0), self._pt(1, 1)]},  # fewer than 3 points
            {"exterior": [self._pt(0, 0), self._pt(1, 0), "bad point"]},
            {"exterior": tri(), "holes": "not a list"},
            {"exterior": tri(), "holes": ["hole not a list"]},
            # valid dict exterior so execution reaches the holes branches:
            {"exterior": tri(), "holes": [[self._pt(0, 0), self._pt(1, 1)]]},  # < 3
            {"exterior": tri(), "holes": [[self._pt(0, 0), self._pt(1, 0), "bad"]]},
        ]
        for bad in cases:
            assert validate_contour(bad) is False, bad  # type: ignore[arg-type]


class TestNumpyFromStruct:
    """The numpy-sink interop reconstructs arrays and rejects malformed structs."""

    @staticmethod
    def _struct(arr: np.ndarray, *, with_strides: bool = True) -> dict[str, object]:
        row: dict[str, object] = {
            "data": arr.tobytes(),
            "dtype": arr.dtype.name,
            "shape": list(arr.shape),
            "offset": 0,
        }
        if with_strides:
            row["strides"] = list(arr.strides)
        return row

    def test_dict_roundtrip_with_strides(self) -> None:
        arr = np.arange(6, dtype=np.uint8).reshape(2, 3)
        out = numpy_from_struct(self._struct(arr))
        assert np.array_equal(out, arr)

    def test_dict_roundtrip_without_strides_assumes_c_contiguous(self) -> None:
        arr = np.arange(6, dtype=np.uint8).reshape(2, 3)
        out = numpy_from_struct(self._struct(arr, with_strides=False))
        assert np.array_equal(out, arr)

    def test_copy_false_returns_a_view_over_the_buffer(self) -> None:
        arr = np.arange(6, dtype=np.uint8).reshape(2, 3)
        out = numpy_from_struct(self._struct(arr), copy=False)
        assert np.array_equal(out, arr)
        assert out.base is not None  # a view, not an owned copy

    def test_series_struct_input(self) -> None:
        arr = np.arange(6, dtype=np.uint8).reshape(2, 3)
        series = pl.Series("out", [self._struct(arr)])
        out = numpy_from_struct(series)
        assert np.array_equal(out, arr)

    def test_non_struct_series_raises(self) -> None:
        with pytest.raises(ValueError, match="Expected Struct Series"):
            numpy_from_struct(pl.Series("x", [1, 2, 3]))

    def test_null_data_raises(self) -> None:
        row = self._struct(np.arange(4, dtype=np.uint8).reshape(2, 2))
        row["data"] = None
        with pytest.raises(ValueError, match="'data' is null"):
            numpy_from_struct(row)

    def test_unknown_dtype_raises(self) -> None:
        row = self._struct(np.arange(4, dtype=np.uint8).reshape(2, 2))
        row["dtype"] = "not_a_dtype"
        with pytest.raises(ValueError, match="Unsupported dtype"):
            numpy_from_struct(row)

    def test_misaligned_offset_raises(self) -> None:
        arr = np.arange(4, dtype=np.uint16).reshape(2, 2)
        row = self._struct(arr)
        row["offset"] = 1  # not a multiple of itemsize (2)
        with pytest.raises(ValueError, match="not a multiple of itemsize"):
            numpy_from_struct(row)

    def test_rank_mismatch_raises(self) -> None:
        arr = np.arange(6, dtype=np.uint8).reshape(2, 3)
        row = self._struct(arr)
        row["strides"] = [3, 1, 1]  # rank 3 against shape rank 2
        with pytest.raises(ValueError, match="different rank"):
            numpy_from_struct(row)

    def test_mapping_like_object_input(self) -> None:
        # Neither a dict nor a Series: a subscriptable object without `.get`,
        # so strides default to None (the C-contiguous branch) and offset to 0.
        arr = np.arange(6, dtype=np.uint8).reshape(2, 3)

        class _StructLike:
            def __init__(self, d: dict[str, object]) -> None:
                self._d = d

            def __getitem__(self, key: str) -> object:
                return self._d[key]  # raises KeyError on a missing field

        row = self._struct(arr, with_strides=False)
        row.pop("offset")
        out = numpy_from_struct(_StructLike(row))
        assert np.array_equal(out, arr)

    def test_mapping_like_object_missing_field_raises(self) -> None:
        class _Empty:
            def __getitem__(self, key: str) -> object:
                raise KeyError(key)

        with pytest.raises(ValueError, match="Cannot extract struct fields"):
            numpy_from_struct(_Empty())

    def test_buffer_protocol_data_without_copy(self) -> None:
        # `data` that supports the buffer protocol but is not bytes/bytearray/
        # memoryview exercises the memoryview branch of `_as_buffer`.
        arr = np.arange(6, dtype=np.uint8).reshape(2, 3)
        row = self._struct(arr, with_strides=False)
        row["data"] = np.frombuffer(arr.tobytes(), dtype=np.uint8)
        out = numpy_from_struct(row, copy=False)
        assert np.array_equal(out, arr)

    def test_negative_strides_reconstruct_a_flip(self) -> None:
        arr = np.arange(6, dtype=np.uint8).reshape(2, 3)
        flipped = arr[:, ::-1]  # negative stride on the last axis
        row = {
            "data": arr.tobytes(),  # backing buffer is the original, contiguous
            "dtype": "uint8",
            "shape": list(flipped.shape),
            "strides": list(flipped.strides),
            "offset": flipped.__array_interface__["data"][0]
            - arr.__array_interface__["data"][0],
        }
        out = numpy_from_struct(row)
        assert np.array_equal(out, flipped)


class TestBBoxMatcherValidation:
    """Construction and argument validation before any frame is touched."""

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_iou_threshold_out_of_range_raises(self, bad: float) -> None:
        with pytest.raises(ValueError, match="iou_threshold"):
            BBoxMatcher(iou_threshold=bad)

    def test_match_without_score_col_raises(self) -> None:
        df = pl.DataFrame({"pred": [[]], "gt": [[]]})
        with pytest.raises(ValueError, match="requires `score_col`"):
            BBoxMatcher().match(df, pred_col="pred", gt_col="gt")


@plugin_required
class TestBBoxMatcherOptionalColumns:
    """`match` wires through every optional column (class / weight / group).

    The image-id path and the optional-column branches are otherwise not
    exercised; this drives the full happy path with all of them set.
    """

    @staticmethod
    def _frame() -> pl.DataFrame:
        return pl.DataFrame(
            {
                "pred": [
                    [{"x": 0.0, "y": 0.0, "width": 10.0, "height": 10.0}],
                    [{"x": 5.0, "y": 5.0, "width": 4.0, "height": 4.0}],
                ],
                "gt": [
                    [{"x": 0.0, "y": 0.0, "width": 10.0, "height": 10.0}],
                    [{"x": 50.0, "y": 50.0, "width": 4.0, "height": 4.0}],
                ],
                "scores": [[0.9], [0.4]],
                "cls": ["cat", "dog"],
                "img": ["a", "b"],
                "w": [1.0, 2.0],
                "grp": ["g1", "g2"],
            }
        ).cast({"pred": pl.List(_BBOX_SCHEMA), "gt": pl.List(_BBOX_SCHEMA)})

    def test_all_optional_columns_are_honoured(self) -> None:
        table = BBoxMatcher(iou_threshold=0.5).match(
            self._frame(),
            pred_col="pred",
            gt_col="gt",
            score_col="scores",
            class_col="cls",
            image_id_col="img",
            weight_col="w",
            group_col="grp",
        )
        # A DetectionTable was produced from the fully-specified call.
        assert table is not None


@plugin_required
def test_build_info_reports_agreeing_versions() -> None:
    # Without the extension, plugin_version is None and filtered out, making the
    # assertion vacuously true — so this needs the built plugin to mean anything.
    info = build_info()
    assert info["version"] is not None
    # In a built checkout the plugin and distribution versions agree with __version__.
    non_null = {v for k, v in info.items() if "hash" not in k and v is not None}
    assert non_null == {info["version"]}
