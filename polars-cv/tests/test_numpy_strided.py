"""
Strided reconstruction for the numpy/torch struct sink.

The Rust sink can hand back a *non-contiguous* view of a shared buffer:
transpose produces permuted (positive) strides, flip/rotate produce negative
strides. `numpy_from_struct` must reproduce the true layout from the struct's
byte `strides`/`offset` in both copy modes — a plain `frombuffer().reshape()`
would silently read the bytes in C-order.

`TestNumpyFromStructStrided` unit-tests the consumer directly (full control over
strides/offset/dtype, no plugin needed). `TestEndToEndStridedSink` runs an
actual transpose pipeline through the numpy sink and checks the zero-copy path.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from polars_cv import Pipeline, numpy_from_struct
from tests.conftest import plugin_required

DTYPES = ["uint8", "uint16", "float32", "float64"]


def _row(orig: np.ndarray, view: np.ndarray, offset: int) -> dict[str, object]:
    """Struct dict mirroring what the numpy sink emits for `view`, whose data
    lives in `orig`'s contiguous bytes."""
    return {
        "data": orig.tobytes(),
        "dtype": str(orig.dtype),
        "shape": tuple(int(x) for x in view.shape),
        "strides": tuple(int(x) for x in view.strides),
        "offset": int(offset),
    }


class TestNumpyFromStructStrided:
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_transpose_positive_strides(self, dtype: str) -> None:
        orig = np.arange(24, dtype=dtype).reshape(2, 3, 4)
        view = np.transpose(orig, (2, 0, 1))  # permuted positive strides
        row = _row(orig, view, offset=0)
        np.testing.assert_array_equal(numpy_from_struct(row), view)
        np.testing.assert_array_equal(numpy_from_struct(row, copy=False), view)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_flip_axis0_negative_stride(self, dtype: str) -> None:
        orig = np.arange(12, dtype=dtype).reshape(3, 4)
        view = orig[::-1]  # negative stride on axis 0
        offset = (orig.shape[0] - 1) * orig.strides[0]  # first logical element
        row = _row(orig, view, offset)
        np.testing.assert_array_equal(numpy_from_struct(row), view)
        np.testing.assert_array_equal(numpy_from_struct(row, copy=False), view)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_flip_both_axes_negative_strides(self, dtype: str) -> None:
        orig = np.arange(12, dtype=dtype).reshape(3, 4)
        view = orig[::-1, ::-1]  # 180-degree rotate == flip both axes
        offset = (orig.shape[0] - 1) * orig.strides[0] + (
            orig.shape[1] - 1
        ) * orig.strides[1]
        row = _row(orig, view, offset)
        np.testing.assert_array_equal(numpy_from_struct(row), view)
        np.testing.assert_array_equal(numpy_from_struct(row, copy=False), view)

    def test_contiguous_still_correct(self) -> None:
        orig = np.arange(24, dtype="float32").reshape(2, 3, 4)
        row = _row(orig, orig, offset=0)
        np.testing.assert_array_equal(numpy_from_struct(row), orig)

    def test_copy_true_owns_data_copy_false_is_view(self) -> None:
        orig = np.arange(12, dtype="float32").reshape(3, 4)
        view = orig.T
        row = _row(orig, view, offset=0)

        owned = numpy_from_struct(row, copy=True)
        assert owned.flags["OWNDATA"]

        v = numpy_from_struct(row, copy=False)
        assert not v.flags["OWNDATA"]  # a view over the backing buffer
        assert v.base is not None
        np.testing.assert_array_equal(v, owned)


@plugin_required
class TestEndToEndStridedSink:
    def test_transpose_pipeline_zero_copy_and_correct(self) -> None:
        import polars as pl

        arr = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
        buf = io.BytesIO()
        Image.fromarray(arr, "RGB").save(buf, format="PNG")  # lossless
        df = pl.DataFrame({"img": [buf.getvalue()]})

        pipe = Pipeline().source("image_bytes").transpose([2, 0, 1])
        out = df.select(o=pl.col("img").cv.pipe(pipe).sink("numpy"))
        row = out["o"][0]

        expected = np.transpose(arr, (2, 0, 1))
        owned = numpy_from_struct(row, copy=True)
        view = numpy_from_struct(row, copy=False)

        np.testing.assert_array_equal(owned, expected)
        np.testing.assert_array_equal(view, owned)
        # The non-contiguous transpose output is delivered zero-copy (a view
        # over the plugin's buffer), not materialised to contiguous.
        assert view.base is not None
        assert not view.flags["C_CONTIGUOUS"]
