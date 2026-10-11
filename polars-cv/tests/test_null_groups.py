"""A null group key is a group like any other, at every grouped metrics entry point.

Sometimes the group is unknown (a site not recorded), and those images still
need their own curve. Each case evaluates the same table twice: once with
some images in a null group, once with that group named by a sentinel that
sorts where null does (first), so the bootstrap's draws, which follow the
sorted order, are the same. The results must agree once the null is named.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import polars_cv.metrics as M

from .conftest import plugin_required

_SENTINEL = "\x00null"
_N = 60
_IDS = [f"i{i:03d}" for i in range(_N)]


def _table(named: bool) -> M.DetectionTable:
    rng = np.random.default_rng(1)
    site = [None if i % 3 == 0 else "ab"[i % 3 - 1] for i in range(_N)]
    if named:
        site = [_SENTINEL if s is None else s for s in site]
    preds, gts = [], []
    for image in _IDS:
        for _ in range(rng.integers(0, 4)):
            x = float(rng.integers(0, 50))
            preds.append(
                {"image_id": image, "bbox": [x, x, 10.0, 10.0], "score": rng.random()}
            )
        for _ in range(rng.integers(0, 3)):
            x = float(rng.integers(0, 50))
            gts.append(
                {"image_id": image, "bbox": [x + rng.integers(0, 3), x, 10.0, 10.0]}
            )
    images = pl.DataFrame(
        {"image_id": _IDS, "weight": rng.uniform(0.5, 2, _N), "site": site}
    )
    return M.match_detections(
        pl.DataFrame(preds),
        pl.DataFrame(gts),
        box_format="xywh",
        class_id=None,
        images=images,
        weight="weight",
        group="site",
    )


_B = {"n_bootstrap": 8, "seed": 0}
_G = {"group_by": "group_id"}

CASES: dict[str, Callable[[M.DetectionTable], pl.LazyFrame]] = {
    "froc_curve_lazy": lambda t: M.froc_curve_lazy(t, **_G),
    "froc_summary_table": lambda t: M.froc_summary_table(t, [0.5, 1.0], **_G),
    "froc_sensitivity_at_fp": lambda t: M.froc_sensitivity_at_fp(t, 0.5, **_G),
    "froc_auc(partial)": lambda t: M.froc_auc(t, fp_range=(0, 1), **_G),
    "froc_auc(mann_whitney)": lambda t: M.froc_auc(t, method="mann_whitney", **_G),
    "lroc_curve_lazy": lambda t: M.lroc_curve_lazy(t, **_G),
    "lroc_auc": lambda t: M.lroc_auc(t, **_G),
    "lroc_sensitivity_at_fpf": lambda t: M.lroc_sensitivity_at_fpf(t, 0.3, **_G),
    "AP": lambda t: M.AP().by_group(t, "group_id"),
    "AP(11_point)": lambda t: M.AP("11_point").by_group(t, "group_id"),
    "Recall": lambda t: M.Recall().by_group(t, "group_id"),
    "FROCAUC": lambda t: M.FROCAUC(fp_range=(0, 1)).by_group(t, "group_id"),
    "froc_auc_ci_lazy": lambda t: M.froc_auc_ci_lazy(t, fp_range=(0, 1), **_G, **_B),
    "froc_auc_ci_lazy(fixed)": lambda t: M.froc_auc_ci_lazy(
        t, fp_range=(0, 1), weight_scheme="fixed", **_G, **_B
    ),
    "froc_auc_ci_lazy(strata)": lambda t: M.froc_auc_ci_lazy(
        t, method="mann_whitney", strata="group_id", **_B
    ),
    "froc_auc_ci_lazy(sample_col)": lambda t: M.froc_auc_ci_lazy(
        t, method="mann_whitney", sample_col="group_id", **_B
    ),
    "lroc_auc_ci_lazy": lambda t: M.lroc_auc_ci_lazy(t, **_G, **_B),
    "average_precision_ci_lazy": lambda t: M.average_precision_ci_lazy(t, **_G, **_B),
    "bootstrap_ci(AP)": lambda t: M.bootstrap_ci(t, M.AP(), **_G, **_B),
}


@plugin_required
@pytest.mark.parametrize("name", list(CASES))
def test_a_null_group_is_a_group(name: str) -> None:
    null = CASES[name](_table(named=False)).collect()
    named = CASES[name](_table(named=True)).collect()
    strings = [c for c, t in null.schema.items() if t == pl.String]
    null = null.with_columns(pl.col(c).fill_null(_SENTINEL) for c in strings)
    if "group_id" in null.columns:
        assert _SENTINEL in null["group_id"].to_list()
    if strings:
        null, named = null.sort(strings), named.sort(strings)
    assert_frame_equal(null, named, check_exact=False, rel_tol=1e-9, abs_tol=1e-12)
