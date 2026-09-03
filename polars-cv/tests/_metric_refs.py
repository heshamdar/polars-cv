"""Independent NumPy references for FROC / LROC AUC.

These reimplement the FROC/LROC curve + integral from scratch so tests can check
the Polars expression code against a second implementation, rather than against
the removed eager API. Single-class (pooled) tables only — grouped correctness is
covered by group-vs-filtered-sub-table parity within the public API.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from polars_cv.metrics import DetectionTable


def _collapse_and_trapz(
    xs: np.ndarray, ys: np.ndarray, lo: float | None = None, hi: float | None = None
) -> float:
    """Upper-envelope collapse on ties, then (optionally clipped) trapezoid."""
    order = np.argsort(xs, kind="stable")
    xs, ys = xs[order], ys[order]
    # collapse ties in x to their max y
    ux: list[float] = []
    uy: list[float] = []
    for x, y in zip(xs, ys):
        if ux and x == ux[-1]:
            uy[-1] = max(uy[-1], y)
        else:
            ux.append(float(x))
            uy.append(float(y))
    ax = np.array(ux)
    ay = np.array(uy)
    if lo is None:
        return float(np.trapezoid(ay, ax))
    # clipped trapezoid over [lo, hi] with flat extension at the endpoints
    grid = np.clip(np.concatenate([[lo, hi], ax]), lo, hi)
    grid = np.unique(grid)

    def interp(q: float) -> float:
        if q <= ax[0]:
            return float(ay[0])
        if q >= ax[-1]:
            return float(ay[-1])
        return float(np.interp(q, ax, ay))

    gy = np.array([interp(q) for q in grid])
    return float(np.trapezoid(gy, grid))


def _froc_points(table: DetectionTable) -> tuple[np.ndarray, np.ndarray]:
    det, meta = table.collect()
    twg = max(float((meta["n_gts"] * meta["weight"]).sum()), 1.0)
    wsum = max(
        float(meta.unique(subset=["image_id"], keep="first")["weight"].sum()), 1.0
    )
    wmap = dict(zip(meta["image_id"].to_list(), meta["weight"].to_list()))

    d = det.filter(pl.col("score").is_not_null())
    scores = d["score"].to_numpy()
    tp = d["is_tp"].to_numpy().astype(bool)
    iw = np.array([wmap[i] for i in d["image_id"].to_list()], dtype=float)

    fps = [0.0]
    sens = [0.0]
    cum_wtp = 0.0
    cum_wfp = 0.0
    for s in sorted(set(scores.tolist()), reverse=True):
        m = scores == s
        cum_wtp += float((iw[m] * tp[m]).sum())
        cum_wfp += float((iw[m] * (~tp[m])).sum())
        fps.append(cum_wfp / wsum)
        sens.append(cum_wtp / twg)
    return np.array(fps), np.array(sens)


def _mcclish(raw: float, lo: float, hi: float) -> float:
    span = hi - lo
    if span <= 0:
        return 0.5
    min_pauc = (lo + hi) * span / 2.0
    denom = span - min_pauc
    if denom <= 0:
        return 0.5
    return (1.0 + (raw - min_pauc) / denom) / 2.0


def ref_froc_auc(
    table: DetectionTable,
    *,
    fp_range: tuple[float, float] | None = None,
    correction: str | None = None,
) -> float:
    """Reference trapezoidal FROC AUC (raw or partial) for a single-class table."""
    fps, sens = _froc_points(table)
    if fp_range is None:
        return _collapse_and_trapz(fps, sens)
    raw = _collapse_and_trapz(fps, sens, fp_range[0], fp_range[1])
    if correction == "normalize":
        span = fp_range[1] - fp_range[0]
        return raw / span if span > 0 else 0.0
    if correction == "mcclish":
        return _mcclish(raw, fp_range[0], fp_range[1])
    return raw


def _collapsed_xy(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(xs, kind="stable")
    xs, ys = xs[order], ys[order]
    ux: list[float] = []
    uy: list[float] = []
    for x, y in zip(xs, ys):
        if ux and x == ux[-1]:
            uy[-1] = max(uy[-1], y)
        else:
            ux.append(float(x))
            uy.append(float(y))
    return np.array(ux), np.array(uy)


def ref_froc_sensitivity_at_fp(table: DetectionTable, fp: float) -> float | None:
    """Reference interpolated sensitivity at ``fp`` (None outside the range)."""
    fps, sens = _froc_points(table)
    ax, ay = _collapsed_xy(fps, sens)
    if fp < ax[0] or fp > ax[-1]:
        return None
    return float(np.interp(fp, ax, ay))


def ref_mann_whitney(scores: list[float], labels: list[bool]) -> float:
    """Brute-force Mann-Whitney AUC: P(pos > neg) + 0.5 P(pos == neg)."""
    pos = [s for s, lb in zip(scores, labels) if lb]
    neg = [s for s, lb in zip(scores, labels) if not lb]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def ref_weighted_mann_whitney(
    scores: list[float],
    labels: list[bool],
    weights: list[float],
) -> float:
    """Brute-force *weighted* Mann-Whitney AUC — the independent oracle.

    Sums ``w_p * w_n`` over every positive/negative pair the positive wins (½ on
    ties), divided by the product of the class weight masses. With unit weights
    this equals :func:`ref_mann_whitney`, and it is a second, pairwise
    implementation of exactly the quantity the vectorized ``collapse_scores`` +
    ``mann_whitney_auc_expr`` path computes.
    """
    pos = [(s, w) for s, lb, w in zip(scores, labels, weights) if lb]
    neg = [(s, w) for s, lb, w in zip(scores, labels, weights) if not lb]
    if not pos or not neg:
        return 0.5
    w_pos = sum(w for _, w in pos)
    w_neg = sum(w for _, w in neg)
    num = 0.0
    for sp, wp in pos:
        for sn, wn in neg:
            if sp > sn:
                num += wp * wn
            elif sp == sn:
                num += 0.5 * wp * wn
    return num / (w_pos * w_neg)


def ref_froc_mw_detection(table: DetectionTable) -> float:
    """Weighted detection-level MW: each detection carries its image's weight."""
    det, meta = table.collect()
    wmap = {
        (iid, cid): float(w)
        for iid, cid, w in zip(
            meta["image_id"].to_list(),
            meta["class_id"].to_list(),
            meta["weight"].to_list(),
        )
    }
    d = det.filter(pl.col("score").is_not_null())
    weights = [
        wmap.get((iid, cid), 1.0)
        for iid, cid in zip(d["image_id"].to_list(), d["class_id"].to_list())
    ]
    return ref_weighted_mann_whitney(
        d["score"].to_list(), d["is_tp"].to_list(), weights
    )


# ---------------------------------------------------------------------------
# LROC reference (single-class / pooled)
# ---------------------------------------------------------------------------


def _lroc_per_image(table: DetectionTable, variant: str) -> list[dict]:
    """Per-image (gt_label, weight, max_score, top_is_tp) for a variant."""
    det, meta = table.collect()
    by_image: dict[str, list[tuple[float, bool]]] = {}
    for s, tp, iid in zip(
        det["score"].to_list(), det["is_tp"].to_list(), det["image_id"].to_list()
    ):
        if s is not None:
            by_image.setdefault(iid, []).append((float(s), bool(tp)))

    rows: list[dict] = []
    for iid, gt, w in zip(
        meta["image_id"].to_list(),
        meta["gt_label"].to_list(),
        meta["weight"].to_list(),
    ):
        dets = by_image.get(iid, [])
        tp_scores = [s for s, tp in dets if tp]
        all_scores = [s for s, _tp in dets]
        best_tp = max(tp_scores) if tp_scores else None
        max_det = max(all_scores) if all_scores else None
        top_is_tp_flag = dets[0][1] if dets else None  # first == highest score
        if variant == "best_tp":
            max_score = best_tp if gt else max_det
            top_is_tp = bool(best_tp is not None) if gt else False
        else:  # top_scoring
            max_score = max_det
            top_is_tp = bool(top_is_tp_flag) if gt else False
        rows.append(
            {
                "gt": bool(gt),
                "w": float(w),
                "max_score": max_score,
                "top_is_tp": top_is_tp,
            }
        )
    return rows


def _lroc_points(table: DetectionTable, variant: str) -> tuple[np.ndarray, np.ndarray]:
    rows = _lroc_per_image(table, variant)
    tw_pos = sum(r["w"] for r in rows if r["gt"])
    tw_neg = sum(r["w"] for r in rows if not r["gt"])
    n_pos = max(sum(1 for r in rows if r["gt"]), 1)
    n_neg = max(sum(1 for r in rows if not r["gt"]), 1)

    scored = [r for r in rows if r["max_score"] is not None]
    fpf = [0.0]
    sens = [0.0]
    cum_wpos = cum_wneg = cum_pos = cum_neg = 0.0
    max_sens = 0.0
    for s in sorted({r["max_score"] for r in scored}, reverse=True):
        for r in [r for r in scored if r["max_score"] == s]:
            if r["gt"] and r["top_is_tp"]:
                cum_pos += 1
                cum_wpos += r["w"]
            if not r["gt"]:
                cum_neg += 1
                cum_wneg += r["w"]
        se = cum_wpos / tw_pos if tw_pos > 0 else cum_pos / n_pos
        fp = cum_wneg / tw_neg if tw_neg > 0 else cum_neg / n_neg
        fpf.append(fp)
        sens.append(se)
        max_sens = max(max_sens, se)
    # lower-right endpoint (fpf=1, sensitivity=max achievable)
    fpf.append(1.0)
    sens.append(max_sens)
    return np.array(fpf), np.array(sens)


def ref_lroc_auc(
    table: DetectionTable,
    *,
    variant: str = "best_tp",
    fpf_range: tuple[float, float] | None = None,
) -> float:
    """Reference trapezoidal LROC AUC (raw or partial), single-class table."""
    fpf, sens = _lroc_points(table, variant)
    if fpf_range is None:
        return _collapse_and_trapz(fpf, sens)
    return _collapse_and_trapz(fpf, sens, fpf_range[0], fpf_range[1])


def ref_lroc_sensitivity_at_fpf(
    table: DetectionTable, fpf: float, variant: str = "best_tp"
) -> float | None:
    fps, sens = _lroc_points(table, variant)
    ax, ay = _collapsed_xy(fps, sens)
    if fpf < ax[0] or fpf > ax[-1]:
        return None
    return float(np.interp(fpf, ax, ay))


def ref_lroc_image_mw(table: DetectionTable, variant: str = "best_tp") -> float:
    """Weighted image-level MW: each per-image observation carries its weight."""
    rows = _lroc_per_image(table, variant)
    scores: list[float] = []
    labels: list[bool] = []
    weights: list[float] = []
    for r in rows:
        ms = r["max_score"]
        if r["gt"]:
            s = (ms if r["top_is_tp"] else 0.0) if ms is not None else 0.0
        else:
            s = ms if ms is not None else 0.0
        scores.append(float(s))
        labels.append(r["gt"])
        weights.append(float(r["w"]))
    return ref_weighted_mann_whitney(scores, labels, weights)
