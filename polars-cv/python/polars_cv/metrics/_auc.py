"""Correction vocabulary for FROC/LROC/PR AUC.

The AUC integrals themselves are lazy Polars expressions in
:mod:`polars_cv.metrics._auc_expr` (``trapz_auc_expr`` / ``partial_auc_expr``),
the single authority every caller reduces through. The eager, Series-based
``trapz_auc`` / ``partial_auc`` that once lived here were removed once every
consumer routed through the lazy path; only the shared ``correction`` vocabulary
remains, so a value nothing recognises still fails loudly in one place.
"""

from __future__ import annotations

from typing import Literal

CorrectionMethod = Literal["normalize"] | None


def validate_correction(correction: CorrectionMethod) -> None:
    """Reject any correction outside the supported vocabulary.

    The single authority for the ``correction`` vocabulary, so an unrecognised
    value fails loudly at the entry point rather than being silently treated as
    the raw area (the ``if correction == "normalize": ...; return raw`` shape
    would otherwise swallow it). ``"mcclish"`` -- McClish's standardized ROC
    partial-AUC correction -- was removed here: it standardizes against the ROC
    chance diagonal ``y = x`` on the unit square, which is only meaningful when
    the x-axis is a probability bounded to ``[0, 1]``. FROC's x-axis (false
    positives per image) is unbounded, LROC's chance line is not the diagonal,
    and PR's chance line is horizontal at prevalence -- so none of the curves in
    this package is the ROC curve McClish assumes.

    Args:
        correction: The correction requested by the caller.

    Raises:
        ValueError: If ``correction`` is neither ``None`` nor ``"normalize"``.
    """
    if correction is not None and correction != "normalize":
        raise ValueError(
            f"Unknown correction {correction!r}. Expected 'normalize' or None. "
            "(The McClish standardized correction was removed: it assumes a "
            "bounded [0,1] ROC x-axis with a y=x chance diagonal, which the "
            "FROC/LROC/PR curves are not.)"
        )
