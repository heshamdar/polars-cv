"""The user guide's vocabularies must match the code's.

Three places in ``docs/`` restate something the code already owns. All three
were hand-maintained with nothing comparing them to anything, which is how
``metrics/AGENTS.md`` came to document a ``rasterize(anti_alias=)`` parameter
that had been removed and guarded against.

These do not check prose. They check the specific claims a reader would act on:
which domains exist, what ``source("auto")`` resolves a column dtype to, and
whether the methods in the copy-pasteable examples exist.
"""

from __future__ import annotations

import inspect

import polars as pl
import pytest

from polars_cv import Pipeline
from polars_cv._types import Domain, SourceFormat
from polars_cv.expressions import CvNamespace
from polars_cv.geometry.bbox import BBoxNamespace
from polars_cv.geometry.contours import ContourNamespace
from polars_cv.geometry.points import PointNamespace
from polars_cv.lazy import LazyPipelineExpr

from ._discovery import doc_page
from ._doc_tables import cell_code, fenced_python_method_calls, table_with_header


def test_domains_page_lists_exactly_the_domain_vocabulary() -> None:
    """``domains.md``'s table must equal ``_types.Domain``.

    Equality rather than containment: a domain the code has and the page omits
    is as much a documentation bug as one the page invents.
    """
    text = doc_page("user-guide/concepts/domains.md").read_text()
    rows = table_with_header(text, "Domain", "Description")
    assert rows, (
        "domains.md has no table headed 'Domain | Description'. If the table "
        "was restructured, update this guard; it is not checking anything now."
    )

    documented = {cell_code(row[0]) for row in rows}
    assert None not in documented, (
        f"every row of the domain table must name the domain in a code span; "
        f"got {sorted(str(d) for d in documented)}"
    )
    assert documented == {d.value for d in Domain}


def test_auto_source_table_names_real_source_formats() -> None:
    """Every format ``auto`` is documented to resolve to must exist.

    A subset check, not equality: the table's left column is Polars dtypes, so
    it covers only the formats ``auto`` can *infer* — ``raw`` and ``contour``
    are real formats that a user names explicitly and that this table has no
    reason to mention.
    """
    text = doc_page("user-guide/concepts/sources.md").read_text()
    rows = table_with_header(text, "Column dtype", "Resolves to")
    assert rows, (
        "sources.md has no table headed 'Column dtype | Resolves to'. If the "
        "auto-resolution table was restructured, update this guard."
    )

    documented = {cell_code(row[1]) for row in rows}
    assert None not in documented, (
        f"every row must name the resolved format in a code span; got "
        f"{sorted(str(d) for d in documented)}"
    )

    known = {f.value for f in SourceFormat}
    unknown = documented - known
    assert not unknown, (
        f"sources.md says `auto` resolves to {sorted(unknown)}, which "
        f"SourceFormat does not have. Known formats: {sorted(known)}."
    )


#: Method names in the docs' Python blocks that are not ours. Each is a Polars
#: or builtin call appearing in an example, so resolving it against `Pipeline`
#: would fail for a reason that says nothing about our documentation.
_FOREIGN_METHODS = frozenset(
    {
        # Polars expression / frame API used to set an example up.
        "col",
        "lit",
        "when",
        "then",
        "otherwise",
        "fill_null",
        "select",
        "filter",
        "with_columns",
        "head",
        "collect",
        "explode",
        "struct",
        "field",
        "DataFrame",
        "LazyFrame",
        "read_parquet",
        "scan_parquet",
        # Builtins and stdlib.
        "append",
        "format",
        "join",
        "keys",
        "values",
        "items",
        "get",
        "min",
        "max",
        "sum",
        "abs",
        "round",
        "astype",
        "tolist",
    }
)

#: Every class a documented method may belong to: the builder, its lazy twin,
#: and the four expression namespaces (`.cv`, `.point`, `.contour`, `.bbox`).
#:
#: The geometry three are here because the docs use them heavily — the
#: geometry page alone calls twelve of their methods — and leaving them out
#: made the sweep report real methods as stale.
_OUR_SURFACES = (
    Pipeline,
    LazyPipelineExpr,
    CvNamespace,
    PointNamespace,
    ContourNamespace,
    BBoxNamespace,
)


def _resolves(name: str) -> bool:
    return any(callable(getattr(cls, name, None)) for cls in _OUR_SURFACES)


@pytest.mark.parametrize(
    "page",
    [
        "user-guide/operations/image-ops.md",
        "user-guide/operations/geometry.md",
        "user-guide/operations/reductions.md",
        "user-guide/operations/hashing.md",
        "user-guide/concepts/pipelines.md",
    ],
)
def test_documented_methods_exist(page: str) -> None:
    """Every method called in a page's Python examples must resolve.

    The operations pages name operations in prose headings ("Warp Affine",
    "Channel Select"), not in a table — so there is no list to diff, and a
    guard written as "the table's rows are a subset of ``OP_NAMES``" would
    match nothing and pass forever. What a reader actually copies is the code
    blocks, so those are what this reads.

    Extraction is textual and cannot tell our methods from Polars', which is
    why each name is resolved against the real classes and anything left over
    must be an acknowledged foreign name rather than assumed to be fine.
    """
    text = doc_page(page).read_text()
    called = fenced_python_method_calls(text)
    assert called, (
        f"{page} has no method calls in any ```python block. Either the page "
        f"lost its examples or the fence syntax changed; this guard is "
        f"checking nothing."
    )

    unresolved = {
        name
        for name in called
        if not _resolves(name)
        and name not in _FOREIGN_METHODS
        and not hasattr(pl.Expr, name)
        and not hasattr(pl.DataFrame, name)
    }
    assert not unresolved, (
        f"{page} calls {sorted(unresolved)}, which is not a method of "
        f"Pipeline, LazyPipelineExpr, CvNamespace, or the Polars API. Either "
        f"the documentation is stale or the name belongs in _FOREIGN_METHODS."
    )


def test_foreign_methods_are_all_actually_foreign() -> None:
    """Nothing in ``_FOREIGN_METHODS`` may be one of ours.

    The exemption list is the way this guard gets weakened: adding our own
    method to it would silence a real staleness report. A name is only foreign
    if we do not define it.
    """
    ours = {name for name in _FOREIGN_METHODS if _resolves(name)}
    assert not ours, (
        f"these are exempted as foreign but are methods of ours: {sorted(ours)}. "
        f"Remove them from _FOREIGN_METHODS — the guard is meant to check them."
    )


def test_our_surfaces_are_populated() -> None:
    """The classes resolved against must actually carry their methods.

    ``_resolves`` answering ``False`` for everything makes
    ``test_documented_methods_exist`` fail loudly, which is fine. Answering
    ``True`` for everything makes it pass silently, which is not — and an
    import that quietly resolved to the wrong object would do exactly that. So
    pin one method per surface that must be there, and that the surfaces are
    distinct objects rather than three names for one.
    """
    required = {
        Pipeline: "resize",
        LazyPipelineExpr: "sink",
        CvNamespace: "read_bytes",
        PointNamespace: "distance",
        ContourNamespace: "area",
        BBoxNamespace: "pairwise_iou",
    }
    assert set(required) == set(_OUR_SURFACES), (
        "every surface must be pinned by a known method; otherwise a new one "
        "could resolve to the wrong object unnoticed"
    )
    for cls, method in required.items():
        assert callable(getattr(cls, method, None)), (
            f"{cls.__name__}.{method} is missing, so this class is not the "
            f"surface the method resolution assumes it is."
        )
        public = [
            n for n, _ in inspect.getmembers(cls, callable) if not n.startswith("_")
        ]
        assert public, (
            f"{cls.__name__} exposes no public methods, so resolving against "
            f"it can only ever answer False."
        )

    assert len(set(_OUR_SURFACES)) == len(_OUR_SURFACES), (
        "_OUR_SURFACES lists the same class twice, so one surface is not being "
        "resolved against at all."
    )
