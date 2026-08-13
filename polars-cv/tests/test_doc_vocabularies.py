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

import ast
import inspect
import re

import polars as pl
import pytest

import polars_cv
from polars_cv import Pipeline, metrics
from polars_cv._types import Domain, SourceFormat
from polars_cv.expressions import CvNamespace
from polars_cv.geometry.bbox import BBoxNamespace
from polars_cv.geometry.contours import ContourNamespace
from polars_cv.geometry.points import PointNamespace
from polars_cv.lazy import LazyPipelineExpr

from ._discovery import doc_page, doc_pages, repo_file
from ._doc_tables import cell_code, fenced_python_method_calls, table_with_header

#: Every test here is a structural guard: it checks the *shape* of the codebase
#: -- registries, authorities, removed surfaces, documented vocabularies --
#: rather than the numerical behaviour of a pipeline. `-m structural` is the
#: lane pre-commit runs; see `tests/AGENTS.md`. Note that the lane as a whole
#: does need the compiled extension: many structural facts are only observable
#: through the FFI, and those tests fail rather than skip without it.
pytestmark = pytest.mark.structural


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


#: Names called in a doc example that belong to neither us nor Polars, with the
#: reason each is allowed. **Empty, and that is the point.**
#:
#: This began as a 34-name set, which was an escape hatch rather than a
#: contract: a stale method name is by definition not one of ours, so anything
#: a reviewer added here would silence exactly the staleness the sweep exists
#: to report. Once ``_resolves`` learned about the metrics API and
#: ``_POLARS_SURFACES`` learned about module-level ``pl.col``/``pl.DataFrame``,
#: every one of the 34 turned out to be either resolvable or absent from the
#: documentation entirely — so the hatch closed on its own.
#:
#: Keep it that way. An entry needs a reason, and
#: ``test_foreign_methods_are_all_actually_foreign`` rejects one that is ours
#: or that no page actually calls, so a dead exemption cannot accumulate here
#: waiting to hide something.
_FOREIGN_METHODS: "dict[str, str]" = {}

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

#: Public classes the metrics package exposes — `DetectionTable`, the matchers,
#: and the result objects whose methods the metrics page calls (`auc`,
#: `sensitivity_at_fp`, `summary_table`, …). Read from the module rather than
#: listed, so a renamed result class cannot leave its methods unresolvable.
_METRICS_SURFACES = tuple(v for v in vars(metrics).values() if inspect.isclass(v))

#: Polars objects a documented call may legitimately belong to. `pl` itself
#: matters most: `pl.col(...)` and `pl.DataFrame(...)` are module-level, and
#: checking only `pl.Expr`/`pl.DataFrame` left `col` unresolvable on nearly
#: every page — which is why the sweep once covered five hand-picked pages
#: instead of all of them.
_POLARS_SURFACES = (pl, pl.Expr, pl.DataFrame, pl.LazyFrame, pl.Series)


def _resolves(name: str) -> bool:
    """True when *name* is a method of ours."""
    if any(callable(getattr(cls, name, None)) for cls in _OUR_SURFACES):
        return True
    if any(callable(getattr(cls, name, None)) for cls in _METRICS_SURFACES):
        return True
    return callable(getattr(polars_cv, name, None)) or callable(
        getattr(metrics, name, None)
    )


@pytest.mark.parametrize("page", [p.name for p in doc_pages()], ids=lambda n: n)
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
    path = next(p for p in doc_pages() if p.name == page)
    called = fenced_python_method_calls(path.read_text())
    if not called:
        pytest.skip(f"{page} has no ```python examples to check")

    unresolved = {
        name
        for name in called
        if not _resolves(name)
        and name not in _FOREIGN_METHODS
        and not any(hasattr(obj, name) for obj in _POLARS_SURFACES)
    }
    assert not unresolved, (
        f"{page} calls {sorted(unresolved)}, which is not a method of "
        f"Pipeline, LazyPipelineExpr, the expression namespaces, the metrics "
        f"API, or Polars. Either the documentation is stale or the name "
        f"belongs in _FOREIGN_METHODS."
    )


def test_foreign_methods_are_all_actually_foreign() -> None:
    """``_FOREIGN_METHODS`` may hold neither our own names nor dead ones.

    The exemption list is how this guard gets weakened, so it is itself
    guarded in both directions:

    * **Ours.** Exempting one of our own methods silences a real staleness
      report. This is not hypothetical — `alias`, `cast` and `reshape` were all
      wrongly exempted when the list was first written, and `collect` joined
      them once the metrics API entered the resolution.
    * **Dead.** An entry no page calls is an exemption sitting ready to hide
      something later. Every entry must earn its place from a real call site.
    """
    called: set[str] = set()
    for page in doc_pages():
        called |= fenced_python_method_calls(page.read_text())

    ours = sorted(name for name in _FOREIGN_METHODS if _resolves(name))
    assert not ours, (
        f"these are exempted as foreign but are methods of ours: {ours}. "
        f"Remove them from _FOREIGN_METHODS — the guard is meant to check them."
    )

    dead = sorted(name for name in _FOREIGN_METHODS if name not in called)
    assert not dead, (
        f"these exemptions match no call in any doc page: {dead}. Remove them; "
        f"a dead exemption is one that will silently cover a future mistake."
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


# ---------------------------------------------------------------------------
# benchmarks/AGENTS.md: the single-op list
# ---------------------------------------------------------------------------


def _declared_benchmark_names() -> list[str]:
    """The ``name=`` of every config ``get_single_op_benchmarks()`` returns.

    Read by parsing the source rather than importing it: ``benchmarks`` pulls
    in every framework adapter (OpenCV, torch, torchvision) through its
    ``frameworks`` package, and none of that is installed for the structural
    lane. Parsing keeps this guard dependency-free and fast.
    """
    tree = ast.parse(
        repo_file("polars-cv/benchmarks/scenarios/single_ops.py").read_text()
    )
    fn = next(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "get_single_op_benchmarks"
        ),
        None,
    )
    assert fn is not None, (
        "get_single_op_benchmarks() is gone from scenarios/single_ops.py; this "
        "guard reads it as the authority for the benchmark list."
    )
    return [
        kw.value.value
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        for kw in node.keywords
        if kw.arg == "name" and isinstance(kw.value, ast.Constant)
    ]


def test_benchmark_list_is_current() -> None:
    """``benchmarks/AGENTS.md`` must name the benchmarks that actually exist.

    That sentence has been wrong twice: it said "21 benchmarks" while listing
    19, and the names had drifted (``flip`` for what are really
    ``flip_horizontal`` and ``flip_vertical``, ``crop`` for ``crop_center``).
    It was corrected by hand, and the correction *claimed* this test pinned it
    — while this test did not exist. So the sentence went straight back to
    being unpinned prose asserting it could not go stale.

    The count and every name are checked, because both were wrong before.
    """
    names = _declared_benchmark_names()
    assert names, (
        "no `name=` arguments were found in get_single_op_benchmarks(); the "
        "parse has rotted and this guard is checking nothing."
    )

    text = repo_file("polars-cv/benchmarks/AGENTS.md").read_text()

    counts = set(re.findall(r"(\d+)\s+benchmarks", text))
    assert counts == {str(len(names))}, (
        f"benchmarks/AGENTS.md claims {sorted(counts)} benchmarks; "
        f"get_single_op_benchmarks() returns {len(names)}."
    )

    listed = re.search(r"benchmarks:\s*([^)]*)\)", text)
    assert listed, "the sentence listing the benchmark names is gone from AGENTS.md"
    documented = [n.strip() for n in listed.group(1).split(",")]
    assert documented == names, (
        f"benchmarks/AGENTS.md lists {documented}, but "
        f"get_single_op_benchmarks() returns {names}."
    )
