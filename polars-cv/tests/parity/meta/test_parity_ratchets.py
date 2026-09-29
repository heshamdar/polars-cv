"""Coverage ratchets: the parity tables against the engine's own catalogues.

Every list the parity framework keeps is held to the authority it mirrors,
read at run time — never to a second list written here:

* the reference table (``framework/oracle.OPS`` + ``EXEMPT``) against the
  chainable ``Pipeline`` methods (``polars_cv.lazy._chainable_pipeline_ops``,
  the authority ``tests/_op_cases.py`` is also held to);
* the binary table (``BINARY`` + ``BINARY_EXEMPT``) against the op
  catalogue's ``lazy_only`` ops;
* the source and sink registries against the I/O catalogue;
* the image dtypes against the ``cast`` op's dtype enum, which is
  ``dtype_table!``'s spelling list.

A new op, format or dtype therefore fails here until it has a parity entry
or an exemption that says why not; a removed one fails until its stale entry
goes. Each check names both directions.
"""

from __future__ import annotations

import json

import pytest

from tests.conftest import plugin_required
from tests.parity.framework.images import DTYPES
from tests.parity.framework.io import SINK_EXEMPT, SINKS, SOURCE_EXEMPT, SOURCES
from tests.parity.framework.oracle import BINARY, BINARY_EXEMPT, EXEMPT, OPS

pytestmark = [pytest.mark.structural, plugin_required]


def _op_catalog() -> list[dict]:
    from polars_cv._lib import op_catalog

    return json.loads(op_catalog())


def _io_catalog() -> dict:
    from polars_cv._lib import io_catalog

    return json.loads(io_catalog())


def _both_ways(
    label: str, authority: set[str], covered: set[str], exempt: set[str]
) -> None:
    missing = sorted(authority - covered - exempt)
    stale = sorted((covered | exempt) - authority)
    overlap = sorted(covered & exempt)
    assert not missing, f"{label}: no parity entry and no exemption for {missing}"
    assert not stale, f"{label}: entries for names that no longer exist: {stale}"
    assert not overlap, f"{label}: both an entry and an exemption: {overlap}"


def test_every_chainable_op_has_a_reference_entry_or_a_reason() -> None:
    """``OPS`` + ``EXEMPT`` is exactly the chainable Pipeline vocabulary."""
    from polars_cv.lazy import _chainable_pipeline_ops

    authority = set(_chainable_pipeline_ops())
    assert len(authority) > 50, "the chainable-op scan found almost nothing"
    _both_ways("chainable ops", authority, set(OPS), set(EXEMPT))


def test_every_lazy_only_op_has_a_binary_entry_or_a_reason() -> None:
    """``BINARY`` + ``BINARY_EXEMPT`` is exactly the lazy-only vocabulary."""
    authority = {
        op.get("python") or op["name"]
        for op in _op_catalog()
        if op.get("visibility") == "lazy_only"
    }
    assert authority, "the catalogue lists no lazy-only op: the scan is broken"
    _both_ways("lazy-only ops", authority, set(BINARY), set(BINARY_EXEMPT))


def test_every_catalogue_source_is_fed_or_exempt() -> None:
    """Each source format is some SourceSpec's format, or exempted."""
    authority = {s["name"] for s in _io_catalog()["sources"]}
    covered = {spec.format for spec in SOURCES.values()}
    _both_ways("sources", authority, covered, set(SOURCE_EXEMPT))


def test_every_catalogue_sink_is_read_or_exempt() -> None:
    """Each sink format has a SinkSpec (its decoder), or is exempted."""
    authority = {s["name"] for s in _io_catalog()["sinks"]}
    _both_ways("sinks", authority, set(SINKS), set(SINK_EXEMPT))


def test_image_dtypes_are_the_engine_dtypes() -> None:
    """The images span exactly the dtypes ``cast`` accepts (``dtype_table!``)."""
    (cast,) = [op for op in _op_catalog() if op["name"] == "cast"]
    (field,) = [f for f in cast["fields"] if f["name"] == "dtype"]
    assert set(field["type"]["variants"]) == set(DTYPES)


def test_exemptions_say_why() -> None:
    """An exemption is a decision; it must record its reason."""
    for table in (EXEMPT, BINARY_EXEMPT, SOURCE_EXEMPT, SINK_EXEMPT):
        blank = [k for k, reason in table.items() if len(reason.strip()) < 20]
        assert not blank, f"exemptions without a real reason: {blank}"


def test_entries_without_a_reference_say_why() -> None:
    """``OpSpec`` refuses construction without one; this keeps the reasons
    substantive rather than a placeholder."""
    thin = [s.method for s in OPS.values() if s.ref is None and len(s.no_ref) < 20]
    assert not thin, f"entries with no reference and no real reason: {thin}"
