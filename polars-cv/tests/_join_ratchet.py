"""Every keyed join in ``polars_cv.metrics`` matches null keys to each other.

A null group key is a group of its own throughout the metrics (``group_by``
and ``.over`` already treat it so). A join defaults to ``nulls_equal=False``,
which silently drops a null group's rows from one side, so the curve of a
group named null came out empty while the same group under any other name
did not. :func:`unmatched_joins` reports every ``.join(...)`` with ``on=`` or
``left_on=`` that does not pass ``nulls_equal=True``; a cross join has no
keys and is skipped.

Limits: this is a syntactic scan (``ast``) of the source. It sees a method
named ``join`` called with ``on=`` / ``left_on=`` keywords; a join built with
the keys passed positionally, or through ``**kwargs``, is reported (it cannot
show ``nulls_equal=True``), and ``str.join`` (no such keywords) is ignored.
``tests/test_null_groups.py`` holds the behaviour itself at the public API.
"""

from __future__ import annotations

import ast


def unmatched_joins(source: str) -> list[int]:
    """Line numbers of the keyed joins in ``source`` without ``nulls_equal=True``."""
    tree = ast.parse(source)
    bad = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "join"
        ):
            continue
        kw = {k.arg: k.value for k in node.keywords}
        how = kw.get("how")
        if isinstance(how, ast.Constant) and how.value == "cross":
            continue
        keyed = "on" in kw or "left_on" in kw or len(node.args) >= 2 or None in kw
        if not keyed:
            continue
        ne = kw.get("nulls_equal")
        if not (isinstance(ne, ast.Constant) and ne.value is True):
            bad.append(node.lineno)
    return bad
