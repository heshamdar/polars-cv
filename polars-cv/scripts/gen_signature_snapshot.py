#!/usr/bin/env python
"""(Re)record ``tests/golden/signatures.json``: the public builder call surface.

The typed-op migration (``TYPED_OPS_PLAN.md``) replaces hand-written builder
methods with generated ones, and must not change how they are *called* until
its API phase (P8). This snapshot freezes, for every public method of
``Pipeline`` and ``LazyPipelineExpr``, each parameter's name, kind and default.
Annotations are deliberately not recorded: generated methods will spell their
types differently while accepting the same calls.

Re-record only in the API phase, deliberately, with the diff reviewed.

Usage::

    python scripts/gen_signature_snapshot.py
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parents[1]
FIXTURE = _PKG / "tests" / "golden" / "signatures.json"


def surface() -> dict[str, dict[str, list[list[str]]]]:
    """Class -> public method -> [[name, kind, default-repr], ...]."""
    from polars_cv import LazyPipelineExpr, Pipeline

    out: dict[str, dict[str, list[list[str]]]] = {}
    for cls in (Pipeline, LazyPipelineExpr):
        methods: dict[str, list[list[str]]] = {}
        for name in sorted(dir(cls)):
            if name.startswith("_"):
                continue
            attr = inspect.getattr_static(cls, name)
            func = (
                attr.__func__ if isinstance(attr, (staticmethod, classmethod)) else attr
            )
            if not callable(func):
                continue
            params = []
            for p in inspect.signature(func).parameters.values():
                default = (
                    "" if p.default is inspect.Parameter.empty else repr(p.default)
                )
                params.append([p.name, p.kind.name, default])
            methods[name] = params
        out[cls.__name__] = methods
    return out


def main() -> int:
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(surface(), indent=1, sort_keys=True) + "\n")
    print(f"wrote {FIXTURE.relative_to(_PKG)}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(_PKG / "python"))
    sys.exit(main())
