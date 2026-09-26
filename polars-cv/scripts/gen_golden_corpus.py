#!/usr/bin/env python
"""(Re)record ``tests/golden/op_corpus.json`` from the current build.

The corpus is the behavioural arbiter for the typed-op migration
(``TYPED_OPS_PLAN.md``). Recording it is a deliberate act: run this only when a
change is *meant* to alter an op's plan or output, and say so in the commit —
the fixture diff is the review.

Requires the compiled extension (``maturin develop``).

Usage::

    python scripts/gen_golden_corpus.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from tests._golden_cases import golden_cases  # noqa: E402

FIXTURE = _PKG / "tests" / "golden" / "op_corpus.json"


def record() -> dict:
    cases = {}
    for case in golden_cases():
        outcome = case.run()
        # A rejection's message is checked against the case's `expect`
        # substring at test time; storing the full text would pin wording.
        outcome.pop("message", None)
        cases[case.case_id] = outcome
    return {"platform": sys.platform, "cases": cases}


def main() -> int:
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    data = record()
    # One case per line, so a fixture diff names exactly the cases that changed.
    lines = [
        f"  {json.dumps(case_id)}: {json.dumps(outcome, sort_keys=True)}"
        for case_id, outcome in sorted(data["cases"].items())
    ]
    FIXTURE.write_text(
        "{\n"
        f' "platform": {json.dumps(data["platform"])},\n'
        ' "cases": {\n' + ",\n".join(lines) + "\n }\n}\n"
    )
    print(f"wrote {FIXTURE.relative_to(_PKG)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
