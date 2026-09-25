"""Print the runtime dependencies pinned to their declared floors.

``pyproject.toml``'s ``[project] dependencies`` is the one authority for the
oldest versions polars-cv claims to support. The ``dependency-floors`` CI job
installs this script's output over the locked environment, so the floors are
tested rather than asserted (CR-44)::

    uv pip install $(python scripts/dependency_floors.py)

Every dependency must carry a ``>=`` floor: one without cannot be tested at
its oldest version, so it is an error rather than something to skip.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # Python 3.10: `tomllib` is 3.11+ stdlib; `tomli` is its exact predecessor.
    import tomli as tomllib

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

# `name[extras] ... >=version ...; marker` — the floor is the `>=` clause.
_REQUIREMENT = re.compile(r"^\s*(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?")
_FLOOR = re.compile(r">=\s*(?P<version>[0-9][0-9A-Za-z.+!-]*)")


def floors(pyproject_text: str) -> list[str]:
    """``name==floor`` for every ``[project] dependencies`` entry."""
    deps = tomllib.loads(pyproject_text)["project"]["dependencies"]
    pinned: list[str] = []
    for dep in deps:
        requirement = dep.split(";", 1)[0]
        name = _REQUIREMENT.match(requirement)
        floor = _FLOOR.search(requirement)
        if name is None or floor is None:
            msg = f"dependency {dep!r} declares no '>=' floor, so it cannot be tested"
            raise ValueError(msg)
        pinned.append(f"{name['name']}=={floor['version']}")
    return pinned


if __name__ == "__main__":
    print(" ".join(floors(PYPROJECT.read_text())))
