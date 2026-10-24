"""The CI decision for a missing compiled plugin.

Locally, ``plugin_required`` tests skip when ``maturin develop`` has not run,
so the suite stays usable mid-edit. Under CI the same skip hides a broken
build, so the session must fail instead. ``conftest.py`` asks
:func:`ci_requires_plugin` which of the two applies.

GitHub Actions and most other CI systems set ``CI=true``; the accepted values
follow the common convention rather than one platform's.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def is_ci(env: Mapping[str, str]) -> bool:
    """Whether ``env`` marks this run as CI."""
    return env.get("CI", "").strip().lower() in _TRUTHY


def ci_requires_plugin(env: Mapping[str, str], plugin_available: bool) -> bool:
    """True when the run must fail because CI would otherwise skip the plugin tests."""
    return is_ci(env) and not plugin_available


#: The source directory that holds the built extension after ``maturin develop``
#: (``python/polars_cv/_lib*``). Subprocess tests import from here rather than
#: through the editable install, which ``uv sync`` removes; the build is what
#: they depend on.
PYTHON_SRC = Path(__file__).resolve().parents[1] / "python"


def subprocess_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment for a child interpreter that must import this checkout.

    ``PYTHONPATH`` puts :data:`PYTHON_SRC` first, so the child imports the source
    tree and its built extension whether or not the package is installed.
    Anything already on ``PYTHONPATH`` follows it.
    """
    env = dict(os.environ if base is None else base)
    prior = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(PYTHON_SRC), prior) if p)
    return env
