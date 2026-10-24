"""The CI decision for a missing compiled plugin.

Locally, ``plugin_required`` tests skip when ``maturin develop`` has not run,
so the suite stays usable mid-edit. Under CI the same skip hides a broken
build, so the session must fail instead. ``conftest.py`` asks
:func:`ci_requires_plugin` which of the two applies.

GitHub Actions and most other CI systems set ``CI=true``; the accepted values
follow the common convention rather than one platform's.
"""

from __future__ import annotations

from collections.abc import Mapping

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def is_ci(env: Mapping[str, str]) -> bool:
    """Whether ``env`` marks this run as CI."""
    return env.get("CI", "").strip().lower() in _TRUTHY


def ci_requires_plugin(env: Mapping[str, str], plugin_available: bool) -> bool:
    """True when the run must fail because CI would otherwise skip the plugin tests."""
    return is_ci(env) and not plugin_available
