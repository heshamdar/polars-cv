"""
The plugin allocates through polars' own allocator (CR-60).

pyo3-polars' `PolarsAllocator` relays every allocation to the allocator polars
exports as the `polars.polars._allocator` capsule, and falls back to the
system `malloc` -- silently -- when that capsule cannot be imported. A polars
release that moves the capsule would put the plugin back on glibc, whose heap
trimming made a call that holds many large rows re-fault its memory on every
call after the first (~40% slower). This makes that fallback a failure.
"""

import pytest

from tests.conftest import plugin_required

pytestmark = [pytest.mark.structural, plugin_required]


def test_the_plugin_allocates_with_polars_allocator() -> None:
    import polars_cv._lib as lib

    assert lib.__allocator__ == "polars", (
        "the plugin's PolarsAllocator could not import polars' allocator "
        "capsule and fell back to the system allocator"
    )
