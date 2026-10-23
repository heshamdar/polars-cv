"""No test may pin the in-memory engine on its own.

The suite runs every bare ``.collect()`` under the lane's engine: streaming by
default, in-memory in the second lane (``scripts/verify.sh``). A test that
switches to in-memory itself runs in-memory in *both* lanes, so a failure that
only streaming shows (a morsel-local row number, a chunk-boundary bug) is
hidden from the suite exactly where it occurs. That is how a ``(row n)`` that
was wrong under streaming shipped with tests that only asserted it in-memory.

What is allowed: a test that runs the in-memory engine *beside* streaming to
compare them (it names both). What is rejected:

- changing the engine affinity (``set_engine_affinity``, ``engine_affinity=``,
  ``POLARS_ENGINE_AFFINITY``) anywhere but ``conftest.py``'s module level, which
  only sets the lane's default;
- an ``"in-memory"`` engine literal in a function (or a parameter default) that
  does not also name ``"streaming"``.

The checker is :func:`tests._engine_pin_guard.engine_pin_offenders`; the
fixtures below are run through the same function the repo-wide test uses.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tests._discovery import suite_files
from tests._engine_pin_guard import engine_pin_offenders

pytestmark = pytest.mark.structural


def offenders(src: str, name: str = "test_x.py") -> list[str]:
    return engine_pin_offenders(textwrap.dedent(src), name)


@pytest.mark.parametrize(
    "src",
    [
        # The fixture this replaced.
        """
        @pytest.fixture
        def in_memory_engine():
            pl.Config.set_engine_affinity("in-memory")
            yield
        """,
        # Affinity through the environment.
        """
        def test_x(monkeypatch):
            monkeypatch.setenv("POLARS_ENGINE_AFFINITY", "in-memory")
        """,
        """
        def test_x():
            os.environ["POLARS_ENGINE_AFFINITY"] = "in-memory"
        """,
        # A config context.
        """
        def test_x():
            with pl.Config(engine_affinity="in-memory"):
                pass
        """,
        # One engine only, by argument.
        """
        def test_x(df):
            df.lazy().collect(engine="in-memory")
        """,
        # One engine only, by a helper's default.
        """
        def run(df, *, engine: str = "in-memory"):
            return df.lazy().collect(engine=engine)
        """,
        # Nested in a method.
        """
        class TestX:
            def test_y(self, lf):
                assert lf.collect(engine="in-memory").height
        """,
    ],
)
def test_a_pin_is_rejected(src: str) -> None:
    assert offenders(src), src


@pytest.mark.parametrize(
    "src",
    [
        # Both engines, compared.
        """
        def test_x(lf):
            assert lf.collect(engine="in-memory").equals(lf.collect(engine="streaming"))
        """,
        """
        @pytest.mark.parametrize("engine", ["in-memory", "streaming"])
        def test_x(lf, engine):
            lf.collect(engine=engine)
        """,
        # The lane's engine.
        """
        def test_x(lf):
            lf.collect()
        """,
        # Prose is not a pin.
        '''
        def test_x(lf):
            """Unlike the in-memory engine, ..."""
            lf.collect()
        ''',
    ],
)
def test_a_comparison_or_the_lanes_engine_is_accepted(src: str) -> None:
    assert offenders(src) == [], src


def test_conftest_may_set_the_lanes_default_at_module_level() -> None:
    src = 'pl.Config.set_engine_affinity(os.environ.setdefault("X", "streaming"))\n'
    assert offenders(src, "conftest.py") == []
    inside = """
    def fixture():
        pl.Config.set_engine_affinity("in-memory")
    """
    assert offenders(inside, "conftest.py")


def test_no_test_pins_the_in_memory_engine() -> None:
    found = [
        hit
        for path in suite_files()
        if path.name != Path(__file__).name
        for hit in engine_pin_offenders(path.read_text(), path.name, path=path)
    ]
    assert not found, "tests that pin the in-memory engine:\n  " + "\n  ".join(found)
