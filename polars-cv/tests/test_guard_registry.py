"""Every guard test module is registered, and every registered guard has tests.

A guard that is deleted, renamed or emptied stops protecting anything without
any test going red, because nothing else refers to it. This test is the
ratchet on that: each ``test_*guard*.py`` / ``test_*ratchet*.py`` module must
appear in ``GUARDS`` below, and each listed module must exist and define at
least one ``test_`` function. Adding a guard therefore means adding it here,
in the same change.

The checks themselves live in :mod:`tests._guard_registry` and are exercised
on temporary directories first, so the failing cases are shown to fail.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._discovery import suite_files
from tests._guard_registry import (
    count_tests,
    empty_or_missing,
    guard_modules,
    unregistered,
)

pytestmark = pytest.mark.structural

GUARDS = (
    "test_build_efficiency.py",
    "test_dtype_ratchet_fixtures.py",
    "test_engine_pin_guard.py",
    "test_guard_registry.py",
    "test_join_ratchet.py",
    "test_parity_ratchets.py",
    "test_plugin_gate_guard.py",
    "test_row_ref_guard.py",
    "test_version_consistency.py",
)


def test_every_guard_module_is_registered() -> None:
    assert unregistered(suite_files(), GUARDS) == set()


def test_every_registered_guard_exists_and_has_tests() -> None:
    assert empty_or_missing(suite_files(), GUARDS) == []


def test_unregistered_guard_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "test_new_guard.py"
    path.write_text("def test_a():\n    pass\n")
    assert unregistered([path], ()) == {"test_new_guard.py"}


def test_registered_guard_with_no_tests_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "test_old_guard.py"
    path.write_text("X = 1\n")
    assert empty_or_missing([path], ("test_old_guard.py",)) == ["test_old_guard.py"]


def test_registered_guard_that_was_deleted_is_reported(tmp_path: Path) -> None:
    assert empty_or_missing([], ("test_gone_guard.py",)) == ["test_gone_guard.py"]


def test_helper_modules_are_not_guards(tmp_path: Path) -> None:
    path = tmp_path / "_some_guard.py"
    path.write_text("def test_a():\n    pass\n")
    assert guard_modules([path]) == set()


def test_count_sees_methods_in_classes(tmp_path: Path) -> None:
    path = tmp_path / "test_x_guard.py"
    path.write_text("class TestX:\n    def test_a(self):\n        pass\n")
    assert count_tests(path) == 1
