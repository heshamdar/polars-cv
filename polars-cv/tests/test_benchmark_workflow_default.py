"""The manual benchmark workflow's thread default is the suite's default.

The dispatch form names its own default. A second copy that drifts from
``SuiteConfig.num_threads`` would benchmark a pool nobody asked for, so this
reads the workflow file and holds the one number to the suite's.

This is a source scan of one named file, so it is structural: a fast check
that pre-commit runs. Its limit: it recognises the ``threads`` input's
``default:`` written in the layout ``benchmark.yml`` uses (six-space key,
eight-space fields); a reformatted file fails loudly rather than passing.
"""

from __future__ import annotations

import re

import pytest
from benchmarks.regression.config import DEFAULT

from tests._discovery import REPO_ROOT

pytestmark = pytest.mark.structural


def test_the_workflow_thread_default_is_the_suite_default() -> None:
    text = (REPO_ROOT / ".github" / "workflows" / "benchmark.yml").read_text()
    match = re.search(r'\n {6}threads:\n(?: {8}[^\n]*\n)+? {8}default: "(\d+)"', text)
    assert match is not None, "benchmark.yml has no threads input default"
    assert int(match.group(1)) == DEFAULT.num_threads
