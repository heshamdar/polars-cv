"""CI installs one pinned uv, named once in ``.tool-versions``.

``astral-sh/setup-uv`` with no ``version`` resolves ``latest`` on every run:
an extra network lookup per job (a macOS leg once died on its connect
timeout before any test ran) and a uv that changes under CI without a
commit. Every setup-uv step therefore reads the repo-root ``.tool-versions``,
the one place the version is written; a step that omits it would silently
fall back to ``latest``, so it fails here instead.

``.tool-versions`` must carry no ``python`` entry: setup-uv exports one as
``UV_PYTHON``, which would override each job's ``setup-python`` matrix
version.

These are source scans of hand-written workflow YAML (the weakest guard
kind, used because neither the compiler nor a runtime check can see CI
config). They read text rather than parse YAML, like the other workflow
guards; the fixtures below pin what the scan accepts and rejects.
"""

from __future__ import annotations

import re

import pytest

from tests._discovery import REPO_ROOT, workflow_files

pytestmark = pytest.mark.structural

_TOOL_VERSIONS = ".tool-versions"


def _setup_uv_steps(workflow: str) -> list[tuple[str, dict[str, str]]]:
    """Each ``astral-sh/setup-uv`` step: its ref and its ``with:`` inputs.

    A step's keys share the ``uses:`` line's indent and its inputs sit
    deeper; the step ends at the first non-blank line indented less than
    ``uses:`` (the next ``- `` item, or the end of the job).
    """
    steps = []
    lines = workflow.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^(\s*)(?:-\s+)?uses:\s*astral-sh/setup-uv@(\S+)", line)
        if not m:
            continue
        indent = line.index("uses:")
        inputs: dict[str, str] = {}
        in_with = False
        for body in lines[i + 1 :]:
            if not body.strip():
                continue
            depth = len(body) - len(body.lstrip())
            if depth < indent:
                break
            if depth == indent:
                in_with = body.strip() == "with:"
                continue
            if in_with:
                key, _, value = body.strip().partition(":")
                inputs[key.strip()] = value.split("#")[0].strip().strip("'\"")
        steps.append((m[2], inputs))
    return steps


def _unpinned(workflow: str) -> list[str]:
    """Why each setup-uv step in ``workflow`` would not install the pinned uv."""
    problems = []
    for ref, inputs in _setup_uv_steps(workflow):
        if "version" in inputs:
            problems.append(f"@{ref}: `version:` restates the uv version")
        if inputs.get("version-file") != _TOOL_VERSIONS:
            problems.append(f"@{ref}: no `version-file: {_TOOL_VERSIONS}`")
    return problems


def _uv_pin(tool_versions: str) -> str:
    """The exact uv version ``tool_versions`` pins; raises on anything else."""
    entries = dict(
        line.split(None, 1)
        for line in tool_versions.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if "python" in entries:
        raise ValueError("a `python` entry would be exported as UV_PYTHON")
    version = entries.get("uv", "").strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError(f"uv is not pinned to an exact version: {version!r}")
    return version


# ---------------------------------------------------------------------------
# The repository
# ---------------------------------------------------------------------------


def test_every_setup_uv_step_installs_the_pinned_uv() -> None:
    steps = {wf.name: _setup_uv_steps(wf.read_text()) for wf in workflow_files()}
    assert any(steps.values()), "no setup-uv step found — the scan drifted"
    problems = {
        wf.name: p for wf in workflow_files() if (p := _unpinned(wf.read_text()))
    }
    assert not problems, problems
    refs = {ref for found in steps.values() for ref, _ in found}
    assert len(refs) == 1, f"setup-uv is pinned to several versions: {refs}"


def test_tool_versions_pins_an_exact_uv() -> None:
    _uv_pin((REPO_ROOT / _TOOL_VERSIONS).read_text())


# ---------------------------------------------------------------------------
# Fixtures: the scan rejects what it claims to and accepts the fixed form
# ---------------------------------------------------------------------------

_STEP = """
jobs:
  test:
    steps:
      - name: Install uv
        uses: astral-sh/setup-uv@v10.2.0
        with:
          enable-cache: true
{extra}
      - name: Next
        with:
          version-file: {tv}
"""


@pytest.mark.parametrize(
    ("extra", "reason"),
    [
        ("", "no `version-file"),
        ("          version-file: pyproject.toml", "no `version-file"),
        ("          version: latest", "restates the uv version"),
        (
            "          version-file: .tool-versions  # pin\n          version: 0.1.0",
            "restates the uv version",
        ),
    ],
)
def test_a_step_without_the_pin_is_rejected(extra: str, reason: str) -> None:
    # The next step's `version-file` must not be read as this step's.
    problems = _unpinned(_STEP.format(extra=extra, tv=_TOOL_VERSIONS))
    assert any(reason in p for p in problems), problems


def test_the_pinned_step_is_accepted() -> None:
    extra = f"          version-file: {_TOOL_VERSIONS}"
    workflow = _STEP.format(extra=extra, tv="other")
    assert _setup_uv_steps(workflow) == [
        ("v10.2.0", {"enable-cache": "true", "version-file": _TOOL_VERSIONS})
    ]
    assert _unpinned(workflow) == []


@pytest.mark.parametrize(
    ("tool_versions", "reason"),
    [
        ("uv latest\n", "exact version"),
        ("uv 0.12\n", "exact version"),
        ("nodejs 22.0.0\n", "exact version"),
        ("uv 0.12.23\npython 3.12.0\n", "UV_PYTHON"),
    ],
)
def test_a_loose_or_python_pin_is_rejected(tool_versions: str, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        _uv_pin(tool_versions)


def test_an_exact_uv_pin_is_accepted() -> None:
    assert _uv_pin("# CI's uv\nuv 0.12.23\n") == "0.12.23"
