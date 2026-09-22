"""Everything the regression harness imports must be installed by `dev`.

The regression suite is dev tooling: `scripts/verify.sh` and
`.github/workflows/benchmark.yml` both install `--group dev` and nothing else.
So a module under `benchmarks/regression/` or `benchmarks/utils/` that imports
a package declared only in `bench` is depending on an install nobody performs.

Why a `try/except ImportError` does not excuse it
-------------------------------------------------

Both offenders this guard was written for are try-guarded, and that is the
problem rather than the mitigation. `psutil` is reached today only through a
four-hop transitive chain — `jupyterlab-execute-time -> jupyterlab -> ipykernel
-> psutil` — that nothing records and no one maintains on purpose. Drop the
Jupyter cell-timing plugin from `dev`, an entirely plausible cleanup, and the
`except ImportError` branch takes over: every memory measurement becomes
unavailable, silently, with the suite still exiting 0.

So the rule is on the *declaration*, not on the import style. A runtime
fallback is a way to degrade gracefully; it is not permission to leave the
dependency unstated.
"""

from __future__ import annotations

import ast
import sys

import pytest

from tests._discovery import REPO_ROOT, regression_harness_modules

pytestmark = pytest.mark.structural

PACKAGE_ROOT = REPO_ROOT / "polars-cv"

#: Import roots that are this repo's own code rather than a distribution.
FIRST_PARTY = frozenset({"benchmarks", "polars_cv", "tests"})

#: Distribution name for an import root that differs from it. Only the ones
#: this tree actually imports; a guess for a package nobody imports would be a
#: second authority on packaging metadata.
IMPORT_TO_DISTRIBUTION = {
    "cv2": "opencv-python",
    "PIL": "pillow",
    "sklearn": "scikit-learn",
}


def _imported_roots(source: str) -> set[str]:
    """Top-level import roots in *source*, including function-level imports.

    Function-level and `try`-guarded imports count. Deferring an import moves
    when it fails, not whether the dependency exists.
    """
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # `level > 0` is a relative import, so first-party by construction.
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    return roots


def _third_party_imports() -> dict[str, set[str]]:
    """Distribution name -> the harness modules importing it."""
    found: dict[str, set[str]] = {}
    for module in regression_harness_modules():
        rel = module.relative_to(PACKAGE_ROOT).as_posix()
        for root in _imported_roots(module.read_text()):
            if root in sys.stdlib_module_names or root in FIRST_PARTY:
                continue
            dist = IMPORT_TO_DISTRIBUTION.get(root, root)
            found.setdefault(dist, set()).add(rel)
    return found


def _installed_distributions() -> set[str]:
    """Everything `uv sync --group dev` puts in the environment.

    That is the `dev` group *plus* the project's own runtime dependencies,
    which are installed unconditionally. Checking only the group would report
    `polars` and `numpy` as missing, which would be wrong and would teach the
    next reader to add an exemption list.
    """
    if sys.version_info >= (3, 11):
        import tomllib
    else:  # pragma: no cover - only on 3.10
        import tomli as tomllib

    manifest = PACKAGE_ROOT / "pyproject.toml"
    data = tomllib.loads(manifest.read_text())
    entries = list(data["dependency-groups"]["dev"]) + list(
        data["project"].get("dependencies", [])
    )

    names: set[str] = set()
    for entry in entries:
        if not isinstance(entry, str):
            continue
        # "pytest>=7.0", "tomli>=2.0; python_version < '3.11'", "mkdocstrings[python]"
        name = entry.split(";")[0].strip()
        for sep in (">=", "<=", "==", "!=", "~=", ">", "<", "["):
            name = name.split(sep)[0]
        names.add(name.strip().lower().replace("_", "-"))
    return names


def test_the_scan_found_the_harness() -> None:
    """Discovery is asserted before anything is concluded from it."""
    modules = regression_harness_modules()
    assert len(modules) >= 8, (
        f"only {len(modules)} harness module(s) found; the scan has rotted and "
        f"this guard is checking nothing."
    )
    assert _third_party_imports(), (
        "no third-party imports found anywhere in the regression harness — "
        "implausible, so the import walk is broken."
    )


def test_every_harness_import_is_declared_in_the_dev_group() -> None:
    """`--group dev` must install everything the regression harness imports.

    Watched failing before `psutil` and `rich` moved into `dev`: it named both,
    and named the modules importing them.
    """
    declared = _installed_distributions()
    missing = {
        dist: sorted(users)
        for dist, users in sorted(_third_party_imports().items())
        if dist.lower().replace("_", "-") not in declared
    }
    assert not missing, (
        "these distributions are imported by the regression harness but are "
        "neither a project dependency nor in pyproject.toml's `dev` group, "
        "which is all that verify.sh and benchmark.yml install:\n"
        + "\n".join(
            f"  {dist}: imported by {', '.join(users)}"
            for dist, users in missing.items()
        )
        + "\n\nA `try/except ImportError` around the import does not excuse "
        "this: the fallback branch degrades a measurement silently."
    )


def test_the_benchmark_workflow_installs_the_dev_group() -> None:
    """The guard above is only meaningful if CI installs what it checks.

    A regex over the workflow text rather than a YAML parse: PyYAML is not a
    dev dependency, and adding one to read one line would be a heavier
    commitment than the check is worth. The limit is that this sees the literal
    command, so a refactor into a composite action would need updating here.
    """
    workflow = (REPO_ROOT / ".github" / "workflows" / "benchmark.yml").read_text()
    assert "uv sync --group dev" in workflow, (
        "benchmark.yml no longer installs `--group dev`, so the dependency "
        "set this guard checks is not the one the workflow creates."
    )
