"""Build the parity report: what polars-cv is checked against, and how closely.

Two sources, both committed, both produced by checks CI runs:

- ``docs/assets/parity/reference.json``: one entry per comparison declared with
  ``@pytest.mark.parity`` in ``tests/reference/`` (``tests/_parity.py``), with
  how many of its tests passed, and the versions of every reference library.
- ``../benchmarks/results/parity.json``: the benchmark's end-to-end diffs of
  each library against PIL on 40 COCO images (``benchmarks/bench/parity.py``).

From them it renders ``docs/user-guide/parity.md`` and the README section
between ``<!-- parity:start -->`` and ``<!-- parity:end -->``.

    cd polars-cv
    uv run --no-sync --with pycocotools python scripts/parity_report.py

runs the reference suite and rewrites all three. ``--render`` re-renders from
the committed JSON without running anything. A run with any failed or skipped
test writes nothing and exits non-zero: the report never claims a comparison
that did not run (``pycocotools`` is not a dependency, hence ``--with``).
``tests/test_parity_report.py`` holds the committed files to each other and to
the declarations the suite collects.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_JSON = ROOT / "docs" / "assets" / "parity" / "reference.json"
BENCH_JSON = ROOT.parent / "benchmarks" / "results" / "parity.json"
PAGE = ROOT / "docs" / "user-guide" / "parity.md"
README = ROOT / "README.md"
START, END = "<!-- parity:start -->", "<!-- parity:end -->"
DOCS_URL = "https://heshamdar.github.io/polars-cv/user-guide/parity/"
REPO_URL = "https://github.com/heshamdar/polars-cv/blob/main/polars-cv/"

#: Distributions whose versions the report records: polars-cv, its runtime,
#: and every reference implementation the declarations name.
VERSIONED = (
    "polars-cv",
    "polars",
    "numpy",
    "opencv-python",
    "scipy",
    "pillow",
    "imagehash",
    "pycocotools",
)

#: The benchmark's end-to-end workloads, what each compares, and the metrics
#: shown, all from ``benchmarks/bench/workloads.py`` and ``parity.py``.
WORKLOADS = (
    ("w1", "decode → bilinear 224² → gray", ("mean_abs", "max_abs"), "levels"),
    (
        "w2",
        "thumbnail → JPEG, re-decoded",
        ("mean_abs", "worst_image_mean_abs"),
        "levels",
    ),
    ("w3", "64-bit pHash", ("hash_mean_bits", "hash_max_bits"), "bits"),
    ("w4", "per-row crop + bilinear resize", ("mean_abs", "max_abs"), "levels"),
)
#: Benchmark library ids, as the report names them. ``pil`` is the reference.
LIBRARIES = (
    ("polars-cv-eager", "polars-cv"),
    ("opencv", "OpenCV"),
    ("torchvision", "torchvision"),
    ("daft", "Daft"),
)


#: Reference words in ``tests/_parity.py``'s ``REFERENCES`` that are not a
#: third-party library, left out where the README names the libraries.
_NOT_LIBRARIES = ("Python", "Cross-check", "Definition", "Hand-computed")


def reference_library(against: str) -> str:
    """The library a declaration compares with: the text before its first space.

    ``tests/_parity.py`` refuses an ``against`` that does not start with one of
    its ``REFERENCES``, so this is never a stray word.
    """
    return against.split(" ", 1)[0].rstrip(":")


# --- Run ----------------------------------------------------------------------


def run_suite() -> list[dict]:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "parity.json"
        cmd = [
            sys.executable, "-m", "pytest", "tests/reference", "-q", "-n", "auto",
            "-m", "not network", "-p", "no:cacheprovider", f"--parity-report={out}",
        ]  # fmt: skip
        code = subprocess.call(cmd, cwd=ROOT)
        groups = json.loads(out.read_text()) if out.exists() else []
    bad = [g for g in groups if g.get("failed") or g.get("skipped")]
    if code != 0 or bad or not groups:
        for g in bad:
            print(
                f"  {g['file']} {g['ops']}: failed={g.get('failed', 0)} "
                f"skipped={g.get('skipped', 0)}",
                file=sys.stderr,
            )
        sys.exit(
            f"pytest exited {code}; nothing written (every test must run and pass)"
        )
    return groups


def versions() -> dict[str, str]:
    out = {}
    for dist in VERSIONED:
        try:
            out[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            continue
    return out


def commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
    ).strip()


# --- Render -------------------------------------------------------------------


def _code_list(names: list[str]) -> str:
    return ", ".join(f"`{n}`" for n in names)


def _fmt(v: float) -> str:
    return f"{v:.2f}" if isinstance(v, float) and not v.is_integer() else f"{v:g}"


def _summary(ref: dict) -> tuple[int, int, dict[str, set[str]], dict[str, int]]:
    ops_by_lib: dict[str, set[str]] = defaultdict(set)
    tests_by_lib: dict[str, int] = defaultdict(int)
    for g in ref["groups"]:
        lib = reference_library(g["against"])
        ops_by_lib[lib].update(g["ops"])
        tests_by_lib[lib] += g["passed"]
    ops = {o for g in ref["groups"] for o in g["ops"]}
    return len(ops), sum(tests_by_lib.values()), ops_by_lib, tests_by_lib


def _versions_line(ref: dict) -> str:
    v = ref["versions"]
    return ", ".join(f"{k} {v[k]}" for k in VERSIONED if k in v)


def library_table(ref: dict, *, names: bool) -> str:
    """Tests and ops per reference; ``names`` lists the ops, else counts them."""
    _, _, ops_by_lib, tests_by_lib = _summary(ref)
    order = sorted(ops_by_lib, key=lambda lib: -tests_by_lib[lib])
    lines = ["| Checked against | Tests passing | Ops |", "|---|---:|---|"]
    for lib in order:
        ops = sorted(ops_by_lib[lib])
        cell = _code_list(ops) if names else f"{len(ops)}"
        lines.append(f"| {lib} | {tests_by_lib[lib]:,} | {cell} |")
    return "\n".join(lines)


def end_to_end_table(bench: dict) -> str:
    head = (
        "| Workload (40 COCO images) | Unit | "
        + " | ".join(n for _, n in LIBRARIES)
        + " |"
    )
    lines = [head, "|---|---|" + "---:|" * len(LIBRARIES)]
    for w, what, (mean, worst), unit in WORKLOADS:
        cells = []
        for lib, _ in LIBRARIES:
            d = bench.get(f"{lib}/{w}/path")
            cells.append("n/a" if d is None else f"{_fmt(d[mean])} / {_fmt(d[worst])}")
        lines.append(f"| {what} | {unit} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def readme_block(ref: dict, bench: dict) -> str:
    n_ops, n_tests, ops_by_lib, _ = _summary(ref)
    libs = [lib for lib in ops_by_lib if lib not in _NOT_LIBRARIES]
    return f"""{START}
<!-- Generated by scripts/parity_report.py from the test suite; do not edit. -->
{n_tests:,} reference tests hold {n_ops} ops to an independent implementation
({", ".join(sorted(libs))}) or to a written definition, and all of them pass.
Each test states what it compares and how close the outputs must be, and
the [parity report]({DOCS_URL}) lists every comparison with its tolerance.

{library_table(ref, names=False)}

End to end, the [benchmark](https://github.com/heshamdar/polars-cv/tree/main/benchmarks)
holds every library to PIL's output before timing it. Mean / worst difference
from PIL (0–255 levels, or hash bits):

{end_to_end_table(bench)}

OpenCV's larger gap is its `INTER_LINEAR` resize, which doesn't antialias when
downscaling, not a bug. polars-cv and Daft decode JPEGs with pure-Rust
decoders, so they differ from libjpeg-turbo by a level or two at chroma edges.
{END}"""


def page(ref: dict, bench: dict) -> str:
    n_ops, n_tests, _, _ = _summary(ref)
    rows = [
        "| Ops | Checked against | Tolerance | Tests | Source |",
        "|---|---|---|---:|---|",
    ]
    for g in ref["groups"]:
        src = f"[{Path(g['file']).name}]({REPO_URL}{g['file']})"
        rows.append(
            f"| {_code_list(g['ops'])} | {g['against']} | {g['tolerance']} | "
            f"{g['passed']} | {src} |"
        )
    return f"""# Parity report

<!-- Generated by scripts/parity_report.py; do not edit. -->

polars-cv's outputs are checked against independent implementations: OpenCV,
SciPy, NumPy, PIL, imagehash and pycocotools, or a written definition where
no library computes the same thing. {n_tests:,} tests over {n_ops} ops ran and
passed at commit `{ref["commit"]}` on {ref["generated"]}, with
{_versions_line(ref)}.

Every test declares what it compares and the tolerance it asserts
(`@pytest.mark.parity` in `tests/reference/`). A reference test without a
declaration fails collection, so nothing is checked that this page does not
list, and nothing is listed that is not checked. A declared test that never
runs polars-cv fails, so a reference checked against itself cannot pass as
coverage. CI runs these tests on every push, except the pycocotools check,
which needs a package polars-cv does not depend on (CR-79 in `ISSUES.md`);
this page is regenerated with it installed.

## By reference

{library_table(ref, names=True)}

## End to end against PIL

The [benchmark](https://github.com/heshamdar/polars-cv/tree/main/benchmarks)
runs four real workloads on 40 COCO val2017 images and holds every library to
PIL's output before anything is timed. Mean / worst difference from PIL:

{end_to_end_table(bench)}

Where the differences come from (from `benchmarks/bench/parity.py`):

- **JPEG decoding.** PIL, OpenCV and torchvision use libjpeg-turbo; polars-cv
  and Daft use pure-Rust decoders. The same file decodes a level or two apart
  at chroma edges.
- **Resize.** PIL, torchvision, polars-cv and Daft widen the bilinear filter
  when downscaling. OpenCV's `INTER_LINEAR` doesn't, so it aliases on a 640 to
  224 downscale.
- **Thumbnails** are re-encoded JPEGs, so both sides are lossy.
- **pHash.** polars-cv follows imagehash's algorithm; its grayscale and
  Lanczos resize round differently from Pillow's, so a few bits move. OpenCV's
  `img_hash` is another variant, so its bits are not comparable, but every
  library finds the same duplicate groups.

## Every comparison

{chr(10).join(rows)}

## Reproduce

```bash
cd polars-cv
uv run --no-sync --with pycocotools python scripts/parity_report.py
```

It runs the reference suite, refuses to write anything if a test fails or is
skipped, and regenerates this page and the README summary.
"""


def render(ref: dict, bench: dict) -> tuple[str, str]:
    """The docs page and the README with its parity block replaced."""
    readme = README.read_text()
    head, sep, rest = readme.partition(START)
    _, sep2, tail = rest.partition(END)
    if not (sep and sep2):
        raise SystemExit(f"README.md has no {START} … {END} block")
    return page(ref, bench), head + readme_block(ref, bench) + tail


def load() -> tuple[dict, dict]:
    return json.loads(REFERENCE_JSON.read_text()), json.loads(BENCH_JSON.read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--render", action="store_true", help="re-render from the committed JSON"
    )
    args = ap.parse_args()
    if not args.render:
        ref = {
            "generated": dt.date.today().isoformat(),
            "commit": commit(),
            "versions": versions(),
            "groups": run_suite(),
        }
        REFERENCE_JSON.parent.mkdir(parents=True, exist_ok=True)
        REFERENCE_JSON.write_text(json.dumps(ref, indent=1, ensure_ascii=False) + "\n")
    ref, bench = load()
    doc, readme = render(ref, bench)
    PAGE.write_text(doc)
    README.write_text(readme)
    print(f"wrote {PAGE.relative_to(ROOT)} and README.md", file=sys.stderr)


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
