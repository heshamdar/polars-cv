"""An op's domain contract as docstring prose.

One renderer for every method that states a ``Domain:`` line: the generated
builder methods (``scripts/gen_ops.py``, which loads this file by path) and
the hand-written sugar over them (``Pipeline``'s ``_sugar``), which composes
the contracts of the ops it appends. The contracts themselves are the op
catalogue's ``domains`` (``OP_DOMAINS``), read off each Rust op.

Imports nothing from the package, so the generator can load it before the
modules it writes exist.
"""

from __future__ import annotations

import re
import textwrap
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

#: Docstring section headers a ``Domain:`` paragraph goes before.
SECTION = re.compile(
    r"^\s*(Args|Returns|Raises|Example|Examples|Note|Notes|Warning):\s*$"
)

Case = Mapping[str, Any]


def domain_line(cases: Iterable[Case]) -> str:
    """``Domain: a → b``, from an op's domain cases.

    Transitions the op's base form makes come first; one that needs a
    structural choice (an optional field set, an enum value) follows, after
    the choice that makes it.
    """
    groups: dict[str, list[str]] = {}
    for case in cases:
        key = " / ".join(case.get("with", []))
        groups.setdefault(key, []).append(f"{case['input']} → {case['output']}")
    parts = [
        f"with {key}: {', '.join(arrows)}" if key else ", ".join(arrows)
        for key, arrows in groups.items()
    ]
    return "Domain: " + "; ".join(parts)


def wrap(line: str, width: int) -> list[str]:
    """The line as docstring lines, continuation indented."""
    return textwrap.wrap(line, width=width, subsequent_indent="    ")


def compose(
    op_domains: Mapping[str, Sequence[Case]], ops: Sequence[str]
) -> list[dict[str, str]]:
    """The base transitions of *ops* applied in order: each input the first
    op accepts, carried through every op (a structural choice an op's own
    parameters make is the op's, not the chain's)."""
    first = [c for c in op_domains[ops[0]] if not c.get("with")]
    out = []
    for case in first:
        domain = case["output"]
        for op in ops[1:]:
            step = [
                c for c in op_domains[op] if not c.get("with") and c["input"] == domain
            ]
            if not step:
                msg = f"{op} does not accept the {domain} {ops[0]} produces"
                raise ValueError(msg)
            domain = step[0]["output"]
        out.append({"input": case["input"], "output": domain})
    return out


def with_domain(doc: str, line: str, width: int = 80) -> str:
    """*doc* with *line* as a paragraph before its first section header (or
    at its end), at the docstring's own indentation."""
    lines = doc.split("\n")
    at = next((i for i, text in enumerate(lines) if SECTION.match(text)), None)
    anchor = (
        lines[at]
        if at is not None
        else next((text for text in reversed(lines) if text.strip()), "")
    )
    indent = anchor[: len(anchor) - len(anchor.lstrip())]
    block = [indent + text for text in wrap(line, width - len(indent))]
    if at is None:
        body = doc.rstrip()
        return body + "\n\n" + "\n".join(block) + doc[len(body) :]
    return "\n".join([*lines[:at], *block, "", *lines[at:]])
