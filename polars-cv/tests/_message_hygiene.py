"""What a user-facing message must not contain, as one checker.

A message is written for the person who built the pipeline. Three things have
reached them that were written for nobody: a Rust ``Debug`` rendering
(``UnsupportedError { format: Unknown, kind: ... }``), a position in the JSON
the builder sends the plugin (``at line 1 column 203``), and a node id made
up at build time (``node_e16ec35b``).

Limits: this reads text, so it knows a ``Debug`` value only by its shape, a
name followed by `` { field:``. A ``Debug`` tuple (``Foo(Bar)``) or a quoted
string's escapes are not recognised, and polars' own trace under "This error
occurred in the following expression" is not the plugin's to fix: pass
:func:`own_words` of the exception.
"""

from __future__ import annotations

import re

#: A Rust ``Debug`` struct: ``Name { field: ...``.
_DEBUG_STRUCT = re.compile(r"\b[A-Z]\w* \{ \w+:")
#: The position ``serde_json`` appends to a parse error.
_JSON_POSITION = re.compile(r"\bline \d+ column \d+")
#: A build-time node id (``lazy._generate_node_id``).
_NODE_ID = re.compile(r"\bnode_[0-9a-f]{8}\b")

RULES = {
    "a Rust Debug value": _DEBUG_STRUCT,
    "a JSON position": _JSON_POSITION,
    "an internal node id": _NODE_ID,
}


def own_words(error: BaseException) -> str:
    """The message without the expression trace polars appends to it."""
    return str(error).split("\n\nThis error occurred")[0]


def leaks(message: str) -> list[str]:
    """Each rule *message* breaks, with the text that breaks it."""
    found = []
    for what, pattern in RULES.items():
        hit = pattern.search(message)
        if hit:
            found.append(f"{what}: {hit.group(0)!r}")
    return found
