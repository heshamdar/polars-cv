# Duplication detection: exploration notes

Status: exploratory. Nothing here is wired into `verify.sh`, CI or pre-commit.
Written so the work can be picked up later without re-running it.

## Question

Which duplication tool can find the duplicates this repo cares about, and in
which languages? In particular:

1. The same list declared twice, across Rust and Python, where a human must
   check that the two copies agree (generated copies are fine).
2. Fragmented operations that should be combined, which needs semantic
   understanding rather than textual matching.

## Tool: jscpd 5.4.2

Run through npm, with no install in the repo:

```sh
npx -y jscpd@5.4.2 polars-cv/src view-buffer/src polars-cv/python \
  --format rust,python --min-lines 5 --min-tokens 50 \
  --reporters console,json --output <dir>
```

### Languages

- Language is chosen by file extension, through `--format`. Without a format
  flag it scanned four formats here, including Markdown, which had no clones.
- **It never compares a Rust file with a Python file.** Every clone pair in the
  report had the same extension on both sides (123 Rust-Rust, 90 Python-Python,
  0 cross-language). So question 1 is not answered by this tool at all.

### Results on this tree

| Language | Clones | Duplicated lines | Share of code |
|----------|--------|------------------|---------------|
| Rust (`polars-cv/src`, `view-buffer/src`) | 123 | 1,105 | 1.8% |
| Python (`polars-cv/python`) | 90 | 997 | 4.6% |

- The largest Python clones are in `_ops_generated.py` and
  `_lazy_forwarders.py`. Both are generated (`scripts/gen_ops.py`), so they
  are noise. Excluding them with `--ignore` is untested.
- Clone kind is `exact` for every record. Other kinds (`renamed`, `similar`,
  `semantic`) need extra flags and were not run.

### One confirmed duplicate

`metrics/_metrics/_froc.py` lines 313-333 and `metrics/_metrics/_lroc.py`
lines 270-290 are the same 21 lines (diff is empty). It is the same
join-then-`collapse_scores` block in two metric modules. A shared helper would
remove it. This is a design choice, so it is not done.

### Reliability of the report

- The reported line ranges are often wrong. For the Python clones, the
  matched text (the `fragment` field) appears in both files for 76 of 90
  pairs. For Rust, it does for 96 of 123, with 5 pairs unresolved because the
  basename was ambiguous.
- Most of the unverified pairs are not explained yet. A constant line offset
  does not account for them.
- **Rule for later use:** verify a clone by its `fragment` against the file
  before acting on it. Do not trust the reported line numbers.

## Concern 1: the same list declared in two languages

jscpd cannot see this, for the reason above. What the repo already has:

- Generated declarations are the single authority, per CLAUDE.md. `_ops_generated.py`
  comes from the Rust catalogue; `_dtype_names.py` comes from
  `scripts/gen_dtype_names.py`; `geom_catalog.json` drives the geometry
  accessors. Duplicates that come from these are acceptable.
- Hand-written lists are the risk. Nothing checks them against the other
  language today.

Options, not yet tried:

- A small script that extracts identifier sets from Rust enums and arrays and
  from Python literal lists, then reports overlaps that are not generated. It
  would need an allowlist of intentional overlaps.
- Extend the existing registry tests (the canonical-paths rule) to assert that
  each hand-written list has a single source.

## Concern 2: fragmented operations that should be combined

Textual matching cannot find this. Options:

- jscpd's `semantic` mode (`--semantic`, experimental). It compares embeddings
  of functions and can match across languages. It needs a local model, which
  the tool downloads once. Not run.
- Manual review of the clone list, grouped by module, once the generated code
  is excluded.

## Results: the experimental modes

### jscpd `--semantic`: blocked

The local model (`nomic-ai/CodeRankEmbed`, 548 MB) is hosted on
`huggingface.co`. The egress proxy refused the connection with a policy 403,
recorded in `/__agentproxy/status`. The proxy's README says policy denials are
reported rather than worked around, so it was not retried another way. The
`--semantic-provider http` option needs an embeddings API, which this session
does not have. **Not run.** To try it later, run `npx jscpd --semantic-download`
from an environment that can reach Hugging Face.

### `similarity-rs` 0.5.0: ran

Installed from crates.io (`cargo install similarity-rs --locked`). It compares
function syntax trees with tree-edit distance, so it catches structural
similarity that text matching misses.

```sh
similarity-rs polars-cv/src view-buffer/src --skip-test --threshold 0.85 --min-lines 5
```

- 319 function pairs at 85% or higher: 166 at 85–90%, 122 at 90–95%, 31 at 95%
  or higher.
- The top matches are mostly expected: per-type `clone`, `describe` and `new`
  implementations, and inverse pairs such as `rgb_to_gray` / `gray_to_rgb`.
  Those are not duplicates to remove.
- **A fragmented-operations candidate:** `geom_contour_from_coords` and
  `geom_contour_set_from_coords` in `polars-cv/src/geom_construct.rs` (97%).
  They are the same operation at two nesting depths. They are not safe to merge
  blindly: the set version treats a null ring as an error
  (`polars_bail!(ComputeError: "a ring is null")`), while the single-contour
  version does not. A shared helper would need to keep that difference
  deliberately. This is the semantic judgement the task needs.
- Also worth a look, not yet read: `into_polars_buffer_with_policy` and
  `into_polars_buffer_strided_with_policy` in `view-buffer/src/core/buffer.rs`
  (93%).

## Next steps when picking this up

1. Re-run the command above with `--ignore` for the generated Python files, and
   record the counts again.
2. Run `--semantic` once from an environment that can reach Hugging Face (blocked
   from this session). Check whether it finds the froc/lroc-style cases and
   anything cross-language.
3. Decide on the list-duplication check: a script or an extension of the
   registry tests.
4. Decide thresholds before any gate. Report-only is the safe first step.

## What was not done

- No change was made to the repo's code or CI.
- The raw JSON reports were written to `/tmp` and are not committed.

## Results: ast-grep 0.51.0 (cross-language lists)

Installed with `uvx --from ast-grep-cli ast-grep`, nothing added to the repo.
Two rules matched list and array literals, one per language:

- Python: `kind: list | tuple | set`, on `polars-cv/python`
- Rust: `kind: array_expression`, on `polars-cv/src` and `view-buffer/src`

Then a small script kept lists of at least four string literals and paired
Python with Rust lists that share at least three strings. Scratch files were
in `/tmp/ast_exp` and are not committed.

- 166 Python and 52 Rust string lists, and 21 cross-language pairs that share
  three or more strings.
- Most of the strong pairs are the generated Python op-name list
  (`_ops_generated.py`) against Rust lists of op names. The generated side is
  fine. The Rust side needs a check:
  - `graph/copy_census.rs`, the `KNOWN` table, is a hand-written list of op
    names with expected copy counts. It is guarded both ways: the census fails
    if a listed op did not run, and fails if a run op is missing. So the
    duplication is checked, which is the pattern to copy.
  - The matches in `ops/mod.rs`, `passes.rs` and `plan.rs` are test fixtures
    that name ops. Repeating names there is expected.
- Generic matches, such as `height, width, x, y` in geometry schemas and
  operator tokens, are false positives.

What this shows: the extraction works on both languages, and the cross-language
list question can be answered with a script over ast-grep output. What it
does not do is decide whether a shared list is safe, which still needs a
person or a guard.
