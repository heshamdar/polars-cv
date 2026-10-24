# Architecture tooling: investigation notes

Status: exploratory, nothing wired in. Written so the work can be picked up later.

## Question

Can an architecture-rules tool express the module boundaries this repo cares
about with more control than the current guard tests? Scope: Python
(import-linter) and Rust (cargo-deny, module-level tools).

## What the suite already does

- Registry and ratchet tests (`test_guard_registry`, `test_dtype_ratchet_fixtures`,
  `test_join_ratchet`) pin specific shapes of the code.
- `test_sanitation` walks ASTs to enforce conventions such as discovery and lane
  markers.
- The copy census checks its `KNOWN` table in both directions against the registry.

These are strong but bespoke: each rule is a test written for one invariant.

## import-linter (Python): tried

Version: `uvx --from import-linter lint-imports` (installed by uvx, not added to the repo).
Config: an INI file passed with `--config`.

### Works

- **Positive control:** a function-level import of `polars_cv.lazy` inside
  `geometry/` was detected and broke the contract. So the tool sees imports
  inside functions, not only at module level.
- **Stale exceptions are errors.** An `ignore_imports` entry that matches no
  import fails the run (exit 1). That is the same "a checker that matches nothing
  reads as green" failure the repo guards against, handled by the tool.

### Does not work

- **Compiled-module imports are invisible.** `polars_cv._lib` is a `.so`, so
  import-linter's static graph has no node for it. A planted
  `from polars_cv._lib import Plan` in a new module did not break a contract
  forbidding `polars_cv._lib`. This is a real limit: the tool cannot enforce
  "only `_plugin` touches the extension". That rule needs a source scan, as now.
- **Indirect chains are noise.** The package `__init__.py` imports the whole API,
  so almost everything reaches everything. Use `allow_indirect_imports = true`
  for a usable contract (direct imports only).

### Findings from a direct-import layering check

Contract: `polars_cv.metrics` and `polars_cv.geometry` must not import
`polars_cv.pipeline`, `polars_cv.lazy` or `polars_cv._graph`.

- `geometry`: kept.
- `metrics`: broken, three direct imports:
  - `metrics/_matching/_contour.py` imports `polars_cv.lazy` (line 13) and
    `polars_cv.pipeline` (line 14). **Documented:** `metrics/AGENTS.md` says
    matchers accept a pre-decoded `LazyPipelineExpr`, so these are intended.
    They need an explicit exception in the contract.
  - `metrics/_segmentation.py` imports `polars_cv.pipeline` (line 26). **Not
    documented** in `metrics/AGENTS.md`. This is a question for the owner: either
    document the dependency or move the segmentation helper behind a
    `LazyPipelineExpr` the caller passes in.

The point of the run: a contract forces the undocumented dependency into the
open, and the documented exception has to be written down to pass.

## Rust

- **cargo-deny `[bans]` with `wrappers`** restricts which crates may depend on
  a given crate. This is crate-level, not module-level. A Torrust PR uses it for
  layer boundaries. The exact semantics of `wrappers` were not confirmed against
  the official reference, so check the docs before relying on it.
- **archaven** (module-path rules as a Rust test) and **cargo_pup** (rustc-based
  rules) were found by search. Neither was installed or run. archaven is a
  dev-dependency test, so it would fit the existing test style.
- **Orphan-item detection** (cargo-modules) was not found by the search.

## Comparison

| | Current guard tests | import-linter | cargo-deny wrappers (Rust) |
|---|---|---|---|
| Expresses a new rule | Write a test each time | One config entry | Config entry, crate level only |
| Sees compiled modules | Yes (source scan, AST) | No | n/a |
| Stale exceptions | Own checks | Fails the run | Unknown |
| Maintenance | Repo code | External tool | External tool |

Assessment: import-linter is better for declarative Python layering than
bespoke tests, but it is not a replacement. It cannot see the extension, and
it needs `allow_indirect_imports` plus documented exceptions. The current suite
covers things it cannot.

## Next steps when picking this up

1. Decide on `_segmentation -> pipeline`: document it, or move it behind an
   argument.
2. If adopting import-linter, write the contract with the documented exception
   for `_contour`, add it to pre-commit and CI, and keep the `_lib` rule as a
   source scan.
3. For Rust, install cargo-deny's bans check with `wrappers` and test it on a
   throwaway crate before using it on the workspace.
