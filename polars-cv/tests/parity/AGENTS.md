# AGENTS.md — Parity suite (`polars-cv/tests/parity/`)

> Read [`tests/AGENTS.md`](../AGENTS.md) first. This file covers the
> generative parity framework only.

## What it is

A Hypothesis-driven framework that checks polars-cv three ways, over a drawn
state space rather than fixed inputs:

| Directory | Evidence | Question |
|-----------|----------|----------|
| `oracle/` | an independent reference (NumPy, OpenCV, Pillow, SciPy) | Is the answer right? |
| `invariance/` | the engine against itself | Does the answer depend on how it was asked? |
| `laws/` | algebraic relations between ops | Do ops compose as they must? |
| `meta/` | the engine's catalogues; committed fixtures | Does the framework cover everything, and can its guards fail? |

`framework/` is the machinery all four share. It holds no tests.

The fixed-input reference tests in `tests/reference/` stay: they pin specific
guarantees. This suite searches around them.

## The state space

| Axis | Where | Values |
|------|-------|--------|
| dtype | `images.DTYPES` | all ten engine dtypes (ratcheted to `cast`'s enum) |
| size, channels, content | `images.image_specs` | 1–48 px (biased to 1, 2, primes), 1–4 channels, noise / smooth / constant / extremes / edges / gradient / half-integer ties |
| rows | `cases.batches` | 1–4 rows, optional nulls, optional mixed sizes |
| source | `io.SOURCES` | PNG, PNG with no declared dtype, TIFF, WebP, JPEG (lossy), file path, `auto` over bytes and lists, `list`, `array`, `raw`, `blob` |
| sink | `io.SINKS` | `numpy`, `ndarray`, `torch` (if installed), `list`, `array`, `blob`, PNG / TIFF / WebP / JPEG, `native` |
| engine | `run.ENGINES` | eager (`DataFrame.select`, pinned to in-memory), lazy in-memory, lazy streaming |
| parameters | `run.PARAM_STYLES` | literal, `pl.col`, `pl.lit`, a computed expression; per-row values (`PerRow`) |
| composition | `run.COMPOSITIONS` | one pipeline, `.pipe()` continuation, every prefix aliased, every step materialized through `blob` |
| optimizer | `run.OPTIMIZATION` | all, none, and each pass in `PASS_NAMES` alone (`only:`) and removed (`without:`) |
| frame layout | `run.Axes.chunked` | one chunk / a morsel boundary mid-column |

Codec sinks decode through OpenCV, not polars-cv (Pillow cannot hold 16-bit
colour). `blob` is the one format only the engine can read or write.

The suite's `conftest.py` defaults every bare collect to the streaming engine
(and CI runs a second in-memory lane). A `DataFrame.select` follows that
default, so the harness pins each engine explicitly (`run._collect`).
Otherwise "eager" would silently mean streaming, which is how two row-mixing
bugs first hid from these tests.

## The reference table (`framework/oracle.py`)

One `OpSpec` per chainable `Pipeline` method. It gives:

- `accepts`: what the op's contract admits.
- `params`: a strategy for valid arguments on a given input.
- `ref`: the reference, which states its output dtype on its own authority.
- `tol`, `gain` and `kind`: the error model.
- `note`: every convention modelled.

Lazy-only binary ops have a `BinarySpec`. An op without a trustworthy
reference sets `ref=None` and says why in `no_ref`. It is still exercised by
`invariance/`.

**Model conventions; never widen a tolerance to absorb one.** Conventions
currently modelled:

- Resize resamples height before width, where Pillow goes width first.
- 2- and 4-channel resizes are alpha-premultiplied.
- Nearest-neighbour ties may go either way. Only the tie pixels are excused,
  computed exactly.
- Rotation is clockwise about `(w/2, h/2)`.
- `convolve2d`'s `"reflect"` repeats the edge pixel, which `pad` calls
  `"symmetric"`.
- `histogram` clamps out-of-range values into the edge bins.
- int→int `cast` wraps.
- `rgb→gray` via `convert_color` keeps alpha, while `grayscale()` drops it.

If a new mismatch is a convention, model it in the reference and write it in
the entry's `note`. If it is a bug, record it (below).

### Why chains need two checks (`framework/tolerance.py`)

`grayscale` is ±1 against OpenCV. Put `threshold` after it and a pixel one
level either side of the cut moves by 255. So a chain is checked two ways
(`checks.ChainChecker`):

1. **Each step** is compared with its reference applied to the engine's *own*
   previous output. The incoming error is zero by construction, so the step's
   own tolerance applies however long the chain is.
2. **The whole chain** is compared with the composed references.
   `tolerance.propagate` carries the bound through each step's `gain` until a
   discontinuous step (gain ∞) or a sparse error through a spatial step makes
   it meaningless.

## Recording a bug you are not fixing (`framework/known.py`)

A generative suite cannot leave a known bug failing: every run would find it
again. So each confirmed divergence is one `Divergence` entry, with:

- a **predicate**, in one of three forms:
  - `affects_axes` for a defect of an execution axis (source, sink, layout,
    parameter style); it also sees the steps, for one that bites only some
    ops;
  - `affects_step` for one op on one kind of input;
  - `affects_chain` for a defect that depends on the steps before it.

  The checks consult it and withhold only that comparison, counted with
  `hypothesis.event`.
- `avoid=True` when the case cannot be carried forward: it raises, or it
  leaves a state such as a 0×0 image. Generators never produce such a case.
- a **repro** that asserts the *correct* behaviour.
  `oracle/test_parity_known_divergences.py` runs it as a strict `xfail`, so a
  fix XPASSes, the suite goes red, and the entry (predicate included) must be
  deleted. That deletion puts the fixed path back under the sweeps.

Confirm a divergence against the running engine with an independent decoder
or reference before adding it, as for `tests/test_known_gaps.py`.

## Lanes and budgets (`framework/budget.py`)

Declare each property once with `@property_lanes(weight=..., **strategies)`
and it runs twice:

- **fast**: collected under its own name in the per-push lane. It uses
  `FAST_EXAMPLES × weight` examples and is **derandomized**, so the same
  examples run every time and `scripts/verify.sh` is reproducible.
- **deep**: registered as `<name>_deep` and marked `slow` (the weekly lane).
  It uses `DEEP_FACTOR` times as many examples with fresh random draws. It is
  the lane that searches. When it finds a failure, understand it, then fix
  it, record it, or pin it.

Two environment variables support local hunting:

- `POLARS_CV_PARITY_SCALE=20` multiplies every budget.
- `POLARS_CV_PARITY_RANDOM=1` makes the fast lane draw fresh examples.

```bash
POLARS_CV_PARITY_RANDOM=1 POLARS_CV_PARITY_SCALE=10 \
  uv run pytest tests/parity -k "not deep" --hypothesis-show-statistics
```

The statistics' `event` lines show what was withheld and why: known
divergences, inputs a reference does not model, planner refusals mid-chain.
A property whose events say it mostly skipped is not testing much.

## Adding to it

- **A new op**: `meta/test_parity_ratchets.py` fails until the op has an
  `OpSpec` or an `EXEMPT` reason. Give it the tightest honest tolerance and a
  `gain`, then run the single-op and chain suites with a large random budget
  before committing.
- **A new source or sink**: the same ratchet, against the I/O catalogue. Add
  an encoder or decoder that does not go through polars-cv wherever a
  third-party one exists.
- **A new axis**: add it to `run.Axes`, `run.execute` and
  `cases.axes_variants`. Every invariance test picks it up.
- **Changing framework logic**: extend `meta/test_parity_framework_fixtures.py`
  with a bad input the logic must reject and a good one it must accept. Watch
  the new fixture fail first.
