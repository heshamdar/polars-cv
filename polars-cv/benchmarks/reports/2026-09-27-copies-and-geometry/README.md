# Copies, geometry readers and row splitting: base vs branch

Base: `c7ca10c`. Candidate: `cf87f78` (this branch, including the three
regression fixes the first run found). Both built with `maturin develop
--release` (fat LTO) into one venv, so the harness, polars 1.42.0 and every
other dependency are identical. Machine: 4-core cloud container.

## What was measured

| file | what |
|---|---|
| `t1-base.json`, `t1-head-abf89e0.json` | regression harness, `single_ops,pipelines,e2e,zero_copy`, 1 thread. The head run predates the regression fixes (`abf89e0`); the scenarios they touch are re-measured below |
| `t4-base-a.json`, `t4-base-b.json`, `t4-head.json` | harness, `single_ops,zero_copy,pipelines`, 4 threads; base run twice as a same-binary noise check |
| `t1-zero-copy-{base,head}.json` | harness `zero_copy` at 1 thread, final head |
| `targets-t{1,4}-{base,head}.json` | `targeted.py`: the paths the harness does not cover (encode, list/array sinks, blob source, geometry), 200 × 512² images / 100k geometries, best of 5 |
| `interleaved_*.py` | base and head `.so` swapped into the same venv, alternated, for every result the harness flagged at 4 threads |

## Noise

At 4 threads the harness is not a reliable gate on this container: the same
base binary run twice back to back put 18 of 57 results outside ±7%, up to
±17% (`band.py t4-base-a.json t4-base-b.json`). Every 4-thread flag was
therefore re-measured by alternating the two binaries (`interleaved_*.py`),
and that is the number reported.

## Results

Targeted (final head vs base, milliseconds, lower is better):

| case | base t1 | head t1 | | base t4 | head t4 | |
|---|---:|---:|---:|---:|---:|---:|
| PNG decode → numpy | 652 | 636 | 1.02x | 200 | 193 | 1.04x |
| PNG → PNG | 1448 | 1348 | 1.07x | 433 | 415 | 1.04x |
| JPEG → JPEG | 3035 | 2858 | 1.06x | 836 | 722 | 1.16x |
| PNG → `array` sink | 798 | 764 | 1.05x | 319 | 224 | 1.42x |
| PNG → `list` sink | 1475 | 1426 | 1.03x | 1005 | 897 | 1.12x |
| PNG → transpose → `array` | 1177 | 1072 | 1.10x | 426 | 302 | 1.41x |
| f32 blob → numpy | 564 | 0.9 | 634x | 110 | 1.0 | 105x |
| f32 blob → scale → numpy | 656 | 540 | 1.21x | 128 | 154 | see below |
| `.contour.area` | 141 | 96 | 1.47x | 131 | 44 | 2.98x |
| `.contour.translate` | 254 | 206 | 1.24x | 235 | 142 | 1.65x |
| `.contour.iou` | 1050 | 962 | 1.09x | 1082 | 263 | 4.11x |
| `.contour.contains_point` | 110 | 66 | 1.66x | 110 | 32 | 3.43x |
| `.point.translate` | 5.9 | 4.5 | 1.31x | 5.5 | 2.9 | 1.86x |
| `.point.distance` | 7.7 | 6.9 | 1.12x | 7.7 | 3.0 | 2.52x |
| `.point.distance_to_contour` | 160 | 121 | 1.32x | 156 | 40 | 3.92x |
| `contour` source → rasterize | 152 | 147 | 1.03x | 83 | 83 | 1.00x |

Harness, 1 thread: 40 improved, 19 neutral. Pipelines +4% to +41% (streaming
light/medium/heavy/medical +30/+41/+21/+18%). `zero_copy_blob` (blob source →
numpy → `to_list`) +119% on the final head.

Harness, 4 threads, against the band of both base runs: 21 improved beyond
the band (flips/crop/rotate_90 +20–53%, pipelines +9–23%).

## Flags that did not reproduce

Re-measured interleaved (medians, ms, two or three rounds each):

| flag | base | head |
|---|---|---|
| eager `invert` 4t (harness −21%) | 88 / 72 | 76 / 71 |
| eager `normalize` 4t (harness −19%) | 186 / 167 | 149 / 151 |
| eager grayscale+`threshold` 4t (harness −12%) | 76 / 82 | 75 / 73 |
| eager decode → numpy 4t (harness −9%) | 94 / 87 | 84 / 72 |
| f32 blob → scale → numpy 4t (targeted 0.83x) | 128 / 135 | 93 / 98 |
| streaming `invert` 1t (harness −11%) | 258 / 251 / 257 | 254 / 247 / 253 |
| numpy output → `to_list` 1t (harness −7.5%) | 97 / 93 | 95 / 92 |

## Regressions found and fixed on the way (`cf87f78`)

| cause | cost | fix |
|---|---|---|
| a blob row read in place handed on the column's whole shared data buffer | `zero_copy_blob` −96%: `to_list` of n rows materialised n copies of the column | slice the buffer to the row |
| copying each tensor-sink row once moved the strided gather out of the parallel row phase | transposed `array` sink 0.62x at 4 threads | fill each row's slot on the call's row split |
| the list source's grid check built a lengths column per level per row | `zero_copy_list_*` −17% to −24% | read lengths from the offsets |
