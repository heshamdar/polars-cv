#!/bin/bash
# Same harness (main's benchmarks/), branch-built polars_cv. polars-cv frameworks
# only — opencv/pillow/torchvision numbers are identical to the main run.
set -x
cd /home/user/pcv-main/polars-cv
VENV=/home/user/polars-cv/polars-cv/.venv/bin/python
FW="polars-cv-eager,polars-cv-streaming"
OUT=/home/user/bench-results

$VENV -m benchmarks.run_benchmarks --scenario single_ops \
  --frameworks "$FW" --counts 10,100 --sizes 256,512 \
  --warmup 2 --iterations 5 --output json --quiet \
  > $OUT/branch_single_ops.json 2> $OUT/branch_single_ops.err

$VENV -m benchmarks.run_benchmarks --scenario pipelines \
  --frameworks "$FW" --counts 10,100 --sizes 256,512 \
  --warmup 2 --iterations 5 --output json --quiet \
  > $OUT/branch_pipelines.json 2> $OUT/branch_pipelines.err

$VENV -m benchmarks.run_benchmarks --scenario pipelines \
  --frameworks "$FW" --counts 1000 --sizes 256 \
  --warmup 1 --iterations 3 --output json --quiet \
  > $OUT/branch_pipelines_1000.json 2> $OUT/branch_pipelines_1000.err

$VENV -m benchmarks.run_benchmarks --scenario e2e \
  --frameworks "$FW" --counts 10,100 --sizes 256,512 \
  --warmup 2 --iterations 5 --output json --quiet \
  > $OUT/branch_e2e.json 2> $OUT/branch_e2e.err
echo DONE
