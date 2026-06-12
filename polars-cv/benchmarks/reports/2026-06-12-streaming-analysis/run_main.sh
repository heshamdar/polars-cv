#!/bin/bash
# Comprehensive benchmark run of polars-cv@main vs other frameworks.
set -x
cd /home/user/pcv-main/polars-cv
VENV=.venv/bin/python
FW="polars-cv-eager,polars-cv-streaming,opencv,pillow,torchvision-cpu"
OUT=/home/user/bench-results

# 1. Single ops: 21 ops, counts 10/100, sizes 256/512
$VENV -m benchmarks.run_benchmarks --scenario single_ops \
  --frameworks "$FW" --counts 10,100 --sizes 256,512 \
  --warmup 2 --iterations 5 --output json --quiet \
  > $OUT/main_single_ops.json 2> $OUT/main_single_ops.err

# 2. Pipelines: counts 10/100, sizes 256/512
$VENV -m benchmarks.run_benchmarks --scenario pipelines \
  --frameworks "$FW" --counts 10,100 --sizes 256,512 \
  --warmup 2 --iterations 5 --output json --quiet \
  > $OUT/main_pipelines.json 2> $OUT/main_pipelines.err

# 3. Pipelines at scale: count 1000, size 256 (streaming parallelism territory)
$VENV -m benchmarks.run_benchmarks --scenario pipelines \
  --frameworks "$FW" --counts 1000 --sizes 256 \
  --warmup 1 --iterations 3 --output json --quiet \
  > $OUT/main_pipelines_1000.json 2> $OUT/main_pipelines_1000.err

# 4. E2E (includes decode): counts 10/100, sizes 256/512
$VENV -m benchmarks.run_benchmarks --scenario e2e \
  --frameworks "$FW" --counts 10,100 --sizes 256,512 \
  --warmup 2 --iterations 5 --output json --quiet \
  > $OUT/main_e2e.json 2> $OUT/main_e2e.err
echo DONE
