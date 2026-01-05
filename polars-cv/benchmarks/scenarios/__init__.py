"""
Benchmark scenarios for testing vision processing frameworks.

Scenarios include:
- single_ops: Individual operation benchmarks
- pipelines: Chained operation benchmarks
- e2e_workflow: End-to-end file-to-memory workflow
"""

from __future__ import annotations

from .e2e_workflow import get_e2e_workflows, run_all_e2e_workflows
from .pipelines import get_pipeline_benchmarks, run_all_pipelines
from .single_ops import get_single_op_benchmarks, run_all_single_ops

__all__ = [
    "get_e2e_workflows",
    "get_pipeline_benchmarks",
    "get_single_op_benchmarks",
    "run_all_e2e_workflows",
    "run_all_pipelines",
    "run_all_single_ops",
]
