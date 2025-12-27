#!/usr/bin/env python
"""
Main CLI entry point for running benchmarks.

Usage:
    python -m benchmarks.run_benchmarks [options]

Examples:
    # Run all benchmarks with defaults
    python -m benchmarks.run_benchmarks

    # Run only single operation benchmarks
    python -m benchmarks.run_benchmarks --scenario single_ops

    # Run with specific frameworks
    python -m benchmarks.run_benchmarks --frameworks opencv,pillow

    # Custom image counts and sizes
    python -m benchmarks.run_benchmarks --counts 10,100 --sizes 256,512

    # Output as JSON
    python -m benchmarks.run_benchmarks --output json > results.json
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run polars-vision benchmarks against other frameworks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--scenario",
        type=str,
        choices=["all", "single_ops", "pipelines", "e2e"],
        default="all",
        help="Benchmark scenario to run (default: all)",
    )

    parser.add_argument(
        "--frameworks",
        type=str,
        default=None,
        help=(
            "Comma-separated list of frameworks to benchmark "
            "(default: all available). Options: opencv, pillow, "
            "polars-vision-eager, polars-vision-streaming, "
            "torchvision-cpu, torchvision-mps"
        ),
    )

    parser.add_argument(
        "--counts",
        type=str,
        default="10,100,1000",
        help="Comma-separated list of image counts (default: 10,100,1000)",
    )

    parser.add_argument(
        "--sizes",
        type=str,
        default="256,512",
        help="Comma-separated list of image sizes (default: 256,512)",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup iterations (default: 3)",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of benchmark iterations (default: 10)",
    )

    parser.add_argument(
        "--output",
        type=str,
        choices=["table", "json", "csv"],
        default="table",
        help="Output format (default: table)",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run output validation after benchmarks",
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Tolerance for output validation (default: 1e-5)",
    )

    parser.add_argument(
        "--complexity",
        type=str,
        choices=["light", "medium", "heavy"],
        default=None,
        help="Filter pipeline benchmarks by complexity",
    )

    parser.add_argument(
        "--list-frameworks",
        action="store_true",
        help="List available frameworks and exit",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    return parser.parse_args()


def list_available_frameworks() -> None:
    """List all available framework adapters."""
    from benchmarks.frameworks import get_available_adapters

    print("\nAvailable framework adapters:")
    print("-" * 40)

    adapters = get_available_adapters()
    for adapter in adapters:
        gpu_info = f" (GPU: {adapter.gpu_device})" if adapter.supports_gpu else ""
        print(f"  - {adapter.name}{gpu_info}")

    if not adapters:
        print("  (none available - check dependencies)")

    print()


def get_adapters(framework_names: list[str] | None) -> list:
    """
    Get framework adapters by name or all available.

    Args:
        framework_names: List of framework names or None for all.

    Returns:
        List of framework adapters.
    """
    from benchmarks.frameworks import get_adapter, get_available_adapters

    if framework_names is None:
        return get_available_adapters()

    adapters = []
    for name in framework_names:
        try:
            adapter = get_adapter(name.strip())
            if adapter.is_available():
                adapters.append(adapter)
            else:
                print(f"Warning: {name} is not available (missing dependencies)")
        except ValueError as e:
            print(f"Warning: {e}")

    return adapters


def run_benchmarks(args: argparse.Namespace) -> int:
    """
    Run the benchmarks based on command-line arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    from benchmarks.scenarios.e2e_workflow import run_all_e2e_workflows
    from benchmarks.scenarios.pipelines import run_all_pipelines
    from benchmarks.scenarios.single_ops import run_all_single_ops
    from benchmarks.utils.data_gen import generate_image_bytes
    from benchmarks.utils.results import ResultsCollector
    from benchmarks.utils.validation import OutputValidator

    # Parse configuration
    counts = [int(c.strip()) for c in args.counts.split(",")]
    sizes = [(int(s.strip()), int(s.strip())) for s in args.sizes.split(",")]
    framework_names = args.frameworks.split(",") if args.frameworks else None

    # Get adapters
    adapters = get_adapters(framework_names)

    if not adapters:
        print("Error: No framework adapters available")
        return 1

    if not args.quiet:
        print("\n" + "=" * 60)
        print("POLARS-VISION BENCHMARK SUITE")
        print("=" * 60)
        print(f"\nScenario: {args.scenario}")
        print(f"Frameworks: {', '.join(a.name for a in adapters)}")
        print(f"Image counts: {counts}")
        print(f"Image sizes: {sizes}")
        print(f"Warmup iterations: {args.warmup}")
        print(f"Benchmark iterations: {args.iterations}")
        print()

    # Collect results
    collector = ResultsCollector()

    # Run benchmarks based on scenario
    if args.scenario in ("all", "single_ops"):
        if not args.quiet:
            print("Running single operation benchmarks...")
        results = run_all_single_ops(
            adapters=adapters,
            image_counts=counts,
            image_sizes=sizes,
            warmup_iterations=args.warmup,
            benchmark_iterations=args.iterations,
        )
        collector.add_many(results)

    if args.scenario in ("all", "pipelines"):
        if not args.quiet:
            print("Running pipeline benchmarks...")
        results = run_all_pipelines(
            adapters=adapters,
            image_counts=counts,
            image_sizes=sizes,
            warmup_iterations=args.warmup,
            benchmark_iterations=args.iterations,
            complexity_filter=args.complexity,
        )
        collector.add_many(results)

    if args.scenario in ("all", "e2e"):
        if not args.quiet:
            print("Running end-to-end workflow benchmarks...")
        results = run_all_e2e_workflows(
            adapters=adapters,
            image_counts=counts,
            image_sizes=sizes,
            warmup_iterations=args.warmup,
            benchmark_iterations=args.iterations,
        )
        collector.add_many(results)

    # Output results
    if args.output == "json":
        print(collector.to_json())
    elif args.output == "csv":
        print(collector.to_csv())
    else:
        collector.print_tables()
        collector.print_summary()

    # Run validation if requested
    if args.validate:
        if not args.quiet:
            print("\nRunning output validation...")

        validator = OutputValidator(
            tolerance=args.tolerance,
            reference_framework="opencv",
        )

        # Generate test image
        test_image = generate_image_bytes(256, 256, 3, "gradient")

        # Validate single ops
        from benchmarks.scenarios.single_ops import get_single_op_benchmarks

        for bench in get_single_op_benchmarks():
            validator.validate(
                adapters=adapters,
                test_image_bytes=test_image,
                operations=[bench.params],
                operation_name=bench.name,
            )

        validator.print_summary()

        if not validator.all_passed():
            return 1

    return 0


def main() -> int:
    """
    Main entry point.

    Returns:
        Exit code.
    """
    args = parse_args()

    if args.list_frameworks:
        list_available_frameworks()
        return 0

    try:
        return run_benchmarks(args)
    except KeyboardInterrupt:
        print("\nBenchmark interrupted.")
        return 130
    except Exception as e:
        print(f"Error: {e}")
        if not args.quiet:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
