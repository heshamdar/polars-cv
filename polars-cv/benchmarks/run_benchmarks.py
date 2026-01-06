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
        description="Run polars-cv benchmarks against other frameworks",
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
            "polars-cv-eager, polars-cv-streaming, "
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


def get_adapters(framework_names: list[str] | None, quiet: bool = False) -> list:
    """
    Get framework adapters by name or all available.

    Args:
        framework_names: List of framework names or None for all.
        quiet: Whether to suppress progress output.

    Returns:
        List of framework adapters.
    """
    if not quiet:
        print("  Importing framework adapters (this may take a moment)...", flush=True)

    from benchmarks.frameworks import get_adapter, get_available_adapters

    if not quiet:
        print("  Framework adapters imported.", flush=True)

    if framework_names is None:
        if not quiet:
            print("  Checking adapter availability...", flush=True)
        return get_available_adapters()

    adapters = []
    for name in framework_names:
        if not quiet:
            print(f"  Loading adapter: {name}...", end="", flush=True)
        try:
            adapter = get_adapter(name.strip())
            if adapter.is_available():
                adapters.append(adapter)
                if not quiet:
                    print(" OK", flush=True)
            else:
                if not quiet:
                    print(" UNAVAILABLE (missing dependencies)", flush=True)
                else:
                    print(
                        f"Warning: {name} is not available (missing dependencies)",
                        flush=True,
                    )
        except ValueError as e:
            if not quiet:
                print(f" ERROR: {e}", flush=True)
            else:
                print(f"Warning: {e}", flush=True)

    return adapters


def run_benchmarks(args: argparse.Namespace) -> int:
    """
    Run the benchmarks based on command-line arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    if not args.quiet:
        print("Loading benchmark modules...", flush=True)

    from benchmarks.scenarios.e2e_workflow import run_all_e2e_workflows
    from benchmarks.scenarios.pipelines import run_all_pipelines
    from benchmarks.scenarios.single_ops import run_all_single_ops
    from benchmarks.utils.data_gen import generate_image_bytes
    from benchmarks.utils.results import ResultsCollector
    from benchmarks.utils.validation import OutputValidator

    if not args.quiet:
        print("Modules loaded.", flush=True)

    # Parse configuration
    if not args.quiet:
        print("Parsing configuration...", flush=True)

    counts = [int(c.strip()) for c in args.counts.split(",")]
    sizes = [(int(s.strip()), int(s.strip())) for s in args.sizes.split(",")]
    framework_names = args.frameworks.split(",") if args.frameworks else None

    # Get adapters
    if not args.quiet:
        print("Loading framework adapters...", flush=True)

    adapters = get_adapters(framework_names, quiet=args.quiet)

    if not adapters:
        print("Error: No framework adapters available", flush=True)
        return 1

    if not args.quiet:
        print("\n" + "=" * 60, flush=True)
        print("polars-cv BENCHMARK SUITE", flush=True)
        print("=" * 60, flush=True)
        print(f"\nScenario: {args.scenario}", flush=True)
        print(f"Frameworks: {', '.join(a.name for a in adapters)}", flush=True)
        print(f"Image counts: {counts}", flush=True)
        print(f"Image sizes: {sizes}", flush=True)
        print(f"Warmup iterations: {args.warmup}", flush=True)
        print(f"Benchmark iterations: {args.iterations}", flush=True)
        print(flush=True)

    # Collect results
    collector = ResultsCollector()

    # Run benchmarks based on scenario
    if args.scenario in ("all", "single_ops"):
        if not args.quiet:
            print("\nRunning single operation benchmarks...", flush=True)
        results = run_all_single_ops(
            adapters=adapters,
            image_counts=counts,
            image_sizes=sizes,
            warmup_iterations=args.warmup,
            benchmark_iterations=args.iterations,
            verbose=not args.quiet,
        )
        collector.add_many(results)

    if args.scenario in ("all", "pipelines"):
        if not args.quiet:
            print("\nRunning pipeline benchmarks...", flush=True)
        results = run_all_pipelines(
            adapters=adapters,
            image_counts=counts,
            image_sizes=sizes,
            warmup_iterations=args.warmup,
            benchmark_iterations=args.iterations,
            complexity_filter=args.complexity,
            verbose=not args.quiet,
        )
        collector.add_many(results)

    if args.scenario in ("all", "e2e"):
        if not args.quiet:
            print("\nRunning end-to-end workflow benchmarks...", flush=True)
        results = run_all_e2e_workflows(
            adapters=adapters,
            image_counts=counts,
            image_sizes=sizes,
            warmup_iterations=args.warmup,
            benchmark_iterations=args.iterations,
            verbose=not args.quiet,
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
            reference_framework="pillow",
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
    # Print immediately so user knows script started
    print("Starting polars-cv benchmark suite...", flush=True)

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
