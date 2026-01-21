#!/usr/bin/env python3
"""
Test matrix runner for polars-cv.

Tests the library against multiple Python versions using uv to manage
isolated environments dynamically.

Usage:
    python scripts/test_multiple_python.py                    # Use current Python environment
    python scripts/test_multiple_python.py --all              # Test all versions
    python scripts/test_multiple_python.py --versions 3.10 3.13  # Test specific versions
    python scripts/test_multiple_python.py --fast             # Test only min and max versions
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Python versions to test against
DEFAULT_VERSIONS = [
    "3.10",  # Minimum supported version
    "3.11",
    "3.12",
    "3.13",  # Latest version
]


def run_command(cmd: List[str], cwd: Optional[Path] = None) -> Tuple[int, str, str]:
    """Run a command and return exit code, stdout, and stderr."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return 1, "", str(e)


def _test_python_version(version: str, project_root: Path) -> Tuple[bool, str, str]:
    """
    Test the library against a specific Python version using uv.

    Returns:
        (success: bool, output: str, actual_version: str)
    """
    print(f"\n{'=' * 80}")
    print(f"Testing with Python {version}")
    print(f"{'=' * 80}")

    # Check if this is the current Python version (no version specified)
    current_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    use_current_env = version == current_version

    if use_current_env:
        # Use current environment directly
        print(f"Using current Python environment: {current_version}")
        actual_version = current_version

        # Build the package (without cloud feature for speed)
        print("Building package (without cloud feature for speed)...")
        build_cmd = [
            "maturin",
            "develop",
            "--release",
            "--no-default-features",
            "--features",
            "pyo3/extension-module,pyo3/abi3-py39",
        ]
        exit_code, stdout, stderr = run_command(build_cmd, cwd=project_root)
        if exit_code != 0:
            error_msg = f"Failed to build package:\n{stderr}\n{stdout}"
            print(f"❌ {error_msg}")
            return False, error_msg, actual_version

        # Install test dependencies
        print("Installing test dependencies...")
        install_test_cmd = [
            "uv",
            "pip",
            "install",
            "-e",
            ".[dev]",
        ]
        exit_code, stdout, stderr = run_command(install_test_cmd, cwd=project_root)
        if exit_code != 0:
            error_msg = f"Failed to install test dependencies:\n{stderr}"
            print(f"❌ {error_msg}")
            return False, error_msg, actual_version

        # Run pytest
        print(f"Running tests with Python {actual_version}...")
        test_cmd = [
            "pytest",
            "tests/",
            "-v",
            "--tb=short",
        ]
    else:
        # Use uv run with specific Python version (creates isolated env automatically)
        python_spec = f"python{version}"

        # Get the actual Python version being used
        version_cmd = [
            "uv",
            "run",
            f"--python={python_spec}",
            "python",
            "-c",
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
        ]
        exit_code, version_stdout, version_stderr = run_command(
            version_cmd, cwd=project_root
        )
        if exit_code != 0:
            error_msg = f"Python {version} not available. Install with: uv python install {version}"
            print(f"❌ {error_msg}")
            return False, error_msg, version

        actual_version = version_stdout.strip()
        print(f"Using Python version: {actual_version}")

        # Build the package using uv run (installs maturin automatically if needed)
        print("Building package (without cloud feature for speed)...")
        build_cmd = [
            "uv",
            "run",
            f"--python={python_spec}",
            "--with",
            "maturin",
            "maturin",
            "develop",
            "--release",
            "--no-default-features",
            "--features",
            "pyo3/extension-module,pyo3/abi3-py39",
        ]
        exit_code, stdout, stderr = run_command(build_cmd, cwd=project_root)
        if exit_code != 0:
            error_msg = f"Failed to build package:\n{stderr}\n{stdout}"
            print(f"❌ {error_msg}")
            return False, error_msg, actual_version

        # Install test dependencies
        print("Installing test dependencies...")
        install_test_cmd = [
            "uv",
            "run",
            f"--python={python_spec}",
            "uv",
            "pip",
            "install",
            "-e",
            ".[dev]",
        ]
        exit_code, stdout, stderr = run_command(install_test_cmd, cwd=project_root)
        if exit_code != 0:
            error_msg = f"Failed to install test dependencies:\n{stderr}"
            print(f"❌ {error_msg}")
            return False, error_msg, actual_version

        # Run pytest using uv run
        print(f"Running tests with Python {actual_version}...")
        test_cmd = [
            "uv",
            "run",
            f"--python={python_spec}",
            "--with",
            "pytest",
            "pytest",
            "tests/",
            "-v",
            "--tb=short",
        ]

    exit_code, stdout, stderr = run_command(test_cmd, cwd=project_root)

    if exit_code == 0:
        print(f"✅ Python {actual_version}: All tests passed!")
        return True, stdout, actual_version
    else:
        error_msg = f"Tests failed for Python {actual_version}:\n{stderr}\n{stdout}"
        print(f"❌ {error_msg}")
        return False, error_msg, actual_version


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test polars-cv against multiple Python versions"
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        help="Specific Python versions to test (e.g., --versions 3.10 3.13)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Test only minimum and maximum versions (3.10 and 3.13)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Test all default Python versions (3.10, 3.11, 3.12, 3.13)",
    )
    parser.add_argument(
        "--skip-versions",
        nargs="+",
        help="Versions to skip",
        default=[],
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop testing on first failure",
    )

    args = parser.parse_args()

    # Determine which versions to test
    if args.versions:
        versions_to_test = args.versions
    elif args.fast:
        versions_to_test = [DEFAULT_VERSIONS[0], DEFAULT_VERSIONS[-1]]
    elif args.all:
        versions_to_test = DEFAULT_VERSIONS
    else:
        # No arguments passed - use current Python environment
        current_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        print(
            f"No versions specified, using current Python environment: {current_version}"
        )
        versions_to_test = [current_version]

    # Filter out skipped versions
    versions_to_test = [v for v in versions_to_test if v not in args.skip_versions]

    if not versions_to_test:
        print("No versions to test!")
        return 1

    # Get project root (parent of scripts directory)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    if (project_root / "polars-cv").exists():
        project_root = project_root / "polars-cv"

    print(f"Testing polars-cv against {len(versions_to_test)} Python version(s)")
    print(f"Versions: {', '.join(versions_to_test)}")
    print(f"Project root: {project_root}")

    results = {}
    for version in versions_to_test:
        success, output, actual_version = _test_python_version(version, project_root)
        results[version] = (success, output, actual_version)

        if not success and args.stop_on_failure:
            print(f"\n❌ Stopping on first failure (Python {actual_version})")
            break

    # Print summary
    print(f"\n{'=' * 80}")
    print("TEST SUMMARY")
    print(f"{'=' * 80}")

    passed = sum(1 for success, _, _ in results.values() if success)
    failed = len(results) - passed

    for version, (success, _, actual_version) in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"Python {actual_version:6s}: {status}")

    print(f"\nTotal: {len(results)} | Passed: {passed} | Failed: {failed}")

    if failed > 0:
        print("\nFailed versions:")
        for version, (success, output, actual_version) in results.items():
            if not success:
                print(f"\n  Python {actual_version}:")
                # Show first 500 chars of error output
                error_preview = output[:500] if output else "No output"
                print(f"  {error_preview}...")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
