"""
Output validation utilities.

This module provides functions for verifying that all frameworks
produce equivalent results within acceptable tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    from benchmarks.frameworks import BaseFrameworkAdapter


@dataclass
class ValidationResult:
    """Result of output validation between frameworks."""

    passed: bool
    reference_framework: str
    compared_frameworks: list[str]
    max_absolute_error: float
    max_relative_error: float
    tolerance_used: float
    failures: list[str]
    details: str
    operation_name: str = "unknown"

    def __repr__(self) -> str:
        """Return string representation."""
        status = "PASS" if self.passed else "FAIL"
        return (
            f"ValidationResult({status}, "
            f"op={self.operation_name}, "
            f"max_abs_err={self.max_absolute_error:.6f}, "
            f"failures={len(self.failures)})"
        )


def normalize_output(
    output: Any,
    adapter: "BaseFrameworkAdapter",
) -> "npt.NDArray[np.float32]":
    """
    Normalize framework output to a comparable NumPy array.

    Args:
        output: Output from a framework (tensor, PIL image, bytes, etc.).
        adapter: The framework adapter that produced the output.

    Returns:
        Normalized NumPy array with float32 dtype and [0, 1] range.
    """
    # Convert to numpy
    arr = adapter.to_numpy(output)

    # Ensure float32
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)

    # Normalize to [0, 1] if needed
    if arr.max() > 1.0:
        arr = arr / 255.0

    # Squeeze trailing singleton dimension (e.g., (H, W, 1) -> (H, W))
    # This aligns grayscale outputs from different frameworks
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr.squeeze(axis=2)

    # Ensure 2D or 3D array
    if arr.ndim == 1:
        # Try to infer shape from size (assume square)
        side = int(np.sqrt(len(arr)))
        if side * side == len(arr):
            arr = arr.reshape(side, side)

    return arr


def is_binary_image(arr: "npt.NDArray[np.float32]") -> bool:
    """
    Check if an array appears to be a binary image (only 0 and 1 values).

    Args:
        arr: Array to check.

    Returns:
        True if the array contains only 0 and 1 values (within tolerance).
    """
    unique_vals = np.unique(arr)
    if len(unique_vals) > 2:
        return False
    # Check if values are approximately 0 and/or 1
    return all(
        np.isclose(v, 0.0, atol=0.01) or np.isclose(v, 1.0, atol=0.01)
        for v in unique_vals
    )


def compare_arrays(
    arr1: "npt.NDArray[np.float32]",
    arr2: "npt.NDArray[np.float32]",
    tolerance: float = 1e-5,
    binary_tolerance: float = 0.01,
) -> tuple[bool, float, float]:
    """
    Compare two arrays for approximate equality.

    Args:
        arr1: First array.
        arr2: Second array.
        tolerance: Maximum allowed relative difference.
        binary_tolerance: For binary images, max fraction of differing pixels allowed.

    Returns:
        Tuple of (is_equal, max_absolute_error, max_relative_error).
    """
    # Check shapes match
    if arr1.shape != arr2.shape:
        # Try to make shapes compatible
        if arr1.ndim != arr2.ndim:
            # Flatten and compare lengths
            if arr1.size != arr2.size:
                return False, float("inf"), float("inf")
            arr1 = arr1.flatten()
            arr2 = arr2.flatten()

    # Compute absolute error
    abs_error = np.abs(arr1 - arr2)
    max_abs_error = float(abs_error.max())

    # Compute relative error (avoid division by zero)
    denom = np.maximum(np.abs(arr1), np.abs(arr2))
    denom = np.where(denom < 1e-10, 1.0, denom)
    rel_error = abs_error / denom
    max_rel_error = float(rel_error.max())

    # For binary images, use fraction of differing pixels instead of max error
    # This is because grayscale boundary differences cause individual pixels
    # to differ by 1.0, but the overall image is still correct
    if is_binary_image(arr1) and is_binary_image(arr2):
        num_diff = np.sum(abs_error > 0.5)
        diff_fraction = num_diff / arr1.size
        is_equal = diff_fraction <= binary_tolerance
        # Report the fraction as the error for binary images
        max_abs_error = diff_fraction
        max_rel_error = diff_fraction
    else:
        # Check if within tolerance
        is_equal = np.allclose(arr1, arr2, rtol=tolerance, atol=tolerance)

    return is_equal, max_abs_error, max_rel_error


def validate_outputs(
    outputs: dict[str, "npt.NDArray[np.float32]"],
    reference_name: str = "opencv",
    tolerance: float = 1e-5,
) -> ValidationResult:
    """
    Validate that all framework outputs are equivalent.

    Args:
        outputs: Dictionary mapping framework names to output arrays.
        reference_name: Name of the framework to use as reference.
        tolerance: Maximum allowed relative difference.

    Returns:
        ValidationResult with comparison details.
    """
    if reference_name not in outputs:
        # Use first available as reference
        reference_name = next(iter(outputs.keys()))

    reference = outputs[reference_name]
    failures: list[str] = []
    max_abs_error = 0.0
    max_rel_error = 0.0
    compared: list[str] = []

    for name, arr in outputs.items():
        if name == reference_name:
            continue

        compared.append(name)

        try:
            is_equal, abs_err, rel_err = compare_arrays(reference, arr, tolerance)
            max_abs_error = max(max_abs_error, abs_err)
            max_rel_error = max(max_rel_error, rel_err)

            if not is_equal:
                failures.append(
                    f"{name}: max_abs_err={abs_err:.6f}, max_rel_err={rel_err:.6f}"
                )
        except Exception as e:
            failures.append(f"{name}: comparison failed - {e}")
            max_abs_error = float("inf")
            max_rel_error = float("inf")

    passed = len(failures) == 0
    details = "All outputs match within tolerance" if passed else "; ".join(failures)

    return ValidationResult(
        passed=passed,
        reference_framework=reference_name,
        compared_frameworks=compared,
        max_absolute_error=max_abs_error,
        max_relative_error=max_rel_error,
        tolerance_used=tolerance,
        failures=failures,
        details=details,
    )


def validate_framework_outputs(
    adapters: list["BaseFrameworkAdapter"],
    test_image_bytes: bytes,
    operations: list[Any],  # OperationParams
    tolerance: float = 1e-5,
    reference_adapter_name: str = "opencv",
) -> ValidationResult:
    """
    Run the same operations on all adapters and validate outputs match.

    Args:
        adapters: List of framework adapters to validate.
        test_image_bytes: Test image as bytes.
        operations: List of operations to apply.
        tolerance: Maximum allowed relative difference.
        reference_adapter_name: Name of adapter to use as reference.

    Returns:
        ValidationResult with comparison details.
    """
    outputs: dict[str, npt.NDArray[np.float32]] = {}

    for adapter in adapters:
        if not adapter.is_available():
            continue

        try:
            # Run pipeline
            results = adapter.run_pipeline_batch([test_image_bytes], operations)

            if results:
                # Normalize output
                arr = normalize_output(results[0], adapter)
                outputs[adapter.name] = arr
        except Exception as e:
            print(f"Warning: {adapter.name} failed: {e}")

    if len(outputs) < 2:
        return ValidationResult(
            passed=True,
            reference_framework=reference_adapter_name,
            compared_frameworks=[],
            max_absolute_error=0.0,
            max_relative_error=0.0,
            tolerance_used=tolerance,
            failures=[],
            details="Not enough outputs to compare",
        )

    return validate_outputs(
        outputs,
        reference_name=reference_adapter_name,
        tolerance=tolerance,
    )


def print_validation_result(result: ValidationResult) -> None:
    """
    Print a formatted validation result.

    Args:
        result: Validation result to print.
    """
    status = "PASS" if result.passed else "FAIL"
    color = "\033[92m" if result.passed else "\033[91m"
    reset = "\033[0m"

    print(f"\n{color}Output Validation: {status}{reset}")
    print(f"  Reference: {result.reference_framework}")
    print(f"  Compared: {', '.join(result.compared_frameworks)}")
    print(f"  Tolerance: {result.tolerance_used:.1e}")
    print(f"  Max absolute error: {result.max_absolute_error:.6f}")
    print(f"  Max relative error: {result.max_relative_error:.6f}")

    if not result.passed:
        print("  Failures:")
        for failure in result.failures:
            print(f"    - {failure}")


class OutputValidator:
    """
    Validator for checking output consistency across frameworks.
    """

    def __init__(
        self,
        tolerance: float = 1e-5,
        reference_framework: str = "opencv",
    ) -> None:
        """
        Initialize the validator.

        Args:
            tolerance: Maximum allowed relative difference.
            reference_framework: Name of framework to use as reference.
        """
        self.tolerance = tolerance
        self.reference_framework = reference_framework
        self.validation_results: list[ValidationResult] = []

    def validate(
        self,
        adapters: list["BaseFrameworkAdapter"],
        test_image_bytes: bytes,
        operations: list[Any],
        operation_name: str = "unknown",
    ) -> ValidationResult:
        """
        Validate outputs from all adapters.

        Args:
            adapters: List of framework adapters.
            test_image_bytes: Test image as bytes.
            operations: Operations to apply.
            operation_name: Name of the operation for reporting.

        Returns:
            ValidationResult.
        """
        result = validate_framework_outputs(
            adapters=adapters,
            test_image_bytes=test_image_bytes,
            operations=operations,
            tolerance=self.tolerance,
            reference_adapter_name=self.reference_framework,
        )

        # Store the operation name in the result
        result.operation_name = operation_name

        self.validation_results.append(result)
        return result

    def print_summary(self) -> None:
        """Print summary of all validation results."""
        total = len(self.validation_results)
        passed = sum(1 for r in self.validation_results if r.passed)
        failed = total - passed

        print("\n" + "=" * 50)
        print("VALIDATION SUMMARY")
        print("=" * 50)
        print(f"Reference framework: {self.reference_framework}")
        print(f"Tolerance: {self.tolerance:.1e}")
        print(f"Total validations: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")

        if failed > 0:
            print("\nFailed validations:")
            for result in self.validation_results:
                if not result.passed:
                    print(f"  [{result.operation_name}] {result.details}")

        print("=" * 50)

    def all_passed(self) -> bool:
        """
        Check if all validations passed.

        Returns:
            True if all validations passed.
        """
        return all(r.passed for r in self.validation_results)
