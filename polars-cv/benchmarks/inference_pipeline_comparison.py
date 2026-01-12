"""
Batch Inference Pipeline Comparison: HuggingFace/torchvision vs polars-cv.

This script compares batch preprocessing performance for inference workloads:
1. Standard HuggingFace + torchvision pipeline (with upfront .map() preprocessing)
2. polars-cv batch preprocessing pipeline

Both pipelines use the same PNG source files and produce equivalent outputs
suitable for batch inference or serving.

NOTE: This benchmark focuses on deterministic preprocessing for inference.
For training workloads with random augmentation, polars-cv handles
heavy preprocessing (decode, resize, normalize) while PyTorch handles
per-sample random transforms (flips, rotations, color jitter).
See the ML Integration guide for the recommended hybrid pattern.

Usage:
    python -m benchmarks.inference_pipeline_comparison [OPTIONS]

Options:
    --num-images N      Number of images to generate (default: 1000)
    --image-size S      Image size in pixels (default: 224)
    --num-classes C     Number of classification categories (default: 10)
    --batch-size B      DataLoader batch size (default: 32)
    --num-workers W     DataLoader workers (default: 4)
    --epochs E          Iteration epochs for throughput test (default: 1)
    --skip-serving      Skip the model serving comparison
    --keep-data         Don't delete generated data after benchmark
"""

from __future__ import annotations

import os

os.environ["POLARS_IDEAL_MORSEL_SIZE"] = "10"

import argparse
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    pass

# NOTE ON MULTIPROCESSING:
# Polars uses internal threading which is incompatible with `fork` multiprocessing.
# Using fork after Polars has read files causes deadlocks (inherited mutex locks).
# See: https://docs.pola.rs/user-guide/misc/multiprocessing/
#
# For PyTorch DataLoader with num_workers > 0:
# - On macOS: Uses 'spawn' by default (safe, but slower startup)
# - On Linux: Uses 'fork' by default (CAUSES DEADLOCKS with Polars!)
#
# Workarounds:
# 1. Use num_workers=0 (single process, no forking)
# 2. On Linux, set: torch.multiprocessing.set_start_method('spawn')
# 3. Preprocess data BEFORE creating the DataLoader (what polars-cv does)


def log(message: str, level: str = "INFO") -> None:
    """Log a message with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] [{level}] {message}")


def log_step(step: str) -> None:
    """Log a major step."""
    print()
    log(f"{'=' * 60}", "STEP")
    log(step, "STEP")
    log(f"{'=' * 60}", "STEP")


def log_timing(operation: str, duration: float) -> None:
    """Log a timing result."""
    log(f"⏱️  {operation}: {duration:.3f}s", "TIME")


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    name: str
    batch_preprocessing_time_s: float
    dataloader_throughput_img_per_s: float
    first_batch_latency_s: float
    total_batches: int
    memory_mb: float | None = None


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import psutil

        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


def benchmark_huggingface_pipeline(
    images_dir: Path,
    batch_size: int = 32,
    num_workers: int = 4,
    num_epochs: int = 1,
) -> BenchmarkResult:
    """
    Benchmark HuggingFace + torchvision pipeline with upfront preprocessing.

    This uses HuggingFace datasets with .map(batched=True) for upfront batch
    preprocessing, which is the fair comparison to polars-cv's approach.

    Args:
        images_dir: Directory containing the ImageFolder dataset (class subdirs).
        batch_size: DataLoader batch size.
        num_workers: DataLoader workers.
        num_epochs: Number of epochs to iterate.

    Returns:
        BenchmarkResult with timing information.
    """
    import torch
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from torchvision import transforms

    log("Setting up HuggingFace + torchvision pipeline...")

    # Standard ImageNet-style preprocessing
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def apply_transform(examples: dict[str, Any]) -> dict[str, Any]:
        """Apply transforms to a batch of examples."""
        examples["pixel_values"] = [
            transform(img.convert("RGB")) for img in examples["image"]
        ]
        return examples

    # Load dataset and preprocess ALL images upfront with .map()
    # This is the fair comparison to polars-cv's batch preprocessing
    log(f"Loading and preprocessing dataset from {images_dir}...")
    preprocess_start = time.perf_counter()
    dataset = load_dataset("imagefolder", data_dir=str(images_dir), split="train")
    # Use batched=True with num_proc for parallel preprocessing
    dataset = dataset.map(
        apply_transform,
        batched=True,
        batch_size=100,
        remove_columns=["image"],  # Remove original PIL images to save memory
    )
    dataset.set_format("torch")
    preprocessing_time = time.perf_counter() - preprocess_start
    log_timing("Batch preprocessing", preprocessing_time)
    log(f"  Preprocessed {len(dataset)} images")
    log(f"  Rate: {len(dataset) / preprocessing_time:.1f} images/sec")

    # Create DataLoader
    def collate_fn(batch: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
        """Collate batch of samples."""
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        labels = torch.tensor([item["label"] for item in batch])
        return pixel_values, labels

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Measure first batch latency
    log("Measuring first batch latency...")
    first_batch_start = time.perf_counter()
    first_batch = next(iter(dataloader))
    first_batch_latency = time.perf_counter() - first_batch_start
    log_timing("First batch", first_batch_latency)
    log(f"  Batch shape: {first_batch[0].shape}")
    del first_batch

    # Measure throughput over epochs
    log(f"Measuring DataLoader throughput over {num_epochs} epoch(s)...")
    total_images = 0
    total_batches = 0
    throughput_start = time.perf_counter()

    for epoch in range(num_epochs):
        epoch_start = time.perf_counter()
        epoch_images = 0
        for batch_images, batch_labels in dataloader:
            total_images += batch_images.shape[0]
            epoch_images += batch_images.shape[0]
            total_batches += 1
        epoch_time = time.perf_counter() - epoch_start
        log(
            f"  Epoch {epoch + 1}: {epoch_images} images in {epoch_time:.2f}s "
            f"({epoch_images / epoch_time:.1f} img/s)"
        )

    throughput_time = time.perf_counter() - throughput_start
    throughput = total_images / throughput_time if throughput_time > 0 else 0
    log_timing(f"Total throughput ({total_batches} batches)", throughput_time)
    log(f"  Average: {throughput:.1f} images/sec")

    memory_mb = get_memory_usage_mb()
    log(f"  Memory usage: {memory_mb:.1f} MB")

    return BenchmarkResult(
        name="HuggingFace + torchvision",
        batch_preprocessing_time_s=preprocessing_time,
        dataloader_throughput_img_per_s=throughput,
        first_batch_latency_s=first_batch_latency,
        total_batches=total_batches,
        memory_mb=memory_mb,
    )


def benchmark_polars_cv_pipeline(
    metadata_path: Path,
    batch_size: int = 32,
    num_workers: int = 4,
    num_epochs: int = 1,
    use_streaming: bool = False,
) -> BenchmarkResult:
    """
    Benchmark polars-cv batch preprocessing pipeline.

    This uses polars-cv for batch preprocessing with the file_path source,
    then wraps in a PyTorch Dataset for DataLoader compatibility.

    Args:
        metadata_path: Path to Parquet metadata file with paths and labels.
        batch_size: DataLoader batch size.
        num_workers: DataLoader workers.
        num_epochs: Number of epochs to iterate.
        use_streaming: Use Polars streaming engine.

    Returns:
        BenchmarkResult with timing information.
    """
    import torch
    from polars_cv import IMAGENET_MEAN, IMAGENET_STD, Pipeline, numpy_from_struct
    from torch.utils.data import DataLoader, Dataset

    mode = "streaming" if use_streaming else "eager"
    log(f"Setting up polars-cv pipeline (mode={mode})...")

    # Define preprocessing pipeline
    preprocess_pipe = (
        Pipeline()
        .source("file_path")  # Load from paths in DataFrame
        .resize(height=256, width=256, filter="bilinear")
        .crop(top=16, left=16, height=224, width=224)  # Center crop
        .scale(1 / 255.0)  # [0, 255] -> [0, 1]
        .normalize(method="preset", mean=IMAGENET_MEAN, std=IMAGENET_STD)
        .sink("torch")
    )

    # Load metadata
    log(f"Loading metadata from {metadata_path}...")
    df = pl.read_parquet(metadata_path)
    log(f"  Loaded {len(df)} rows")

    # Preprocess all images
    log("Preprocessing all images with polars-cv...")
    preprocess_start = time.perf_counter()

    if use_streaming:
        log("  Using streaming engine...")
        processed_df = (
            df.lazy()
            .with_columns(tensor=pl.col("path").cv.pipeline(preprocess_pipe))
            .collect(engine="streaming")
        )
    else:
        log("  Using eager engine...")
        processed_df = df.with_columns(
            tensor=pl.col("path").cv.pipeline(preprocess_pipe)
        )

    preprocessing_time = time.perf_counter() - preprocess_start
    log_timing("Batch preprocessing", preprocessing_time)
    log(f"  Preprocessed {len(processed_df)} images")
    log(f"  Rate: {len(processed_df) / preprocessing_time:.1f} images/sec")

    # Create PyTorch Dataset wrapper
    class PreprocessedPolarsDataset(Dataset):
        """PyTorch Dataset backed by preprocessed Polars DataFrame."""

        def __init__(self, df: pl.DataFrame) -> None:
            self.df = df

        def __len__(self) -> int:
            return len(self.df)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
            row = self.df.row(idx, named=True)
            arr = numpy_from_struct(row["tensor"])
            tensor = torch.from_numpy(arr.copy()).permute(2, 0, 1)  # HWC -> CHW
            label = row["label"]
            return tensor, label

    log("Creating PyTorch Dataset wrapper...")
    dataset = PreprocessedPolarsDataset(processed_df)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Measure first batch latency
    log("Measuring first batch latency...")
    first_batch_start = time.perf_counter()
    first_batch = next(iter(dataloader))
    first_batch_latency = time.perf_counter() - first_batch_start
    log_timing("First batch", first_batch_latency)
    log(f"  Batch shape: {first_batch[0].shape}")
    del first_batch

    # Measure throughput over epochs
    log(f"Measuring DataLoader throughput over {num_epochs} epoch(s)...")
    total_images = 0
    total_batches = 0
    throughput_start = time.perf_counter()

    for epoch in range(num_epochs):
        epoch_start = time.perf_counter()
        epoch_images = 0
        for batch_images, batch_labels in dataloader:
            total_images += batch_images.shape[0]
            epoch_images += batch_images.shape[0]
            total_batches += 1
        epoch_time = time.perf_counter() - epoch_start
        log(
            f"  Epoch {epoch + 1}: {epoch_images} images in {epoch_time:.2f}s "
            f"({epoch_images / epoch_time:.1f} img/s)"
        )

    throughput_time = time.perf_counter() - throughput_start
    throughput = total_images / throughput_time if throughput_time > 0 else 0
    log_timing(f"Total throughput ({total_batches} batches)", throughput_time)
    log(f"  Average: {throughput:.1f} images/sec")

    memory_mb = get_memory_usage_mb()
    log(f"  Memory usage: {memory_mb:.1f} MB")

    name = f"polars-cv ({mode})"

    return BenchmarkResult(
        name=name,
        batch_preprocessing_time_s=preprocessing_time,
        dataloader_throughput_img_per_s=throughput,
        first_batch_latency_s=first_batch_latency,
        total_batches=total_batches,
        memory_mb=memory_mb,
    )


def verify_output_equivalence(
    images_dir: Path,
    metadata_path: Path,
    tolerance: float = 1e-5,
) -> bool:
    """
    Verify that both pipelines produce equivalent outputs.

    Note: Some differences are expected due to different resize interpolation
    algorithms between torchvision (PIL-based) and polars-cv (Rust-based).
    Use a tolerance of ~0.1 to account for these implementation differences.

    Args:
        images_dir: Directory with ImageFolder data.
        metadata_path: Path to Parquet metadata.
        tolerance: Maximum allowed difference.

    Returns:
        True if outputs are equivalent within tolerance.
    """
    import torch
    from datasets import load_dataset
    from polars_cv import IMAGENET_MEAN, IMAGENET_STD, Pipeline, numpy_from_struct
    from torchvision import transforms

    log("Verifying output equivalence between pipelines...")
    log("  Note: Minor differences expected due to resize interpolation algorithms")

    # HuggingFace transform
    hf_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # polars-cv pipeline
    pv_pipe = (
        Pipeline()
        .source("file_path")
        .resize(height=256, width=256, filter="bilinear")
        .crop(top=16, left=16, height=224, width=224)
        .scale(1 / 255.0)
        .normalize(method="preset", mean=IMAGENET_MEAN, std=IMAGENET_STD)
        .sink("torch")
    )

    # Load samples from each
    log(f"  Loading HuggingFace dataset from {images_dir}...")
    hf_dataset = load_dataset("imagefolder", data_dir=str(images_dir), split="train")
    log(f"  Loading polars-cv metadata from {metadata_path}...")
    pv_df = pl.read_parquet(metadata_path)

    # Compare first 5 samples
    log("  Comparing first 5 samples...")
    max_diff = 0.0
    for i in range(min(5, len(hf_dataset))):
        # HuggingFace output
        hf_img = hf_dataset[i]["image"].convert("RGB")
        hf_tensor = hf_transform(hf_img)

        # polars-cv output
        sample_df = pv_df.slice(i, 1)
        processed = sample_df.with_columns(tensor=pl.col("path").cv.pipeline(pv_pipe))
        pv_arr = numpy_from_struct(processed["tensor"][0])
        pv_tensor = torch.from_numpy(pv_arr.copy()).permute(2, 0, 1)

        # Compare
        diff = (hf_tensor - pv_tensor).abs().max().item()
        max_diff = max(max_diff, diff)
        log(f"    Sample {i}: max diff = {diff:.6f}")

    log(f"  Maximum difference: {max_diff:.6f}")

    if max_diff < tolerance:
        log(f"✅ Outputs are equivalent (within tolerance {tolerance})")
        return True
    else:
        log(f"⚠️ Outputs differ by {max_diff:.6f} (tolerance: {tolerance})", "WARN")
        log("   This is expected due to resize interpolation differences", "WARN")
        log("   (torchvision uses PIL, polars-cv uses Rust/Lanczos)", "WARN")
        return False


def run_inference_serving(
    images_dir: Path,
    metadata_path: Path,
    batch_size: int = 32,
    num_workers: int = 4,
    num_epochs: int = 1,
    num_classes: int = 10,
) -> dict[str, float]:
    """
    Simulate batch inference serving with both pipelines.

    This simulates a model serving scenario where preprocessed data is fed
    to a model for inference. Unlike training, there is no augmentation
    and preprocessing is deterministic.

    Note: This is a fair comparison since both pipelines:
    1. Preprocess all data upfront
    2. Feed preprocessed batches to the model
    3. Use the same deterministic preprocessing

    Args:
        images_dir: ImageFolder directory.
        metadata_path: Parquet metadata path.
        batch_size: Batch size.
        num_workers: DataLoader workers.
        num_epochs: Number of inference passes.
        num_classes: Number of output classes.

    Returns:
        Dict with inference throughput for each pipeline.
    """
    import torch
    from datasets import load_dataset
    from polars_cv import IMAGENET_MEAN, IMAGENET_STD, Pipeline, numpy_from_struct
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from torchvision.models import resnet18

    log_step(f"INFERENCE SERVING COMPARISON ({num_epochs} passes)")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    log(f"Using device: {device}")

    results: dict[str, float] = {}

    # --- HuggingFace Inference ---
    log_step("INFERENCE: HuggingFace + torchvision")

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def apply_transform(examples: dict) -> dict:
        examples["pixel_values"] = [
            transform(img.convert("RGB")) for img in examples["image"]
        ]
        return examples

    log("Loading and preprocessing HuggingFace dataset...")
    preprocess_start = time.perf_counter()
    hf_dataset = load_dataset("imagefolder", data_dir=str(images_dir), split="train")
    hf_dataset = hf_dataset.map(
        apply_transform,
        batched=True,
        batch_size=100,
        remove_columns=["image"],
    )
    hf_dataset.set_format("torch")
    hf_preprocess_time = time.perf_counter() - preprocess_start
    log_timing("Batch preprocessing", hf_preprocess_time)
    log(f"  Preprocessed {len(hf_dataset)} images")

    def hf_collate(batch: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        labels = torch.tensor([item["label"] for item in batch])
        return pixel_values, labels

    hf_loader = DataLoader(
        hf_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for inference
        num_workers=num_workers,
        collate_fn=hf_collate,
    )

    log("Initializing ResNet18 model...")
    model = resnet18(weights=None, num_classes=num_classes).to(device)
    model.eval()

    log("Starting inference...")
    total_samples = 0
    start_time = time.perf_counter()

    with torch.no_grad():
        for epoch in range(num_epochs):
            epoch_start = time.perf_counter()
            epoch_samples = 0

            for images, labels in hf_loader:
                images = images.to(device)
                _ = model(images)  # Run inference (output intentionally unused)
                total_samples += images.shape[0]
                epoch_samples += images.shape[0]

            epoch_time = time.perf_counter() - epoch_start
            log(
                f"  Pass {epoch + 1}: {epoch_samples} samples, "
                f"time={epoch_time:.2f}s, "
                f"throughput={epoch_samples / epoch_time:.1f} samples/sec"
            )

    hf_time = time.perf_counter() - start_time
    results["huggingface"] = total_samples / hf_time
    log_timing("Total inference time", hf_time)
    log(f"  Final throughput: {results['huggingface']:.1f} samples/sec")

    # --- polars-cv Inference ---
    log_step("INFERENCE: polars-cv")

    pv_pipe = (
        Pipeline()
        .source("file_path")
        .resize(height=256, width=256, filter="bilinear")
        .crop(top=16, left=16, height=224, width=224)
        .scale(1 / 255.0)
        .normalize(method="preset", mean=IMAGENET_MEAN, std=IMAGENET_STD)
        .sink("torch")
    )

    log("Loading and preprocessing with polars-cv...")
    preprocess_start = time.perf_counter()
    df = pl.read_parquet(metadata_path)
    processed_df = df.with_columns(tensor=pl.col("path").cv.pipeline(pv_pipe))
    pv_preprocess_time = time.perf_counter() - preprocess_start
    log_timing("Batch preprocessing", pv_preprocess_time)
    log(f"  Preprocessed {len(processed_df)} images")

    class PVDataset(Dataset):
        def __init__(self, df: pl.DataFrame) -> None:
            self.df = df

        def __len__(self) -> int:
            return len(self.df)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
            row = self.df.row(idx, named=True)
            arr = numpy_from_struct(row["tensor"])
            tensor = torch.from_numpy(arr.copy()).permute(2, 0, 1)
            return tensor, row["label"]

    pv_dataset = PVDataset(processed_df)
    pv_loader = DataLoader(
        pv_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for inference
        num_workers=num_workers,
    )

    log("Initializing ResNet18 model...")
    model = resnet18(weights=None, num_classes=num_classes).to(device)
    model.eval()

    log("Starting inference...")
    total_samples = 0
    start_time = time.perf_counter()

    with torch.no_grad():
        for epoch in range(num_epochs):
            epoch_start = time.perf_counter()
            epoch_samples = 0

            for images, labels in pv_loader:
                images = images.to(device)
                _ = model(images)  # Run inference (output intentionally unused)
                total_samples += images.shape[0]
                epoch_samples += images.shape[0]

            epoch_time = time.perf_counter() - epoch_start
            log(
                f"  Pass {epoch + 1}: {epoch_samples} samples, "
                f"time={epoch_time:.2f}s, "
                f"throughput={epoch_samples / epoch_time:.1f} samples/sec"
            )

    pv_time = time.perf_counter() - start_time
    results["polars_cv"] = total_samples / pv_time
    log_timing("Total inference time", pv_time)
    log(f"  Final throughput: {results['polars_cv']:.1f} samples/sec")

    return results


def print_results(results: list[BenchmarkResult]) -> None:
    """Print benchmark results in a formatted table."""
    print()
    print("=" * 95)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 95)

    # Header
    print(
        f"{'Pipeline':<30} {'Batch Preproc (s)':<18} {'Throughput (img/s)':<20} "
        f"{'1st Batch (s)':<15} {'Memory (MB)':<12}"
    )
    print("-" * 95)

    for r in results:
        memory_str = f"{r.memory_mb:.1f}" if r.memory_mb else "N/A"
        print(
            f"{r.name:<30} {r.batch_preprocessing_time_s:<18.3f} "
            f"{r.dataloader_throughput_img_per_s:<20.1f} "
            f"{r.first_batch_latency_s:<15.3f} {memory_str:<12}"
        )

    print("=" * 95)


def main() -> None:
    """Main entry point for the inference pipeline comparison."""
    parser = argparse.ArgumentParser(
        description="Compare HuggingFace and polars-cv for batch inference preprocessing"
    )
    parser.add_argument(
        "--num-images", type=int, default=1000, help="Number of images to generate"
    )
    parser.add_argument(
        "--image-size", type=int, default=224, help="Image size in pixels"
    )
    parser.add_argument(
        "--num-classes", type=int, default=10, help="Number of categories"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="DataLoader batch size"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (default 0 to avoid fork/Polars deadlocks)",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Inference passes")
    parser.add_argument(
        "--skip-serving", action="store_true", help="Skip inference serving comparison"
    )
    parser.add_argument(
        "--keep-data", action="store_true", help="Don't delete generated data"
    )
    args = parser.parse_args()

    print()
    print("=" * 95)
    print("BATCH INFERENCE PIPELINE COMPARISON: HuggingFace vs polars-cv")
    print("=" * 95)
    print()
    print(
        "This benchmark compares deterministic preprocessing for inference workloads."
    )
    print("For training with random augmentation, see the ML Integration guide.")
    print()
    log("Configuration:")
    log(f"  Images: {args.num_images}")
    log(f"  Size: {args.image_size}x{args.image_size}")
    log(f"  Classes: {args.num_classes}")
    log(f"  Batch size: {args.batch_size}")
    log(f"  Workers: {args.num_workers}")
    log(f"  Inference passes: {args.epochs}")
    log(f"  Skip serving: {args.skip_serving}")

    # Warn about multiprocessing issues
    if args.num_workers > 0:
        import sys

        if sys.platform == "linux":
            log("⚠️  WARNING: num_workers > 0 on Linux may deadlock!", "WARN")
            log("   Linux uses 'fork' which is incompatible with Polars.", "WARN")
            log("   If it hangs, restart with --num-workers 0", "WARN")

    # Generate dataset
    from benchmarks.utils.data_gen import generate_imagefolder_dataset

    log_step("GENERATING SYNTHETIC DATASET")
    gen_start = time.perf_counter()
    dataset = generate_imagefolder_dataset(
        output_dir="./benchmark_data",
        num_images=args.num_images,
        num_classes=args.num_classes,
        height=args.image_size,
        width=args.image_size,
        pattern="mixed",
    )
    gen_time = time.perf_counter() - gen_start
    log_timing("Dataset generation", gen_time)
    log(f"  Created {dataset.image_count} images")
    log(f"  Images directory: {dataset.images_dir}")
    log(f"  Metadata file: {dataset.metadata_path}")

    try:
        # Verify equivalence
        log_step("VERIFYING OUTPUT EQUIVALENCE")
        verify_output_equivalence(
            dataset.images_dir, dataset.metadata_path, tolerance=0.1
        )

        # Run benchmarks
        results: list[BenchmarkResult] = []

        log_step("BENCHMARK: HuggingFace + torchvision (batch preprocessing)")
        hf_result = benchmark_huggingface_pipeline(
            dataset.images_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_epochs=args.epochs,
        )
        results.append(hf_result)

        log_step("BENCHMARK: polars-cv (eager)")
        pv_result = benchmark_polars_cv_pipeline(
            dataset.metadata_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_epochs=args.epochs,
            use_streaming=False,
        )
        results.append(pv_result)

        log_step("BENCHMARK: polars-cv (streaming)")
        pv_streaming_result = benchmark_polars_cv_pipeline(
            dataset.metadata_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_epochs=args.epochs,
            use_streaming=True,
        )
        results.append(pv_streaming_result)

        # Print results
        print_results(results)

        # Inference serving comparison
        if not args.skip_serving:
            serving_results = run_inference_serving(
                dataset.images_dir,
                dataset.metadata_path,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                num_epochs=args.epochs,
                num_classes=args.num_classes,
            )
            print()
            print("=" * 95)
            print("INFERENCE SERVING THROUGHPUT COMPARISON")
            print("=" * 95)
            hf_tp = serving_results["huggingface"]
            pv_tp = serving_results["polars_cv"]
            print(f"  HuggingFace:    {hf_tp:.1f} samples/sec")
            print(f"  polars-cv:  {pv_tp:.1f} samples/sec")
            if hf_tp > 0:
                speedup = pv_tp / hf_tp
                print(f"  Speedup:        {speedup:.2f}x")
            print("=" * 95)

    finally:
        if not args.keep_data:
            log_step("CLEANUP")
            log("Deleting generated data...")
            dataset.cleanup()
            log("  Done")

    print()
    log("✅ Benchmark complete!")
    print()


if __name__ == "__main__":
    main()
    main()
