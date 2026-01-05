"""
Torchvision framework adapter for benchmarking.

This module provides adapters for torchvision image processing,
supporting both CPU and MPS (Metal) GPU execution.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .base import BaseFrameworkAdapter, OperationParams

if TYPE_CHECKING:
    import numpy.typing as npt
    import torch


class TorchvisionAdapter(BaseFrameworkAdapter):
    """
    Adapter for torchvision image processing.

    Uses torchvision.transforms and torch tensors for image operations.

    Attributes:
        name: Human-readable name of the adapter.
        device: Torch device to use ("cpu", "mps", "cuda").
        supports_gpu: Whether GPU is available and being used.
    """

    def __init__(self, device: str = "cpu") -> None:
        """
        Initialize the torchvision adapter.

        Args:
            device: Device to use ("cpu", "mps", "cuda", "cuda:0", etc.).
        """
        self._requested_device = device
        self._torch: Any = None
        self._transforms: Any = None
        self._F: Any = None  # functional transforms
        self._device: Any = None

        # Set name based on device
        if device == "cpu":
            self.name = "torchvision-cpu"
            self.supports_gpu = False
            self.gpu_device = None
        elif device == "mps":
            self.name = "torchvision-mps"
            self.supports_gpu = True
            self.gpu_device = "mps"
        elif device.startswith("cuda"):
            self.name = f"torchvision-{device}"
            self.supports_gpu = True
            self.gpu_device = device
        else:
            self.name = f"torchvision-{device}"
            self.supports_gpu = False
            self.gpu_device = None

    def is_available(self) -> bool:
        """
        Check if torchvision is available with the requested device.

        Returns:
            True if torchvision can be used with the device, False otherwise.
        """
        try:
            import torch
            import torchvision.transforms  # noqa: F401

            # Check device availability
            if self._requested_device == "mps":
                return torch.backends.mps.is_available()
            elif self._requested_device.startswith("cuda"):
                return torch.cuda.is_available()
            return True  # CPU always available
        except ImportError:
            return False

    def _get_modules(self) -> tuple[Any, Any, Any]:
        """Get torch and torchvision modules."""
        if self._torch is None:
            import torch
            import torchvision.transforms as transforms
            import torchvision.transforms.functional as F

            self._torch = torch
            self._transforms = transforms
            self._F = F

            # Initialize device
            self._device = torch.device(self._requested_device)

        return self._torch, self._transforms, self._F

    def _to_tensor(self, img: Any) -> "torch.Tensor":
        """
        Convert image to torch tensor on the correct device.

        Args:
            img: Image (PIL, numpy, or tensor).

        Returns:
            Torch tensor on the configured device.
        """
        torch, transforms, _ = self._get_modules()

        if isinstance(img, torch.Tensor):
            return img.to(self._device)

        # Use torchvision's to_tensor which handles PIL and numpy
        if isinstance(img, np.ndarray):
            # Ensure uint8 for to_tensor
            if img.dtype != np.uint8:
                img = (img * 255).clip(0, 255).astype(np.uint8)
            # Add batch dimension if needed
            if img.ndim == 2:
                img = img[:, :, np.newaxis]
            # Convert HWC to CHW
            tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        else:
            # PIL Image
            tensor = transforms.functional.to_tensor(img)

        return tensor.to(self._device)

    def _from_tensor(self, tensor: "torch.Tensor") -> "npt.NDArray[np.float32]":
        """
        Convert tensor back to numpy array.

        Args:
            tensor: Torch tensor (C, H, W) format.

        Returns:
            NumPy array (H, W, C) format.
        """
        # Move to CPU if on GPU
        arr = tensor.cpu().numpy()

        # Convert CHW to HWC
        if arr.ndim == 3:
            arr = arr.transpose(1, 2, 0)

        return arr.astype(np.float32)

    def load_from_file(self, path: Path) -> "torch.Tensor":
        """
        Load an image from a file path.

        Args:
            path: Path to the image file.

        Returns:
            Image as torch tensor.
        """
        from PIL import Image

        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self._to_tensor(img)

    def load_from_bytes(self, data: bytes) -> "torch.Tensor":
        """
        Load an image from bytes.

        Args:
            data: Image bytes (PNG, JPEG, etc.).

        Returns:
            Image as torch tensor on the configured device.
        """
        from PIL import Image

        img = Image.open(io.BytesIO(data))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self._to_tensor(img)

    def resize(self, img: "torch.Tensor", height: int, width: int) -> "torch.Tensor":
        """
        Resize an image.

        Args:
            img: Image tensor (C, H, W).
            height: Target height.
            width: Target width.

        Returns:
            Resized image tensor.
        """
        torch, transforms, F = self._get_modules()

        # Use bilinear interpolation for consistency across frameworks
        return F.resize(
            img,
            [height, width],
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        )

    def grayscale(self, img: "torch.Tensor") -> "torch.Tensor":
        """
        Convert image to grayscale.

        Args:
            img: Image tensor (C, H, W).

        Returns:
            Grayscale image tensor (1, H, W).
        """
        _, _, F = self._get_modules()
        return F.rgb_to_grayscale(img)

    def normalize(self, img: "torch.Tensor") -> "torch.Tensor":
        """
        Apply min-max normalization.

        Args:
            img: Image tensor.

        Returns:
            Normalized tensor with values in [0, 1].
        """
        torch, _, _ = self._get_modules()
        min_val = img.min()
        max_val = img.max()
        if max_val - min_val > 0:
            return (img - min_val) / (max_val - min_val)
        return img

    def flip_horizontal(self, img: "torch.Tensor") -> "torch.Tensor":
        """
        Flip image horizontally.

        Args:
            img: Image tensor (C, H, W).

        Returns:
            Horizontally flipped image tensor.
        """
        _, _, F = self._get_modules()
        return F.hflip(img)

    def flip_vertical(self, img: "torch.Tensor") -> "torch.Tensor":
        """
        Flip image vertically.

        Args:
            img: Image tensor (C, H, W).

        Returns:
            Vertically flipped image tensor.
        """
        _, _, F = self._get_modules()
        return F.vflip(img)

    def crop(
        self,
        img: "torch.Tensor",
        top: int,
        left: int,
        height: int,
        width: int,
    ) -> "torch.Tensor":
        """
        Crop image.

        Args:
            img: Image tensor (C, H, W).
            top: Top offset.
            left: Left offset.
            height: Crop height.
            width: Crop width.

        Returns:
            Cropped image tensor.
        """
        _, _, F = self._get_modules()
        return F.crop(img, top, left, height, width)

    def blur(self, img: "torch.Tensor", sigma: float) -> "torch.Tensor":
        """
        Apply Gaussian blur.

        Args:
            img: Image tensor (C, H, W).
            sigma: Blur sigma.

        Returns:
            Blurred image tensor.
        """
        _, _, F = self._get_modules()
        # Kernel size should be odd and related to sigma
        ksize = int(sigma * 6) | 1
        if ksize < 3:
            ksize = 3
        return F.gaussian_blur(img, kernel_size=[ksize, ksize], sigma=sigma)

    def threshold(self, img: "torch.Tensor", value: int) -> "torch.Tensor":
        """
        Apply binary threshold.

        Args:
            img: Image tensor.
            value: Threshold value (0-255, will be scaled to 0-1).

        Returns:
            Thresholded image tensor.
        """
        torch, _, F = self._get_modules()
        # Convert to grayscale if needed
        if img.shape[0] == 3:
            img = F.rgb_to_grayscale(img)

        # Threshold (value is 0-255, tensor is 0-1)
        threshold_val = value / 255.0
        return (img > threshold_val).float()

    def to_numpy(self, img: "torch.Tensor") -> "npt.NDArray[np.float32]":
        """
        Convert tensor to NumPy array.

        Args:
            img: Image tensor.

        Returns:
            NumPy array (H, W, C).
        """
        return self._from_tensor(img)

    def run_pipeline_batch(
        self,
        image_bytes_list: list[bytes],
        operations: list[OperationParams],
    ) -> list["torch.Tensor"]:
        """
        Run a pipeline on a batch of images.

        Args:
            image_bytes_list: List of image bytes.
            operations: Operations to apply.

        Returns:
            List of processed image tensors.
        """
        results = []
        for data in image_bytes_list:
            img = self.load_from_bytes(data)
            for op in operations:
                img = self.apply_operation(img, op)
            results.append(img)
        return results

    def run_pipeline_batch_cold(
        self,
        image_bytes_list: list[bytes],
        operations: list[OperationParams],
    ) -> list["torch.Tensor"]:
        """
        Run pipeline with cold start (includes data transfer time).

        This is the same as run_pipeline_batch but named explicitly
        for benchmark clarity.

        Args:
            image_bytes_list: List of image bytes.
            operations: Operations to apply.

        Returns:
            List of processed image tensors.
        """
        return self.run_pipeline_batch(image_bytes_list, operations)

    def run_pipeline_batch_warm(
        self,
        tensors: list["torch.Tensor"],
        operations: list[OperationParams],
    ) -> list["torch.Tensor"]:
        """
        Run pipeline with warm start (data already on device).

        Args:
            tensors: List of image tensors already on the device.
            operations: Operations to apply.

        Returns:
            List of processed image tensors.
        """
        results = []
        for img in tensors:
            for op in operations:
                img = self.apply_operation(img, op)
            results.append(img)
        return results

    def preload_to_device(self, image_bytes_list: list[bytes]) -> list["torch.Tensor"]:
        """
        Preload images to the device for warm-start benchmarking.

        Args:
            image_bytes_list: List of image bytes.

        Returns:
            List of tensors on the device.
        """
        return [self.load_from_bytes(data) for data in image_bytes_list]

    def synchronize(self) -> None:
        """
        Synchronize GPU operations (wait for completion).

        Important for accurate timing on GPU.
        """
        torch, _, _ = self._get_modules()
        if self._requested_device == "mps":
            torch.mps.synchronize()
        elif self._requested_device.startswith("cuda"):
            torch.cuda.synchronize()


class TorchvisionCPUAdapter(TorchvisionAdapter):
    """Torchvision adapter for CPU execution."""

    def __init__(self) -> None:
        """Initialize CPU adapter."""
        super().__init__(device="cpu")


class TorchvisionMPSAdapter(TorchvisionAdapter):
    """Torchvision adapter for MPS (Metal) GPU execution."""

    def __init__(self) -> None:
        """Initialize MPS adapter."""
        super().__init__(device="mps")


class TorchvisionCUDAAdapter(TorchvisionAdapter):
    """Torchvision adapter for CUDA GPU execution."""

    def __init__(self, device_id: int = 0) -> None:
        """
        Initialize CUDA adapter.

        Args:
            device_id: CUDA device ID.
        """
        super().__init__(device=f"cuda:{device_id}")
