# Installation

## Prerequisites

- Python 3.10+
- Polars 0.46+

## Installing with pip

```bash
pip install polars-vision
```

## Installing with uv

```bash
uv add polars-vision
```

## Installing from Source

If you want to build from source (e.g., for development):

```bash
# Clone the repository
git clone https://github.com/your-org/polars-vision.git
cd polars-vision

# Install with uv (recommended)
uv sync

# Or install with pip in development mode
pip install -e .
```

## Verifying Installation

```python
import polars as pl
from polars_vision import Pipeline

# Create a simple test pipeline
pipe = Pipeline().source("image_bytes").grayscale().sink("png")
print("polars-vision installed successfully!")
```

## Optional Dependencies

### For Cloud Storage (S3, GCS, Azure)

Cloud storage support is built into polars-vision by default. For optimal performance, ensure your cloud credentials are configured:

**AWS S3:**
```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_REGION=us-east-1
```

**Google Cloud Storage:**
```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

**Azure Blob Storage:**
```bash
export AZURE_STORAGE_ACCOUNT=your_account
export AZURE_STORAGE_ACCESS_KEY=your_key
```

### For Documentation (development only)

```bash
pip install polars-vision[docs]
# or
uv add polars-vision --extra docs
```

### For PyTorch Integration

PyTorch is not a required dependency, but polars-vision integrates seamlessly with it:

```python
import torch
from polars_vision import Pipeline, numpy_from_bytes

# Pipeline with torch-compatible output
pipe = Pipeline().source("image_bytes").normalize().sink("torch")

# Convert to PyTorch tensor
tensor = torch.from_numpy(numpy_from_bytes(result))
```

