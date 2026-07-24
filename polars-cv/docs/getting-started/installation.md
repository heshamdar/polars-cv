# Installation

## Prerequisites

- Python 3.10+
- Polars 1.0+

## Installing with pip

```bash
pip install polars-cv
```

## Installing with uv

```bash
uv add polars-cv
```

## Installing from Source

If you want to build from source (e.g., for development):

```bash
# Clone the repository
git clone https://github.com/heshamdar/polars-cv.git
cd polars-cv

# Install with uv (recommended)
uv sync

# Or install with pip in development mode
pip install -e .
```

## Verifying Installation

```python
import polars as pl
from polars_cv import Pipeline

# Create a simple test pipeline
pipe = Pipeline().source("image_bytes").grayscale()
print("polars-cv installed successfully!")
```

## Optional Dependencies

### For Cloud Storage (S3, GCS, Azure)

Cloud storage support is built into polars-cv by default. For optimal performance, ensure your cloud credentials are configured:

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

Instead of (or in addition to) environment variables, credentials can be
passed explicitly with `CloudOptions` on a `file_path` source. Named fields
cover the common cases, and `storage_options` forwards any option straight
through to the underlying `object_store` backend using its native config keys:

```python
import polars_cv as cv

# GCS service-account JSON file path, inline JSON, or an ADC file.
cv.CloudOptions(gcs_service_account_key="/path/to/service-account.json")
cv.CloudOptions(storage_options={"google_service_account_key": inline_json})
cv.CloudOptions(storage_options={"google_application_credentials": "/path/adc.json"})
```

**Federated / brokered Google credentials (workforce & workload identity).**
Application Default Credentials of type `external_account` and
`external_account_authorized_user` (e.g. an OIDC identity from an external
provider, exchanged into Google through a workload/workforce identity pool)
cannot be parsed by `object_store`. polars-cv handles them by delegating to the
`gcloud` CLI, which understands the full federation matrix. When it detects a
federated ADC — from `GOOGLE_APPLICATION_CREDENTIALS`, an explicit
`google_application_credentials` option, or the well-known
`~/.config/gcloud/application_default_credentials.json` — it runs
`gcloud auth application-default print-access-token` and uses the resulting
token (cached until shortly before it expires). This requires the
[`gcloud` CLI](https://cloud.google.com/sdk/docs/install) on `PATH`; set
`POLARS_CV_DISABLE_GCS_FEDERATION=1` to disable it.

To obtain the token another way — a custom broker, a wrapper script, or a
different CLI — point `gcs_token_command` at any shell command that prints a GCS
access token, or supply a pre-obtained token via `gcs_bearer_token`:

```python
import polars_cv as cv

# Any command whose stdout is a GCS access token:
cv.CloudOptions(gcs_token_command="my-broker get-gcs-token --audience ...")

# Or a token you already hold (access tokens are ~1h; supply a fresh one):
cv.CloudOptions(gcs_bearer_token=token)
```

### For Documentation (development only)

When working in the project repository, install the docs dependency group:

```bash
uv sync --group docs
```

### For PyTorch Integration

PyTorch is not a required dependency, but polars-cv integrates seamlessly with it:

```python
import torch
from polars_cv import Pipeline, numpy_from_struct

# Pipeline with torch-compatible output
pipe = Pipeline().source("image_bytes").normalize()

# Convert to PyTorch tensor (result is a struct with data, dtype, shape fields)
tensor = torch.from_numpy(numpy_from_struct(result))
```
