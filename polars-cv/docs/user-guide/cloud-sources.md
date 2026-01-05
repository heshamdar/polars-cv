# Cloud Sources

polars-cv supports reading images directly from cloud storage providers.

## Supported Providers

| Provider | URL Scheme | Environment Variables |
|----------|------------|----------------------|
| Amazon S3 | `s3://` | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` |
| Google Cloud Storage | `gs://` | `GOOGLE_APPLICATION_CREDENTIALS` |
| Azure Blob Storage | `az://`, `abfs://` | `AZURE_STORAGE_ACCOUNT`, `AZURE_STORAGE_ACCESS_KEY` |
| Local Files | `/path/...` or `file://` | None |

## Basic Usage

Use the `file_path` source format:

```python
import polars as pl
from polars_cv import Pipeline

# Pipeline that reads from file paths
pipe = Pipeline().source("file_path").resize(224, 224).sink("numpy")

# DataFrame with cloud paths
df = pl.DataFrame({
    "path": [
        "s3://my-bucket/images/photo1.jpg",
        "gs://my-bucket/images/photo2.png",
        "/local/path/photo3.png",
    ]
})

result = df.with_columns(tensor=pl.col("path").cv.pipeline(pipe))
```

## Authentication

### Default Credential Chain

By default, polars-cv uses the standard credential chain:

1. **Anonymous access** (for public buckets)
2. **Environment variables**
3. **Instance metadata / IAM roles**

### AWS S3

```bash
# Set environment variables
export AWS_ACCESS_KEY_ID=your_key_id
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_REGION=us-east-1

# Or use AWS CLI configuration
aws configure
```

### Google Cloud Storage

```bash
# Set service account key path
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# Or use application default credentials
gcloud auth application-default login
```

### Azure Blob Storage

```bash
export AZURE_STORAGE_ACCOUNT=your_account_name
export AZURE_STORAGE_ACCESS_KEY=your_access_key
```

## Explicit Credentials

For fine-grained control, use `CloudOptions`:

```python
from polars_cv import Pipeline, CloudOptions

# AWS S3 with explicit credentials
s3_options = CloudOptions(
    aws_region="us-east-1",
    aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
    aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
)

pipe = Pipeline().source("file_path", cloud_options=s3_options).sink("numpy")
```

### CloudOptions Fields

| Field | Provider | Description |
|-------|----------|-------------|
| `aws_region` | S3 | AWS region (e.g., "us-east-1") |
| `aws_access_key_id` | S3 | AWS access key ID |
| `aws_secret_access_key` | S3 | AWS secret access key |
| `aws_session_token` | S3 | Session token for temporary credentials |
| `gcs_service_account_key` | GCS | Path to service account JSON file |
| `azure_storage_account` | Azure | Storage account name |
| `azure_storage_access_key` | Azure | Storage access key |
| `anonymous` | All | Force anonymous access (default: auto) |

### Using Dict Syntax

```python
# Alternative: pass options as dict
pipe = Pipeline().source(
    "file_path",
    cloud_options={
        "aws_region": "eu-west-1",
        "anonymous": True,  # Access public bucket
    }
).sink("numpy")
```

## URL Formats

### Amazon S3

```
s3://bucket-name/path/to/object.jpg
```

### Google Cloud Storage

```
gs://bucket-name/path/to/object.jpg
```

### Azure Blob Storage

```
az://container-name/path/to/object.jpg
abfs://container-name/path/to/object.jpg
```

### Local Files

```
/absolute/path/to/file.jpg
relative/path/to/file.jpg
file:///absolute/path/to/file.jpg
```

## Mixing Sources

Process mixed local and cloud paths:

```python
df = pl.DataFrame({
    "source": ["local", "s3", "gcs"],
    "path": [
        "/local/image.jpg",
        "s3://bucket/image.jpg",
        "gs://bucket/image.jpg",
    ]
})

# Same pipeline works for all
pipe = Pipeline().source("file_path").resize(224, 224).sink("numpy")
result = df.with_columns(tensor=pl.col("path").cv.pipeline(pipe))
```

## Error Handling

Cloud read errors are propagated as Polars errors:

```python
try:
    result = df.with_columns(tensor=pl.col("path").cv.pipeline(pipe))
except pl.exceptions.ComputeError as e:
    print(f"Failed to read: {e}")
```

Common errors:

- **Access denied**: Check credentials and bucket policies
- **Not found**: Verify path exists
- **Network error**: Check connectivity

## Performance Tips

1. **Batch Processing**: Process many files in one DataFrame for efficiency
2. **Regional Colocation**: Run compute in same region as storage
3. **Credential Caching**: Credentials are cached per-session
4. **Pre-filter**: Filter DataFrame before processing to avoid unnecessary reads

## Next Steps

- [ML Integration](ml-integration.md) - Use cloud data for training
- [Quickstart](../getting-started/quickstart.md) - Basic usage examples

