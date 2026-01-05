//! Cloud storage abstraction for file_path source.
//!
//! This module provides support for reading files from:
//! - Local filesystem (file:// or absolute/relative paths)
//! - Amazon S3 (s3://)
//! - Google Cloud Storage (gs://)
//! - Azure Blob Storage (az:// or abfs://)
//!
//! Credentials are resolved using the default chain:
//! 1. Anonymous access (for public buckets)
//! 2. Environment variables (AWS_ACCESS_KEY_ID, GOOGLE_APPLICATION_CREDENTIALS, etc.)
//! 3. Instance metadata / IAM roles

use object_store::aws::AmazonS3Builder;
use object_store::azure::MicrosoftAzureBuilder;
use object_store::gcp::GoogleCloudStorageBuilder;
use object_store::path::Path as ObjectPath;
use object_store::ObjectStore;
use std::collections::HashMap;
use std::path::Path;
use thiserror::Error;
use tokio::runtime::Runtime;
use url::Url;

/// Errors that can occur during cloud file operations.
#[derive(Error, Debug)]
pub enum CloudError {
    #[error("Failed to parse URL: {0}")]
    UrlParse(String),

    #[error("Unsupported URL scheme: {0}")]
    UnsupportedScheme(String),

    #[error("Failed to read file: {0}")]
    ReadError(String),

    #[error("Failed to build object store: {0}")]
    StoreError(String),

    #[error("Failed to create runtime: {0}")]
    RuntimeError(String),
}

/// Cloud storage options for explicit credential configuration.
#[derive(Debug, Clone, Default)]
pub struct CloudOptions {
    /// AWS region (e.g., "us-east-1")
    pub aws_region: Option<String>,
    /// AWS access key ID
    pub aws_access_key_id: Option<String>,
    /// AWS secret access key
    pub aws_secret_access_key: Option<String>,
    /// AWS session token (for temporary credentials)
    pub aws_session_token: Option<String>,
    /// GCS service account key path
    pub gcs_service_account_key: Option<String>,
    /// Azure storage account name
    pub azure_storage_account: Option<String>,
    /// Azure storage access key
    pub azure_storage_access_key: Option<String>,
    /// Whether to allow anonymous access (default: true for first attempt)
    pub anonymous: Option<bool>,
}

impl CloudOptions {
    /// Create options from a HashMap (for Python interop).
    #[allow(dead_code)]
    pub fn from_map(map: &HashMap<String, String>) -> Self {
        CloudOptions {
            aws_region: map.get("aws_region").cloned(),
            aws_access_key_id: map.get("aws_access_key_id").cloned(),
            aws_secret_access_key: map.get("aws_secret_access_key").cloned(),
            aws_session_token: map.get("aws_session_token").cloned(),
            gcs_service_account_key: map.get("gcs_service_account_key").cloned(),
            azure_storage_account: map.get("azure_storage_account").cloned(),
            azure_storage_access_key: map.get("azure_storage_access_key").cloned(),
            anonymous: map.get("anonymous").map(|s| s == "true"),
        }
    }
}

/// Read a file from a path (local or cloud).
///
/// # Arguments
/// * `path` - The file path (local path, or URL like s3://, gs://, az://)
/// * `options` - Optional cloud configuration
///
/// # Returns
/// The file contents as bytes.
pub fn read_file(path: &str, options: Option<&CloudOptions>) -> Result<Vec<u8>, CloudError> {
    // Try to parse as URL first
    if let Ok(url) = Url::parse(path) {
        match url.scheme() {
            "file" => read_local_file(url.path()),
            "s3" => read_s3(&url, options),
            "gs" => read_gcs(&url, options),
            "az" | "abfs" | "abfss" => read_azure(&url, options),
            scheme => Err(CloudError::UnsupportedScheme(scheme.to_string())),
        }
    } else {
        // Not a valid URL, treat as local path
        read_local_file(path)
    }
}

/// Read a file from the local filesystem.
fn read_local_file(path: &str) -> Result<Vec<u8>, CloudError> {
    std::fs::read(Path::new(path)).map_err(|e| CloudError::ReadError(e.to_string()))
}

/// Create a tokio runtime for async operations.
fn create_runtime() -> Result<Runtime, CloudError> {
    Runtime::new().map_err(|e| CloudError::RuntimeError(e.to_string()))
}

/// Read a file from Amazon S3.
fn read_s3(url: &Url, options: Option<&CloudOptions>) -> Result<Vec<u8>, CloudError> {
    let bucket = url
        .host_str()
        .ok_or_else(|| CloudError::UrlParse("Missing bucket name in S3 URL".to_string()))?;
    let key = url.path().trim_start_matches('/');

    let runtime = create_runtime()?;

    // Build S3 client with credentials
    let mut builder = AmazonS3Builder::new().with_bucket_name(bucket);

    if let Some(opts) = options {
        if let Some(region) = &opts.aws_region {
            builder = builder.with_region(region);
        }
        if let Some(key_id) = &opts.aws_access_key_id {
            builder = builder.with_access_key_id(key_id);
        }
        if let Some(secret) = &opts.aws_secret_access_key {
            builder = builder.with_secret_access_key(secret);
        }
        if let Some(token) = &opts.aws_session_token {
            builder = builder.with_token(token);
        }
        if opts.anonymous == Some(true) {
            builder = builder.with_skip_signature(true);
        }
    }

    // Try with default credentials from environment
    let store = builder
        .build()
        .map_err(|e| CloudError::StoreError(e.to_string()))?;

    let path = ObjectPath::from(key);
    runtime.block_on(async {
        let result = store.get(&path).await;
        match result {
            Ok(get_result) => get_result
                .bytes()
                .await
                .map(|b| b.to_vec())
                .map_err(|e| CloudError::ReadError(e.to_string())),
            Err(e) => Err(CloudError::ReadError(e.to_string())),
        }
    })
}

/// Read a file from Google Cloud Storage.
fn read_gcs(url: &Url, options: Option<&CloudOptions>) -> Result<Vec<u8>, CloudError> {
    let bucket = url
        .host_str()
        .ok_or_else(|| CloudError::UrlParse("Missing bucket name in GCS URL".to_string()))?;
    let key = url.path().trim_start_matches('/');

    let runtime = create_runtime()?;

    // Build GCS client
    let mut builder = GoogleCloudStorageBuilder::new().with_bucket_name(bucket);

    if let Some(opts) = options {
        if let Some(key_path) = &opts.gcs_service_account_key {
            builder = builder.with_service_account_path(key_path);
        }
        // Note: GCS builder doesn't have anonymous_credentials method in this version
        // Anonymous access is the default when no credentials are provided
    }

    let store = builder
        .build()
        .map_err(|e| CloudError::StoreError(e.to_string()))?;

    let path = ObjectPath::from(key);
    runtime.block_on(async {
        let result = store.get(&path).await;
        match result {
            Ok(get_result) => get_result
                .bytes()
                .await
                .map(|b| b.to_vec())
                .map_err(|e| CloudError::ReadError(e.to_string())),
            Err(e) => Err(CloudError::ReadError(e.to_string())),
        }
    })
}

/// Read a file from Azure Blob Storage.
fn read_azure(url: &Url, options: Option<&CloudOptions>) -> Result<Vec<u8>, CloudError> {
    // Azure URLs: az://container/path or abfs://container@account.dfs.core.windows.net/path
    let container = url
        .host_str()
        .ok_or_else(|| CloudError::UrlParse("Missing container name in Azure URL".to_string()))?;
    let key = url.path().trim_start_matches('/');

    let runtime = create_runtime()?;

    // Build Azure client
    let mut builder = MicrosoftAzureBuilder::new().with_container_name(container);

    if let Some(opts) = options {
        if let Some(account) = &opts.azure_storage_account {
            builder = builder.with_account(account);
        }
        if let Some(key) = &opts.azure_storage_access_key {
            builder = builder.with_access_key(key);
        }
        if opts.anonymous == Some(true) {
            builder = builder.with_skip_signature(true);
        }
    }

    let store = builder
        .build()
        .map_err(|e| CloudError::StoreError(e.to_string()))?;

    let path = ObjectPath::from(key);
    runtime.block_on(async {
        let result = store.get(&path).await;
        match result {
            Ok(get_result) => get_result
                .bytes()
                .await
                .map(|b| b.to_vec())
                .map_err(|e| CloudError::ReadError(e.to_string())),
            Err(e) => Err(CloudError::ReadError(e.to_string())),
        }
    })
}

/// Check if a path is a cloud URL.
#[allow(dead_code)]
pub fn is_cloud_path(path: &str) -> bool {
    if let Ok(url) = Url::parse(path) {
        matches!(url.scheme(), "s3" | "gs" | "az" | "abfs" | "abfss")
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_cloud_path() {
        assert!(is_cloud_path("s3://bucket/key"));
        assert!(is_cloud_path("gs://bucket/key"));
        assert!(is_cloud_path("az://container/path"));
        assert!(is_cloud_path("abfs://container/path"));
        assert!(!is_cloud_path("/local/path/file.png"));
        assert!(!is_cloud_path("relative/path.png"));
        assert!(!is_cloud_path("file:///local/path.png"));
    }

    #[test]
    fn test_read_local_file() {
        // Create a temp file for testing
        let temp_dir = std::env::temp_dir();
        let test_path = temp_dir.join("polars_cv_test.txt");
        std::fs::write(&test_path, b"test content").unwrap();

        let result = read_file(test_path.to_str().unwrap(), None);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), b"test content");

        // Cleanup
        std::fs::remove_file(test_path).ok();
    }
}
