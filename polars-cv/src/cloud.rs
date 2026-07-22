//! Cloud storage abstraction for file_path source.
//!
//! This module provides support for reading files from:
//! - Local filesystem (file:// or absolute/relative paths)
//! - Amazon S3 (s3://)
//! - Google Cloud Storage (gs://)
//! - Azure Blob Storage (az:// or abfs://)
//! - HTTP/HTTPS URLs (http://, https://)
//!
//! Credentials are resolved using the default chain:
//! 1. Explicit options passed via `CloudOptions` (see below)
//! 2. Environment variables (AWS_ACCESS_KEY_ID, GOOGLE_APPLICATION_CREDENTIALS, etc.)
//! 3. Instance metadata / IAM roles
//!
//! `CloudOptions.config` is a generic pass-through map keyed by
//! `object_store`'s own configuration keys (e.g. `aws_region`,
//! `google_service_account`, `google_application_credentials`,
//! `azure_storage_account_name`). Each entry is forwarded verbatim to the
//! backend builder via `with_config`, so any option the underlying
//! `object_store` backend understands is available without bespoke plumbing.
//!
//! Two keys are reserved and handled explicitly rather than passed through:
//! - `anonymous` → skip request signing for public buckets (S3, GCS, Azure).
//! - `bearer_token` → install a pre-obtained OAuth access token as a static
//!   GCS credential. This is the escape hatch for credential types
//!   `object_store` cannot parse natively (e.g. workforce/federated
//!   `external_account_authorized_user` Application Default Credentials): mint
//!   a token out of band and hand it over directly.

use object_store::aws::{AmazonS3Builder, AmazonS3ConfigKey};
use object_store::azure::{AzureConfigKey, MicrosoftAzureBuilder};
use object_store::gcp::{GcpCredential, GoogleCloudStorageBuilder, GoogleConfigKey};
use object_store::path::Path as ObjectPath;
use object_store::{ObjectStore, StaticCredentialProvider};
use std::collections::HashMap;
use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;
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
///
/// `config` holds pass-through options keyed by `object_store`'s native
/// configuration keys; the reserved `anonymous` and `bearer_token` inputs are
/// lifted out into their own fields (see the module docs).
#[derive(Debug, Clone, Default)]
pub struct CloudOptions {
    /// Options forwarded verbatim to the backend builder via `with_config`,
    /// keyed by `object_store`'s native config keys.
    pub config: HashMap<String, String>,
    /// Pre-obtained OAuth bearer token, installed as a static GCS credential.
    pub bearer_token: Option<String>,
    /// Skip request signing for public buckets (opt-in; default: signed
    /// requests using the credential chain).
    pub anonymous: Option<bool>,
}

impl CloudOptions {
    /// Create options from the wire map (string key/value pairs from Python).
    ///
    /// Most keys pass straight through to `config` using `object_store`'s
    /// native names. Two keys are reserved (`anonymous`, `bearer_token`), and a
    /// handful of historical polars-cv field names are translated to their
    /// canonical `object_store` equivalents for backwards compatibility. When a
    /// legacy name and its canonical name are both present, the explicit
    /// canonical value wins.
    pub fn from_map(map: &HashMap<String, String>) -> Self {
        let mut config: HashMap<String, String> = HashMap::new();

        // Pass 1: legacy field names -> canonical keys (lower precedence).
        for (k, v) in map {
            let canonical = match k.as_str() {
                // Historically a *path* to a service-account JSON file.
                "gcs_service_account_key" => "google_service_account",
                "azure_storage_account" => "azure_storage_account_name",
                "azure_storage_access_key" => "azure_storage_account_key",
                _ => continue,
            };
            config.insert(canonical.to_string(), v.clone());
        }

        // Pass 2: reserved keys + verbatim pass-through (higher precedence, so
        // an explicit canonical key overrides a legacy alias from pass 1).
        let mut bearer_token = None;
        let mut anonymous = None;
        for (k, v) in map {
            match k.as_str() {
                "anonymous" => anonymous = Some(v == "true"),
                "bearer_token" => bearer_token = Some(v.clone()),
                "gcs_service_account_key"
                | "azure_storage_account"
                | "azure_storage_access_key" => {} // already translated in pass 1
                _ => {
                    config.insert(k.clone(), v.clone());
                }
            }
        }

        CloudOptions {
            config,
            bearer_token,
            anonymous,
        }
    }
}

/// Read a file from a path (local, cloud, or HTTP URL).
///
/// # Arguments
/// * `path` - The file path (local path, or URL like s3://, gs://, az://, http://, https://)
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
            "http" | "https" => read_http(path),
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

/// Read a path already known to be local (not a remote/cloud URL).
///
/// Strips a `file://` prefix if present, otherwise reads the literal path.
/// Unlike [`read_file`], this never treats the path as a cloud URL, so a
/// bare local filename that happens to contain a colon (e.g. `img:2.png`,
/// legal on Unix) is read as-is rather than being mis-parsed as a URL with
/// scheme `img`.
pub fn read_local_path(path: &str) -> Result<Vec<u8>, CloudError> {
    if let Some(rest) = path.strip_prefix("file://") {
        // Parse so percent-encoding and the `file://host/path` form resolve
        // to a real filesystem path, matching read_file's `file` arm.
        match Url::parse(path) {
            Ok(url) => read_local_file(url.path()),
            // Malformed file:// URL — fall back to the literal remainder.
            Err(_) => read_local_file(rest),
        }
    } else {
        read_local_file(path)
    }
}

/// Get or create a tokio runtime for async operations.
///
/// Reuses a thread-local runtime to avoid the overhead of creating a new
/// runtime for every cloud file read.
fn get_runtime() -> Result<&'static Runtime, CloudError> {
    use std::sync::OnceLock;
    static RUNTIME: OnceLock<Runtime> = OnceLock::new();
    if let Some(rt) = RUNTIME.get() {
        return Ok(rt);
    }
    let rt = Runtime::new().map_err(|e| CloudError::RuntimeError(e.to_string()))?;
    // Race is fine - OnceLock guarantees only one wins, others drop theirs
    Ok(RUNTIME.get_or_init(|| rt))
}

/// Read a file from Amazon S3.
fn read_s3(url: &Url, options: Option<&CloudOptions>) -> Result<Vec<u8>, CloudError> {
    let bucket = url
        .host_str()
        .ok_or_else(|| CloudError::UrlParse("Missing bucket name in S3 URL".to_string()))?;
    let key = url.path().trim_start_matches('/');

    let runtime = get_runtime()?;

    // Build S3 client with credentials
    let mut builder = AmazonS3Builder::new().with_bucket_name(bucket);

    if let Some(opts) = options {
        for (k, v) in &opts.config {
            let key = AmazonS3ConfigKey::from_str(k).map_err(|e| {
                CloudError::StoreError(format!("unknown S3 storage option '{k}': {e}"))
            })?;
            builder = builder.with_config(key, v);
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

    let runtime = get_runtime()?;

    // Build GCS client
    let mut builder = GoogleCloudStorageBuilder::new().with_bucket_name(bucket);

    if let Some(opts) = options {
        for (k, v) in &opts.config {
            let key = GoogleConfigKey::from_str(k).map_err(|e| {
                CloudError::StoreError(format!("unknown GCS storage option '{k}': {e}"))
            })?;
            builder = builder.with_config(key, v);
        }
        // A pre-obtained OAuth token bypasses object_store's credential loader,
        // which cannot parse federated `external_account_authorized_user` ADC.
        if let Some(token) = &opts.bearer_token {
            let provider = StaticCredentialProvider::new(GcpCredential {
                bearer: token.clone(),
            });
            builder = builder.with_credentials(Arc::new(provider));
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

/// Read a file from Azure Blob Storage.
fn read_azure(url: &Url, options: Option<&CloudOptions>) -> Result<Vec<u8>, CloudError> {
    // Azure URLs: az://container/path or abfs://container@account.dfs.core.windows.net/path
    let container = url
        .host_str()
        .ok_or_else(|| CloudError::UrlParse("Missing container name in Azure URL".to_string()))?;
    let key = url.path().trim_start_matches('/');

    let runtime = get_runtime()?;

    // Build Azure client
    let mut builder = MicrosoftAzureBuilder::new().with_container_name(container);

    if let Some(opts) = options {
        for (k, v) in &opts.config {
            let key = AzureConfigKey::from_str(k).map_err(|e| {
                CloudError::StoreError(format!("unknown Azure storage option '{k}': {e}"))
            })?;
            builder = builder.with_config(key, v);
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

/// Read a file from an HTTP or HTTPS URL.
///
/// Uses async reqwest within a tokio runtime to avoid blocking issues
/// when called from within Polars plugin execution context.
///
/// # Arguments
/// * `url` - The HTTP/HTTPS URL to fetch
///
/// # Returns
/// The file contents as bytes.
///
/// # Example
/// ```ignore
/// let bytes = read_http("https://example.com/image.png")?;
/// ```
fn read_http(url: &str) -> Result<Vec<u8>, CloudError> {
    let runtime = get_runtime()?;
    let url_string = url.to_string();

    runtime.block_on(async {
        let client = reqwest::Client::new();
        let response = client
            .get(&url_string)
            .send()
            .await
            .map_err(|e| CloudError::ReadError(format!("HTTP request failed: {e}")))?;

        if !response.status().is_success() {
            return Err(CloudError::ReadError(format!(
                "HTTP {} for URL: {url_string}",
                response.status()
            )));
        }

        response
            .bytes()
            .await
            .map(|b| b.to_vec())
            .map_err(|e| CloudError::ReadError(format!("Failed to read response body: {e}")))
    })
}

/// Fetch many files concurrently with bounded parallelism.
///
/// Each path is fetched with the same logic (and credentials) as
/// [`read_file`]; results are keyed by path, errors carried per path as
/// strings. Graph execution uses this to prefetch a batch's remote sources
/// before the row loop, converting per-row network latency into per-batch
/// latency.
///
/// Implementation: bounded scoped OS threads, each performing one blocking
/// [`read_file`] (which itself parks on the shared tokio runtime). This keeps
/// per-file behavior — auth, retries, error text — byte-identical to the
/// sequential path.
pub fn read_files_concurrent(
    paths: &[String],
    options: Option<&CloudOptions>,
    max_concurrency: usize,
) -> HashMap<String, Result<Vec<u8>, String>> {
    let mut results: HashMap<String, Result<Vec<u8>, String>> = HashMap::with_capacity(paths.len());
    // Fetch each distinct path once, even when many rows repeat it.
    let unique: Vec<&String> = {
        let mut seen = std::collections::HashSet::new();
        paths.iter().filter(|p| seen.insert(p.as_str())).collect()
    };
    for chunk in unique.chunks(max_concurrency.max(1)) {
        let fetched: Vec<(String, Result<Vec<u8>, String>)> = std::thread::scope(|s| {
            let handles: Vec<_> = chunk
                .iter()
                .map(|path| {
                    s.spawn(move || {
                        let result = read_file(path, options).map_err(|e| e.to_string());
                        ((*path).clone(), result)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("prefetch thread panicked"))
                .collect()
        });
        results.extend(fetched);
    }
    results
}

/// Check if a path is a remote URL (cloud storage or HTTP).
///
/// Returns true for:
/// - Cloud storage URLs: s3://, gs://, az://, abfs://, abfss://
/// - HTTP URLs: http://, https://
#[allow(dead_code)]
pub fn is_remote_path(path: &str) -> bool {
    if let Ok(url) = Url::parse(path) {
        matches!(
            url.scheme(),
            "s3" | "gs" | "az" | "abfs" | "abfss" | "http" | "https"
        )
    } else {
        false
    }
}

/// Check if a path is a cloud storage URL (S3, GCS, Azure).
///
/// Does NOT include HTTP/HTTPS URLs - use `is_remote_path` for that.
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
        // Cloud storage paths
        assert!(is_cloud_path("s3://bucket/key"));
        assert!(is_cloud_path("gs://bucket/key"));
        assert!(is_cloud_path("az://container/path"));
        assert!(is_cloud_path("abfs://container/path"));
        // HTTP is NOT a cloud path (use is_remote_path)
        assert!(!is_cloud_path("http://example.com/image.png"));
        assert!(!is_cloud_path("https://example.com/image.png"));
        // Local paths
        assert!(!is_cloud_path("/local/path/file.png"));
        assert!(!is_cloud_path("relative/path.png"));
        assert!(!is_cloud_path("file:///local/path.png"));
    }

    #[test]
    fn test_is_remote_path() {
        // Cloud storage paths
        assert!(is_remote_path("s3://bucket/key"));
        assert!(is_remote_path("gs://bucket/key"));
        assert!(is_remote_path("az://container/path"));
        assert!(is_remote_path("abfs://container/path"));
        // HTTP/HTTPS paths
        assert!(is_remote_path("http://example.com/image.png"));
        assert!(is_remote_path("https://example.com/image.png"));
        // Local paths
        assert!(!is_remote_path("/local/path/file.png"));
        assert!(!is_remote_path("relative/path.png"));
        assert!(!is_remote_path("file:///local/path.png"));
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

    #[test]
    fn test_read_local_path_variants() {
        let dir = std::env::temp_dir().join("polars_cv_local_path_test");
        std::fs::create_dir_all(&dir).unwrap();

        // Bare path.
        let bare = dir.join("plain.txt");
        std::fs::write(&bare, b"bare").unwrap();
        assert_eq!(
            read_local_path(bare.to_str().unwrap()).unwrap(),
            b"bare".to_vec()
        );

        // file:// URL for the same path.
        let file_url = format!("file://{}", bare.to_str().unwrap());
        assert_eq!(read_local_path(&file_url).unwrap(), b"bare".to_vec());

        // Bare filename containing a colon (legal on Unix): must be read
        // literally, not mis-parsed as a URL with scheme before the colon.
        let colon = dir.join("frame:2.txt");
        std::fs::write(&colon, b"colon").unwrap();
        assert_eq!(
            read_local_path(colon.to_str().unwrap()).unwrap(),
            b"colon".to_vec()
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    fn map(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn test_from_map_passthrough() {
        // Native object_store keys flow straight into `config`.
        let opts = CloudOptions::from_map(&map(&[
            ("aws_region", "eu-west-1"),
            ("google_application_credentials", "/adc.json"),
        ]));
        assert_eq!(opts.config.get("aws_region").unwrap(), "eu-west-1");
        assert_eq!(
            opts.config.get("google_application_credentials").unwrap(),
            "/adc.json"
        );
        assert!(opts.bearer_token.is_none());
        assert!(opts.anonymous.is_none());
    }

    #[test]
    fn test_from_map_reserved_keys() {
        let opts = CloudOptions::from_map(&map(&[
            ("anonymous", "true"),
            ("bearer_token", "ya29.token"),
        ]));
        assert_eq!(opts.anonymous, Some(true));
        assert_eq!(opts.bearer_token.as_deref(), Some("ya29.token"));
        // Reserved keys are lifted out, not forwarded to the backend config.
        assert!(!opts.config.contains_key("anonymous"));
        assert!(!opts.config.contains_key("bearer_token"));
    }

    #[test]
    fn test_from_map_legacy_aliases() {
        let opts = CloudOptions::from_map(&map(&[
            ("gcs_service_account_key", "/sa.json"),
            ("azure_storage_account", "acct"),
            ("azure_storage_access_key", "secret"),
        ]));
        // Legacy names are translated to canonical object_store keys.
        assert_eq!(
            opts.config.get("google_service_account").unwrap(),
            "/sa.json"
        );
        assert_eq!(
            opts.config.get("azure_storage_account_name").unwrap(),
            "acct"
        );
        assert_eq!(
            opts.config.get("azure_storage_account_key").unwrap(),
            "secret"
        );
        assert!(!opts.config.contains_key("gcs_service_account_key"));
    }

    #[test]
    fn test_from_map_explicit_key_overrides_legacy_alias() {
        // When both a legacy alias and its canonical key are supplied, the
        // explicit canonical value wins regardless of map iteration order.
        let opts = CloudOptions::from_map(&map(&[
            ("gcs_service_account_key", "/legacy.json"),
            ("google_service_account", "/explicit.json"),
        ]));
        assert_eq!(
            opts.config.get("google_service_account").unwrap(),
            "/explicit.json"
        );
    }

    #[test]
    fn test_unknown_gcs_option_errors() {
        // An unrecognized key should fail loudly at build time rather than be
        // silently dropped.
        let opts = CloudOptions::from_map(&map(&[("not_a_real_key", "x")]));
        let url = Url::parse("gs://bucket/obj.png").unwrap();
        let err = read_gcs(&url, Some(&opts)).unwrap_err();
        assert!(
            matches!(err, CloudError::StoreError(_)),
            "expected StoreError, got {err:?}"
        );
        assert!(err.to_string().contains("not_a_real_key"));
    }

    // Integration test for HTTP - requires network access
    // Run with: cargo test --features cloud -- --ignored
    #[test]
    #[ignore]
    fn test_read_http_url() {
        // Use httpbin.org which returns known content
        let result = read_http("https://httpbin.org/bytes/100");
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 100);
    }

    #[test]
    #[ignore]
    fn test_read_http_image() {
        // Test with httpbin's PNG image endpoint
        let result = read_file("https://httpbin.org/image/png", None);
        assert!(result.is_ok());
        // PNG files start with these magic bytes
        let bytes = result.unwrap();
        assert!(bytes.len() > 8);
        assert_eq!(&bytes[0..4], &[0x89, 0x50, 0x4E, 0x47]); // PNG magic
    }
}
