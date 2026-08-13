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
//! # Who builds the store
//!
//! For `s3://`, `gs://` and `az://` this module builds **nothing**: it
//! translates our options ([`polars_options`]) and hands them to
//! `polars-io`'s [`build_object_store`], which owns a process-wide store cache,
//! credential-expiry refresh and rebuild-on-error retry. That is not a detail —
//! this crate previously built a store per *file*, so every image paid a fresh
//! DNS lookup, TLS handshake and connection pool, and under the streaming engine
//! it paid them again on every morsel.
//!
//! `http(s)://` is the exception, and deliberately: polars does not cache HTTP
//! stores (only Aws/Gcp/Azure take a cache key), which suits reading a few large
//! Parquet files and not many small images. That path keeps its own pooled
//! [`http_client`].
//!
//! `CloudOptions.config` is a generic pass-through map keyed by
//! `object_store`'s own configuration keys (e.g. `aws_region`,
//! `google_service_account`, `google_application_credentials`,
//! `azure_storage_account_name`). Entries are validated against the backend's
//! own key vocabulary and then handed to polars — validated *by us* because
//! polars silently drops keys it does not recognise, which would turn a
//! misspelled option into a request that quietly does the wrong thing.
//!
//! Two keys are reserved and handled explicitly rather than passed through:
//! - `anonymous` → skip request signing for public buckets (S3, GCS, Azure).
//! - `bearer_token` → install a pre-obtained OAuth access token as a static
//!   GCS credential. This is the escape hatch for credential types
//!   `object_store` cannot parse natively (e.g. workforce/federated
//!   `external_account_authorized_user` Application Default Credentials): mint
//!   a token out of band and hand it over directly.

use object_store::aws::AmazonS3ConfigKey;
use object_store::azure::AzureConfigKey;
use object_store::gcp::GoogleConfigKey;
use object_store::path::Path as ObjectPath;
use object_store::ObjectStoreExt;
use polars::io::cloud::{build_object_store, CloudOptions as PlCloudOptions, CloudType};
use polars_utils::pl_path::{CloudScheme, PlRefPath};
use std::collections::HashMap;
use std::path::Path;
use std::str::FromStr;
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
/// configuration keys; the reserved `anonymous`, `bearer_token`, and
/// `token_command` inputs are lifted out into their own fields (see the module
/// docs).
#[derive(Debug, Clone, Default)]
pub struct CloudOptions {
    /// Options forwarded verbatim to the backend builder via `with_config`,
    /// keyed by `object_store`'s native config keys.
    pub config: HashMap<String, String>,
    /// Pre-obtained OAuth bearer token, installed as a static GCS credential.
    pub bearer_token: Option<String>,
    /// Command whose stdout is a GCS OAuth access token. Run to obtain a bearer
    /// credential for federated/brokered setups object_store can't load itself.
    pub token_command: Option<String>,
    /// Skip request signing for public buckets (opt-in; default: signed
    /// requests using the credential chain).
    pub anonymous: Option<bool>,
}

impl CloudOptions {
    /// Create options from the wire map (string key/value pairs from Python).
    ///
    /// Most keys pass straight through to `config` using `object_store`'s
    /// native names. Three keys are reserved (`anonymous`, `bearer_token`,
    /// `token_command`), and a handful of historical polars-cv field names are
    /// translated to their canonical `object_store` equivalents for backwards
    /// compatibility. When a legacy name and its canonical name are both
    /// present, the explicit canonical value wins.
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
        let mut token_command = None;
        let mut anonymous = None;
        for (k, v) in map {
            match k.as_str() {
                "anonymous" => anonymous = Some(v == "true"),
                "bearer_token" => bearer_token = Some(v.clone()),
                "token_command" => token_command = Some(v.clone()),
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
            token_command,
            anonymous,
        }
    }
}

/// Reject a storage option no backend for this scheme recognises.
///
/// This exists because **polars does not do it**. `parse_untyped_config` is a
/// `filter_map` over `from_str(..).ok()` — its own comment reads *"Silently
/// ignores custom upstream storage_options"* — so a misspelled key reaches the
/// builder as nothing at all. A user who writes `aws_regionn="us-east-1"` would
/// get an unsigned or differently-signed request and no indication why.
///
/// polars can afford that: `storage_options` there is a shared bag that may
/// carry keys meant for a different layer. Ours is a checked surface and has
/// been since `reject_inapplicable_params`, so the key check has to be ours
/// too. Validating *before* handing the map over keeps the error at the caller's
/// spelling rather than at a request that quietly did the wrong thing.
///
/// Keys are lower-cased first because `parse_untyped_config` does, so this
/// accepts exactly what polars would then act on — no wider, no narrower.
fn validate_config_keys(
    cloud_type: &CloudType,
    config: &HashMap<String, String>,
) -> Result<(), CloudError> {
    for key in config.keys() {
        let lowered = key.to_ascii_lowercase();
        let (known, backend) = match cloud_type {
            CloudType::Aws => (AmazonS3ConfigKey::from_str(&lowered).is_ok(), "S3"),
            CloudType::Gcp => (GoogleConfigKey::from_str(&lowered).is_ok(), "GCS"),
            CloudType::Azure => (AzureConfigKey::from_str(&lowered).is_ok(), "Azure"),
            // `http(s)://` and local paths take no backend configuration at all.
            // Options are accepted and inert there, which is long-standing
            // behaviour: a column may mix schemes, so one set of credentials has
            // to be passable for the subset that needs them.
            _ => (true, ""),
        };
        if !known {
            return Err(CloudError::StoreError(format!(
                "unknown {backend} storage option '{key}'"
            )));
        }
    }
    Ok(())
}

/// Translate our wire options into the ones `polars-io` builds a store from.
///
/// The single place a [`PlCloudOptions`] is constructed. The reserved keys, the
/// strict key check and the credential choice are one act — split them and the
/// next caller gets a store built from a map nothing validated.
///
/// The scheme→backend mapping is read from polars' own
/// [`CloudType::from_cloud_scheme`] rather than re-enumerated here; `s3a`,
/// `abfss` and `azure` are all spellings this crate would otherwise have to
/// remember.
fn polars_options(
    scheme: CloudScheme,
    options: Option<&CloudOptions>,
) -> Result<PlCloudOptions, CloudError> {
    let cloud_type = CloudType::from_cloud_scheme(scheme);
    let mut config: HashMap<String, String> = options.map(|o| o.config.clone()).unwrap_or_default();

    if let Some(opts) = options {
        // S3 authenticates with SigV4, not an OAuth bearer token, so neither
        // bearer input can drive it. Reject rather than silently ignore a
        // credential the caller believes is in effect — `bearer_token` *was*
        // silently ignored here before, which is the same bug one field over.
        if matches!(cloud_type, CloudType::Aws) {
            if opts.token_command.is_some() {
                return Err(CloudError::StoreError(
                    "token_command produces an OAuth bearer token, which S3 does not \
                     use; supply AWS credentials via aws_access_key_id / \
                     aws_secret_access_key / aws_session_token or storage_options"
                        .to_string(),
                ));
            }
            if opts.bearer_token.is_some() {
                return Err(CloudError::StoreError(
                    "gcs_bearer_token is an OAuth bearer token, which S3 does not use; \
                     supply AWS credentials via aws_access_key_id / \
                     aws_secret_access_key / aws_session_token or storage_options"
                        .to_string(),
                ));
            }
        }

        validate_config_keys(&cloud_type, &config)?;

        // `anonymous` is our spelling of what every backend calls
        // `skip_signature`. polars has no `anonymous` concept, but the config
        // key exists for all three and is exactly what our builders used to set
        // through `with_skip_signature`.
        if opts.anonymous == Some(true) {
            let key = match cloud_type {
                CloudType::Aws => Some("aws_skip_signature"),
                CloudType::Gcp => Some("google_skip_signature"),
                CloudType::Azure => Some("azure_skip_signature"),
                _ => None,
            };
            if let Some(key) = key {
                config.insert(key.to_string(), "true".to_string());
            }
        }
    }

    // Sorted, because the map is a `HashMap` and polars' object-store cache key
    // is a *serialization* of the config rather than a hash of an unordered
    // set. Two identical option sets iterated in different orders would key two
    // separate cache entries, and the symptom would be a silent loss of
    // connection reuse rather than an error.
    let mut pairs: Vec<(&String, &String)> = config.iter().collect();
    pairs.sort_unstable_by(|a, b| a.0.cmp(b.0));

    let built = PlCloudOptions::from_untyped_config(Some(scheme), pairs)
        .map_err(|e| CloudError::StoreError(e.to_string()))?;

    // The bespoke credentials `object_store` cannot load itself: a supplied
    // bearer token, a token command, or a federated ADC delegated to `gcloud`.
    // `None` leaves the credential chain to `object_store`, which is what makes
    // an ordinary service-account setup keep working untouched.
    match crate::cloud_auth::credential_provider(&cloud_type, options)? {
        Some(provider) => Ok(built.with_credential_provider(Some(provider))),
        None => Ok(built),
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
            "http" | "https" => read_http(path),
            "s3" | "s3a" | "gs" | "gcs" | "az" | "azure" | "abfs" | "abfss" | "adl" => {
                read_object(path, options)
            }
            scheme => Err(CloudError::UnsupportedScheme(scheme.to_string())),
        }
    } else {
        // Not a valid URL, treat as local path
        read_local_file(path)
    }
}

/// Read one object from S3, GCS or Azure through `polars-io`'s cached store.
///
/// The whole reason this delegates rather than building its own client: the
/// store comes from `polars-io`'s process-wide `OBJECT_STORE_CACHE`, so the
/// second and every later read of a bucket is a map lookup rather than a fresh
/// DNS lookup, TLS handshake and connection pool. Under the streaming engine the
/// plugin is invoked once per morsel, so a per-call store is rebuilt on every
/// morsel however wide the batch is — which is what this crate did, and what the
/// benchmark measured as one connection per file.
///
/// `exec_with_rebuild_retry_on_err` is what makes caching a *credentialed* store
/// safe: on any `object_store` error it rebuilds the store with cached
/// credentials cleared and retries once, swapping the new store in place through
/// the `Arc` the cache holds — so a refresh reaches every reader without the
/// cache having to be invalidated.
///
/// Whole-object `get` rather than `head` + `get_range`: we read many small
/// files, so sizing first would double the request count for no benefit. It also
/// avoids nesting concurrency-budget acquisitions — `PolarsObjectStore::head`
/// and `get_range` take permits internally, and holding one while asking for
/// another is a deadlock at any budget smaller than twice the in-flight count.
fn read_object(path: &str, options: Option<&CloudOptions>) -> Result<Vec<u8>, CloudError> {
    let url = PlRefPath::new(path);
    let scheme = url
        .scheme()
        .ok_or_else(|| CloudError::UrlParse(format!("no scheme in remote path '{path}'")))?;
    let pl_options = polars_options(scheme, options)?;
    let runtime = get_runtime()?;

    runtime.block_on(async {
        let (location, store) = build_object_store(url, Some(&pl_options), false)
            .await
            .map_err(|e| CloudError::StoreError(e.to_string()))?;
        // `location.prefix` is the raw key after the authority. Parsed rather
        // than `Path::from`: `from` percent-encodes, and this string has already
        // been through the URL, so encoding it again turned `a b.png` into
        // `a%2520b.png` on the old S3 path.
        let key = ObjectPath::parse(&location.prefix).map_err(|e| {
            CloudError::UrlParse(format!("bad object key '{}': {e}", location.prefix))
        })?;

        store
            .exec_with_rebuild_retry_on_err(|store| {
                let key = &key;
                async move { store.get(key).await?.bytes().await }
            })
            .await
            .map(|bytes| bytes.to_vec())
            .map_err(|e| CloudError::ReadError(e.to_string()))
    })
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
pub(crate) fn get_runtime() -> Result<&'static Runtime, CloudError> {
    use std::sync::OnceLock;
    static RUNTIME: OnceLock<Runtime> = OnceLock::new();
    if let Some(rt) = RUNTIME.get() {
        return Ok(rt);
    }
    let rt = Runtime::new().map_err(|e| CloudError::RuntimeError(e.to_string()))?;
    // Race is fine - OnceLock guarantees only one wins, others drop theirs
    Ok(RUNTIME.get_or_init(|| rt))
}

/// The process-wide HTTP client, and therefore the process-wide connection pool.
///
/// A `reqwest::Client` *is* the pool: it owns the idle connections, the DNS
/// resolver and the TLS session cache, and cloning it is an `Arc` bump. Building
/// one per file — which is what this module used to do, inside [`read_http`] —
/// meant every file paid a fresh TCP handshake, and off loopback a fresh TLS
/// handshake too. The benchmark server counted it exactly: one connection per
/// request, no reuse at all.
///
/// Shared for the same reason [`get_runtime`] is, and note where the sharing
/// has to happen: the streaming engine calls the plugin once per morsel, so a
/// client scoped to a call is rebuilt on every morsel however wide the batch is.
///
/// Deliberately *not* delegated to polars. `polars-io` builds an object-store
/// for `http://` but does not cache it (`object_store_setup.rs`: only Aws/Gcp/
/// Azure take a cache key), which is a reasonable trade for reading a few large
/// Parquet files and the wrong one for reading many small images.
fn http_client() -> &'static reqwest::Client {
    use std::sync::OnceLock;
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .user_agent(concat!("polars-cv/", env!("CARGO_PKG_VERSION")))
            // Bound only the *connect* phase. A read timeout would have to bound
            // the whole body, and this client fetches images of unknown size
            // over links of unknown speed.
            .connect_timeout(std::time::Duration::from_secs(30))
            .build()
            // `build()` fails only on a bad TLS backend configuration, which is
            // fixed at compile time by the `rustls-tls` feature — not something
            // a caller's input can reach.
            .expect("failed to build the shared HTTP client")
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
        let client = http_client();
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
///
/// This is the only remote/local split the fetch path makes. A second
/// predicate `is_cloud_path` sat here excluding HTTP, called by nothing but
/// its own test; the distinction it drew is not one any caller needed, and an
/// unused classifier next to a used one is an invitation to pick the wrong
/// one.
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

#[cfg(test)]
mod tests {
    use super::*;

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

    // -- polars_options: the one translation into polars-io's CloudOptions ----

    /// A key no backend knows must be refused, not dropped.
    ///
    /// This is the guard for the one thing delegation would otherwise lose.
    /// polars' `parse_untyped_config` is a `filter_map` over `from_str(..).ok()`
    /// ("Silently ignores custom upstream storage_options"), so handing it an
    /// unvalidated map turns a misspelled option into a request that quietly
    /// does the wrong thing. Watched failing by removing the
    /// `validate_config_keys` call from `polars_options`: `from_untyped_config`
    /// returns `Ok` and this test reports "expected an error for an unknown key".
    #[test]
    fn unknown_storage_option_is_refused_for_every_backend() {
        for (scheme, backend) in [
            (CloudScheme::S3, "S3"),
            (CloudScheme::Gs, "GCS"),
            (CloudScheme::Az, "Azure"),
        ] {
            let opts = CloudOptions::from_map(&map(&[("not_a_real_key", "x")]));
            let err = polars_options(scheme, Some(&opts))
                .expect_err("expected an error for an unknown key");
            assert!(
                matches!(err, CloudError::StoreError(_)),
                "expected StoreError, got {err:?}"
            );
            let message = err.to_string();
            assert!(
                message.contains("not_a_real_key") && message.contains(backend),
                "message should name the key and the backend, got {message:?}"
            );
        }
    }

    /// A key the backend *does* know must survive the same path.
    ///
    /// Without this the test above is satisfied by a validator that rejects
    /// everything, which would be green and useless.
    #[test]
    fn a_known_storage_option_passes_validation() {
        let opts = CloudOptions::from_map(&map(&[("aws_region", "us-east-1")]));
        polars_options(CloudScheme::S3, Some(&opts)).expect("aws_region is a real S3 config key");
    }

    /// Keys are matched case-insensitively, because polars lower-cases before
    /// its own lookup — so accepting a narrower set here would reject options
    /// polars would have honoured.
    #[test]
    fn storage_option_keys_are_case_insensitive() {
        let opts = CloudOptions::from_map(&map(&[("AWS_REGION", "us-east-1")]));
        polars_options(CloudScheme::S3, Some(&opts)).expect("keys are lower-cased first");
    }

    /// `anonymous` is our spelling of `skip_signature`, and it has to reach the
    /// built options or a public-bucket read starts getting signed.
    #[test]
    fn anonymous_becomes_skip_signature() {
        for scheme in [CloudScheme::S3, CloudScheme::Gs, CloudScheme::Az] {
            let anon = CloudOptions::from_map(&map(&[("anonymous", "true")]));
            let signed = CloudOptions::from_map(&map(&[]));
            assert_ne!(
                polars_options(scheme, Some(&anon)).unwrap(),
                polars_options(scheme, Some(&signed)).unwrap(),
                "anonymous must change the built options for {scheme:?}"
            );
        }
    }

    /// Bearer credentials are an OAuth concept; S3 signs with SigV4. Both
    /// inputs must be refused rather than accepted and ignored.
    #[test]
    fn s3_refuses_both_bearer_inputs() {
        for key in ["token_command", "bearer_token"] {
            let opts = CloudOptions::from_map(&map(&[(key, "whatever")]));
            let err = polars_options(CloudScheme::S3, Some(&opts))
                .expect_err("S3 must refuse an OAuth bearer credential");
            assert!(
                err.to_string().contains("S3 does not use"),
                "message should say why S3 cannot use it, got {err}"
            );
        }

        // The same inputs are accepted on the schemes that *do* use a bearer
        // token, so the rejection is about S3 and not about the key existing.
        for key in ["token_command", "bearer_token"] {
            let opts = CloudOptions::from_map(&map(&[(key, "whatever")]));
            polars_options(CloudScheme::Gs, Some(&opts))
                .unwrap_or_else(|e| panic!("GCS should accept {key}, got {e}"));
        }
    }

    /// The same option map must always produce the same options, whatever order
    /// the `HashMap` iterates in. polars keys its object-store cache on a
    /// *serialization* of the config, so an order-dependent translation would
    /// silently split one cache entry into many — losing exactly the connection
    /// reuse this delegation exists to gain, with no error anywhere.
    #[test]
    fn translation_is_order_independent() {
        let pairs: &[(&str, &str)] = &[
            ("aws_region", "us-east-1"),
            ("aws_access_key_id", "AKIA"),
            ("aws_secret_access_key", "secret"),
            ("aws_endpoint", "https://minio.local:9000"),
        ];
        let forward = CloudOptions::from_map(&map(pairs));
        let reversed: Vec<(&str, &str)> = pairs.iter().rev().copied().collect();
        let backward = CloudOptions::from_map(&map(&reversed));

        assert_eq!(
            polars_options(CloudScheme::S3, Some(&forward)).unwrap(),
            polars_options(CloudScheme::S3, Some(&backward)).unwrap(),
        );
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
        //
        // Driven through the user-facing `read_file` rather than a helper: the
        // point is that the *query* fails, and polars' own `parse_untyped_config`
        // would silently drop this key, so what is pinned here is our check
        // sitting in front of it. No store is built and no network is touched —
        // validation happens first — which is also what keeps this hermetic in a
        // shared test process.
        let opts = CloudOptions::from_map(&map(&[("not_a_real_key", "x")]));
        let err = read_file("gs://bucket/obj.png", Some(&opts)).unwrap_err();
        assert!(
            matches!(err, CloudError::StoreError(_)),
            "expected StoreError, got {err:?}"
        );
        assert!(err.to_string().contains("not_a_real_key"));
    }

    #[test]
    fn test_s3_rejects_token_command() {
        // token_command is an OAuth-bearer concept; S3 uses SigV4. The S3 path
        // must reject it (before any network I/O) rather than silently ignore a
        // credential the user believes is in effect.
        let opts = CloudOptions::from_map(&map(&[("token_command", "printf tok")]));
        let err = read_file("s3://bucket/obj.png", Some(&opts)).unwrap_err();
        assert!(
            matches!(err, CloudError::StoreError(_)),
            "expected StoreError, got {err:?}"
        );
        assert!(err.to_string().contains("S3 does not"));
    }

    #[test]
    fn test_bearer_token_bypasses_unparseable_adc() {
        // Regression: a supplied `bearer_token` must let the GCS store build
        // even when the ambient Application Default Credentials are a federated
        // `external_account_authorized_user` file that object_store cannot parse.
        // In object_store 0.12 `build()` parsed ADC unconditionally and failed
        // here before honoring the static credential; 0.13 skips the ADC parse
        // whenever an explicit credential provider is installed.
        //
        // We point `google_application_credentials` at a poisoned file (rather
        // than touching the process-global HOME env) so the test is hermetic and
        // parallel-safe.
        let dir = std::env::temp_dir().join(format!("pcv_adc_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let adc = dir.join("application_default_credentials.json");
        std::fs::write(
            &adc,
            r#"{"type": "external_account_authorized_user", "audience": "x", "refresh_token": "y"}"#,
        )
        .unwrap();

        let opts = CloudOptions::from_map(&map(&[
            ("google_application_credentials", adc.to_str().unwrap()),
            ("bearer_token", "ya29.fake-token"),
        ]));

        // With the bearer token installed, building must succeed despite the
        // unparseable ADC file. (An explicit bearer also short-circuits our own
        // federated auto-mint, so this stays hermetic — no token exchange is
        // attempted, and no request is made.)
        //
        // The *reason* the property holds moved with the delegation and this
        // now pins the new one: it is no longer our builder's ordering but
        // polars' `build_gcp`, which uses `GoogleCloudStorageBuilder::new()`
        // rather than `from_env()` whenever a credential provider is installed,
        // so the poisoned ADC is never read. The user-visible fact is identical.
        let options = polars_options(CloudScheme::Gs, Some(&opts))
            .expect("translating options must not read the ADC");
        let built = get_runtime().unwrap().block_on(build_object_store(
            PlRefPath::new("gs://bucket/obj.png"),
            Some(&options),
            false,
        ));
        assert!(
            built.is_ok(),
            "bearer_token should bypass unparseable ADC, got {built:?}"
        );

        std::fs::remove_dir_all(&dir).ok();
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
