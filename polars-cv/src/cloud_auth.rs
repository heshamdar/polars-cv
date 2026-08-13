//! Bearer-token acquisition for federated / brokered cloud credentials,
//! obtained without reimplementing any provider's token-exchange protocol.
//!
//! Some credential types the OAuth-bearer backends accept cannot be loaded by
//! `object_store` itself — most notably Google Workload / Workforce Identity
//! Federation (`external_account*` ADC), which its GCS loader rejects. Rather
//! than own those protocols, polars-cv obtains an access token out of band and
//! installs it as a static credential. Two general sources exist, both yielding
//! an OAuth bearer token usable by the bearer backends (GCS and Azure):
//!
//! - a user-configured command (Python: `CloudOptions.token_command`) whose
//!   stdout is the token. Provider-agnostic: any broker, script, or CLI works.
//! - a detected federated `external_account*` ADC, delegated to
//!   `gcloud auth application-default print-access-token`, which understands the
//!   full federation matrix. Requires the `gcloud` CLI on `PATH`; set
//!   `POLARS_CV_DISABLE_GCS_FEDERATION=1` to turn it off.
//!
//! [`credential_provider`] is the single entry point: it decides which of those
//! applies and hands back a [`PlCredentialProvider`] for `polars-io` to install
//! on the store, or `None` to leave the chain to `object_store`.
//!
//! S3 uses SigV4 (access key / secret / session token), not bearer tokens, so
//! neither applies there — `cloud::polars_options` rejects a bearer input on an
//! `s3://` path rather than silently ignore it. Non-federated ambient GCS
//! credentials (`service_account` / `authorized_user`) are left to
//! `object_store`.
//!
//! # Why providers are memoized
//!
//! A provider is not just a way to get a token: its `Arc` address *is* the
//! identity polars keys its process-wide object-store cache on
//! (`PlCredentialProvider::stable_cache_key`). Handing back a fresh provider for
//! the same credential would therefore defeat that cache entirely — every read
//! would rebuild its store and its connection pool — while handing back a shared
//! provider for a *different* credential would serve a rotated token from a store
//! still holding the old one. Both are silent. See [`credential_provider`] and
//! its identity test.

use object_store::azure::AzureCredential;
use object_store::gcp::GcpCredential;
use polars::io::cloud::credential_provider::{ObjectStoreCredential, PlCredentialProvider};
use polars::io::cloud::CloudType;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use crate::cloud::{CloudError, CloudOptions};

/// The delegation command used for a detected federated GCS ADC.
const GCLOUD_ADC_COMMAND: &str = "gcloud auth application-default print-access-token";
/// Assumed lifetime of a command-sourced token. GCS/Azure access tokens live
/// ~1h; we cache for a shorter window so a token that is a little shorter-lived
/// than that is still refreshed before it lapses.
const ASSUMED_TOKEN_TTL: Duration = Duration::from_secs(55 * 60);

/// GCS-specific: the path of an ambient ADC file that is a federated credential
/// `object_store` cannot parse, if one applies.
///
/// Returns `None` when there is nothing federated to handle — auto-delegation
/// disabled, an explicit service account was given, no ADC found, or the ADC is
/// a plain `service_account` / `authorized_user` file `object_store` handles
/// itself.
///
/// Returns the *path* rather than a token because the path (with its mtime) is
/// the credential's identity, and identity is what the provider memo is keyed
/// on. Rotating the file changes the key, which yields a new provider `Arc`,
/// which is what makes polars build a fresh store rather than reuse one holding
/// the superseded credential.
fn federated_adc_path(options: Option<&CloudOptions>) -> Option<PathBuf> {
    if std::env::var_os("POLARS_CV_DISABLE_GCS_FEDERATION").is_some() {
        return None;
    }
    // An explicit service account is the caller's stated intent — don't override.
    if let Some(opts) = options {
        if opts.config.contains_key("google_service_account")
            || opts.config.contains_key("google_service_account_key")
        {
            return None;
        }
    }

    let path = locate_adc_path(options)?;
    // Unreadable ADC: leave it to object_store to report in context.
    let bytes = std::fs::read(&path).ok()?;
    if is_federated_adc(&bytes) {
        Some(path)
    } else {
        // service_account / authorized_user / unknown -> object_store's job.
        None
    }
}

/// Locate the ADC file, mirroring `object_store`/gcloud resolution order:
/// explicit `google_application_credentials`, then GOOGLE_APPLICATION_CREDENTIALS,
/// then the well-known gcloud path.
fn locate_adc_path(options: Option<&CloudOptions>) -> Option<PathBuf> {
    if let Some(opts) = options {
        if let Some(p) = opts.config.get("google_application_credentials") {
            if !p.is_empty() {
                return Some(PathBuf::from(p));
            }
        }
    }
    if let Ok(p) = std::env::var("GOOGLE_APPLICATION_CREDENTIALS") {
        if !p.is_empty() {
            return Some(PathBuf::from(p));
        }
    }
    let home_var = if cfg!(windows) { "APPDATA" } else { "HOME" };
    let rel = if cfg!(windows) {
        "gcloud/application_default_credentials.json"
    } else {
        ".config/gcloud/application_default_credentials.json"
    };
    if let Some(home) = std::env::var_os(home_var) {
        let p = Path::new(&home).join(rel);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// Whether an ADC file is a federated type `object_store` cannot load itself.
///
/// Only the `type` field is inspected, so `service_account` files (whose bodies
/// differ) don't trip a full-schema parse.
fn is_federated_adc(bytes: &[u8]) -> bool {
    #[derive(Deserialize)]
    struct Probe {
        #[serde(rename = "type")]
        typ: Option<String>,
    }
    matches!(
        serde_json::from_slice::<Probe>(bytes)
            .ok()
            .and_then(|p| p.typ)
            .as_deref(),
        Some("external_account") | Some("external_account_authorized_user")
    )
}

// ---------------------------------------------------------------------------
// Credential providers
// ---------------------------------------------------------------------------

/// Cache key suffix for an ADC file: path plus mtime, so rotating the file
/// forces a refresh.
fn cache_key(path: &Path) -> String {
    let mtime = std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{}|{mtime}", path.display())
}

/// Seconds since the UNIX epoch, which is the unit polars states expiries in.
fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Which backend a bearer token is being minted for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Bearer {
    Gcs,
    Azure,
}

impl Bearer {
    fn wrap(self, token: String) -> ObjectStoreCredential {
        match self {
            Bearer::Gcs => ObjectStoreCredential::Gcp(Arc::new(GcpCredential { bearer: token })),
            Bearer::Azure => {
                ObjectStoreCredential::Azure(Arc::new(AzureCredential::BearerToken(token)))
            }
        }
    }
}

/// Credential identity → provider, so the same credential always yields the
/// **same `Arc`**.
///
/// This is the load-bearing part of delegating to polars, and it is invisible in
/// the type system. `PlCredentialProvider::stable_cache_key` is
/// `Arc::as_ptr(..) as usize` — the pointer address — and polars folds that into
/// the key of its process-wide object-store cache. So a provider constructed
/// fresh on each call produces a different key every time, the store cache never
/// hits, and every read rebuilds its connection pool: exactly the bug this
/// delegation exists to fix, one layer down and with nothing to show for it.
///
/// Bounded and MRU like the compiled-graph cache in `graph/compiled.rs`, for the
/// same reason: a caller minting a fresh `bearer_token` per query would
/// otherwise grow this without limit.
const PROVIDER_CACHE_CAP: usize = 32;

type ProviderEntries = Vec<(String, PlCredentialProvider)>;

fn provider_cache() -> &'static Mutex<ProviderEntries> {
    static CACHE: OnceLock<Mutex<ProviderEntries>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(Vec::new()))
}

/// Return the memoized provider for `key`, or build, store and return one.
fn memoized(key: String, build: impl FnOnce() -> PlCredentialProvider) -> PlCredentialProvider {
    let mut cache = provider_cache().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(pos) = cache.iter().position(|(k, _)| *k == key) {
        let entry = cache.remove(pos);
        let provider = entry.1.clone();
        cache.insert(0, entry);
        return provider;
    }
    let provider = build();
    cache.insert(0, (key, provider.clone()));
    cache.truncate(PROVIDER_CACHE_CAP);
    provider
}

/// A provider that returns a fixed token which never expires.
fn static_provider(kind: Bearer, token: String) -> PlCredentialProvider {
    PlCredentialProvider::from_func(move || {
        let credential = kind.wrap(token.clone());
        Box::pin(async move { Ok((credential, u64::MAX)) })
    })
}

/// A provider that runs `command` and uses its stdout as a bearer token.
///
/// The token is cached *inside the provider*, alongside polars' own
/// `FetchedCredentialsCache`, and that duplication is deliberate: polars builds
/// a fresh `FetchedCredentialsCache` every time it builds the store, and
/// `exec_with_rebuild_retry_on_err` rebuilds the store on **any** object-store
/// error — including a 404 for a missing key. Without a cache that outlives the
/// store, a batch of missing objects would spawn one subprocess per row.
///
/// The lock is held across the command run, which is what collapses a burst of
/// concurrent first-time reads into a single invocation instead of a stampede.
fn command_provider(kind: Bearer, command: String) -> PlCredentialProvider {
    let cached: Arc<Mutex<Option<(String, u64)>>> = Arc::new(Mutex::new(None));
    PlCredentialProvider::from_func(move || {
        let command = command.clone();
        let cached = cached.clone();
        Box::pin(async move {
            let mut guard = cached.lock().unwrap_or_else(|e| e.into_inner());
            if let Some((token, expiry)) = guard.as_ref() {
                if *expiry > now_unix() {
                    return Ok((kind.wrap(token.clone()), *expiry));
                }
            }
            let token = run_token_command(&command)
                .map_err(|e| polars::prelude::polars_err!(ComputeError: "{e}"))?;
            // A token command's stdout is opaque, so a real expiry is not
            // obtainable in general. The assumed window is unchanged from when
            // it was this module's own cache TTL — what changes is that polars
            // now acts on it rather than it being invisible.
            let expiry = now_unix() + ASSUMED_TOKEN_TTL.as_secs();
            *guard = Some((token.clone(), expiry));
            Ok((kind.wrap(token), expiry))
        })
    })
}

/// Choose the credential provider for a remote read, if any.
///
/// `Ok(None)` means "nothing bespoke applies" and is the signal to leave the
/// credential chain to `object_store`. That distinction matters to polars:
/// `build_gcp` calls `GoogleCloudStorageBuilder::from_env()` **only** when no
/// provider is installed, which is what makes the bearer-token escape hatch
/// bypass an unparseable ambient ADC.
pub(crate) fn credential_provider(
    cloud_type: &CloudType,
    options: Option<&CloudOptions>,
) -> Result<Option<PlCredentialProvider>, CloudError> {
    let Some(opts) = options else {
        return Ok(None);
    };

    // Anonymous access is requested by skipping signing entirely; installing a
    // credential alongside it would contradict the request.
    if opts.anonymous == Some(true) {
        return Ok(None);
    }

    let kind = match cloud_type {
        CloudType::Gcp => Bearer::Gcs,
        CloudType::Azure => Bearer::Azure,
        // S3 signs with SigV4; `cloud::polars_options` rejects a bearer input
        // there before this is reached.
        _ => return Ok(None),
    };

    if kind == Bearer::Gcs {
        if let Some(token) = opts.bearer_token.as_deref() {
            let token = token.to_string();
            return Ok(Some(memoized(format!("gcp:bearer:{token}"), || {
                static_provider(kind, token.clone())
            })));
        }
    }

    if let Some(command) = opts.token_command.as_deref() {
        if !command.trim().is_empty() {
            let command = command.to_string();
            let prefix = match kind {
                Bearer::Gcs => "gcp",
                Bearer::Azure => "azure",
            };
            return Ok(Some(memoized(format!("{prefix}:cmd:{command}"), || {
                command_provider(kind, command.clone())
            })));
        }
    }

    if kind == Bearer::Gcs {
        if let Some(path) = federated_adc_path(options) {
            let key = format!("gcp:adc:{}", cache_key(&path));
            return Ok(Some(memoized(key, || {
                command_provider(kind, GCLOUD_ADC_COMMAND.to_string())
            })));
        }
    }

    Ok(None)
}

/// Run a token command through the platform shell and return its trimmed
/// stdout. Using the shell keeps the option flexible (arguments, pipes, env)
/// and matches how users write such commands.
fn run_token_command(command: &str) -> Result<String, CloudError> {
    let output = if cfg!(windows) {
        std::process::Command::new("cmd")
            .args(["/C", command])
            .output()
    } else {
        std::process::Command::new("sh")
            .args(["-c", command])
            .output()
    }
    .map_err(|e| CloudError::StoreError(format!("failed to run token command `{command}`: {e}")))?;

    if !output.status.success() {
        return Err(CloudError::StoreError(format!(
            "token command `{command}` exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    let token = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if token.is_empty() {
        return Err(CloudError::StoreError(format!(
            "token command `{command}` produced no output"
        )));
    }
    Ok(token)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn opts_with(pairs: &[(&str, &str)]) -> CloudOptions {
        let map = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect::<HashMap<_, _>>();
        CloudOptions::from_map(&map)
    }

    fn write_adc(name: &str, body: &str) -> (PathBuf, PathBuf) {
        let dir = std::env::temp_dir().join(format!("pcv_cauth_{}_{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("adc.json");
        std::fs::write(&path, body).unwrap();
        (dir, path)
    }

    #[test]
    fn test_is_federated_adc() {
        assert!(is_federated_adc(br#"{"type": "external_account"}"#));
        assert!(is_federated_adc(
            br#"{"type": "external_account_authorized_user"}"#
        ));
        assert!(!is_federated_adc(br#"{"type": "service_account"}"#));
        assert!(!is_federated_adc(br#"{"type": "authorized_user"}"#));
        assert!(!is_federated_adc(b"not json"));
    }

    /// Ask a provider for a GCS credential exactly the way `object_store` will.
    ///
    /// Goes through `into_gcp_provider()` rather than the closure directly, so
    /// what is exercised is the whole path a real read takes — including polars'
    /// own `FetchedCredentialsCache`, which this wraps the closure in.
    ///
    /// Note that each call to `into_gcp_provider()` builds a *fresh*
    /// `FetchedCredentialsCache`, which is precisely what happens every time
    /// polars rebuilds the store. That is what makes it the right tool for
    /// testing the cache that has to survive a rebuild.
    fn fetch(provider: &PlCredentialProvider) -> Result<String, String> {
        use polars::io::cloud::credential_provider::IntoCredentialProvider;

        let os_provider = provider.clone().into_gcp_provider();
        crate::cloud::get_runtime()
            .unwrap()
            .block_on(os_provider.get_credential())
            .map(|c| c.bearer.clone())
            .map_err(|e| e.to_string())
    }

    #[test]
    fn test_token_command_is_run_and_used() {
        let opts = opts_with(&[("token_command", "printf 'ya29.from-command'")]);
        let provider = credential_provider(&CloudType::Gcp, Some(&opts))
            .unwrap()
            .expect("a token_command must install a provider");
        assert_eq!(fetch(&provider).unwrap(), "ya29.from-command");
    }

    #[test]
    fn test_no_token_command_returns_none() {
        // `aws_region` is not a credential, and S3 is not a bearer backend.
        let opts = opts_with(&[("aws_region", "eu-west-1")]);
        assert!(credential_provider(&CloudType::Aws, Some(&opts))
            .unwrap()
            .is_none());
        assert!(credential_provider(&CloudType::Gcp, None)
            .unwrap()
            .is_none());
    }

    /// The same credential must yield the *same* `Arc`, and a different one a
    /// different `Arc`.
    ///
    /// This is the guard for the property the whole delegation rests on.
    /// `PlCredentialProvider::stable_cache_key` is the `Arc`'s address, and
    /// polars folds it into the key of its process-wide object-store cache. So:
    ///
    /// * if the same credential produced a fresh `Arc` each time, the store
    ///   cache would never hit and every read would rebuild its connection pool
    ///   — silently, with no test failing;
    /// * if a *different* credential reused an `Arc`, a rotated token would be
    ///   served by a store still holding the superseded one.
    ///
    /// Watched failing in both directions: keying the memo on the scheme alone
    /// fails the `assert_ne`, and removing the memo fails the `assert_eq`.
    #[test]
    fn provider_identity_tracks_the_credential_and_only_the_credential() {
        let key = |provider: &PlCredentialProvider| provider.stable_cache_key().unwrap();

        let a = credential_provider(
            &CloudType::Gcp,
            Some(&opts_with(&[("bearer_token", "tok-a")])),
        )
        .unwrap()
        .unwrap();
        let a_again = credential_provider(
            &CloudType::Gcp,
            Some(&opts_with(&[("bearer_token", "tok-a")])),
        )
        .unwrap()
        .unwrap();
        let b = credential_provider(
            &CloudType::Gcp,
            Some(&opts_with(&[("bearer_token", "tok-b")])),
        )
        .unwrap()
        .unwrap();

        assert_eq!(
            key(&a),
            key(&a_again),
            "the same credential must reuse one provider, or polars' store cache never hits"
        );
        assert_ne!(
            key(&a),
            key(&b),
            "a changed credential must not reuse the provider, or a rotated token \
             is served by a store holding the old one"
        );
    }

    #[test]
    fn test_token_command_caches() {
        // The command appends to a file on each run; a second call within the
        // TTL must reuse the cached token and not run it again.
        //
        // This is the anti-stampede property: under the streaming engine a burst
        // of concurrent first-time reads must collapse to one subprocess, not
        // one per object. It matters more after delegation, not less —
        // `exec_with_rebuild_retry_on_err` rebuilds the store on *any*
        // object-store error including a 404, and each rebuild asks the provider
        // again.
        let dir = std::env::temp_dir().join(format!("pcv_cmd_cache_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let marker = dir.join("runs");
        let cmd = format!(
            "printf x >> {m}; printf 'tok-cache-{p}'",
            m = marker.display(),
            p = std::process::id()
        );
        let opts = opts_with(&[("token_command", cmd.as_str())]);

        let provider = credential_provider(&CloudType::Gcp, Some(&opts))
            .unwrap()
            .unwrap();
        // Two separate `into_gcp_provider()` conversions, i.e. what two store
        // builds look like. polars' own credential cache is fresh each time, so
        // only the cache *inside* our provider can collapse these.
        let first = fetch(&provider).unwrap();
        let second = fetch(&provider).unwrap();
        assert_eq!(first, second);
        // Command ran exactly once despite two calls.
        assert_eq!(std::fs::read(&marker).unwrap().len(), 1);

        // And once more through a freshly *obtained* provider, which the memo
        // must serve from cache rather than rebuild — otherwise the token cache
        // inside it would be discarded and the command would run again.
        let same = credential_provider(&CloudType::Gcp, Some(&opts))
            .unwrap()
            .unwrap();
        assert_eq!(fetch(&same).unwrap(), first);
        assert_eq!(std::fs::read(&marker).unwrap().len(), 1);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_token_command_reports_a_future_expiry() {
        // polars treats an expiry already in the past as a hard error, so the
        // assumed window must always be ahead of now. Asserted on the value the
        // provider reports rather than on the constant, since it is the sum that
        // has to be in the future.
        let opts = opts_with(&[("token_command", "printf tok")]);
        let provider = credential_provider(&CloudType::Gcp, Some(&opts))
            .unwrap()
            .unwrap();
        // A successful fetch is itself the assertion: `FetchedCredentialsCache`
        // rejects a past expiry with "Invalid credential expiry".
        fetch(&provider).expect("a past expiry would be rejected by polars");
    }

    #[test]
    fn test_failing_token_command_errors() {
        // The failure now surfaces when the credential is *fetched* rather than
        // when the provider is chosen — the provider is a closure, so it cannot
        // report a subprocess failure before being called.
        let opts = opts_with(&[("token_command", "exit 3")]);
        let provider = credential_provider(&CloudType::Gcp, Some(&opts))
            .unwrap()
            .unwrap();
        let err = fetch(&provider).unwrap_err();
        assert!(
            err.contains("exited with"),
            "expected the exit status in the message, got {err}"
        );
    }

    #[test]
    fn test_service_account_adc_is_left_to_object_store() {
        // A non-federated ADC must not be intercepted (no provider installed);
        // no command is run, and object_store's own chain handles it.
        let (dir, path) = write_adc("sa", r#"{"type": "service_account", "client_email": "x"}"#);
        let opts = opts_with(&[("google_application_credentials", path.to_str().unwrap())]);
        assert!(credential_provider(&CloudType::Gcp, Some(&opts))
            .unwrap()
            .is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_explicit_service_account_skips_delegation() {
        // Even with a federated file present, an explicit SA option means the
        // caller chose that credential; we must not override it.
        let (dir, path) = write_adc("fed", r#"{"type": "external_account"}"#);
        let opts = opts_with(&[
            ("google_service_account", "/some/sa.json"),
            ("google_application_credentials", path.to_str().unwrap()),
        ]);
        assert!(credential_provider(&CloudType::Gcp, Some(&opts))
            .unwrap()
            .is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_federated_adc_installs_a_provider() {
        // The positive half of the two tests above: a federated ADC is exactly
        // the case object_store cannot load, so it must install a provider.
        // Without this, deleting the detection entirely would leave both
        // negative tests green.
        let (dir, path) = write_adc("fed_pos", r#"{"type": "external_account"}"#);
        let opts = opts_with(&[("google_application_credentials", path.to_str().unwrap())]);
        assert!(
            credential_provider(&CloudType::Gcp, Some(&opts))
                .unwrap()
                .is_some(),
            "a federated ADC must be delegated"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_anonymous_installs_no_credential() {
        // Anonymous access is requested by skipping signing; installing a
        // credential alongside it would contradict the request.
        let opts = opts_with(&[("anonymous", "true"), ("bearer_token", "tok")]);
        assert!(credential_provider(&CloudType::Gcp, Some(&opts))
            .unwrap()
            .is_none());
    }
}
