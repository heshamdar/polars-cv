//! Federated / brokered GCS credentials, obtained without reimplementing any
//! provider's token-exchange protocol.
//!
//! `object_store`'s GCS credential loader only understands `service_account`
//! and `authorized_user` Application Default Credentials. Federated setups
//! (Workload / Workforce Identity Federation — e.g. an external OIDC identity
//! exchanged into Google) write an `external_account*` ADC its loader rejects.
//!
//! Rather than owning Google's federation protocol, polars-cv gets an access
//! token from an external command and installs it as a static GCS credential:
//!
//! - If `token_command` is set (Python: `CloudOptions.gcs_token_command`), run
//!   it and use its stdout as the bearer token. This is provider-agnostic: any
//!   broker, script, or CLI that prints a GCS access token works.
//! - Otherwise, if the ambient ADC is a federated `external_account*` file,
//!   delegate to `gcloud auth application-default print-access-token`, which
//!   understands the full federation matrix (every `credential_source`,
//!   service-account impersonation, workload/workforce pools). This requires
//!   the `gcloud` CLI on `PATH`; set `POLARS_CV_DISABLE_GCS_FEDERATION=1` to
//!   turn the auto-delegation off.
//!
//! Non-federated ambient credentials (`service_account` / `authorized_user`)
//! are left to `object_store`. Obtained tokens are cached until shortly before
//! their assumed expiry so a batch of concurrent reads runs the command once,
//! not once per object.

use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::cloud::{CloudError, CloudOptions};

/// The delegation command used for a detected federated ADC.
const GCLOUD_ADC_COMMAND: &str = "gcloud auth application-default print-access-token";
/// Assumed lifetime of a command-sourced token. GCS access tokens live ~1h; we
/// cache for a shorter window so a token that is a little shorter-lived than
/// that is still refreshed before it lapses.
const ASSUMED_TOKEN_TTL: Duration = Duration::from_secs(55 * 60);

/// Obtain a GCS access token for a federated/brokered setup, if one applies.
///
/// Returns `Ok(None)` when there is nothing to do — no `token_command`, an
/// explicit service account was given, or the ambient ADC is a plain
/// `service_account` / `authorized_user` file `object_store` handles itself.
/// Returns `Err` when a token was called for but could not be produced (the
/// command failed), so the failure is surfaced clearly rather than as a
/// downstream `object_store` credential error.
pub(crate) fn maybe_obtain_gcs_token(
    options: Option<&CloudOptions>,
) -> Result<Option<String>, CloudError> {
    // 1. An explicit token command is the most direct intent — always honor it.
    if let Some(opts) = options {
        if let Some(cmd) = opts.token_command.as_deref() {
            if !cmd.trim().is_empty() {
                return token_cached(&format!("cmd:{cmd}"), cmd).map(Some);
            }
        }
    }

    // 2. Auto-delegation for a detected federated ADC (opt-out via env).
    if std::env::var_os("POLARS_CV_DISABLE_GCS_FEDERATION").is_some() {
        return Ok(None);
    }
    // An explicit service account is the caller's stated intent — don't override.
    if let Some(opts) = options {
        if opts.config.contains_key("google_service_account")
            || opts.config.contains_key("google_service_account_key")
        {
            return Ok(None);
        }
    }

    let path = match locate_adc_path(options) {
        Some(p) => p,
        None => return Ok(None),
    };
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        // Unreadable ADC: leave it to object_store to report in context.
        Err(_) => return Ok(None),
    };
    if is_federated_adc(&bytes) {
        token_cached(&format!("adc:{}", cache_key(&path)), GCLOUD_ADC_COMMAND).map(Some)
    } else {
        // service_account / authorized_user / unknown -> object_store's job.
        Ok(None)
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
// Token cache + command execution
// ---------------------------------------------------------------------------

struct CachedToken {
    token: String,
    expires_at: Instant,
}

fn cache() -> &'static Mutex<HashMap<String, CachedToken>> {
    static CACHE: OnceLock<Mutex<HashMap<String, CachedToken>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

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

/// Return a cached token for `key` if still valid, otherwise run `command` and
/// cache its output.
///
/// The cache lock is held across the command run so a burst of concurrent
/// first-time reads (each on its own prefetch thread) collapses to a single
/// invocation rather than a stampede of subprocesses.
fn token_cached(key: &str, command: &str) -> Result<String, CloudError> {
    let mut guard = cache().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(c) = guard.get(key) {
        if c.expires_at > Instant::now() {
            return Ok(c.token.clone());
        }
    }
    let token = run_token_command(command)?;
    guard.insert(
        key.to_string(),
        CachedToken {
            token: token.clone(),
            expires_at: Instant::now() + ASSUMED_TOKEN_TTL,
        },
    );
    Ok(token)
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

    fn opts_with(pairs: &[(&str, &str)]) -> CloudOptions {
        let map = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect::<HashMap<_, _>>();
        CloudOptions::from_map(&map)
    }

    fn write_adc(name: &str, body: &str) -> (PathBuf, PathBuf) {
        let dir = std::env::temp_dir().join(format!("pcv_gauth_{}_{name}", std::process::id()));
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

    #[test]
    fn test_token_command_is_run_and_used() {
        let opts = opts_with(&[("token_command", "printf 'ya29.from-command'")]);
        assert_eq!(
            maybe_obtain_gcs_token(Some(&opts)).unwrap(),
            Some("ya29.from-command".to_string())
        );
    }

    #[test]
    fn test_token_command_caches() {
        // The command appends to a file on each run; a second obtain within the
        // TTL must reuse the cache and not run it again.
        let dir = std::env::temp_dir().join(format!("pcv_cmd_cache_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let marker = dir.join("runs");
        // Distinct token per key so this test is independent of others.
        let cmd = format!(
            "printf x >> {m}; printf 'tok-cache-{p}'",
            m = marker.display(),
            p = std::process::id()
        );
        let opts = opts_with(&[("token_command", cmd.as_str())]);

        let first = maybe_obtain_gcs_token(Some(&opts)).unwrap().unwrap();
        let second = maybe_obtain_gcs_token(Some(&opts)).unwrap().unwrap();
        assert_eq!(first, second);
        // Command ran exactly once despite two obtains.
        assert_eq!(std::fs::read(&marker).unwrap().len(), 1);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_failing_token_command_errors() {
        let opts = opts_with(&[("token_command", "exit 3")]);
        let err = maybe_obtain_gcs_token(Some(&opts)).unwrap_err();
        assert!(err.to_string().contains("exited with"));
    }

    #[test]
    fn test_service_account_adc_is_left_to_object_store() {
        // A non-federated ADC must not be intercepted (returns Ok(None)); no
        // command is run.
        let (dir, path) = write_adc("sa", r#"{"type": "service_account", "client_email": "x"}"#);
        let opts = opts_with(&[("google_application_credentials", path.to_str().unwrap())]);
        assert!(maybe_obtain_gcs_token(Some(&opts)).unwrap().is_none());
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
        assert!(maybe_obtain_gcs_token(Some(&opts)).unwrap().is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_token_command_wins_over_service_account() {
        // token_command is the most explicit intent; it applies even alongside
        // other config keys.
        let opts = opts_with(&[
            ("google_service_account", "/some/sa.json"),
            ("token_command", "printf 'ya29.explicit'"),
        ]);
        assert_eq!(
            maybe_obtain_gcs_token(Some(&opts)).unwrap(),
            Some("ya29.explicit".to_string())
        );
    }
}
