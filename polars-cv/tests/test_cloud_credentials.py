"""Ambient cloud credentials, checked at the user-facing entry points.

Each test runs in a subprocess, so the credential environment it builds (an
ADC file, a fake ``gcloud`` on ``PATH``, an isolated ``HOME``) is the one the
plugin reads, and the process-wide credential-provider memo starts empty.

The fake ``gcloud`` fails, so no request ever reaches GCS: that the provider
ran is proven by the marker file it writes and by the stderr it printed
surfacing in the error. These tests are offline.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests.conftest import plugin_required

pytestmark = plugin_required

_FAKE_STDERR = "Reauthentication failed. refresh token expired"


def _federated_env(tmp_path: Path) -> tuple[dict[str, str], Path]:
    """An environment whose ambient ADC is federated, with a failing ``gcloud``."""
    adc = tmp_path / "adc.json"
    adc.write_text(
        json.dumps(
            {
                "type": "external_account_authorized_user",
                "audience": "//iam.googleapis.com/locations/global/workforcePools/x",
                "refresh_token": "r",
            }
        )
    )
    marker = tmp_path / "gcloud-ran"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gcloud = bin_dir / "gcloud"
    gcloud.write_text(
        f'#!/bin/sh\necho "$@" >> "{marker}"\necho "{_FAKE_STDERR}" >&2\nexit 1\n'
    )
    gcloud.chmod(0o755)
    env = {
        k: v for k, v in os.environ.items() if k != "POLARS_CV_DISABLE_GCS_FEDERATION"
    }
    env.update(
        GOOGLE_APPLICATION_CREDENTIALS=str(adc),
        HOME=str(tmp_path),
        PATH=f"{bin_dir}{os.pathsep}{env.get('PATH', '')}",
    )
    return env, marker


def _run(code: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


class TestAmbientFederatedAdc:
    def test_a_read_with_no_cloud_options_uses_the_federated_adc(
        self, tmp_path: Path
    ) -> None:
        # The default gcloud setup under workforce federation: no
        # cloud_options anywhere. The ambient ADC must be delegated to gcloud
        # exactly as it is when an (empty) CloudOptions is passed.
        env, marker = _federated_env(tmp_path)
        result = _run(
            """
            import polars as pl
            import polars_cv  # noqa: F401
            pl.DataFrame({"p": ["gs://bucket/a.png"]}).select(
                pl.col("p").cv.read_bytes()
            )
            """,
            env,
        )
        assert result.returncode != 0
        assert marker.exists(), result.stderr
        assert "print-access-token" in marker.read_text()
        assert _FAKE_STDERR in result.stderr

    def test_the_error_names_the_credential_source(self, tmp_path: Path) -> None:
        env, _ = _federated_env(tmp_path)
        result = _run(
            """
            import polars as pl
            import polars_cv  # noqa: F401
            pl.DataFrame({"p": ["gs://bucket/a.png"]}).select(
                pl.col("p").cv.read_bytes()
            )
            """,
            env,
        )
        assert "gcloud ADC (external_account_authorized_user" in result.stderr


class TestCheckCloudAccess:
    def test_a_failing_credential_raises_before_any_row(self, tmp_path: Path) -> None:
        env, marker = _federated_env(tmp_path)
        result = _run(
            """
            import polars_cv
            try:
                polars_cv.check_cloud_access("gs://bucket/prefix/")
            except polars_cv.CloudCredentialError as e:
                print("RAISED:", e)
            """,
            env,
        )
        assert result.returncode == 0, result.stderr
        assert "RAISED:" in result.stdout
        assert "gcloud ADC (external_account_authorized_user" in result.stdout
        assert str(tmp_path / "adc.json") in result.stdout
        assert _FAKE_STDERR in result.stdout
        assert marker.exists()

    def test_a_failing_token_command_is_named(self, tmp_path: Path) -> None:
        env, _ = _federated_env(tmp_path)
        result = _run(
            """
            import polars_cv
            opts = polars_cv.CloudOptions(token_command="echo nope >&2; exit 3")
            try:
                polars_cv.check_cloud_access("gs://bucket/x.png", cloud_options=opts)
            except polars_cv.CloudCredentialError as e:
                print("RAISED:", e)
            """,
            env,
        )
        assert result.returncode == 0, result.stderr
        assert "RAISED:" in result.stdout
        assert "token_command `echo nope >&2; exit 3`" in result.stdout
        assert "nope" in result.stdout

    def test_a_non_remote_path_is_refused(self) -> None:
        import polars_cv

        with pytest.raises(ValueError, match="not a remote object-store path"):
            polars_cv.check_cloud_access("/tmp/local.png")
