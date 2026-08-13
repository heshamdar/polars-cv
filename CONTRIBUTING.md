
## Development

### Testing Against Multiple Python Versions

To test against multiple Python versions locally using `uv`:

```bash
# Use current Python environment (default - no arguments needed)
python scripts/test_multiple_python.py

# Test all supported Python versions (3.10, 3.11, 3.12, 3.13)
python scripts/test_multiple_python.py --all

# Test only minimum and maximum versions (faster)
python scripts/test_multiple_python.py --fast

# Test specific versions
python scripts/test_multiple_python.py --versions 3.10 3.13
```

**Prerequisites:**
- Install `uv`: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- For multi-version testing, install Python versions: `uv python install 3.10 3.11 3.12 3.13`

The test script will:
1. Use current environment if no versions specified (default behavior)
2. For specified versions, create isolated environments using `uv run --python`
3. Build the package (without cloud feature for speed)
4. Install test dependencies
5. Run the full test suite
6. Report which versions passed/failed

## Development

```bash
# Run Python tests
pytest tests/

# Build for development
maturin develop

# Build release
maturin build --release
```

## CI/CD and Publishing

This project uses GitHub Actions for continuous integration and publishing to PyPI.

### Workflows

- **CI** (`ci.yml`): Runs on push/PR to main
  - Linting (ruff, cargo clippy, cargo fmt)
  - Tests across Python 3.10-3.13 on Linux, macOS, Windows
  - Build verification

- **Publish** (`publish.yml`): Runs on release creation
  - Builds wheels for all platforms (Linux, macOS universal2, Windows)
  - Publishes to TestPyPI first for validation
  - Publishes to PyPI after TestPyPI succeeds

### Required GitHub Secrets

To enable publishing, configure these secrets in your GitHub repository settings
(Settings → Secrets and variables → Actions → New repository secret):

| Secret | Description | Source |
|--------|-------------|--------|
| (none required) | Uses trusted publishing with OIDC | Configure on PyPI |

### PyPI Trusted Publisher Setup

This project uses [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/) with OIDC,
which is more secure than API tokens. To set it up:

1. **PyPI** (https://pypi.org):
   - Go to your account → Publishing → Add a new pending publisher
   - Owner: `<your-github-username>`
   - Repository name: `polars_plugin_dev`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`

2. **TestPyPI** (https://test.pypi.org):
   - Same steps as above
   - Environment name: `testpypi`

### GitHub Environments

Create two environments in your repository (Settings → Environments):

1. **testpypi** - For TestPyPI publishing
2. **pypi** - For production PyPI publishing (consider adding required reviewers)

### Release Process

1. Bump the version in all six places it is recorded — they must agree.
   `polars-cv/tests/test_version_consistency.py` checks the first four for you;
   run it after bumping:
   - `polars-cv/Cargo.toml`
   - `view-buffer/Cargo.toml` (the two crates are versioned together)
   - `polars-cv/pyproject.toml`
   - `polars_cv.__version__` in `polars-cv/python/polars_cv/__init__.py`
   - `Cargo.lock` — refresh with `cargo update -p polars-cv -p view-buffer`
   - `polars-cv/uv.lock` — refresh with `uv lock` from `polars-cv/`

   The compiled extension's `polars_cv._lib.__version__` needs no action: it is
   baked in from `polars-cv/Cargo.toml` at build time. That is what makes a stale
   `.so` detectable — the install is editable, so Python edits are live but the
   extension is not rebuilt until you run `maturin develop`. See
   `polars_cv.build_info()`.
2. Roll the `CHANGELOG.md` `[Unreleased]` section into a dated entry for the new
   version, and leave a fresh empty `[Unreleased]` heading above it
3. Commit and push to main
4. Create a GitHub release with a version tag matching the bumped version,
   prefixed with `v` (e.g. `v0.1.0`). `.github/workflows/publish.yml` checks the
   tag against `polars-cv/pyproject.toml` and refuses to build if they disagree,
   so a release tagged ahead of (or behind) the manifests fails loudly instead
   of publishing the wrong version.
5. GitHub Actions automatically:
   - Verifies the tag matches the declared version
   - Builds `abi3` wheels for linux-x86_64, linux-aarch64 and macOS-arm64, plus
     an sdist, and rejects any wheel that is not `abi3`
   - Publishes to PyPI via trusted publishing

### Manual Publishing (Alternative)

If you prefer using API tokens instead of trusted publishing:

```bash
# Build wheels
maturin build --release

# Publish to TestPyPI
maturin publish --repository testpypi

# Publish to PyPI
maturin publish
```
