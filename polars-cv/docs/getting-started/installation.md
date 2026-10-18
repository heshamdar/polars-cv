# Installation

## Prerequisites

- Python 3.11+
- Polars 2.0+

## Installing with pip

```bash
pip install polars-cv
```

## Installing with uv

```bash
uv add polars-cv
```

## Installing from Source

Building from source (e.g., for development) needs a Rust toolchain, 1.96 or
newer (`rustup update stable`), and [uv](https://docs.astral.sh/uv/):

```bash
# Clone the repository
git clone https://github.com/heshamdar/polars-cv.git
cd polars-cv/polars-cv

# Install the Python dev dependencies (not the project itself)
uv sync --group dev --no-install-project

# Compile the Rust plugin into the environment (debug build)
uv run --no-sync maturin develop
```

`maturin develop` installs the package in editable mode: Python edits take
effect immediately, but after changing Rust you must run it again.

## Verifying Installation

```python
import polars_cv

print(polars_cv.build_info())
```

`build_info()` reports the package version, the compiled plugin's version and
the installed distribution's; all three should match. From a source checkout
it also compares the Rust source hash the plugin was built from with the
working tree's, so a stale build shows up here.

## Optional Dependencies

### Cloud Storage (S3, GCS, Azure, HTTP)

Remote reads are built in; there is nothing extra to install. Credentials come
from the usual environment variables and config files (`AWS_*`,
`GOOGLE_APPLICATION_CREDENTIALS`, `AZURE_STORAGE_*`, `~/.aws/`) or explicitly
from `CloudOptions`. See
[Sources: `file_path` and Cloud Access](../user-guide/concepts/sources.md#file_path-and-cloud-access)
for every option, including federated Google credentials and token commands.

### Graph Visualization

`show_graph()` needs the `viz` extra and Graphviz's `dot` program, which
pip cannot install:

```bash
pip install 'polars-cv[viz]'
sudo apt-get install graphviz   # or: brew install graphviz, conda install graphviz
```

### PyTorch

PyTorch is not a dependency. An `array` sink of a known shape converts to one
batched array, and from there to a tensor without a copy:

```python
import polars as pl
import torch
from polars_cv import Pipeline

pipe = Pipeline().source("image_bytes", dtype="u8").resize(height=224, width=224)
df = pl.DataFrame({"image": [png_bytes]})
out = df.select(x=pl.col("image").cv.pipe(pipe).sink("array", shape=[224, 224, 3]))

batch = torch.from_numpy(out["x"].to_numpy())  # (rows, 224, 224, 3) uint8
```

For one tensor per row, the `torch` sink and `numpy_from_struct` are covered
in [Functions](../api/functions.md#numpy-conversion).

### Documentation (development only)

From a source checkout, install the docs dependency group and preview the site:

```bash
uv sync --group docs --no-install-project
uv run mkdocs serve
```
