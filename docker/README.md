# Docker build environment

A Debian (bookworm) image with the Rust toolchain, `lld`, `uv` and maturin, for
building polars-cv on Linux from any host. The image copies nothing from the
repository: the repository is mounted at `/workspace`.

Release wheels are built by CI (`.github/workflows/publish.yml`), not here.
This image is for local Linux builds and debugging.

All commands run from the repository root.

## Build a wheel

```bash
docker build -t polars-cv-builder:latest docker/
docker run --rm -v "$PWD":/workspace polars-cv-builder:latest
```

The default command runs `maturin build --release` in `polars-cv/`. Wheels land
in `target/wheels/`.

## Interactive development

```bash
docker compose -f docker/compose.yaml up -d builder
docker compose -f docker/compose.yaml exec builder bash

# inside the container
cd polars-cv
uv venv .venv && source .venv/bin/activate
uv pip install maturin
maturin develop          # debug: the development build
```

The compose service caches the Cargo registry in named volumes. To remove the
container and the caches:

```bash
docker compose -f docker/compose.yaml down -v
```

## Another Python version

The image ships Debian's Python 3.11, which is the minimum polars-cv supports.
To use another version:

```bash
uv python install 3.13
uv venv .venv --python 3.13
```

## Rust toolchain

The image installs the stable Rust current when it was built. Both crates set
`rust-version` (the MSRV) in their `Cargo.toml`, and cargo refuses an older
compiler. If a build fails that way, rebuild the image with `--no-cache` to pick
up a newer stable.
