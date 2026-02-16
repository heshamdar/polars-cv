# Docker Build Environment

This directory contains Docker configuration for building `polars-cv` on Linux.

## Quick Start

### Build the Docker image

```bash
docker build -t polars-cv-builder:latest .
```

### Build the project in Docker

```bash
# Build using docker run
docker run --rm -v $(pwd):/workspace polars-cv-builder:latest

# Or use docker-compose for easier management
docker-compose up builder
```

### Interactive Development

For interactive development, use docker-compose:

```bash
# Start the container (runs in background)
docker-compose up -d builder

# Execute commands in the container
docker-compose exec builder bash

# Inside the container, you can:
cd polars-cv
uv venv .venv
source .venv/bin/activate
uv pip install maturin
maturin develop --release

# Or build wheels
maturin build --release
```

### Build for Specific Python Version

The container includes Python 3.9+ by default. To use a specific Python version:

```bash
# Inside the container
uv python install 3.12
uv venv .venv --python 3.12
source .venv/bin/activate
maturin build --release
```

### Cross-compilation

To build for different Linux targets:

```bash
# Install cross-compilation target
docker-compose exec builder rustup target add x86_64-unknown-linux-gnu
docker-compose exec builder rustup target add aarch64-unknown-linux-gnu

# Build for specific target
docker-compose exec builder bash -c "cd polars-cv && source .venv/bin/activate && maturin build --release --target x86_64-unknown-linux-gnu"
```

### Clean Up

```bash
# Stop and remove containers
docker-compose down

# Remove volumes (clears Rust cache)
docker-compose down -v

# Remove image
docker rmi polars-cv-builder:latest
```

## Image Details

- **Base**: Debian Bookworm Slim (~150MB base)
- **Rust**: Latest stable toolchain
- **Python**: Python 3.11 (Debian default)
- **Tools**: uv, maturin, build-essential

The final image size is approximately 1-2GB including all build dependencies.
