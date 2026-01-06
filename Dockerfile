# Lightweight Linux Docker container for building polars-cv
# Uses Debian slim for better Rust compatibility than Alpine
FROM debian:bookworm-slim

# Install system dependencies in a single layer for better caching
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    ca-certificates \
    git \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable \
    && rustup component add rustfmt clippy \
    && chmod -R a+w $RUSTUP_HOME $CARGO_HOME \
    && rustup --version

# Install uv using official installer
# The installer installs to $HOME/.local/bin by default
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH
ENV PATH=/root/.local/bin:$PATH

# Verify installation
RUN uv --version

# Set working directory
WORKDIR /workspace

# Set environment variables
ENV CARGO_TERM_COLOR=always \
    RUST_BACKTRACE=1

# Default command: build the project
# This assumes the project is mounted at /workspace
CMD ["bash", "-c", "if [ -d polars-cv ]; then cd polars-cv; fi && uv venv .venv && source .venv/bin/activate && uv pip install maturin && maturin build --release"]

