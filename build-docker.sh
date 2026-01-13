#!/bin/bash
# Build script for polars-cv using Docker
colima stop
colima delete -f
colima start --cpu 4 --memory 8
sleep 1 && wait

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="polars-cv-builder:latest"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Building polars-cv in Docker container...${NC}"

# Check Docker version compatibility
if ! docker version &>/dev/null; then
    echo -e "${RED}Error: Docker is not running or not accessible${NC}"
    exit 1
fi

# Build the Docker image if it doesn't exist or if --rebuild is passed
if [[ "$1" == "--rebuild" ]] || ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
    echo -e "${BLUE}Building Docker image...${NC}"
    docker build -t "$IMAGE_NAME" "$SCRIPT_DIR"
fi

# Run the build
# Build in container-local directory to avoid filesystem sync issues with mounted volumes
# The target directory will be in the container, and we'll copy wheels out at the end
echo -e "${BLUE}Running build in container...${NC}"

# Ensure target/wheels directory exists on host (check both possible locations)
mkdir -p "$SCRIPT_DIR/polars-cv/target/wheels"
mkdir -p "$SCRIPT_DIR/target/wheels"

# Try to run the container, handle version mismatch errors
if ! CONTAINER_ID=$(docker run -d \
    -v "$SCRIPT_DIR:/workspace-source:ro" \
    -v "$SCRIPT_DIR/polars-cv/target/wheels:/workspace/wheels-output:rw" \
    -v "$SCRIPT_DIR/target/wheels:/workspace/root-wheels-output:rw" \
    -v cargo-cache:/usr/local/cargo/registry \
    -v cargo-git-cache:/usr/local/cargo/git \
    -e CARGO_TERM_COLOR=always \
    -e RUST_BACKTRACE=1 \
    -e CARGO_BUILD_JOBS=4 \
    "$IMAGE_NAME" \
    bash -c "
        set -e
        # Copy entire workspace to container-local directory (avoids filesystem sync issues)
        # The workspace root contains both view-buffer/ and polars-cv/ subdirectories
        # Use tar to copy while excluding build artifacts
        cd /workspace-source
        tar --exclude='target' --exclude='.venv' --exclude='__pycache__' \
            --exclude='.git' --exclude='site' --exclude='*.pyc' \
            --exclude='.pytest_cache' -cf - . | tar -C /workspace -xf -
        # Build from the polars-cv subdirectory
        cd /workspace/polars-cv
        # Create Cargo config to override workspace LTO settings (reduce memory usage)
        # The workspace Cargo.toml uses 'lto = \"fat\"' which is very memory-intensive
        mkdir -p /workspace/.cargo
        cat > /workspace/.cargo/config.toml << 'EOF'
[profile.release]
lto = false
codegen-units = 16
EOF
        # Build in container-local target directory
        uv venv .venv
        source .venv/bin/activate
        uv pip install maturin
        maturin build --release
        # Copy wheels to writable output directory
        # Maturin may put wheels in workspace root target/ or in polars-cv/target/
        # Check workspace root first (most likely for monorepo)
        WHEEL_COUNT=0
        if [ -d /workspace/target/wheels ] && [ -n \"\$(ls -A /workspace/target/wheels/*.whl 2>/dev/null)\" ]; then
            cp -v /workspace/target/wheels/*.whl /workspace/root-wheels-output/
            WHEEL_COUNT=\$(ls -1 /workspace/target/wheels/*.whl | wc -l)
            echo \"✓ Copied \${WHEEL_COUNT} wheel(s) from /workspace/target/wheels/ to root target/wheels/\"
        fi
        # Also check polars-cv subdirectory
        if [ -d target/wheels ] && [ -n \"\$(ls -A target/wheels/*.whl 2>/dev/null)\" ]; then
            cp -v target/wheels/*.whl /workspace/wheels-output/
            WHEEL_COUNT=\$(ls -1 target/wheels/*.whl | wc -l)
            echo \"✓ Copied \${WHEEL_COUNT} wheel(s) from target/wheels/ to polars-cv/target/wheels/\"
        fi
        # If no wheels found, show debug info
        if [ \$WHEEL_COUNT -eq 0 ]; then
            echo \"Warning: No wheels found. Checking possible locations...\"
            echo \"  /workspace/target/wheels: \$([ -d /workspace/target/wheels ] && ls -la /workspace/target/wheels/ 2>/dev/null | head -5 || echo 'does not exist')\"
            echo \"  /workspace/polars-cv/target/wheels: \$([ -d target/wheels ] && ls -la target/wheels/ 2>/dev/null | head -5 || echo 'does not exist')\"
        fi
    " 2>&1); then
    echo -e "${RED}Error: Failed to start Docker container${NC}"
    echo "$CONTAINER_ID" | head -5
    echo ""
    echo -e "${RED}This might be due to a Docker version mismatch.${NC}"
    echo -e "${BLUE}Please ensure your Docker daemon is up to date.${NC}"
    echo -e "${BLUE}On macOS/Windows: Update Docker Desktop${NC}"
    echo -e "${BLUE}On Linux: Update Docker engine (e.g., sudo apt-get update && sudo apt-get upgrade docker.io)${NC}"
    exit 1
fi

# Wait for build to complete and show logs
docker logs -f "$CONTAINER_ID"
EXIT_CODE=$(docker wait "$CONTAINER_ID")

# Clean up
docker rm "$CONTAINER_ID" > /dev/null 2>&1 || true

if [ "$EXIT_CODE" -ne 0 ]; then
    echo -e "${RED}Build failed with exit code $EXIT_CODE${NC}"
    exit "$EXIT_CODE"
fi

echo -e "${GREEN}Build complete! Check the following locations for the built wheel:${NC}"
echo -e "${BLUE}  - polars-cv/target/wheels/${NC}"
echo -e "${BLUE}  - target/wheels/ (workspace root)${NC}"

echo -e "${RED}Stopping colima...${NC}"
colima stop
echo -e "${RED}Deleting colima...${NC}"
colima delete -f