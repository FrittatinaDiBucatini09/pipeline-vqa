#!/bin/bash
# ==============================================================================
# BUILD SCRIPT: MedCLIP Agentic Routing Docker Image
# ==============================================================================
# Usage: ./build_image.sh
# ==============================================================================
set -e

IMAGE_NAME="medclip_routing:3090"

# Get the absolute path to the medclip_routing directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🏗️  Building Docker Image: $IMAGE_NAME..."
echo "   Stage root: $STAGE_ROOT"

docker build -f "$STAGE_ROOT/docker/Dockerfile.3090" -t "$IMAGE_NAME" "$STAGE_ROOT"

if [ $? -eq 0 ]; then
    echo "✅ Image built successfully."
    echo "   To check: docker images | grep medclip_routing"
else
    echo "❌ Build failed."
    exit 1
fi
