#!/bin/bash
# build_image.sh
# Usage: ./build_image.sh

IMAGE_NAME="pipeline_segmentation:3090"
DOCKERFILE="docker/Dockerfile.3090"

echo "🏗️  Building Docker Image: $IMAGE_NAME..."

cd ..

# Build from attention root to allow copying 'src/'
docker build -f $DOCKERFILE -t $IMAGE_NAME .

if [ $? -eq 0 ]; then
    echo "✅ Image built successfully."
    echo "   To check: docker images | grep pipeline"
else
    echo "❌ Build failed."
    exit 1
fi