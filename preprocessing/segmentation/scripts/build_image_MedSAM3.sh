#!/bin/bash
# build_image.sh
# Usage: ./build_image_MedSAM3.sh

IMAGE_NAME="pipeline_segmentation:sam3"
DOCKERFILE="docker/Dockerfile.sam3"

echo "🏗️  Building Docker Image: $IMAGE_NAME..."

cd ..

# Build from segmentation root to allow copying 'src/'
docker build -f $DOCKERFILE -t $IMAGE_NAME .

if [ $? -eq 0 ]; then
    echo "✅ Image built successfully."
    echo "   To check: docker images | grep pipeline"
else
    echo "❌ Build failed."
    exit 1
fi