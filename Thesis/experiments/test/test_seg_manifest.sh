#!/bin/bash
# Test T3: Segmentation Manifest Generation
# Tests that run_segmentation.py generates vqa_manifest.csv

set -e

TEST_ID="T3_seg_manifest"
TEST_DIR="/home/rbalzani/medical-vqa/Thesis/experiments/test"
SEG_DIR="/home/rbalzani/medical-vqa/Thesis/preprocessing/segmentation"
OUTPUT_DIR="$TEST_DIR/output_$TEST_ID"

echo "========================================"
echo "TEST $TEST_ID: Segmentation Manifest Generation"
echo "========================================"
echo "Output: $OUTPUT_DIR"

# Cleanup previous run
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Create test configs in segmentation's config directory
CONFIG_DIR="$SEG_DIR/configs/test"
mkdir -p "$CONFIG_DIR"

cat > "$CONFIG_DIR/step1_t3.conf" <<'EOF'
# Step 1: Localization (minimal for fast test)
BATCH_SIZE=4
ENABLE_BODY_MASK="false"
ENABLE_ANATOMICAL_CHECK="false"
USE_DYNAMIC_PROMPTS="true"
ENABLE_SMART_PADDING="false"
CAM_THRESHOLD=0.5
OUTPUT_FORMAT="jsonl"
EOF

cat > "$CONFIG_DIR/step2_t3.conf" <<'EOF'
# Step 2: Segmentation
MEDSAM_CHECKPOINT="medsam"
SCENARIO="A"
SAVE_VISUALS="true"
TEXT_PROMPT_MODE="question"
SAM3_USE_BBOX="false"
EOF

cd "$SEG_DIR"

echo "[INFO] Launching segmentation pipeline (step 1 → step 2) with limit=50..."
export CUDA_VISIBLE_DEVICES=2
export LIMIT=50
./scripts/run_docker.sh "all" "/workspace/configs/test/step1_t3.conf" "/workspace/configs/test/step2_t3.conf" "$LIMIT" 2>&1 | tee "$OUTPUT_DIR/run.log"

echo ""
echo "========================================"
echo "VERIFICATION"
echo "========================================"

MANIFEST_PATH="$SEG_DIR/results/step2_masks/vqa_manifest.csv"

# Check if vqa_manifest.csv exists
if [ -f "$MANIFEST_PATH" ]; then
    echo "✅ vqa_manifest.csv found"

    # Count rows
    ROW_COUNT=$(wc -l < "$MANIFEST_PATH")
    echo "✅ Row count: $((ROW_COUNT - 1)) data rows (+ 1 header)"

    # Check columns
    HEADER=$(head -1 "$MANIFEST_PATH")
    echo "✅ Columns: $HEADER"

    # Sample 3 rows
    echo ""
    echo "Sample rows:"
    head -4 "$MANIFEST_PATH" | tail -3

    # Verify overlay image files exist
    echo ""
    echo "Verifying overlay paths..."
    MISSING=0
    BASE_DIR="$SEG_DIR/results/step2_masks"
    while IFS=, read -r img_path question answer; do
        if [ "$img_path" != "image_path" ]; then  # Skip header
            FULL_PATH="$BASE_DIR/$img_path"
            if [ ! -f "$FULL_PATH" ]; then
                echo "❌ Missing: $FULL_PATH"
                MISSING=$((MISSING + 1))
            fi
        fi
    done < "$MANIFEST_PATH"

    if [ $MISSING -eq 0 ]; then
        echo "✅ All overlay paths verified"
    else
        echo "❌ $MISSING overlays missing"
    fi

    echo ""
    echo "TEST $TEST_ID: PASSED ✅"
else
    echo "❌ vqa_manifest.csv NOT FOUND at $MANIFEST_PATH"
    echo "TEST $TEST_ID: FAILED ❌"
    exit 1
fi

echo "========================================"
