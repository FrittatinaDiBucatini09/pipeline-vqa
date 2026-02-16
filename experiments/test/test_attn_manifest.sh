#!/bin/bash
# Test T2: Attention Map Manifest Generation
# Tests that generate_heatmaps.py generates vqa_manifest.csv

set -e

TEST_ID="T2_attn_manifest"
TEST_DIR="/home/rbalzani/medical-vqa/Thesis/experiments/test"
ATTN_DIR="/home/rbalzani/medical-vqa/Thesis/preprocessing/attention_map"
OUTPUT_DIR="$TEST_DIR/output_$TEST_ID"

echo "========================================"
echo "TEST $TEST_ID: Attention Map Manifest Generation"
echo "========================================"
echo "Output: $OUTPUT_DIR"

# Cleanup previous run
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Create minimal test config
cat > "$OUTPUT_DIR/test.conf" <<'EOF'
# Minimal config for fast testing
BATCH_SIZE=2
STOP_AFTER=50
ALPHA=0.5
COLORMAP="jet"
SAVE_RAW_CAM="false"
SKIP_CRF="true"
USE_DYNAMIC_PROMPTS="true"
USE_VISUAL_REGIONS="false"
ENABLE_BODY_MASK="false"
DRAW_LABELS="false"
METADATA_FILENAME="gemex_VQA_mimic_mapped.csv"
NUM_WORKERS=2
EOF

cd "$ATTN_DIR"

echo "[INFO] Launching attention map generation with limit=50 on GPU 3..."
export CUDA_VISIBLE_DEVICES=3
./scripts/run_heatmap_gen.sh "$OUTPUT_DIR/test.conf" 2>&1 | tee "$OUTPUT_DIR/run.log"

echo ""
echo "========================================"
echo "VERIFICATION"
echo "========================================"

# Check if vqa_manifest.csv exists
if [ -f "$OUTPUT_DIR/vqa_manifest.csv" ]; then
    echo "✅ vqa_manifest.csv found"

    # Count rows
    ROW_COUNT=$(wc -l < "$OUTPUT_DIR/vqa_manifest.csv")
    echo "✅ Row count: $((ROW_COUNT - 1)) data rows (+ 1 header)"

    # Check columns
    HEADER=$(head -1 "$OUTPUT_DIR/vqa_manifest.csv")
    echo "✅ Columns: $HEADER"

    # Sample 3 rows
    echo ""
    echo "Sample rows:"
    head -4 "$OUTPUT_DIR/vqa_manifest.csv" | tail -3

    # Verify image files exist
    echo ""
    echo "Verifying image paths..."
    MISSING=0
    while IFS=, read -r img_path question answer; do
        if [ "$img_path" != "image_path" ]; then  # Skip header
            FULL_PATH="$OUTPUT_DIR/$img_path"
            if [ ! -f "$FULL_PATH" ]; then
                echo "❌ Missing: $FULL_PATH"
                MISSING=$((MISSING + 1))
            fi
        fi
    done < "$OUTPUT_DIR/vqa_manifest.csv"

    if [ $MISSING -eq 0 ]; then
        echo "✅ All image paths verified"
    else
        echo "❌ $MISSING images missing"
    fi

    echo ""
    echo "TEST $TEST_ID: PASSED ✅"
else
    echo "❌ vqa_manifest.csv NOT FOUND"
    echo "TEST $TEST_ID: FAILED ❌"
    exit 1
fi

echo "========================================"
