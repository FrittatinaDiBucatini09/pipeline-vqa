#!/bin/bash
# ==============================================================================
# End-to-End Dry Run Test
# Tests the pipeline without loading heavy models (Gemma)
# ==============================================================================
set -e

echo "=" | tr '=' '\075' | head -c 70; echo
echo "TEST 5: End-to-End Dry Run (Structure & Output Verification)"
echo "=" | tr '=' '\075' | head -c 70; echo

cd /home/rbalzani/medical-vqa/Thesis/preprocessing/medclip_routing

# Test 5.1: Check test CSV exists
echo ""
echo "📋 Test 5.1: Verify test CSV"
echo "-------------------------------------------------------------------"
if [ -f "test_sample_3.csv" ]; then
    ROWS=$(wc -l < test_sample_3.csv)
    echo "✅ Test CSV found: test_sample_3.csv ($ROWS lines including header)"
    head -2 test_sample_3.csv | cut -c1-70 | while read line; do
        echo "   $line..."
    done
else
    echo "❌ Test CSV not found"
    exit 1
fi

# Test 5.2: Check test config exists
echo ""
echo "📋 Test 5.2: Verify test config"
echo "-------------------------------------------------------------------"
if [ -f "configs/test_e2e.conf" ]; then
    echo "✅ Test config found: configs/test_e2e.conf"
    grep -E "STOP_AFTER|ENTITY_THRESHOLD|WORD_THRESHOLD" configs/test_e2e.conf | while read line; do
        echo "   $line"
    done
else
    echo "❌ Test config not found"
    exit 1
fi

# Test 5.3: Verify Docker image
echo ""
echo "📋 Test 5.3: Verify Docker image"
echo "-------------------------------------------------------------------"
if docker images | grep -q "medclip_routing.*3090"; then
    echo "✅ Docker image available: medclip_routing:3090"
    docker images | grep "medclip_routing"
else
    echo "❌ Docker image not found"
    exit 1
fi

# Test 5.4: Check script paths and permissions
echo ""
echo "📋 Test 5.4: Verify scripts"
echo "-------------------------------------------------------------------"
for script in submit_routing.sh scripts/run_docker.sh scripts/build_image.sh; do
    if [ -x "$script" ]; then
        echo "✅ $script (executable)"
    elif [ -f "$script" ]; then
        echo "⚠️  $script (exists but not executable)"
        chmod +x "$script"
        echo "   → Fixed: chmod +x applied"
    else
        echo "❌ $script (missing)"
        exit 1
    fi
done

# Test 5.5: Verify output directory structure
echo ""
echo "📋 Test 5.5: Output directory preparation"
echo "-------------------------------------------------------------------"
mkdir -p results
echo "✅ Output directory created: results/"

echo ""
echo "=" | tr '=' '\075' | head -c 70; echo
echo "✅ TEST 5: DRY RUN CHECKS PASSED"
echo ""
echo "⚠️  NOTE: Full E2E test with Gemma requires ~8GB VRAM and is slow."
echo "   To run manually (on faretra with GPU):"
echo "   cd preprocessing/medclip_routing"
echo "   export CUDA_VISIBLE_DEVICES=0"
echo "   ./submit_routing.sh configs/test_e2e.conf"
echo "=" | tr '=' '\075' | head -c 70; echo
