#!/bin/bash
# ==============================================================================
# Run All Tests for MedCLIP Agentic Routing
# ==============================================================================
set -e

REPORT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPORT_DIR/../.."

echo "=" | tr '=' '\075' | head -c 70; echo
echo "MedCLIP Agentic Routing - Comprehensive Test Suite"
echo "=" | tr '=' '\075' | head -c 70; echo
echo ""

# Test Summary
PASSED=0
FAILED=0
SKIPPED=0

# Test 1: File Structure
echo "TEST: File Structure Validation"
echo "-------------------------------------------------------------------"
REQUIRED_FILES=(
    "preprocessing/medclip_routing/docker/Dockerfile.3090"
    "preprocessing/medclip_routing/docker/requirements.txt"
    "preprocessing/medclip_routing/configs/default.conf"
    "preprocessing/medclip_routing/src/main_routing.py"
    "preprocessing/medclip_routing/src/utils.py"
    "preprocessing/medclip_routing/scripts/build_image.sh"
    "preprocessing/medclip_routing/scripts/run_docker.sh"
    "preprocessing/medclip_routing/submit_routing.sh"
    "preprocessing/medclip_routing/README.md"
)

ALL_EXIST=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file MISSING"
        ALL_EXIST=false
    fi
done

if $ALL_EXIST; then
    echo "✅ File structure validation PASSED"
    PASSED=$((PASSED+1))
else
    echo "❌ File structure validation FAILED"
    FAILED=$((FAILED+1))
fi
echo ""

# Test 2: Docker Image
echo "TEST: Docker Image Availability"
echo "-------------------------------------------------------------------"
if docker images | grep -q "medclip_routing.*3090"; then
    echo "✅ Docker image found: medclip_routing:3090"
    docker images | grep "medclip_routing"
    PASSED=$((PASSED+1))
else
    echo "❌ Docker image NOT found"
    echo "   Run: cd preprocessing/medclip_routing && bash scripts/build_image.sh"
    FAILED=$((FAILED+1))
fi
echo ""

# Test 3: Orchestrator Integration
echo "TEST: Orchestrator Integration"
echo "-------------------------------------------------------------------"
if grep -q "medclip_routing" orchestrator/orchestrator.py; then
    echo "✅ medclip_routing found in orchestrator.py"
    PASSED=$((PASSED+1))
else
    echo "❌ medclip_routing NOT in orchestrator.py"
    FAILED=$((FAILED+1))
fi

if grep -q "medclip_routing" orchestrator/slurm_templates.py; then
    echo "✅ medclip_routing found in slurm_templates.py"
    PASSED=$((PASSED+1))
else
    echo "❌ medclip_routing NOT in slurm_templates.py"
    FAILED=$((FAILED+1))
fi
echo ""

# Test 4: Python Imports (Docker)
echo "TEST: Python Imports (in Docker)"
echo "-------------------------------------------------------------------"
if docker run --rm --gpus device=0 medclip_routing:3090 python3 -c "
import sys
sys.path.insert(0, '/workspace/src')
from utils import load_scispacy, load_biomed_clip, CAMWrapper
import main_routing
print('✅ All imports successful')
" 2>/dev/null | grep -q "All imports successful"; then
    echo "✅ Python imports PASSED"
    PASSED=$((PASSED+1))
else
    echo "❌ Python imports FAILED"
    FAILED=$((FAILED+1))
fi
echo ""

# Test 5: Config Discovery
echo "TEST: Config Discovery"
echo "-------------------------------------------------------------------"
CONFIGS=$(find preprocessing/medclip_routing/configs -name "*.conf" | wc -l)
if [ "$CONFIGS" -gt 0 ]; then
    echo "✅ Found $CONFIGS config file(s):"
    find preprocessing/medclip_routing/configs -name "*.conf" -exec basename {} \;
    PASSED=$((PASSED+1))
else
    echo "❌ No config files found"
    FAILED=$((FAILED+1))
fi
echo ""

# Summary
echo "=" | tr '=' '\075' | head -c 70; echo
echo "TEST SUMMARY"
echo "=" | tr '=' '\075' | head -c 70; echo
echo "Total Tests: $((PASSED + FAILED + SKIPPED))"
echo "✅ Passed: $PASSED"
echo "❌ Failed: $FAILED"
echo "⏭️  Skipped: $SKIPPED"
echo ""

if [ "$FAILED" -eq 0 ]; then
    echo "🎉 ALL TESTS PASSED"
    echo ""
    echo "Next Steps:"
    echo "1. Run full E2E test on GPU:"
    echo "   cd preprocessing/medclip_routing"
    echo "   export CUDA_VISIBLE_DEVICES=0"
    echo "   ./submit_routing.sh configs/test_e2e.conf"
    echo ""
    echo "2. Test orchestrator chain:"
    echo "   python orchestrator/orchestrator.py"
    echo ""
    exit 0
else
    echo "❌ SOME TESTS FAILED"
    echo "Review the output above for details."
    exit 1
fi
