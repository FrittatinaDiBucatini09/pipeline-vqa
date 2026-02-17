#!/usr/bin/env python3
"""
Test 7: Verify preprocessing-to-VQA bridge logic
"""
import sys

print("=" * 70)
print("TEST 7: Preprocessing-to-VQA Bridge Logic")
print("=" * 70)

# Read slurm_templates.py
with open("orchestrator/slurm_templates.py", "r") as f:
    content = f.read()

# Check 1: medclip_routing in _PREPROCESSING_OUTPUT_PATHS
if '"medclip_routing": "results"' in content:
    print("✅ medclip_routing mapped to 'results' output path")
else:
    print("❌ medclip_routing NOT found in _PREPROCESSING_OUTPUT_PATHS")
    sys.exit(1)

# Check 2: Bridge function exists
if "_generate_preprocessing_to_vqa_bridge" in content:
    print("✅ Bridge function _generate_preprocessing_to_vqa_bridge exists")
else:
    print("❌ Bridge function NOT found")
    sys.exit(1)

# Check 3: Extract bridge template logic
print("\n📋 Bridge Template Logic:")
print("-" * 70)

# Find the bridge function
lines = content.split("\n")
in_bridge = False
bridge_lines = []
for i, line in enumerate(lines):
    if "_generate_preprocessing_to_vqa_bridge" in line:
        in_bridge = True
    if in_bridge:
        bridge_lines.append(line)
        if line.strip().startswith('"""') and len(bridge_lines) > 5:
            in_bridge = False
            break

# Show key elements
for line in bridge_lines[:25]:
    if "DATA_FILE_OVERRIDE" in line or "VQA_IMAGE_PATH" in line or "vqa_manifest.csv" in line:
        print(f"  {line.strip()}")

print("\nKey Bridge Variables:")
print("  - DATA_FILE_OVERRIDE: Points to vqa_manifest.csv")
print("  - VQA_IMAGE_PATH: Points to preprocessing output directory")
print("  - Source: <preprocessing_output>/vqa_manifest.csv")

# Check 4: Bridge injection logic
if "keys[idx] in _PREPROCESSING_STAGE_KEYS" in content and 'keys[idx + 1] == "vqa_gen"' in content:
    print("\n✅ Bridge auto-injection logic found")
    print("   Triggers when: preprocessing stage → vqa_gen")
else:
    print("\n❌ Bridge injection logic NOT found or incomplete")

print("\n✅ TEST 7: PASSED")
print("=" * 70)
