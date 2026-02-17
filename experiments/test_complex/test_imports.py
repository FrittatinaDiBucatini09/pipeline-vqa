#!/usr/bin/env python3
"""
Test 1: Verify Python imports and module structure
"""
import sys
from pathlib import Path

print("=" * 70)
print("TEST 1: Python Imports and Module Structure")
print("=" * 70)

# Add src to path
src_path = Path(__file__).parent.parent.parent / "preprocessing" / "medclip_routing" / "src"
sys.path.insert(0, str(src_path))

results = {}

# Test 1: Import utils
try:
    import utils
    results["utils import"] = "✅ PASS"
    print(f"✅ utils module imported successfully")
except Exception as e:
    results["utils import"] = f"❌ FAIL: {e}"
    print(f"❌ utils import failed: {e}")

# Test 2: Check utils functions
try:
    from utils import (
        load_scispacy,
        load_gemma,
        load_biomed_clip,
        evaluate_query_quality,
        expand_query,
        generate_cam_bbox,
        CAMWrapper,
        reshape_transform,
    )
    results["utils functions"] = "✅ PASS"
    print(f"✅ All utils functions available")
except Exception as e:
    results["utils functions"] = f"❌ FAIL: {e}"
    print(f"❌ Utils functions check failed: {e}")

# Test 3: Import main_routing
try:
    import main_routing
    results["main_routing import"] = "✅ PASS"
    print(f"✅ main_routing module imported successfully")
except Exception as e:
    results["main_routing import"] = f"❌ FAIL: {e}"
    print(f"❌ main_routing import failed: {e}")

# Test 4: Check dependencies
deps_ok = True
dep_results = []
for dep in ["torch", "numpy", "pandas", "PIL", "cv2", "spacy", "transformers", "open_clip"]:
    try:
        __import__(dep)
        dep_results.append(f"  ✅ {dep}")
    except ImportError as e:
        dep_results.append(f"  ❌ {dep}: {e}")
        deps_ok = False

results["dependencies"] = "✅ PASS" if deps_ok else "❌ FAIL"
print(f"\n📦 Dependencies:")
for res in dep_results:
    print(res)

print("\n" + "=" * 70)
print("SUMMARY:")
for test, result in results.items():
    print(f"  {test}: {result}")
print("=" * 70)
