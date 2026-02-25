# Test Report: Phase 1 - Standalone Preprocessing Manifest Generation

**Date:** 2026-02-15
**Objective:** Verify that all 3 preprocessing stages (bbox, attention_map, segmentation) generate `vqa_manifest.csv` with correct structure for VQA pipeline consumption
**Test Method:** Standalone execution of each preprocessing stage with limit=50 samples

---

## Test Summary

| Test ID | Stage | GPU | Status | Issue |
|---------|-------|-----|--------|-------|
| T1 | bbox_preproc | 0 | ❌ FAILED | CUDA OOM - GPU heavily loaded (22GB/24GB used) |
| T2 | attn_map | 1 | ❌ FAILED | CUDA OOM during model loading |
| T3 | segmentation | 2 | ❌ FAILED | Config path resolution error |

**Overall Result:** ❌ **ALL TESTS FAILED** - No manifests generated

---

## Test Details

### T1: BBox Preprocessing Manifest

**Test Script:** `experiments/test/test_bbox_manifest.sh`
**Output Directory:** `/home/rbalzani/medical-vqa/Thesis/experiments/test/output_T1_bbox_manifest`
**GPU Assigned:** 0
**Configuration:**
```conf
BATCH_SIZE=4
ENABLE_BODY_MASK="false"
ENABLE_ANATOMICAL_CHECK="false"
USE_DYNAMIC_PROMPTS="true"
ENABLE_SMART_PADDING="true"
CAM_THRESHOLD=0.45
OUTPUT_FORMAT="jsonl"
```

**Error:**
```
RuntimeError: CUDA out of memory. Tried to allocate 588.00 MiB.
GPU 0 has a total capacity of 23.68 GiB of which 238.00 MiB is free.
Process 2173212 has 21.77 GiB memory in use.
Process 1831391 has 1.64 GiB memory in use.
```

**Result:**
- Processed: 0 images
- Failed: 40 image-question pairs
- Speed: 0.00 img/s
- **predictions.jsonl NOT GENERATED** (warning: "predictions.jsonl not found at /workspace/data/output/predictions.jsonl")
- **vqa_manifest.csv NOT GENERATED**

**Root Cause:**
GPU 0 is heavily loaded with other processes (total 23.41 GB / 23.68 GB occupied). Insufficient memory to load BiomedCLIP model + process batches of size 4.

**GPU Status at Test Time:**
- GPU 0: 22331 MB used / 24576 MB total (90.9% usage, 99% utilization)
- Available: ~2.2 GB (insufficient for BiomedCLIP inference)

---

### T2: Attention Map Manifest

**Test Script:** `experiments/test/test_attn_manifest.sh`
**Output Directory:** `/home/rbalzani/medical-vqa/Thesis/experiments/test/output_T2_attn_manifest`
**GPU Assigned:** 1
**Configuration:**
```conf
BATCH_SIZE=4
CAM_METHOD="GScoreCAM"
CAM_THRESHOLD=0.45
COLORMAP="jet"
SAVE_ORIGINAL="false"
COMPOSITE_MODE="true"
```

**Error:**
```
RuntimeError: CUDA error: out of memory
  File "/workspace/src/generate_heatmaps.py", line 258, in __init__
    biomed_core.to(self.device).eval()
```

**Result:**
- Pipeline failed during model initialization (before processing any images)
- **vqa_manifest.csv NOT GENERATED**

**Root Cause:**
CUDA OOM when loading BiomedCLIP model to GPU 1. Despite GPU 1 having ~15GB free (9.6GB/24GB used), the model loading failed. This suggests:
1. Another process may have briefly allocated memory
2. Memory fragmentation issues
3. The attention map pipeline may have higher initial memory requirements

**GPU Status at Test Time:**
- GPU 1: 9683 MB used / 24576 MB total (39.4% usage, 91% utilization)
- Available: ~15 GB (should be sufficient, but still failed)

---

### T3: Segmentation Manifest

**Test Script:** `experiments/test/test_seg_manifest.sh`
**Output Directory:** `/home/rbalzani/medical-vqa/Thesis/experiments/test/output_T3_seg_manifest`
**GPU Assigned:** 2
**Configuration:**

*Step 1 (Localization):*
```conf
BATCH_SIZE=4
ENABLE_BODY_MASK="false"
ENABLE_ANATOMICAL_CHECK="false"
USE_DYNAMIC_PROMPTS="true"
ENABLE_SMART_PADDING="false"
CAM_THRESHOLD=0.5
OUTPUT_FORMAT="jsonl"
```

*Step 2 (Segmentation):*
```conf
MEDSAM_CHECKPOINT="medsam"
SCENARIO="A"
SAVE_VISUALS="true"
TEXT_PROMPT_MODE="question"
SAM3_USE_BBOX="false"
```

**Error:**
```
❌ [ERROR] Step 1 configuration file not found at:
/workspace/configs//home/rbalzani/medical-vqa/Thesis/experiments/test/output_T3_seg_manifest/step1.conf
```

**Result:**
- Pipeline failed before execution (config loading phase)
- **vqa_manifest.csv NOT GENERATED**

**Root Cause:**
Config path resolution bug in test script. The test creates config files in the test output directory:
```bash
cat > "$OUTPUT_DIR/step1.conf" <<'EOF'
...
EOF
```

But passes them to `run_docker.sh` as:
```bash
./scripts/run_docker.sh "all" "/workspace/configs/$OUTPUT_DIR/step1.conf" ...
```

This results in an invalid Docker path:
```
/workspace/configs//home/rbalzani/medical-vqa/Thesis/experiments/test/output_T3_seg_manifest/step1.conf
```

The configs are created on the host but not mounted into the Docker container at the expected location.

**GPU Status at Test Time:**
- GPU 2: 12 MB used / 24576 MB total (0.05% usage, 0% utilization)
- Available: ~24 GB (fully available, no memory issue)

---

## Analysis

### Critical Issues

1. **GPU Memory Contention (T1, T2)**
   - GPU 0 is nearly full (90.9% used by other processes)
   - GPU 1 has moderate load but BiomedCLIP loading still fails
   - Both preprocessing stages require ~15-20GB for model + batch processing

2. **Test Script Bug (T3)**
   - Config files created on host but not accessible in Docker
   - Need to either mount test configs or use segmentation's config directory

### GPU Resource Availability

Current status:
- **GPU 0:** Heavily loaded (22GB used) - NOT SUITABLE for testing
- **GPU 1:** Moderately loaded (9.6GB used, 91% utilization) - MARGINAL
- **GPU 2:** Nearly free (12MB used) - IDEAL for testing
- **GPU 3:** Lightly loaded (4.4GB used) - SUITABLE for testing

### Manifest Generation Code Status

Despite test failures, the manifest generation code was successfully integrated into all 3 preprocessing scripts:

1. **bbox_preprocessing.py** (lines 1285-1313): `generate_vqa_manifest()` function added
2. **generate_heatmaps.py** (lines 730-751, 808-814, 848-854, 917): Record collection + manifest generation
3. **run_segmentation.py** (lines 258, 375-381, 387-394): Overlay record collection + manifest generation

The code is ready but **untested** due to runtime failures.

---

## Recommendations

### Immediate Fixes

1. **T3 Fix Priority (Easy Win)**
   - Fix config path handling in `test_seg_manifest.sh`
   - Mount test config directory into Docker
   - Rerun on GPU 2 (fully available)

2. **T1 & T2 Rerun Strategy**
   - **Option A:** Use GPU 2 or GPU 3 (both have sufficient free memory)
   - **Option B:** Reduce BATCH_SIZE from 4 to 1 or 2
   - **Option C:** Kill/move processes on GPU 0/1 (if user authorized)

### GPU Assignment for Retests

Proposed allocation:
- **T1 (bbox):** GPU 2 (24GB free)
- **T2 (attn):** GPU 3 (20GB free)
- **T3 (segmentation):** GPU 2 (can run sequentially after T1, or parallel on GPU 3)

### Next Steps

1. Fix `test_seg_manifest.sh` config path issue
2. Modify all 3 test scripts to use GPU 2/3 instead of 0/1
3. Rerun tests in parallel (T1 on GPU 2, T2 on GPU 3)
4. Verify manifest structure and content
5. Proceed to Phase 2: Bridged pipeline tests (T4-T6)

---

## Files Generated

Test scripts (created):
- `/home/rbalzani/medical-vqa/Thesis/experiments/test/test_bbox_manifest.sh`
- `/home/rbalzani/medical-vqa/Thesis/experiments/test/test_attn_manifest.sh`
- `/home/rbalzani/medical-vqa/Thesis/experiments/test/test_seg_manifest.sh`

Test output directories (created but empty due to failures):
- `/home/rbalzani/medical-vqa/Thesis/experiments/test/output_T1_bbox_manifest/`
- `/home/rbalzani/medical-vqa/Thesis/experiments/test/output_T2_attn_manifest/`
- `/home/rbalzani/medical-vqa/Thesis/experiments/test/output_T3_seg_manifest/`

Logs (captured):
- Task output logs in `/tmp/claude-1224/-home-rbalzani-medical-vqa-Thesis/tasks/`

---

## Status: 🔴 BLOCKED

All Phase 1 tests failed. Must resolve GPU memory contention and test script bugs before proceeding to Phase 2 (bridged pipeline tests).
