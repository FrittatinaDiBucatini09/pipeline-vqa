# Project Validation Report

**Date:** 2026-02-16
**Validator:** Claude Code (Opus 4.6) - Project Validation Agent
**Scope:** `medical-vqa/Thesis` - Full pipeline integrity scan

## Summary
- **Files Scanned:** ~85 (Python: 20+, Shell: 27, Configs: 30+, Dockerfiles: 7, .env: 5)
- **Critical Issues:** 5
- **Warnings:** 8
- **Passed Checks:** 12

---

## 1. Pipeline Connectivity & Data Flow

### Scenario A: Image Output (`OUTPUT_FORMAT='image'`)

**Status: PASS**

The preprocessing stage (`bbox_preprocessing.py`) saves annotated images with unique filenames (`{stem}_idx{N}{ext}`) preserving the original directory structure under `output_root/`. The JSONL written alongside each batch contains `image_path` pointing to the generated annotated image. The VQA generation script (`submit_generation.sh:86-107`) detects `VQA_IMAGE_PATH` (set by the orchestrator bridge) and mounts it as `/preprocessed_images:ro`. The `run_generation.sh` script receives `DATA_FILE_OVERRIDE` which remaps to the container path. `generate_vqa.py:27` reads `VQA_IMAGE_PATH` env var to set `BASE_IMAGE_PATH`, so relative paths from the JSONL/CSV resolve correctly.

**Connectivity: VERIFIED**

### Scenario B: JSONL Output (`OUTPUT_FORMAT='jsonl'`)

**Status: PASS WITH CAVEAT (see Critical Issue #1)**

When `OUTPUT_FORMAT='jsonl'`, `bbox_preprocessing.py` writes a `predictions.jsonl` with each record containing:
- `image_path`: relative path to the **original** source image (not a preprocessed copy)
- `predicted_boxes`: bounding box coordinates in **absolute pixel space** (scaled from CAM resolution to original image dimensions via `_scale_box()`)

The VQA generation script reads this data via `--data_file` and uses the `image_column` to locate images. Since `image_path` points to the original files, the VQA model receives the **raw unprocessed images** - which is the intended design for JSONL mode.

**However, see Critical Issue #1 regarding how bounding box coordinates in JSONL are consumed.**

---

## 2. Bounding Box Scaling Integrity

### Coordinate Flow Analysis

```
PREPROCESSING (bbox_preprocessing.py):
  1. BiomedCLIP generates CAM at model resolution (e.g., 14x14 or 224x224)
  2. _scale_box() scales from CAM space -> original image pixel space
     scale_x = w_orig / w_cam
     scale_y = h_orig / h_cam
  3. Saves absolute pixel coordinates in JSONL "predicted_boxes" field

EVALUATION (evaluate_bbox.py):
  1. Gold boxes stored in 224x224 reference space (GEMeX standard)
  2. scale_gold_to_pixel() scales gold from 224-space -> original pixel space
     scale_x = img_w / 224.0
  3. IoU computed in original pixel space

VQA GENERATION (generate_vqa.py):
  1. vLLM loads images at native resolution (NO explicit resize)
  2. Model handles internal resizing
  3. Bounding box coordinates are NOT consumed by generate_vqa.py
     (the boxes are visual overlays or metadata, not model input)
```

**Status: PASS** - The scaling is internally consistent:
- Predicted boxes: CAM -> original pixels (correct)
- Gold boxes: 224-space -> original pixels (correct)
- IoU comparison: both in original pixel space (correct)
- VQA generation does NOT resize images before passing to vLLM, so no implicit mismatch

---

## Critical Issues

### 1. `vqa/src/generate_vqa.py`: No `try...finally` for GPU Cleanup

**Severity: CRITICAL**
**File:** [generate_vqa.py:783-1079](vqa/src/generate_vqa.py#L783-L1079)

The `main()` function in `generate_vqa.py` initializes a vLLM `LLM` model (line 922) which allocates GPU memory, but the entire execution flow has **no `try...finally` block**. If the script crashes during inference, dataset loading, or metric calculation, the GPU memory is never explicitly released. Compare with:
- `bbox_preprocessing.py:1399` - has `finally:` block for executor shutdown + wandb cleanup
- `generate_heatmaps.py:921` - has `finally:` block for executor shutdown + wandb cleanup
- `run_localization.py:1348` - has `finally:` block with explicit GPU cleanup

**Impact:** A crash during VQA generation will leave a zombie process holding GPU memory on `faretra` until manually killed.

**Fix:**
```python
def main():
    # ... argument parsing ...
    model = None
    try:
        model = LLM(**model_init_kwargs)
        # ... rest of main() ...
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

### 2. `vqa/src/llm_judge.py:507`: Hardcoded `trust_remote_code=True`

**Severity: CRITICAL**
**File:** [llm_judge.py:507](vqa/src/llm_judge.py#L507)

The LLM Judge **always** enables `trust_remote_code=True` regardless of the model being loaded. This is a security risk:
- The default judge model is `Qwen/Qwen2.5-7B-Instruct` (a standard model that does NOT require `trust_remote_code`)
- Per project rules, `trust_remote_code=True` should **ONLY** be enabled for authorized models (e.g., MedGemma 1.5)

In contrast, `generate_vqa.py` correctly makes this a user-controlled flag via `--trust_remote_code` CLI argument.

**Fix:** Add `--trust_remote_code` flag to judge argparse and use `args.trust_remote_code` instead of hardcoded `True`.

### 3. `vqa/submit_generation.sh`: Missing `--time` SLURM Directive

**Severity: CRITICAL**
**File:** [submit_generation.sh:15-20](vqa/submit_generation.sh#L15-L20)

The VQA generation submit script has **no `#SBATCH --time` directive**. This means there is no hard time limit on GPU usage. All other submit scripts have time limits:

| Script | Time Limit |
|--------|-----------|
| `submit_heatmap_gen.sh` | 5:00:00 |
| `submit_bbox_preprocessing.sh` | 5:00:00 |
| `submit_evaluation.sh` | 0:30:00 |
| `submit_segmentation.sh` | 4:00:00 |
| `submit_integration_test.sh` | 1:00:00 |
| **`submit_generation.sh`** | **MISSING** |

**Impact:** A hung VQA generation job could monopolize a GPU indefinitely on `faretra`, violating the 15-hour hard limit mandate.

**Fix:** Add `#SBATCH --time=15:00:00` (or appropriate limit).

### 4. `vqa/submit_generation.sh`: Missing `--memory` Docker Flag

**Severity: CRITICAL**
**File:** [submit_generation.sh:123-136](vqa/submit_generation.sh#L123-L136)

The Docker `run` command in `submit_generation.sh` is missing the `--memory` flag. Compare:

| Script | `--memory` |
|--------|-----------|
| `run_heatmap_gen.sh` | `--memory="30g"` |
| `run_bbox_preprocessing.sh` | `--memory="30g"` |
| `run_evaluation.sh` | `--memory="8g"` |
| `run_docker.sh` (segmentation) | `--memory="30g"` |
| **`submit_generation.sh`** | **MISSING** |

**Impact:** Without a memory limit, a VQA generation container could consume all available host memory, potentially crashing other jobs on `faretra`.

**Fix:** Add `--memory="30g"` to the `docker run` command.

---

### 5. `vqa/submit_integration_test.sh:51`: Incorrect Dataset Mount Path

**Severity: CRITICAL**
**File:** [submit_integration_test.sh:51](vqa/submit_integration_test.sh#L51)

```bash
-v /dataset:/dataset:ro \
```

The mount uses `/dataset` (singular) instead of `/datasets` (plural). All other scripts consistently use `/datasets/MIMIC-CXR`:
- `run_bbox_preprocessing.sh:328`: `-v "/datasets/MIMIC-CXR:/datasets/MIMIC-CXR:ro"`
- `run_heatmap_gen.sh:269`: `-v "/datasets/MIMIC-CXR:/datasets/MIMIC-CXR:ro"`
- `run_docker.sh:86`: `-v "/datasets/MIMIC-CXR:/datasets/MIMIC-CXR:ro"`
- `submit_generation.sh:129`: `-v "/datasets:/datasets"`

**Impact:** Integration tests may silently fail to find dataset files if `/dataset` doesn't exist on the host (it's `/datasets` on `faretra`).

---

## Warnings

### W1. `vqa/src/generate_vqa.py:912`: Unrestricted Filesystem Access

**File:** [generate_vqa.py:912](vqa/src/generate_vqa.py#L912)

```python
allowed_media_path = os.path.abspath("/")  # Allow access to entire filesystem
```

vLLM's `allowed_local_media_path` is set to `/`, granting the model access to the entire filesystem. While this works in a Docker container (which limits the filesystem), it should be restricted to the specific dataset/image directories for defense-in-depth.

**Recommendation:** Use `os.path.abspath(args.images_dir)` or the VQA_IMAGE_PATH.

### W2. `vqa/src/llm_judge.py`: No `try...finally` for GPU Cleanup

**File:** [llm_judge.py:501-515](vqa/src/llm_judge.py#L501-L515)

Similar to Critical Issue #1 but for the judge model. The `try/except` catches loading errors but there's no `finally` block for cleanup after inference completes or crashes.

### W3. `vqa/submit_generation.sh:129`: Dataset Mount Missing `:ro` Flag

**File:** [submit_generation.sh:129](vqa/submit_generation.sh#L129)

```bash
-v "/datasets:/datasets" \
```

The `/datasets` mount is **read-write** instead of read-only. Per `cluster_infrastructure.md`, datasets should be mounted with `:ro`. Other preprocessing scripts correctly use `:ro`:
- `run_heatmap_gen.sh:269`: `-v "/datasets/MIMIC-CXR:/datasets/MIMIC-CXR:ro"`
- `run_bbox_preprocessing.sh:328`: `-v "/datasets/MIMIC-CXR:/datasets/MIMIC-CXR:ro"`
- `run_docker.sh:86`: `-v "/datasets/MIMIC-CXR:/datasets/MIMIC-CXR:ro"`

**Fix:** Change to `-v "/datasets:/datasets:ro"`.

### W4. `data_prep/*.py`: Hardcoded Dataset Paths

**Files:**
- [prepare_gemex.py:26](data_prep/prepare_gemex.py#L26): `/datasets/MIMIC-CXR/files`
- [prepare_gemex_vqa.py:29](data_prep/prepare_gemex_vqa.py#L29): `/datasets/MIMIC-CXR/files`
- [prepare_mimic_ext.py:8](data_prep/prepare_mimic_ext.py#L8): `/datasets/MIMIC-Ext-MIMIC-CXR-VQA/dataset`

These paths are hardcoded in the Python source rather than passed via config or CLI arguments. Per `claude.md`: "Never hardcode paths... rely entirely on the `.conf` files."

**Impact:** Low (data prep is a one-time operation), but inconsistent with project standards.

### W5. `preprocessing/bounding_box/configs/gemex/hard_coded.conf:53`: Empty `USE_DYNAMIC_PROMPTS`

**File:** [hard_coded.conf:53](preprocessing/bounding_box/configs/gemex/hard_coded.conf)

```bash
USE_DYNAMIC_PROMPTS=""  # Use 'question' column as dynamic prompts
```

The comment says it should use the 'question' column, but the value is empty string. If the intent is to enable question-driven prompts, this should be `"true"` or `"question"`.

### W6. `experiments/01_bbox_gridsearch/submit_benchmark.sh:23`: Excessive Time Limit

**File:** [submit_benchmark.sh:23](experiments/01_bbox_gridsearch/submit_benchmark.sh#L23)

```bash
#SBATCH --time=24:00:00
```

This exceeds the 15-hour mandate for GPU usage. While this is in the `experiments/` exclusion zone, it's worth noting for future enforcement.

### W7. `preprocessing/segmentation/src/step2_segmentation/segmentation_utils.py:342`: Hardcoded MedSAM3 Path

**File:** [segmentation_utils.py:342](preprocessing/segmentation/src/step2_segmentation/segmentation_utils.py#L342)

Changes working directory to `/opt/MedSAM3` which is hardcoded to the Docker container layout. This is acceptable within the Docker context but fragile if the container structure changes.

### W8. `preprocessing/bounding_box/configs/gemex/hard_coded.conf:54`: Misleading Comment

**File:** [hard_coded.conf:54](preprocessing/bounding_box/configs/gemex/hard_coded.conf)

```bash
USE_VISUAL_REGIONS="true"  # Disable anatomical region guidance
```

The comment says "Disable" but the value is `"true"` (enable). This is confusing and could lead to misconfiguration if someone uses the comment as guidance.

---

## Passed Checks

| Check | Status | Notes |
|-------|--------|-------|
| Node Pinning (`faretra`) | PASS | All 7 submit scripts contain `#SBATCH -w faretra` |
| `--gpus` SLURM Directive | PASS | All submit scripts specify `nvidia_geforce_rtx_3090:1` |
| `--gpus` Docker Flag | PASS | All `docker run` commands use `--gpus "device=$CUDA_VISIBLE_DEVICES"` |
| Docker `:ro` Mounts (preprocessing) | PASS | All preprocessing scripts mount `/datasets/` as `:ro` |
| `build_all_images.sh` Exists & Executable | PASS | `-rwxrwxr-x` permissions confirmed |
| Python Syntax | PASS | No syntax errors found in any scanned Python files |
| Shell Syntax | PASS | All scripts have proper shebangs and valid bash syntax |
| BBox Scaling (Preprocessing) | PASS | CAM -> original pixel space scaling is correct |
| BBox Scaling (Evaluation) | PASS | 224-space -> original pixel space scaling is correct |
| Config Consistency | PASS | OUTPUT_FORMAT values are valid (`image` or `jsonl`) across all configs |
| Preprocessing `finally` blocks | PASS | `bbox_preprocessing.py`, `generate_heatmaps.py`, `run_localization.py` all have `finally:` cleanup |
| Docker Memory Limits (preprocessing) | PASS | All preprocessing Docker runs use `--memory="30g"` |

---

## Architecture Notes

### Data Flow Summary

```
[Preprocessing Stage]
  attention_map/  ─┐
  segmentation/   ─┤─> JSONL (predicted_boxes + image_path) or Annotated Images
  bounding_box/   ─┘
        │
        ▼
[Orchestrator Bridge]
  VQA_IMAGE_PATH env var ──> submit_generation.sh
  DATA_FILE_OVERRIDE    ──> run_generation.sh
        │
        ▼
[VQA Generation]
  generate_vqa.py ──> vLLM (MedGemma / Qwen-VL)
        │
        ▼
[VQA Judge]
  llm_judge.py ──> vLLM (Qwen2.5-7B-Instruct)
```

### trust_remote_code Audit

| Component | Model | trust_remote_code | Status |
|-----------|-------|-------------------|--------|
| `generate_vqa.py` | MedGemma (configurable) | CLI flag (opt-in) | CORRECT |
| `llm_judge.py` | Qwen2.5-7B-Instruct | **HARDCODED True** | VIOLATION |
| `run_generation.sh` | Configurable | Config-driven | CORRECT |

---

## Remediation Priority

| Priority | Issue | Effort |
|----------|-------|--------|
| P0 | Add `#SBATCH --time=15:00:00` to `submit_generation.sh` | 1 line |
| P0 | Add `--memory="30g"` to `submit_generation.sh` Docker run | 1 line |
| P0 | Fix `/dataset` -> `/datasets` typo in `submit_integration_test.sh` | 1 char |
| P0 | Add `try...finally` GPU cleanup to `generate_vqa.py:main()` | ~10 lines |
| P1 | Fix hardcoded `trust_remote_code=True` in `llm_judge.py` | ~5 lines |
| P1 | Add `:ro` to dataset mount in `submit_generation.sh` | 1 char |
| P2 | Restrict `allowed_local_media_path` in `generate_vqa.py` | 1 line |
| P2 | Add `try...finally` to `llm_judge.py` main | ~10 lines |
| P3 | Externalize hardcoded paths in `data_prep/` scripts | ~15 lines |
