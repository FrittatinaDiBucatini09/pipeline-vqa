# Preprocessing → VQA Bridge Testing Plan

**Date:** 2026-02-15
**Objective:** Verify that all preprocessing stages correctly generate VQA manifests and that the orchestrator bridges connect them to VQA generation.

## Test Matrix

| Test ID | Stages | Expected Behavior | Status |
|---------|--------|-------------------|--------|
| T1 | bbox_preproc only | Generates `vqa_manifest.csv` | ⏳ |
| T2 | attn_map only | Generates `vqa_manifest.csv` | ⏳ |
| T3 | segmentation only | Generates `vqa_manifest.csv` | ⏳ |
| T4 | bbox_preproc → vqa_gen | Bridge activates, VQA reads preprocessed images | ⏳ |
| T5 | attn_map → vqa_gen | Bridge activates, VQA reads heatmaps | ⏳ |
| T6 | segmentation → vqa_gen | Bridge activates, VQA reads overlays | ⏳ |

## Test Parameters

- **Dataset:** Use a small test subset (50-100 samples max)
- **Model:** Lightweight/fast models where possible
- **Dry-run first:** Verify generated meta-job scripts before submission
- **GPU Allocation:** Single GPU per test (3 available in parallel)

## Execution Order

1. **Phase 1 (Dry-runs):** Generate and inspect meta-job scripts for all 6 tests
2. **Phase 2 (Standalone preprocessing):** T1, T2, T3 in parallel
3. **Phase 3 (Bridged pipelines):** T4, T5, T6 sequentially (to verify each bridge)

## Success Criteria

### For Standalone Tests (T1-T3)
- ✅ `vqa_manifest.csv` exists in output directory
- ✅ CSV has columns: `image_path`, `question`, `answer`
- ✅ Row count matches number of generated images
- ✅ Image paths in CSV point to actual files

### For Bridged Tests (T4-T6)
- ✅ Meta-job script contains bridge block
- ✅ Bridge block sets `DATA_FILE_OVERRIDE` and `VQA_IMAGE_PATH`
- ✅ VQA stage logs show bridge mode activation
- ✅ VQA stage processes images from preprocessing output directory
- ✅ VQA results reference preprocessed image paths (not original paths)
