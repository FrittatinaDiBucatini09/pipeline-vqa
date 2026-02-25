# Orchestrator Test Report

**Date:** 2026-02-15 19:23 UTC
**Tester:** Claude (automated)
**Python:** 3.8.10
**Cluster:** faretra (SLURM)

---

## Summary

| # | Test | Result |
|---|------|--------|
| 1 | Module imports & basic setup | PASS |
| 2 | Config discovery for all stages | PASS |
| 3 | Dataset discovery | PASS |
| 4 | Path resolution (config relative to script dir) | **BUG FOUND** |
| 5 | Run directory creation & report writing | PASS |
| 6 | Judge sbatch template generation | PASS |
| 7 | SLURM sbatch validation (`--test-only`) | PASS |
| 8a | Real SLURM submission with dependency chaining | PASS |
| 8b | `DATA_FILE_OVERRIDE` env var propagation through SLURM | PASS |

**Overall: 8/9 PASS, 1 BUG**

---

## Detailed Results

### Test 1: Module Imports & Basic Setup — PASS

All imports from `orchestrator.py` and `slurm_templates.py` succeed. `PROJECT_ROOT` resolves correctly to `/home/rbalzani/medical-vqa/Thesis`. All 6 stages registered.

### Test 2: Config Discovery — PASS

All config directories exist and contain `.conf` files:

| Stage | Config Dir | Files Found |
|-------|-----------|-------------|
| bbox_preproc | `preprocessing/bounding_box/configs/` | 11 |
| attn_map | `preprocessing/attention_map/configs/` | 12 |
| segmentation | `preprocessing/segmentation/configs/` | 7 |
| bbox_eval | N/A (no config dir) | — |
| vqa_gen | `vqa/configs/generation/` | 2 |
| vqa_judge | `vqa/configs/judge/` | 2 |

Configs are returned with correct relative paths (e.g., `gemex/exp_01_vqa.conf`, `step1/gemex/exp_01_vqa.conf`).

### Test 3: Dataset Discovery — PASS

Found 3 CSV files in `preprocessing/bounding_box/`:
- `gemex_VQA_mimic_mapped.csv`
- `gemex_mimic_mapped.csv`
- `mimic_ext_mapped.csv`

### Test 4: Path Resolution — BUG FOUND

**Severity: HIGH** — crashes at runtime when the user selects a config for the VQA Judge stage.

**Root cause:** In `configure_stage()`, the config path is built as:

```python
script_dir = str(Path(stage.script_path).parent)
config_dir_relative = str(Path(stage.config_dir).relative_to(script_dir))
```

For the VQA Judge:
- `script_path` = `vqa/scripts/run_judge.sh` → `script_dir` = `vqa/scripts`
- `config_dir` = `vqa/configs/judge`
- `Path("vqa/configs/judge").relative_to("vqa/scripts")` → **`ValueError`** because `vqa/configs/judge` is not under `vqa/scripts`

**Error message:**
```
ValueError: 'vqa/configs/judge' does not start with 'vqa/scripts'
```

**Why it only affects the judge:** All other stages have their config dir nested under the same parent as their script. The judge is unique because its script is at `vqa/scripts/run_judge.sh` but configs are at `vqa/configs/judge/`.

**Expected behavior:** The generated judge sbatch wrapper does `cd "$VQA_DIR"` (which resolves to `vqa/`), so config paths should be relative to `vqa/`, not `vqa/scripts/`. The correct result should be `configs/judge/hard_coded_judge.conf`.

**Other stages verified OK:**
| Stage | Result | Expected | File Exists |
|-------|--------|----------|-------------|
| bbox_preproc | `configs/gemex/exp_01_vqa.conf` | `configs/gemex/exp_01_vqa.conf` | Yes |
| attn_map | `configs/gemex/heatmap_default.conf` | `configs/gemex/heatmap_default.conf` | Yes |
| vqa_gen | `configs/generation/hard_coded_gen.conf` | `configs/generation/hard_coded_gen.conf` | Yes |
| vqa_judge | **CRASH** | `configs/judge/hard_coded_judge.conf` | Yes |

### Test 5: Run Directory Creation & Report Writing — PASS

- Timestamp-based directory created under `orchestrator_runs/`
- Step subdirectories created correctly (`step_01_bbox_preproc/`, etc.)
- `report.txt` written with correct formatting
- Traceability files (`config_used.txt`, `dataset_override.txt`) written successfully

### Test 6: Judge sbatch Template Generation — PASS

All validation checks passed for the generated script:

| Check | Result |
|-------|--------|
| `#SBATCH -w faretra` | PASS |
| `nvidia_geforce_rtx_3090` GPU constraint | PASS |
| Docker image `med_vqa_project:3090` | PASS |
| Config file passed to `run_judge.sh` | PASS |
| `HF_TOKEN` handling | PASS |
| `cd` to VQA directory | PASS |
| `CUDA_VISIBLE_DEVICES` GPU selection | PASS |
| `--shm-size` set | PASS |
| `/llms` volume mount | PASS |
| Empty config → empty string arg | PASS |

### Test 7: SLURM sbatch Validation — PASS

`sbatch --test-only` accepted all scripts:

| Script | Valid |
|--------|-------|
| `submit_bbox_preprocessing.sh` | Yes |
| `submit_heatmap_gen.sh` | Yes |
| `submit_segmentation.sh` | Yes |
| `submit_evaluation.sh` | Yes |
| `submit_generation.sh` | Yes |
| Generated judge wrapper | Yes |

Job ID regex parsing also validated against test strings.

### Test 8a: Real SLURM Submission — PASS

Submitted two real jobs with dependency chaining:

- **Job 7849025:** `submit_bbox_preprocessing.sh configs/gemex/exp_01_vqa.conf` → submitted successfully
- **Job 7849026:** `submit_generation.sh configs/generation/hard_coded_gen.conf` with `--dependency=afterok:7849025` → submitted successfully

Both jobs were cancelled immediately after confirming submission worked.

### Test 8b: DATA_FILE_OVERRIDE Propagation — PASS

Created a minimal sbatch script that prints env vars. Submitted with:
- `DATA_FILE_OVERRIDE=my_test_dataset.csv`
- `ORCH_OUTPUT_DIR=/tmp/test_orch`

**Job output confirmed both variables were received correctly:**
```
DATA_FILE_OVERRIDE=my_test_dataset.csv
ORCH_OUTPUT_DIR=/tmp/test_orch
TEST_COMPLETE
```

This proves the full chain: `orchestrator subprocess.run(env=...)` → `sbatch` → SLURM job environment works correctly.

---

## Bug List

### BUG-001: `configure_stage()` crashes for VQA Judge config selection

- **File:** `orchestrator/orchestrator.py`, `configure_stage()` function, ~line 169-172
- **Severity:** HIGH — prevents the user from selecting any config for the VQA Judge stage
- **Trigger:** User selects VQA Judge stage and the orchestrator presents the config menu
- **Error:** `ValueError: 'vqa/configs/judge' does not start with 'vqa/scripts'`
- **Root cause:** `Path(stage.config_dir).relative_to(script_dir)` assumes config_dir is nested under the script's parent directory. This is true for all stages except the judge, where the script is at `vqa/scripts/` but configs are at `vqa/configs/`.
- **Fix suggestion:** For stages with `has_submit_script=False`, the working directory of the generated wrapper is different from the script's parent. The judge wrapper `cd`s to `vqa/`, so configs should be relative to `vqa/`. The path resolution needs a special case or a separate `working_dir` field on `PipelineStage`.

---

## Not Tested (Requires Interactive TTY)

The following features require `inquirer` interactive prompts and could not be tested in a non-interactive shell:

- `select_stages()` — checkbox multi-select
- `select_dataset()` — dataset list selection
- `configure_stage()` — config selection prompts (except path resolution logic, tested above)
- `_configure_segmentation()` — TARGET_MODE / LIMIT prompts
- `_configure_positional_args()` — bbox eval directory prompts
- Full end-to-end `main()` flow
- Confirmation prompt before submission

These should be tested manually via `./run_orchestrator.sh --dry-run`.
