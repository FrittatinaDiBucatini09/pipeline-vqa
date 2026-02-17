# Verification Agent: Post-Fix Audit & Testing Brief

## 1. Role & Objective

You are a **Senior QA Engineer** specializing in HPC/Docker/Python infrastructure. Your mission is to **verify that all bugs identified in a recent audit have been correctly fixed**, and to **scan for any regressions or remaining issues** across the project.

You must produce a **structured verification report** at the end.

---

## 2. Project Context

**Repository**: `~/medical-vqa/Thesis/`
**Purpose**: Medical image VQA pipeline (preprocessing + inference + evaluation) running on a SLURM cluster with Docker Rootless containers.

### Key Documentation
- `docs/claude.md` — Project coding standards and constraints
- `docs/cluster_infrastructure.md` — SLURM + Docker Rootless environment details

### Repository Structure (Relevant Modules)
```
Thesis/
├── preprocessing/
│   ├── attention_map/        # Heatmap generation (BiomedCLIP + gScoreCAM)
│   │   ├── src/generate_heatmaps.py
│   │   └── scripts/run_heatmap_gen.sh
│   ├── bounding_box/         # Bounding box detection (BiomedCLIP + CRF)
│   │   ├── src/bbox_preprocessing.py
│   │   └── scripts/run_bbox_preprocessing.sh
│   ├── segmentation/         # MedSAM segmentation (Step 1: localization, Step 2: masks)
│   │   ├── src/step1_localization/run_localization.py
│   │   ├── src/step2_segmentation/run_segmentation.py
│   │   └── scripts/run_docker.sh
│   └── medclip_routing/      # NLP query expansion middleware
│       └── src/main_routing.py
├── vqa/                      # VQA generation (MedGemma via vLLM)
├── orchestrator/             # Pipeline orchestration + SLURM chaining
│   ├── orchestrator.py
│   └── slurm_templates.py
├── experiments/              # Saved experiment logs and reports
└── orchestrator_runs/        # Auto-generated run directories
```

---

## 3. Bugs Found & Fixes Applied

The following bugs were identified and patched. **You must verify each fix is present and correct.**

### 3.1 — Zombie Process / Execution Hang (Python Scripts)

**Root Cause**: OpenMP/multiprocessing deadlock during Python interpreter shutdown, caused by cv2 thread pools conflicting with `ProcessPoolExecutor`, dangling DataLoader workers, and WandB background threads.

#### Fix A: Global OpenCV Threading Control

The following lines must appear **immediately after all imports** (before any constants or function definitions) in each file:

| File | Expected Location |
|------|-------------------|
| `preprocessing/segmentation/src/step1_localization/run_localization.py` | After `pydensecrf` import block |
| `preprocessing/bounding_box/src/bbox_preprocessing.py` | After `pydensecrf` import block |
| `preprocessing/attention_map/src/generate_heatmaps.py` | After `pydensecrf`/`open_clip` import block |
| `preprocessing/segmentation/src/step2_segmentation/run_segmentation.py` | After `import cv2` |

**Expected code** (exact):
```python
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
```

**Verify**: These two lines are present in ALL four files. They must NOT be inside a function or conditional — they must execute at module import time.

#### Fix B: Deterministic Cleanup in `finally` Blocks

The following three files must have a comprehensive `finally` block with **isolated try/except per cleanup step** (not a single bare shutdown call):

1. `preprocessing/segmentation/src/step1_localization/run_localization.py`
2. `preprocessing/bounding_box/src/bbox_preprocessing.py`
3. `preprocessing/attention_map/src/generate_heatmaps.py`

**Required cleanup steps** (in order, each wrapped in its own `try/except`):
1. `io_executor.shutdown(wait=True, cancel_futures=False)` — let pending I/O writes complete
2. `crf_executor.shutdown(wait=True, cancel_futures=True)` — cancel queued CRF tasks (only in files that use CRF: `run_localization.py` and `bbox_preprocessing.py`)
3. `del dataloader` + `del dataset` — release DataLoader worker processes
4. `del model` + `torch.cuda.empty_cache()` — release GPU memory
5. `wandb.finish()` — finalize telemetry
6. Report writing (`report.txt`) — **must be inside the `finally` block**, not after it

**Verify**:
- The report writing code is INSIDE `finally`, not after it
- Each cleanup step has its own `try/except` so one failure doesn't block the rest
- `cancel_futures=True` is used on `crf_executor` but NOT on `io_executor` (I/O writes must complete)

#### Fix C: Forced Exit Strategy

The following files must have a forced exit block in their `if __name__ == "__main__":` section, AFTER the `main()` call:

1. `preprocessing/segmentation/src/step1_localization/run_localization.py`
2. `preprocessing/bounding_box/src/bbox_preprocessing.py`
3. `preprocessing/attention_map/src/generate_heatmaps.py`

**Expected pattern**:
```python
if __name__ == "__main__":
    # ... (set_start_method, etc.)
    main()

    # Forced exit
    try:
        sys.exit(0)
    except SystemExit:
        pass
    finally:
        os._exit(0)
```

**Verify**: `os._exit(0)` is the absolute last line of execution. This guarantees process termination even if third-party threads are deadlocked.

#### Fix D: cv2 Threading in Segmentation Step 2

`preprocessing/segmentation/src/step2_segmentation/run_segmentation.py` must have `cv2.setNumThreads(0)` and `cv2.ocl.setUseOpenCL(False)` after `import cv2`. This file does NOT need the full `finally` cleanup or forced exit (it doesn't use executors or DataLoader).

---

### 3.2 — Segmentation `FileNotFoundError` (Docker Mount Bug)

**Root Cause**: The segmentation Docker wrapper (`run_docker.sh`) mounts `$PHYS_DIR/metadata` → `/workspace/metadata`, but when the orchestrator sets `DATA_FILE_OVERRIDE=mimic_ext_stratified_2000_samples.csv`, this file lives at the project root (`$PHYS_DIR/`), not in the `metadata/` subdirectory.

**File**: `preprocessing/segmentation/scripts/run_docker.sh`

**Fix**: A new block was added BEFORE the Docker `run` command that copies the override file into `metadata/`:

```bash
if [ -n "$DATA_FILE_OVERRIDE" ]; then
    mkdir -p "$PHYS_DIR/metadata"
    if [ -f "$PHYS_DIR/$DATA_FILE_OVERRIDE" ] && [ ! -f "$PHYS_DIR/metadata/$DATA_FILE_OVERRIDE" ]; then
        echo "..."
        cp "$PHYS_DIR/$DATA_FILE_OVERRIDE" "$PHYS_DIR/metadata/$DATA_FILE_OVERRIDE"
    fi
fi
```

**Verify**:
- This block exists AFTER the routing override block and BEFORE the `docker run` command
- The condition checks both that the source exists AND the destination doesn't (idempotent — won't overwrite)
- `mkdir -p` ensures the metadata directory exists

**Cross-check**: Confirm the bbox and attention_map pipelines do NOT have this issue. They use `METADATA_DIR="$PHYS_DIR"` (project root), so their Docker mount already includes project-root-level CSVs.

---

### 3.3 — WandB `permission denied` in Docker Rootless

**Root Cause**: WandB's internal Go process writes to `$HOME/.cache`. When `HOME=/workspace` and the container runs as a non-root user, this path is unwritable. The segmentation wrapper already set `XDG_CACHE_HOME`, but the other two didn't.

**Files fixed**:
- `preprocessing/attention_map/scripts/run_heatmap_gen.sh` — added `-e XDG_CACHE_HOME="/tmp/cache"`
- `preprocessing/bounding_box/scripts/run_bbox_preprocessing.sh` — added `-e XDG_CACHE_HOME="/tmp/cache"`

**Verify**: All three Docker wrapper scripts that run WandB now include `-e XDG_CACHE_HOME="/tmp/cache"`:
1. `preprocessing/segmentation/scripts/run_docker.sh` (already had it)
2. `preprocessing/attention_map/scripts/run_heatmap_gen.sh` (newly added)
3. `preprocessing/bounding_box/scripts/run_bbox_preprocessing.sh` (newly added)

---

## 4. Regression Scan Checklist

Beyond verifying the specific fixes above, perform these broader checks:

### 4.1 — Business Logic Integrity
- [ ] `run_localization.py`: The core processing loop (CAM generation, CRF, bbox extraction, JSONL writing) is **unchanged**. Only the `finally` block, imports section, and `__main__` block were modified.
- [ ] `bbox_preprocessing.py`: Same — core logic untouched, only cleanup/imports/exit modified.
- [ ] `generate_heatmaps.py`: Same — core logic untouched.
- [ ] Output formats (`predictions.jsonl`, `report.txt`, `vqa_manifest.csv`) are unchanged.

### 4.2 — Shell Script Consistency
- [ ] All three Docker wrapper scripts (`run_heatmap_gen.sh`, `run_bbox_preprocessing.sh`, `run_docker.sh`) have consistent environment variable patterns
- [ ] No hardcoded paths that violate `cluster_infrastructure.md` constraints
- [ ] `#SBATCH -w faretra` present in all SLURM submission scripts

### 4.3 — Multiprocessing Safety
- [ ] `multiprocessing.set_start_method('spawn', force=True)` is present in all scripts that use CUDA + multiprocessing (`run_localization.py`, `bbox_preprocessing.py`)
- [ ] No script uses `fork` start method (incompatible with CUDA)

### 4.4 — No Dangling Resources
- [ ] No Python script creates a `ProcessPoolExecutor` or `ThreadPoolExecutor` without a corresponding `shutdown()` in a `finally` block
- [ ] No `DataLoader` with `num_workers > 0` is left without explicit `del` in cleanup
- [ ] `wandb.finish()` is called in every script that calls `wandb.init()`

### 4.5 — Files NOT Modified (Confirm Untouched)
These files should NOT have been modified by the fixes. Verify they are unchanged:
- `orchestrator/orchestrator.py`
- `orchestrator/slurm_templates.py`
- `vqa/src/generate_vqa.py`
- `preprocessing/medclip_routing/src/main_routing.py`
- All Dockerfiles
- All `.conf` config files

---

## 5. Testing Protocol

### 5.1 — Static Analysis (No GPU Required)
Run these checks from the project root:

```bash
# 1. Syntax check all modified Python files
python3 -m py_compile preprocessing/segmentation/src/step1_localization/run_localization.py
python3 -m py_compile preprocessing/bounding_box/src/bbox_preprocessing.py
python3 -m py_compile preprocessing/attention_map/src/generate_heatmaps.py
python3 -m py_compile preprocessing/segmentation/src/step2_segmentation/run_segmentation.py

# 2. Shell script syntax check
bash -n preprocessing/segmentation/scripts/run_docker.sh
bash -n preprocessing/attention_map/scripts/run_heatmap_gen.sh
bash -n preprocessing/bounding_box/scripts/run_bbox_preprocessing.sh

# 3. Verify cv2.setNumThreads(0) is present in all required files
grep -n "cv2.setNumThreads(0)" \
  preprocessing/segmentation/src/step1_localization/run_localization.py \
  preprocessing/bounding_box/src/bbox_preprocessing.py \
  preprocessing/attention_map/src/generate_heatmaps.py \
  preprocessing/segmentation/src/step2_segmentation/run_segmentation.py

# 4. Verify os._exit(0) is present in pipeline scripts
grep -n "os._exit(0)" \
  preprocessing/segmentation/src/step1_localization/run_localization.py \
  preprocessing/bounding_box/src/bbox_preprocessing.py \
  preprocessing/attention_map/src/generate_heatmaps.py

# 5. Verify XDG_CACHE_HOME in Docker wrappers
grep -n "XDG_CACHE_HOME" \
  preprocessing/segmentation/scripts/run_docker.sh \
  preprocessing/attention_map/scripts/run_heatmap_gen.sh \
  preprocessing/bounding_box/scripts/run_bbox_preprocessing.sh

# 6. Verify DATA_FILE_OVERRIDE copy logic in segmentation
grep -n "DATA_FILE_OVERRIDE" preprocessing/segmentation/scripts/run_docker.sh
```

### 5.2 — Structural Verification
- Read each modified file end-to-end and confirm no accidental deletions or duplications occurred
- Confirm indentation is consistent (4 spaces in Python, no mixed tabs)
- Confirm no import was accidentally removed or reordered

---

## 6. Expected Verification Report Format

Produce your report in this structure:

```markdown
# Verification Report

## Fix Verification
| Fix ID | Description | File | Status | Notes |
|--------|-------------|------|--------|-------|
| 3.1-A  | cv2.setNumThreads(0) | run_localization.py | PASS/FAIL | ... |
| ...    | ...         | ...  | ...    | ...   |

## Regression Scan
| Check | Status | Notes |
|-------|--------|-------|
| Business logic unchanged | PASS/FAIL | ... |
| ...   | ...    | ...   |

## Static Analysis
| Test | Result | Output |
|------|--------|--------|
| py_compile | PASS/FAIL | ... |
| ...  | ...    | ...    |

## Issues Found
(List any new issues discovered, or "None")

## Verdict
PASS / FAIL (with explanation)
```
