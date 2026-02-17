# Data Preparation Scripts

**Purpose:** Centralized utilities for preparing and distributing medical VQA datasets across the project.

---

## Architecture Overview

### Unified Design Pattern

All data preparation scripts follow a consistent pattern:

```python
#!/usr/bin/env python3
import sys
import tempfile
from pathlib import Path

# Import centralized utilities
sys.path.append(str(Path(__file__).parent))
import utils

# 1. Process data
OUTPUT_CSV = Path(tempfile.gettempdir()) / "dataset_name.csv"
# ... processing logic ...

# 2. Distribute to all pipeline stages
utils.distribute_file(OUTPUT_CSV)
```

### Core Components

| File | Purpose |
|------|---------|
| **`utils.py`** | Centralized file distribution and registry management |
| **`prepare_gemex.py`** | GEMeX dataset → MIMIC-CXR mapping |
| **`prepare_gemex_vqa.py`** | GEMeX VQA dataset → MIMIC-CXR mapping |
| **`prepare_mimic_ext.py`** | MIMIC-Ext dataset preparation |
| **`create_stratified_samples.py`** | Stratified random sampling for bias elimination |

---

## Usage

### 1. Prepare Full Datasets

**GEMeX Dataset:**
```bash
python3 prepare_gemex.py
# Outputs: gemex_mimic_mapped.csv
# Distributes to: vqa/, preprocessing/{bbox,attention_map,segmentation,medclip_routing}/
```

**GEMeX VQA Dataset:**
```bash
python3 prepare_gemex_vqa.py
# Outputs: gemex_VQA_mimic_mapped.csv
# Distributes to: vqa/, preprocessing/bounding_box/
```

**MIMIC-Ext Dataset:**
```bash
python3 prepare_mimic_ext.py
# Outputs: mimic_ext_mapped.csv
# Distributes to: vqa/, preprocessing/bounding_box/
```

### 2. Create Stratified Samples

**Default (4,000 samples):**
```bash
python3 create_stratified_samples.py --report
# Outputs: mimic_ext_stratified_4000_samples.csv
# Distributes to: vqa/, preprocessing/{bbox,attention_map,segmentation,medclip_routing}/
```

**Custom sample size:**
```bash
python3 create_stratified_samples.py --limit 2000 --report
# Outputs: mimic_ext_stratified_2000_samples.csv
```

**Custom input source:**
```bash
python3 create_stratified_samples.py --input_file ../vqa/gemex_VQA_mimic_mapped.csv --limit 1000
```

### 3. Limit Full Dataset Processing

All `prepare_*.py` scripts support the `--limit` parameter for testing:

```bash
# Process only 100 samples
python3 prepare_mimic_ext.py --limit 100
# Outputs: mimic_ext_mapped_100_samples.csv

# Full dataset (default)
python3 prepare_mimic_ext.py
# Outputs: mimic_ext_mapped.csv
```

---

## Common Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--limit` | Maximum number of samples to process | None (all) |
| `--seed` | Random seed for sampling reproducibility | 42 |
| `--report` | Show distribution analysis (stratified sampling only) | False |
| `--input_file` | Input CSV path (stratified sampling only) | `../vqa/mimic_ext_mapped.csv` |

---

## File Distribution

### Automatic Distribution System

The `utils.distribute_file()` function automatically copies generated CSV files to all pipeline stages:

**Target Directories:**
```python
TARGET_DIRS = [
    "../vqa",                            # VQA generation
    "../preprocessing/bounding_box",     # Bbox preprocessing
    "../preprocessing/attention_map",    # Attention map preprocessing
    "../preprocessing/segmentation",     # Segmentation preprocessing
    "../preprocessing/medclip_routing"   # MedCLIP agentic routing
]
```

**Distribution Flow:**
```
data_prep/
  └── [script generates CSV in /tmp]
       ↓
  utils.distribute_file()
       ↓
  ┌──────────────────────────────────┐
  │ Copies to all TARGET_DIRS         │
  │ + Logs to registry                │
  └──────────────────────────────────┘
       ↓
  vqa/dataset.csv
  preprocessing/bounding_box/dataset.csv
  preprocessing/attention_map/dataset.csv
  preprocessing/segmentation/dataset.csv
  preprocessing/medclip_routing/dataset.csv
```

### Generated Datasets Registry

All distributed files are logged to `generated_datasets_registry.json`:

```json
[
    "/home/user/.../vqa/mimic_ext_mapped.csv",
    "/home/user/.../preprocessing/bounding_box/mimic_ext_mapped.csv",
    "/home/user/.../preprocessing/attention_map/mimic_ext_mapped.csv",
    "/home/user/.../preprocessing/segmentation/mimic_ext_mapped.csv",
    "/home/user/.../preprocessing/medclip_routing/mimic_ext_mapped.csv"
]
```

**Purpose:**
- Track all generated datasets
- Enable cleanup scripts
- Audit data provenance

---

## Stratified Sampling

### Purpose

Eliminates sampling bias while meeting computational constraints (15-hour SLURM limit).

### Key Features

- **99.3% bias reduction:** From +103-272% bias to <0.2% deviation
- **Distribution preservation:** Question types and anatomical regions
- **Reproducible:** Seed-based random sampling (default: 42)
- **Validated:** Complete bias analysis in `../docs/dataset_distribution_analysis.md`

### Categorization Logic

**Question Types:**
- `tubes_devices` - Medical devices, tubes, lines
- `anatomical` - Anatomical structures/findings
- `disease` - Pathologies or abnormalities
- `technical` - Technical/imaging assessments
- `other` - General questions

**Anatomical Regions:**
- `lung` - Pulmonary regions
- `cardiac` - Heart and related structures
- `mediastinal` - Mediastinum, hilar, trachea
- `chest_wall` - Chest wall, hemidiaphragm
- `other` - Other locations

### Distribution Report

Use `--report` flag to see distribution analysis:

```
======================================================================
DISTRIBUTION COMPARISON REPORT
======================================================================

Question Type Distribution:
Category               Original %     Sample %       Diff
----------------------------------------------------------------------
anatomical                   7.7%         7.7%      -0.0%
disease                     10.4%        10.4%      -0.0%
other                       59.2%        59.4%      +0.2%
technical                    3.3%         3.2%      -0.1%
tubes_devices               19.3%        19.2%      -0.1%
...
```

---

## Output Filename Convention

Scripts generate dynamic filenames based on processing:

| Condition | Filename Pattern | Example |
|-----------|------------------|---------|
| **Full dataset** | `{base_name}.csv` | `mimic_ext_mapped.csv` |
| **Limited dataset** | `{base_name}_{N}_samples.csv` | `mimic_ext_mapped_100_samples.csv` |
| **Stratified sample** | `{base}_stratified_{N}_samples.csv` | `mimic_ext_stratified_4000_samples.csv` |

---

## Workflow Examples

### Development Workflow (Small Sample)

```bash
# 1. Create small test dataset
python3 prepare_mimic_ext.py --limit 100

# 2. Test preprocessing pipeline
cd ../preprocessing/bounding_box
bash submit_bbox_preprocessing.sh

# Output uses: mimic_ext_mapped_100_samples.csv
```

### Production Workflow (Full Pipeline)

```bash
# 1. Generate full dataset
python3 prepare_mimic_ext.py

# 2. Create stratified sample for time budget
python3 create_stratified_samples.py --limit 4000 --report

# 3. Update preprocessing configs to use stratified sample
# (already configured by default)

# 4. Run full pipeline
cd ../
./run_orchestrator.sh
```

### Research Workflow (Multiple Sample Sizes)

```bash
# Generate different sample sizes for comparison
for size in 1000 2000 4000 8000; do
    python3 create_stratified_samples.py --limit $size --report
done

# Each creates: mimic_ext_stratified_{size}_samples.csv
# All distributed to all pipeline stages automatically
```

---

## Troubleshooting

### Registry Issues

**Clear registry:**
```bash
rm generated_datasets_registry.json
```

**View registry:**
```bash
cat generated_datasets_registry.json | python3 -m json.tool
```

### Distribution Failures

If distribution fails to specific targets:

1. Check target directory exists:
```bash
ls -la ../vqa
ls -la ../preprocessing/bounding_box
```

2. Check permissions:
```bash
chmod +w ../vqa
```

3. Manual distribution:
```python
from utils import distribute_file
distribute_file("/tmp/your_dataset.csv")
```

### Import Errors

If `import utils` fails:

```bash
# Ensure you're in the data_prep directory
cd /home/rbalzani/medical-vqa/Thesis/data_prep

# Or set PYTHONPATH
export PYTHONPATH=/home/rbalzani/medical-vqa/Thesis/data_prep:$PYTHONPATH
```

---

## Dependencies

```
pandas
tqdm
huggingface_hub
requests
```

Install:
```bash
pip install -r requirements.txt
```

---

## Documentation References

- **Bias Analysis:** [`../docs/dataset_distribution_analysis.md`](../docs/dataset_distribution_analysis.md)
- **Implementation Summary:** [`../docs/STRATIFIED_SAMPLING_IMPLEMENTATION.md`](../docs/STRATIFIED_SAMPLING_IMPLEMENTATION.md)
- **Sampling Strategy:** [`../docs/SAMPLING_STRATEGY.md`](../docs/SAMPLING_STRATEGY.md)
- **Project Validation:** [`../docs/VALIDATION_REPORT.md`](../docs/VALIDATION_REPORT.md)

---

**Last Updated:** 2026-02-17
**Architecture:** Centralized utils + Consistent CLI patterns
