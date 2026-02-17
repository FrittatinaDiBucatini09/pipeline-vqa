# Stratified Sampling Strategy for Bounding Box Preprocessing

## Problem Identified

**Sequential sampling (taking first N rows) introduces significant bias:**

| Category | Sequential (first 4K) | Random Sample | Bias |
|----------|----------------------|---------------|------|
| Tubes/Devices questions | 37.7% | 18.6% | **+103% over-representation** |
| Anatomical questions | 27.5% | 7.4% | **+272% over-representation** |
| Lung regions | 28.9% | 39.0% | **-26% under-representation** |

Using `STOP_AFTER=4000` would have biased the evaluation toward tube/device and anatomical finding questions, underrepresenting disease and lung-related questions.

**Reference:** See [`../../docs/dataset_distribution_analysis.md`](../../docs/dataset_distribution_analysis.md) for complete bias analysis.

---

## Solution Implemented

### Stratified Random Sampling

Created a stratified sampling script that preserves the original distribution across:
- **Question types** (tubes/devices, anatomical, disease, technical, other)
- **Anatomical regions** (lung, cardiac, mediastinal, chest wall, other)
- **Image diversity** (3,907 unique images in 4,000 samples)

### Results for Bounding Box Dataset

**Dataset:** MIMIC-Ext VQA (50,000 samples)
**Sample Size:** 4,000
**Distribution preservation:** <0.2% deviation from original

```
Question Type Distribution:
Category               Original %     Sample %       Diff
----------------------------------------------------------------------
anatomical                   7.7%         7.7%      -0.0%
disease                     10.4%        10.4%      -0.0%
other                       59.2%        59.4%      +0.2%
technical                    3.3%         3.2%      -0.1%
tubes_devices               19.3%        19.2%      -0.1%

Anatomical Region Distribution:
Region                 Original %     Sample %       Diff
----------------------------------------------------------------------
cardiac                      6.0%         6.0%      -0.1%
chest_wall                   5.0%         4.9%      -0.1%
lung                        38.0%        38.2%      +0.2%
mediastinal                 10.1%        10.0%      -0.0%
other                       40.9%        40.8%      -0.0%

Image Diversity:
  Original: 38,307 unique images / 50,000 samples
  Sample:   3,907 unique images / 4,000 samples (97.7%)
```

---

## Files Created

1. **Centralized Sampling Script:** `../../data_prep/create_stratified_samples.py`
   - Single script for all preprocessing methods
   - Implements stratified random sampling
   - Generates distribution reports
   - Reproducible (seed=42)

2. **Stratified Dataset:** `mimic_ext_sample_4k.csv`
   - 4,000 representative samples
   - Preserves original distribution
   - Ready for preprocessing

3. **Updated Config:** `configs/exp/S_question_T0.45_C_crf_on_P_loose.conf`
   - Uses `METADATA_FILENAME="mimic_ext_sample_4k.csv"`
   - Removed biased `STOP_AFTER` parameter

---

## Usage

### Creating New Stratified Samples

**For bbox only:**
```bash
cd /home/rbalzani/medical-vqa/Thesis/data_prep
python3 create_stratified_samples.py --method bbox --report
```

**For all preprocessing methods:**
```bash
cd /home/rbalzani/medical-vqa/Thesis/data_prep
python3 create_stratified_samples.py --report
```

**Custom sample size:**
```bash
python3 create_stratified_samples.py --n_samples 2000 --report
```

### Parameters

- `--input`: Full dataset CSV
- `--output`: Output path for stratified sample
- `--n_samples`: Number of samples to select (default: 4000)
- `--seed`: Random seed for reproducibility (default: 42)
- `--report`: Print distribution comparison report

---

## Time Budget with Stratified Sample

| Stage | Samples | Est. Time |
|-------|---------|-----------|
| Bounding Box Preprocessing | 4,000 | ~9.3h |
| Total Pipeline | 4,000 | **Fits within 15h time budget** ✓ |

---

## Validation

✅ **Question type distribution preserved** (<0.2% deviation)
✅ **Anatomical region distribution preserved** (<0.2% deviation)
✅ **Image diversity maintained** (97.7% unique images)
✅ **Reproducible** (seed-based random sampling)
✅ **Time budget satisfied** (9.3h < 15h limit)

---

## Using Full Dataset

To process the full dataset (50,000 samples), update the config file:

```bash
# Edit configs/exp/S_question_T0.45_C_crf_on_P_loose.conf
METADATA_FILENAME="mimic_ext_mapped.csv"  # Full dataset
```

**Warning:** This will take significantly longer and may exceed SLURM time limits.

---

## See Also

- **Complete Bias Analysis:** [`../../docs/dataset_distribution_analysis.md`](../../docs/dataset_distribution_analysis.md)
- **Bias Calculation Script:** [`../../docs/bias_calculation.py`](../../docs/bias_calculation.py)
- **Project Validation:** [`../../docs/VALIDATION_REPORT.md`](../../docs/VALIDATION_REPORT.md)

---

Generated: 2026-02-16
Author: Claude Code (Project Validation Agent)
