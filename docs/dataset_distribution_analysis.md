# Dataset Distribution Analysis: MIMIC-Ext VQA

**Author:** Research Validation Process
**Date:** 2026-02-16
**Dataset:** MIMIC-Ext Medical VQA (50,000 samples)
**Purpose:** Validate sampling methodology to avoid bias in model evaluation

---

## Table of Contents

1. [Research Question](#research-question)
2. [Dataset Overview](#dataset-overview)
3. [Methodology](#methodology)
4. [Categorization Logic](#categorization-logic)
5. [Sampling Comparison](#sampling-comparison)
6. [Bias Calculation](#bias-calculation)
7. [Results](#results)
8. [Implications](#implications)
9. [Reproducing the Analysis](#reproducing-the-analysis)

---

## Research Question

**Problem:** We need to limit our preprocessing pipeline to 4,000 samples (from 50,000 total) due to computational constraints (15-hour SLURM time limit).

**Critical Question:** Will taking the **first 4,000 rows** (sequential sampling) give us a representative subset, or will it introduce bias?

**Why This Matters:** If the dataset is ordered by patient, question type, or anatomical region, sequential sampling could:
- Over-represent certain question types
- Under-represent important medical conditions
- Produce evaluation metrics that don't generalize to the full dataset
- Compromise research validity

---

## Dataset Overview

**Source:** MIMIC-Ext-MIMIC-CXR-VQA dataset
**Location:** `/home/rbalzani/medical-vqa/Thesis/preprocessing/bounding_box/mimic_ext_mapped.csv`

**Structure:**
```csv
image_path,question,visual_locations,visual_regions,answer_text,grade
files/p17/p17945608/.../image.jpg,Is there any occurrence of...,[],...,yes,Unknown
```

**Key Statistics:**
- Total samples: 50,000 image-question pairs
- Unique images: 38,307 (some images have multiple questions)
- Unique patients: 10
- Answer distribution: 100% "yes" (positive examples dataset)

**Question Types Examples:**
- **Tubes/Devices:** "Are there indications of any tubes/lines within the aortic arch?"
- **Anatomical:** "Is there any occurrence of anatomical findings in the left hilar structures?"
- **Disease:** "Is there any sign of diseases present in the right mid lung zone?"
- **Technical:** "Is the right mid lung zone showing any signs of technical assessments?"

---

## Methodology

### 1. Define Categorization Criteria

We categorized questions along two dimensions:

**Dimension 1: Question Type** (What is being asked about?)
- `tubes_devices` - Questions about medical devices, tubes, lines
- `anatomical` - Questions about anatomical structures/findings
- `disease` - Questions about pathologies or abnormalities
- `technical` - Questions about technical/imaging assessments
- `other` - Questions not matching above patterns

**Dimension 2: Anatomical Region** (Where in the body?)
- `lung` - Pulmonary regions
- `cardiac` - Heart and related structures
- `mediastinal` - Mediastinum, hilar structures, trachea
- `chest_wall` - Chest wall, hemidiaphragm
- `other` - Other anatomical locations

### 2. Create Two Samples

**Sample A: Sequential (First 4,000)**
```python
sequential_sample = data[:4000]  # First 4,000 rows
```
- Simulates what would happen with `STOP_AFTER=4000` parameter
- Fast, simple, no randomization
- **Hypothesis:** May be biased if data is ordered

**Sample B: Random (4,000 from 50,000)**
```python
random.seed(42)  # For reproducibility
random_sample = random.sample(data, 4000)
```
- Simulates true representative sampling
- Acts as ground truth for distribution
- **Hypothesis:** Should match full dataset distribution

### 3. Compare Distributions

For each sample, calculate:
- Percentage of each question type
- Percentage of each anatomical region
- Image diversity (unique images)

### 4. Quantify Bias

Calculate relative bias between sequential and random samples:

```
Bias % = (Sequential % - Random %) / Random % × 100
```

**Interpretation:**
- **Positive bias:** Over-representation (too many of that category)
- **Negative bias:** Under-representation (too few of that category)
- **Near zero:** Balanced representation

---

## Categorization Logic

The categorization is performed using regex pattern matching on the lowercase question text.

### Question Type Classification

```python
def categorize_question_type(question):
    q = question.lower()

    if 'tubes/lines' in q or 'devices' in q:
        return 'tubes_devices'
    elif 'disease' in q or 'abnormal' in q:
        return 'disease'
    elif 'anatomical' in q:
        return 'anatomical'
    elif 'technical' in q:
        return 'technical'
    else:
        return 'other'
```

**Rationale:**
- Questions explicitly mention "tubes/lines" or "devices"
- Disease questions contain "disease" or "abnormal" keywords
- Anatomical questions explicitly state "anatomical findings"
- Technical questions reference "technical assessments"
- Remaining questions are categorized as "other" (e.g., general presence questions)

### Anatomical Region Classification

```python
def categorize_anatomical_region(question):
    q = question.lower()

    if 'lung' in q:
        return 'lung'
    elif any(x in q for x in ['heart', 'cardiac', 'aortic']):
        return 'cardiac'
    elif any(x in q for x in ['mediastinum', 'hilar', 'trachea']):
        return 'mediastinal'
    elif 'chest' in q or 'hemidiaphragm' in q:
        return 'chest_wall'
    else:
        return 'other'
```

**Rationale:**
- Lung questions are very common, get their own category
- Cardiac structures grouped (heart, aortic arch, cardiac silhouette)
- Mediastinal structures grouped (mediastinum, hilar, trachea)
- Chest wall structures grouped (chest wall, hemidiaphragm)
- Other includes regions like SVC, or questions without specific anatomical terms

**Note:** This is a **mutually exclusive** classification - each question gets exactly one type and one region.

---

## Sampling Comparison

### Question Type Distribution

| Question Type | Sequential (first 4K) | Random (4K from 50K) | Difference |
|---------------|----------------------|---------------------|------------|
| tubes_devices | 1,508 (37.7%) | 744 (18.6%) | +19.1 pp |
| anatomical | 1,101 (27.5%) | 295 (7.4%) | +20.1 pp |
| disease | 766 (19.1%) | 442 (11.1%) | +8.0 pp |
| technical | 613 (15.3%) | 125 (3.1%) | +12.2 pp |
| other | 12 (0.3%) | 2,394 (59.9%) | -59.6 pp |

**Key Observation:** The sequential sample has **almost no "other" questions** (0.3%) while the random sample has 59.9%. This is a **massive distributional shift**.

### Anatomical Region Distribution

| Region | Sequential (first 4K) | Random (4K from 50K) | Difference |
|--------|----------------------|---------------------|------------|
| lung | 1,154 (28.9%) | 1,561 (39.0%) | -10.1 pp |
| cardiac | 252 (6.3%) | 222 (5.5%) | +0.8 pp |
| mediastinal | 692 (17.3%) | 387 (9.7%) | +7.6 pp |
| chest_wall | 494 (12.3%) | 207 (5.2%) | +7.1 pp |
| other | 1,408 (35.2%) | 1,623 (40.6%) | -5.4 pp |

**Key Observation:** Sequential sample under-represents **lung questions** by 10.1 percentage points.

---

## Bias Calculation

### Formula

For each category:

```
Sequential % = (Count in Sequential Sample / 4000) × 100
Random %     = (Count in Random Sample / 4000) × 100

Bias % = (Sequential % - Random %) / Random % × 100
```

### Example Calculations

#### Example 1: Tubes/Devices Questions

**Sequential:** 1,508 / 4,000 = 37.7%
**Random:** 744 / 4,000 = 18.6%

```
Bias = (37.7 - 18.6) / 18.6 × 100
     = 19.1 / 18.6 × 100
     = +103%
```

**Interpretation:** The sequential sample has **103% MORE** tubes/devices questions than it should. In other words, it has **2.03× as many** tubes/devices questions compared to a representative sample.

#### Example 2: Anatomical Questions

**Sequential:** 1,101 / 4,000 = 27.5%
**Random:** 295 / 4,000 = 7.4%

```
Bias = (27.5 - 7.4) / 7.4 × 100
     = 20.1 / 7.4 × 100
     = +272%
```

**Interpretation:** The sequential sample has **272% MORE** anatomical questions than it should. It has **3.72× as many** anatomical questions.

#### Example 3: Lung Region Questions

**Sequential:** 1,154 / 4,000 = 28.9%
**Random:** 1,561 / 4,000 = 39.0%

```
Bias = (28.9 - 39.0) / 39.0 × 100
     = -10.1 / 39.0 × 100
     = -26%
```

**Interpretation:** The sequential sample has **26% FEWER** lung questions than it should. It has only **0.74× as many** lung questions.

---

## Results

### Summary Table

| Category | Sequential % | Random % | Bias | Impact |
|----------|-------------|----------|------|--------|
| **Tubes/Devices questions** | 37.7% | 18.6% | **+103%** | 🔴 Severe over-representation |
| **Anatomical questions** | 27.5% | 7.4% | **+272%** | 🔴 Critical over-representation |
| **Lung regions** | 28.9% | 39.0% | **-26%** | 🟡 Moderate under-representation |
| **Other questions** | 0.3% | 59.9% | **-99.5%** | 🔴 Critical under-representation |

### Visualizing the Bias

If we trained a model on the **sequential sample**, it would:

✅ **Over-train on:**
- Medical device/tube detection
- Anatomical structure identification
- Mediastinal and chest wall regions

❌ **Under-train on:**
- General medical questions ("other" category)
- Lung pathology detection
- Diverse question phrasings

**Result:** The model's performance metrics would be **misleading** - high accuracy on tubes/devices, poor on general questions.

---

## Implications

### For Research Validity

1. **Evaluation Bias:** Using sequential sampling would produce metrics that:
   - Overestimate performance on device detection
   - Underestimate performance on lung pathology
   - Don't generalize to the full dataset

2. **Model Selection:** If comparing models:
   - Model A might excel at tube detection
   - Model B might excel at lung pathology
   - Sequential sampling would unfairly favor Model A

3. **Publication Integrity:** Reporting results from a biased sample:
   - Violates reproducibility standards
   - Misleads readers about true model capabilities
   - Reduces scientific value

### For Computational Efficiency

**Naive approach (sequential):**
```python
dataset = dataset[:4000]  # ❌ BIASED
```

**Correct approach (stratified random):**
```python
dataset = stratified_sample(dataset, n=4000, seed=42)  # ✅ REPRESENTATIVE
```

**Cost:** Negligible (sampling takes <1 second)
**Benefit:** Maintains research validity

---

## Reproducing the Analysis

### Prerequisites

```bash
cd /home/rbalzani/medical-vqa/Thesis/docs
```

Ensure you have the dataset:
```bash
ls ../preprocessing/bounding_box/mimic_ext_mapped.csv
```

### Run the Analysis

```bash
python3 bias_calculation.py
```

### Expected Output

```
================================================================================
BIAS CALCULATION METHODOLOGY
================================================================================

SEQUENTIAL (first 4000):
  Question Types:
    tubes_devices       : 1508 ( 37.7%)
    anatomical          : 1101 ( 27.5%)
    disease             :  766 ( 19.1%)
    technical           :  613 ( 15.3%)
    other               :   12 (  0.3%)
  Anatomical Regions:
    lung                : 1154 ( 28.9%)
    cardiac             :  252 (  6.3%)
    mediastinal         :  692 ( 17.3%)
    chest_wall          :  494 ( 12.3%)
    other               : 1408 ( 35.2%)

RANDOM (4000 from 50k):
  Question Types:
    tubes_devices       :  744 ( 18.6%)
    anatomical          :  295 (  7.4%)
    disease             :  442 ( 11.1%)
    technical           :  125 (  3.1%)
    other               : 2394 ( 59.9%)
  Anatomical Regions:
    lung                : 1561 ( 39.0%)
    cardiac             :  222 (  5.5%)
    mediastinal         :  387 (  9.7%)
    chest_wall          :  207 (  5.2%)
    other               : 1623 ( 40.6%)

================================================================================
BIAS CALCULATION
================================================================================

Question Type Bias:
Category                  Sequential %     Random %            Bias
--------------------------------------------------------------------------------
tubes_devices                    37.7%        18.6%           +103%
  → Formula: (37.7 - 18.6) / 18.6 × 100 = +103%
anatomical                       27.5%         7.4%           +273%
  → Formula: (27.5 - 7.4) / 7.4 × 100 = +273%

Anatomical Region Bias:
Region                    Sequential %     Random %            Bias
--------------------------------------------------------------------------------
lung                             28.9%        39.0%            -26%
  → Formula: (28.9 - 39.0) / 39.0 × 100 = -26%

================================================================================
INTERPRETATION
================================================================================

Positive bias = Over-representation in sequential sample
  Example: +103% means sequential has 2.03× more tubes/devices questions

Negative bias = Under-representation in sequential sample
  Example: -26% means sequential has 0.74× fewer lung questions

This demonstrates that sequential sampling (first 4K rows) does NOT
represent the true distribution of the full dataset.
```

### Modify for Different Sample Sizes

Edit `bias_calculation.py` line 43-44:

```python
sequential_sample = data[:N]  # Change N to desired size
random_sample = random.sample(data, N)  # Match N
```

---

## Conclusion

**Finding:** Sequential sampling introduces **severe bias** (up to +272% over-representation).

**Recommendation:** Use **stratified random sampling** to maintain distributional balance.

**Implementation:** See [`../preprocessing/bounding_box/scripts/create_stratified_sample.py`](../preprocessing/bounding_box/scripts/create_stratified_sample.py)

**Validation:** See [`../preprocessing/bounding_box/SAMPLING_STRATEGY.md`](../preprocessing/bounding_box/SAMPLING_STRATEGY.md)

---

## Implementation Across Preprocessing Methods

The stratified sampling strategy has been implemented across all three preprocessing methods:

### 1. Bounding Box Preprocessing

- **Location:** `../preprocessing/bounding_box/`
- **Stratified Sample:** `mimic_ext_sample_4k.csv` (4K from 50K samples)
- **Script:** `scripts/create_stratified_sample.py`
- **Documentation:** `SAMPLING_STRATEGY.md`
- **Config Updated:** `configs/exp/S_question_T0.45_C_crf_on_P_loose.conf`
- **Parameter:** `METADATA_FILENAME="mimic_ext_sample_4k.csv"`

### 2. Attention Map Preprocessing

- **Location:** `../preprocessing/attention_map/`
- **Stratified Sample:** `mimic_ext_sample_4k.csv` (4K from 290K samples)
- **Script:** `scripts/create_stratified_sample.py`
- **Documentation:** `SAMPLING_STRATEGY.md`
- **Script Updated:** `scripts/run_heatmap_gen.sh`
- **Parameters:**
  - `METADATA_FILENAME="mimic_ext_sample_4k.csv"`
  - `STOP_AFTER=""` (deprecated, use stratified CSV instead)

### 3. Segmentation Preprocessing

- **Location:** `../preprocessing/segmentation/`
- **Stratified Sample:** `metadata/mimic_ext_sample_4k.csv` (4K from 290K samples)
- **Script:** `scripts/create_stratified_sample.py`
- **Documentation:** `SAMPLING_STRATEGY.md`
- **Script Updated:** `scripts/run_pipeline.sh`
- **Parameter:** `METADATA_FILE="/workspace/metadata/mimic_ext_sample_4k.csv"`

---

## Quick Reference: Generating Stratified Samples

For any preprocessing method:

```bash
cd /home/rbalzani/medical-vqa/Thesis/preprocessing/<method>/

python3 scripts/create_stratified_sample.py \
    --input <full_dataset>.csv \
    --output <stratified_sample>.csv \
    --n_samples 4000 \
    --seed 42 \
    --report
```

Replace `<method>` with: `bounding_box`, `attention_map`, or `segmentation`

---

## References

- **Dataset:** MIMIC-Ext-MIMIC-CXR-VQA
- **Bias Analysis Script:** `bias_calculation.py` (this directory)
- **Project Validation:** `VALIDATION_REPORT.md` (this directory)

**Preprocessing Method Implementations:**
- **Bounding Box:** [`../preprocessing/bounding_box/SAMPLING_STRATEGY.md`](../preprocessing/bounding_box/SAMPLING_STRATEGY.md)
- **Attention Map:** [`../preprocessing/attention_map/SAMPLING_STRATEGY.md`](../preprocessing/attention_map/SAMPLING_STRATEGY.md)
- **Segmentation:** [`../preprocessing/segmentation/SAMPLING_STRATEGY.md`](../preprocessing/segmentation/SAMPLING_STRATEGY.md)

---

**Last Updated:** 2026-02-16
**Validated By:** Claude Code (Project Validation Agent)
