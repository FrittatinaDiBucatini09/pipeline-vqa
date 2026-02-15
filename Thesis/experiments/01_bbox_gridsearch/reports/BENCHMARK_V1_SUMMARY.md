# Benchmark V1: Initial Hyperparameter Grid Search Results

**Date:** 2026-02-12 (original benchmark)  
**Total Configurations:** 36  
**Test Set Size:** 40 image-question pairs

---

## 📊 V1 Benchmark Results (36 Configurations)

### Performance Summary
- **Best Configuration:** `S_question_T0.30_C_raw_P_loose` → **0.2696 IoU**
- **Worst Configuration:** `S_standard_T0.60_C_crf_on_P_smart` → **0.1086 IoU**
- **Mean IoU:** 0.1576

### Top 10 Performers

| Rank | Configuration | Mean IoU |
|------|---------------|----------|
| 1 | S_question_T0.30_C_raw_P_loose | 0.2696 |
| 2 | S_question_T0.30_C_raw_P_smart | 0.2663 |
| 3 | S_context_T0.30_C_raw_P_loose | 0.2279 |
| 4 | S_standard_T0.30_C_raw_P_loose | 0.2279 |
| 5 | S_context_T0.30_C_raw_P_smart | 0.2217 |
| 6 | S_standard_T0.30_C_raw_P_smart | 0.2217 |
| 7 | S_question_T0.45_C_raw_P_loose | 0.2213 |
| 8 | S_question_T0.45_C_raw_P_smart | 0.2162 |
| 9 | S_question_T0.60_C_raw_P_loose | 0.1804 |
| 10 | S_question_T0.60_C_raw_P_smart | 0.1717 |

### Key Patterns

#### Strategy (S_*)
- **S_question** (USE_DYNAMIC_PROMPTS=true): **0.1923** ✅ WINNER
- S_context (fallback prompt): 0.1403
- S_standard (fixed prompt): 0.1403

#### Refinement (C_*)
- **C_raw** (SKIP_CRF=true): **0.1883** ✅ WINNER
- C_crf_on (SKIP_CRF=false): 0.1270

#### Threshold (T*)
- **T0.30**: **0.1831** ✅ WINNER
- T0.45: 0.1540
- T0.60: 0.1359

#### Padding (P_*)
- **P_loose** (ENABLE_SMART_PADDING=false): **0.1589** ✅ SLIGHT WINNER
- P_smart (ENABLE_SMART_PADDING=true): 0.1563

### Critical Observation
**None of the 36 V1 configurations tested the visual_regions approach!** The user's hypothesis remained completely unexplored:

```bash
USE_VISUAL_REGIONS="true"
COMPOSITE_REGIONS="true"
EXPLODE_REGIONS="false"
INCLUDE_CONTEXT_IN_INFERENCE="false"
```

This gap was addressed in Benchmark V2, which tested 80 additional configurations including the visual_regions approach.

---

## 🎯 Key Takeaways from V1

1. **Question-driven prompting is crucial** — `S_question` strategy dominates top performers
2. **Lower thresholds perform better** — T0.30 > T0.45 > T0.60 consistently  
3. **Skip CRF refinement** — Raw attention maps outperform CRF-refined maps
4. **Padding has minimal impact** — Loose vs smart padding shows < 2% difference
5. **Best combination:** `S_question` + `T0.30` + `C_raw` + `P_loose` → **0.2696 IoU**

---

## 📁 Configuration Details

### Parameter Grid (V1)

- **Strategies (3):** S_standard, S_context, S_question
- **Thresholds (3):** T0.30, T0.45, T0.60
- **CRF (2):** C_raw (skip), C_crf_on (apply)
- **Padding (2):** P_loose, P_smart

**Total:** 3 × 3 × 2 × 2 = **36 configurations**

---

## 🚀 Evolution to V2

Based on V1 findings, Benchmark V2 expanded to 80 configurations to:

1. **Test visual_regions hypothesis** (composite + explode modes) — 60 configs
2. **Refine S_question strategy** with finer-grained thresholds (0.20-0.65) — 20 configs
3. **Validate optimal threshold** across all strategies

**V2 Results:** Best config improved to **0.2908 IoU** using `S_question_T0.20_C_raw_P_loose`

See `BENCHMARK_V2_SUMMARY.md` for complete V2 analysis.

---

**Source:** Original V1 benchmark execution  
**Reference:** Data extracted from BENCHMARK_V2_SUMMARY.md (lines 3-53)  
**Regenerated:** 2026-02-13


**Date:** 2026-02-12  
**Total Configurations:** 36  
**Test Set Size:** 40 image-question pairs

## 📊 Performance Summary

- **Best Configuration:** `S_question_T0.30_C_raw_P_loose` → **0.2696 Mean IoU** ✅
- **Worst Configuration:** `S_standard_T0.60_C_crf_on_P_smart` → **0.1086 Mean IoU**
- **Overall Mean IoU:** 0.1576
- **Overall Median IoU:** 0.1499

---

## 🏆 Top 10 Configurations

| Rank | Configuration | Mean IoU | Median IoU | Acc@0.5 |
|------|---------------|----------|------------|---------|
| 1 | `S_question_T0.30_C_raw_P_loose` | **0.2696** | 0.2351 | 0.4375 |
| 2 | `S_question_T0.30_C_raw_P_smart` | 0.2663 | 0.2335 | 0.4375 |
| 3 | `S_context_T0.30_C_raw_P_loose` | 0.2279 | 0.1832 | 0.3542 |
| 4 | `S_standard_T0.30_C_raw_P_loose` | 0.2279 | 0.1832 | 0.3542 |
| 5 | `S_context_T0.30_C_raw_P_smart` | 0.2217 | 0.1742 | 0.3542 |
| 6 | `S_standard_T0.30_C_raw_P_smart` | 0.2217 | 0.1742 | 0.3542 |
| 7 | `S_question_T0.45_C_raw_P_loose` | 0.2213 | 0.1795 | 0.3125 |
| 8 | `S_question_T0.45_C_raw_P_smart` | 0.2162 | 0.1753 | 0.3125 |
| 9 | `S_question_T0.60_C_raw_P_loose` | 0.1804 | 0.1267 | 0.2292 |
| 10 | `S_question_T0.60_C_raw_P_smart` | 0.1717 | 0.1206 | 0.2083 |

---

## 🔍 Parameter Analysis

### Strategy (S_*)

| Strategy | Avg Mean IoU | Description |
|----------|--------------|-------------|
| **S_question** ✅ | **0.1923** | Dynamic prompts using the question text |
| S_context | 0.1403 | Fallback prompt with context |
| S_standard | 0.1403 | Fixed "anatomical region" prompt |

**Winner:** `S_question` — Using the actual question text for dynamic prompting significantly improves localization (+37% vs other strategies)

---

### Threshold (T*)

| Threshold | Avg Mean IoU | Description |
|-----------|--------------|-------------|
| **T0.30** ✅ | **0.1831** | CAM threshold = 0.30 |
| T0.45 | 0.1540 | CAM threshold = 0.45 |
| T0.60 | 0.1359 | CAM threshold = 0.60 |

**Winner:** `T0.30` — Lower thresholds capture more attention regions, leading to better coverage of ground truth boxes

---

### CRF Refinement (C_*)

| Refinement | Avg Mean IoU | Description |
|------------|--------------|-------------|
| **C_raw** ✅ | **0.1883** | Skip CRF (raw attention maps) |
| C_crf_on | 0.1270 | Apply CRF refinement |

**Winner:** `C_raw` — CRF refinement **reduces performance** (-32% relative). The raw attention maps are already well-localized.

---

### Padding (P_*)

| Padding | Avg Mean IoU | Description |
|---------|--------------|-------------|
| **P_loose** ✅ | **0.1589** | Disable smart padding |
| P_smart | 0.1563 | Enable smart padding |

**Winner:** `P_loose` — Minimal difference (+1.7% relative), but loose padding slightly outperforms

---

## 🎯 Key Insights

1. **Question-driven prompting is crucial** — `S_question` strategy dominates the top 10
2. **Lower thresholds are better** — T0.30 > T0.45 > T0.60 consistently
3. **Skip CRF refinement** — Raw attention maps outperform CRF-refined maps
4. **Padding has minimal impact** — Loose vs smart padding shows < 2% difference
5. **Clear winner combination:** `S_question` + `T0.30` + `C_raw` + `P_loose`

---

## ⚠️ Critical Gap

**None of the 36 V1 configurations tested the visual regions approach!**

The user's hypothesis about leveraging the `visual_regions` column with:
```bash
USE_VISUAL_REGIONS="true"
COMPOSITE_REGIONS="true"
```

...remains completely **unexplored** in V1. This was addressed in Benchmark V2.

---

## 📁 Configuration Details

### Parameter Grid

- **Strategies (3):** S_standard, S_context, S_question
- **Thresholds (3):** T0.30, T0.45, T0.60
- **CRF (2):** C_raw (skip), C_crf_on (apply)
- **Padding (2):** P_loose, P_smart

**Total:** 3 × 3 × 2 × 2 = **36 configurations**

### Configuration Naming Convention

```
S_{strategy}_T{threshold}_C_{crf}_P_{padding}.conf

Examples:
- S_question_T0.30_C_raw_P_loose.conf
- S_standard_T0.45_C_crf_on_P_smart.conf
```

---

## 📊 Files Generated

- **Individual Reports:** `evaluation/results/benchmark/{config_name}/evaluation_report.txt`
- **Per-Sample CSVs:** `evaluation/results/benchmark/{config_name}/per_sample_results.csv`

---

## 🚀 Next Steps → Benchmark V2

Based on V1 findings:

1. ✅ Refine the best strategy (`S_question`) with finer-grained thresholds
2. ✅ Test the **visual regions hypothesis** (composite + explode modes)
3. ✅ Expand to 80 configurations for comprehensive analysis

See `BENCHMARK_V2_SUMMARY.md` for V2 results.

---

**Generated:** 2026-02-13  
**Source:** V1 benchmark execution on 40-image test set  
**Results Location:** `evaluation/results/benchmark/`
