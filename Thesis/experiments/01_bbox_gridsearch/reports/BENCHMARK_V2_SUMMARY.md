# Benchmark V2: Configuration Analysis & New Experiments

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
**None of the 36 V1 configurations tested the visual_regions approach!** The user's hypothesis remains completely unexplored:

```bash
USE_VISUAL_REGIONS="true"
COMPOSITE_REGIONS="true"
EXPLODE_REGIONS="false"
INCLUDE_CONTEXT_IN_INFERENCE="false"
```

## 🧪 V2 Benchmark (80 New Configurations)

### Configuration Groups

#### GROUP 1: Visual Regions + Composite Mode (40 configs)
**Hypothesis:** Using anatomical annotations (`visual_regions` column) with `COMPOSITE_REGIONS=true` will create better bounding boxes by leveraging expert annotations.

**Parameter Grid:**
- `USE_VISUAL_REGIONS="true"`
- `COMPOSITE_REGIONS="true"`
- `EXPLODE_REGIONS="false"`
- `CAM_THRESHOLD`: [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
- `SKIP_CRF`: [true, false]
- `ENABLE_SMART_PADDING`: [true, false]
- `ENABLE_MULTI_BOXES`: [true, false]
- `INCLUDE_CONTEXT_IN_INFERENCE`: [true, false]

**Example configs:**
- `S_regions_composite_T0.20_C_raw_P_loose_M1_I0.conf`
- `S_regions_composite_T0.30_C_raw_P_loose_M1_I0.conf`
- `S_regions_composite_T0.45_C_crf_on_P_smart_M0_I1.conf`

#### GROUP 2: Visual Regions + Explode Mode (20 configs)
**Hypothesis:** Creating individual boxes for each anatomical label (instead of one composite box) may yield better localization.

**Parameter Grid:**
- `USE_VISUAL_REGIONS="true"`
- `COMPOSITE_REGIONS="false"`
- `EXPLODE_REGIONS="true"`
- `CAM_THRESHOLD`: [0.20, 0.30, 0.40, 0.50]
- `SKIP_CRF`: [true, false]
- `ENABLE_SMART_PADDING`: [true, false]
- `ENABLE_MULTI_BOXES`: [true, false]

**Example configs:**
- `S_regions_explode_T0.20_C_raw_P_loose_M0.conf`
- `S_regions_explode_T0.30_C_crf_on_P_smart_M1.conf`

#### GROUP 3: Refined S_question (20 configs)
**Hypothesis:** Fine-tuning the best V1 strategy (`S_question`) with more granular threshold values.

**Parameter Grid:**
- `USE_VISUAL_REGIONS="false"`
- `USE_DYNAMIC_PROMPTS="true"`
- `CAM_THRESHOLD`: [0.20, 0.25, 0.27, 0.32, 0.35, 0.38, 0.42, 0.47, 0.52, 0.65]
- `SKIP_CRF="true"` (always raw, as it performed best)
- `ENABLE_SMART_PADDING`: [true, false]

**Example configs:**
- `S_question_T0.25_C_raw_P_loose.conf`
- `S_question_T0.35_C_raw_P_smart.conf`

## 📁 Output Structure

```
configs/benchmark_v2/
├── S_regions_composite_T0.20_C_raw_P_loose_M0_I0.conf
├── S_regions_composite_T0.20_C_raw_P_loose_M1_I0.conf
├── ... (38 more visual composite configs)
├── S_regions_explode_T0.20_C_raw_P_loose_M0.conf
├── ... (19 more visual explode configs)
├── S_question_T0.20_C_raw_P_loose.conf
└── ... (19 more refined S_question configs)
```

## 🚀 V2 Benchmark Results (80 Configurations)

**Status:** ✅ **COMPLETED** | **Date:** 2026-02-13

### Overall Performance Summary
- **Total Configurations Tested:** 80
- **Best Configuration:** `S_question_T0.20_C_raw_P_loose` → **0.2908 IoU** ✅
- **Worst Configuration:** `S_regions_explode_T0.20_C_crf_on_P_smart_M1` → **0.1005 IoU**
- **Average Mean IoU:** 0.2306

### Top 10 Configurations

| Rank | Configuration | Mean IoU | Strategy | Threshold | CRF | Padding |
|------|---------------|----------|----------|-----------|-----|---------|
| 1 | `S_question_T0.20_C_raw_P_loose` | **0.2908** | question | 0.20 | off | loose |
| 2 | `S_question_T0.20_C_raw_P_smart` | 0.2896 | question | 0.20 | off | smart |
| 3 | `S_regions_composite_T0.20_C_crf_on_P_loose_M0_I0` | 0.2864 | composite | 0.20 | on | loose |
| 4 | `S_regions_composite_T0.20_C_crf_on_P_loose_M1_I0` | 0.2864 | composite | 0.20 | on | loose |
|  5 | `S_regions_composite_T0.20_C_crf_on_P_smart_M0_I0` | 0.2864 | composite | 0.20 | on | smart |
| 6 | `S_regions_composite_T0.20_C_crf_on_P_smart_M1_I0` | 0.2864 | composite | 0.20 | on | smart |
| 7 | `S_regions_composite_T0.20_C_raw_P_loose_M0_I0` | 0.2864 | composite | 0.20 | off | loose |
| 8 | `S_regions_composite_T0.20_C_raw_P_loose_M1_I0` | 0.2864 | composite | 0.20 | off | loose |
| 9 | `S_regions_composite_T0.20_C_raw_P_smart_M0_I0` | 0.2864 | composite | 0.20 | off | smart |
| 10 | `S_regions_composite_T0.20_C_raw_P_smart_M1_I0` | 0.2864 | composite | 0.20 | off | smart |

### Strategy Comparison

#### S_question (Question-Driven, 20 configs)
- **Best:** 0.2908 (T0.20)
- **Worst:** 0.1555 (T0.65)
- **Average:** 0.2394
- **Notes:** Best overall performance. Lower thresholds (0.20-0.27) significantly outperform higher thresholds.

#### S_regions_composite (Visual Regions + Composite, 40 configs)
- **Best:** 0.2864 (T0.20, multiple variants)
- **Worst:** 0.2088 (T0.30, I1 variants)
- **Average:** 0.2625
- **Notes:** Very consistent performance across CRF/padding/mode variations. Threshold T0.20 is clearly optimal. The `I0` (exclude context) variants slightly outperform `I1`.

#### S_regions_explode (Visual Regions + Explode, 20 configs)
- **Best:** 0.2632 (T0.20, raw, loose, M1)
- **Worst:** 0.1005 (T0.20/T0.30, crf_on, M1)
- **Average:** 0.1707
- **Notes:** **CRF hurts performance dramatically** (0.10 IoU vs 0.24 IoU). M1 (multi-boxes) outperforms M0 when CRF is off.

### Key Insights

#### 1. **Threshold Impact** 🎯
- **T0.20 is optimal across ALL strategies** (0.20 > 0.25 > 0.27 > 0.30...)
- Lower thresholds capture more attention regions → better localization

#### 2. **Visual Regions Hypothesis** 🧪
- **Composite mode (GROUP 1) performed very well** → 0.2864 IoU (2nd-3rd place)
- **Explode mode (GROUP 2) underperformed** → 0.1707 avg (significantly worse)
- **Conclusion:** Using anatomical annotations helps, but only in **composite mode**

#### 3. **CRF Refinement** 🔧
- **CRF has NO consistent benefit** and **hurts explode mode severely**
- Best practice: **Skip CRF** (`C_raw`) for all strategies

#### 4. **Padding Strategy** 📏
- **Minimal impact** (loose vs smart difference < 0.01 IoU)
- Slight preference for `P_loose` in top configs

#### 5. **Multi-Region Parameters** 📦
- For composite: `M0/M1` and `I0/I1` show minimal difference
- For explode: **M1 (multi-boxes) + raw significantly outperforms M0**

### Recommended Configuration

Based on all 80 experiments, the **optimal configuration** is:

```bash
# Winner: S_question_T0.20_C_raw_P_loose
USE_VISUAL_REGIONS="false"
USE_DYNAMIC_PROMPTS="true"
CAM_THRESHOLD=0.20
SKIP_CRF=true
ENABLE_SMART_PADDING=false
```

**Alternative (if visual regions are required):**
```bash
# Runner-up: S_regions_composite_T0.20_C_raw_P_loose_M0_I0
USE_VISUAL_REGIONS="true"
COMPOSITE_REGIONS="true"
EXPLODE_REGIONS="false"
CAM_THRESHOLD=0.20
SKIP_CRF=true
ENABLE_SMART_PADDING=false
ENABLE_MULTI_BOXES=false
INCLUDE_CONTEXT_IN_INFERENCE=false
```

### Files Generated

- **Summary CSV:** `evaluation/results/benchmark_v2_summary.csv` (all 80 configs ranked)
- **Individual Reports:** `evaluation/results/benchmark_v2/{config_name}/evaluation_report.txt`

---
**Generated:** 2026-02-13 | **Updated:** 2026-02-13 | **Tool:** `analyze_and_generate_configs.py` + `aggregate_results.py`

