# Attention Map: Dense Heatmap Visualization Pipeline

Generates visual explanations (attention heatmaps) from **BiomedCLIP + gScoreCAM** overlaid on medical X-Ray images. Adapted from the Bounding Box pipeline — same model/data logic, but outputs **colored heatmap overlay images** instead of discrete bounding boxes.

## Output

### Visual Outputs

Each input image produces a `.png` file with a **JET/TURBO colormap** overlay:
- **Red** = High model attention (gScoreCAM activation)
- **Blue** = Low model attention
- Background artifacts (text, markers) are suppressed via Otsu body masking

### VQA Integration

The pipeline automatically generates a `vqa_manifest.csv` file for downstream VQA integration:

```csv
image_path,question,answer
files/p10/p10268877/s50042142/4c3c1335-0fce9b11-027c582b-a0ed8d89-ca614d90_idx1.png,What is the position of the ET tube as seen in the CXR?,C
files/p10/p10268877/s50042142/4c3c1335-0fce9b11-027c582b-a0ed8d89-ca614d90_idx2.png,What could be inferred from the NG tube's positioning as noted in the CXR?,NG tube tip is in the stomach.
```

**Benefits:**
- **Seamless Orchestration:** When chaining `attn_map → vqa_gen` in the orchestrator, the VQA stage automatically uses the heatmap images
- **One Row Per Heatmap:** Multi-region questions create multiple manifest rows
- **Metadata Preserved:** Original questions and answers maintained for evaluation

## Quick Start

```bash
# From the Attention_Map/ directory:
sbatch submit_heatmap_gen.sh [configs/gemex/exp_01_vqa.conf]
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ALPHA` | `0.5` | Blending factor (0=original, 1=heatmap only) |
| `COLORMAP` | `jet` | Colormap: `jet`, `turbo`, `inferno`, `hot` |
| `SAVE_RAW_CAM` | `false` | Also save raw grayscale CAM |
| `ENABLE_BODY_MASK` | `true` | Suppress background with Otsu thresholding |
| `SKIP_CRF` | `true` | Set to `false` to enable Dense CRF refinement |
| `CAM_VERSION` | `gScoreCAM` | Class Activation Map method |
| `STOP_AFTER` | `5` | Limit processing (set empty for full run) |

## Directory Structure

```
Attention_Map/
├── configs/                  # Experiment configs (shared with Bounding_Box)
├── scripts/
│   ├── generate_heatmaps.py  # Main pipeline script
│   ├── utils.py              # CAMWrapper, reshape_transform
│   ├── Dockerfile.3090       # Docker environment
│   ├── run_heatmap_gen.sh    # Docker wrapper
│   └── requirements.txt      # Python dependencies
├── results/                  # Output heatmap images
├── submit_heatmap_gen.sh     # Slurm entry point
└── README.md
```

## WandB Monitoring

Each run logs to the **`GEMeX-VQA-Pipeline`** WandB project under `job_type=step1-attn-map`. Tracked metrics:

| Metric | Description |
|--------|-------------|
| `processed_images` | Running count of successfully processed images |
| `errors` | Cumulative error count |
| `example_heatmap` | Sample heatmap overlay image (logged every 50 batches) |
| `success_rate`, `throughput_img_per_sec`, `total_time_sec` | Written to run summary on completion |

Output artifacts (`vqa_manifest.csv`, `report.txt`) are logged as versioned WandB artifacts. Set `WANDB_MODE` in the environment to control logging mode (`online` / `offline` / `disabled`).

## Differences from Bounding Box Pipeline

| Aspect | Bounding Box | Attention Map |
|--------|-------------|---------------|
| Output | JSONL coordinates + box overlays | PNG heatmap overlay images |
| Post-processing | CRF, contours, adaptive padding | Normalize → Colormap → Alpha blend |
| Dependencies | pydensecrf | None (removed) |
| Script | `bbox_preprocessing.py` | `generate_heatmaps.py` |