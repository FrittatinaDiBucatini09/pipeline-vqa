# Bounding Box Evaluation Module

This directory contains the **Evaluation Sub-system** for the WSOL pipeline. It is responsible for validating bounding box predictions against Ground Truth annotations using IoU (Intersection over Union) metrics.

## 🏗 Architecture

Per the project's **No-Cross-Import Rule**, this module is completely self-contained. It does not import any code from the `generation/` folder.

- **Input:** `predictions.jsonl` (produced by the generation pipeline).
- **Reference:** Ground Truth annotations from the CSV (embedded in `predictions.jsonl`).
- **Processing:**
    1. Parses 224-reference-space gold boxes.
    2. Resolves actual image dimensions from disk.
    3. Scales gold boxes to pixel space.
    4. Computes **Greedy Best-Match IoU**.
- **Output:** Summary Report & Per-Sample CSV.

## 📂 File Structure

```text
evaluation/
├── evaluate_bbox.py        # Core logic: IoU computation & reporting
├── Dockerfile.eval         # Lightweight CPU-only Docker image (Python + Pillow + NumPy)
├── run_evaluation.sh       # Docker wrapper script (mounts datasets/results)
└── submit_evaluation.sh    # Slurm submission script
```

## 🚀 Usage

### 1. Prerequisite
Ensure the generation pipeline has run and produced a `predictions.jsonl` file in `../generation/results/`.

### 2. Run on HPC (Slurm)
Submit the job to the cluster. This will automatically build the Docker image if missing.

```bash
sbatch submit_evaluation.sh
```

### 3. Run Locally (Interactive Docker)
You can also run it interactively on the master node (`faretra`) using the wrapper:

```bash
./run_evaluation.sh
```

## 📊 Metrics

The script reports:
- **Mean IoU:** Average Intersection-over-Union across all samples.
- **Accuracy @ Thresholds:** Percentage of samples with IoU ≥ 0.25, 0.50, 0.75.
- **Per-Type Breakdown:** Metrics separated by Question Type (e.g., `location` vs `abnormality`).

## 🛠 Coordinate System Note

> **⚠️ IMPORTANT ⚠️**
>
> **Gold Coordinates** are stored in a **224×224** reference space.
> **Predicted Boxes** are stored in **Pixel Space** (original image dimensions).
>
> The script automatically handles this mismatch by reading the dimensions of the original images from `/datasets/MIMIC-CXR`.
