# 🏥 Pipeline VQA: Medical Visual Question Answering Framework

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue?logo=docker)](https://www.docker.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)

**Pipeline VQA** is a comprehensive, modular framework designed for automated Medical Visual Question Answering (VQA). This repository handles end-to-end processing of medical imaging datasets (like MIMIC-CXR and GEMEX), from raw data preparation and attention heatmap generation to precise image segmentation, bounding box extraction, and ultimately, VQA generation and automated evaluation via an LLM Judge.

## 📑 Table of Contents
- [Architecture](#-architecture)
- [Repository Structure](#-repository-structure)
- [Key Features](#-key-features)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
  - [Running the Orchestrator](#running-the-orchestrator)
  - [Running Individual Modules](#running-individual-modules)
- [Experiments & Benchmarks](#-experiments--benchmarks)
- [License](#-license)

---

## 🏗 Architecture

The pipeline is composed of highly decoupled modules, communicating via shared file structures and orchestrated seamlessly. Below is the data flow:

```mermaid
graph TD
    %% Define styles
    classDef data fill:#e1f5fe,stroke:#03a9f4,stroke-width:2px;
    classDef process fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px;
    classDef model fill:#fff3e0,stroke:#ff9800,stroke-width:2px;
    classDef eval fill:#e8f5e9,stroke:#4caf50,stroke-width:2px;

    %% Nodes
    A[(Raw Datasets\nGEMEX / MIMIC)]:::data
    B[Data Prep Module]:::process
    
    subgraph Preprocessing
        C[Attention Maps / Heatmaps]:::process
        D[Step 1: Localization]:::process
        E[Step 2: Segmentation\nSAM / MedSAM]:::model
        F[Bounding Box Generation]:::process
    end
    
    subgraph Visual Question Answering
        G[VQA Generation]:::model
        H[LLM Judge Evaluation]:::eval
    end
    
    %% Connections
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    A -.-> G
    G --> H
    
    %% Labels
    click E "[https://github.com/bowang-lab/MedSAM](https://github.com/bowang-lab/MedSAM)" "MedSAM Integration"

```

### Agentic Query Expansion / Routing (Optional Middleware)

The pipeline includes an optional **NLP middleware stage** (`medclip_routing`) that intercepts the raw dataset before visual preprocessing. When enabled via the orchestrator, it:

1. **Evaluates query quality** using SciSpacy clinical entity extraction (CPU). Queries with fewer than 2 clinical entities or fewer than 5 words are flagged as "brief".
2. **Expands brief queries** using Gemma-2-2B-it (GPU, ~5GB VRAM), rewriting them into detailed clinical prompts suitable for downstream vision models.
3. **Outputs an enriched JSONL dataset** (`expanded_queries.jsonl`) with the `question` column overwritten by expanded text, preserving the original in `original_question`.

The enriched dataset is automatically bridged to the next preprocessing stage via the `ROUTED_DATASET_OVERRIDE` environment variable. This architecture isolates NLP models (Gemma) from vision models (BiomedCLIP/MedSAM) across sequential GPU execution steps, preventing OOM issues on the RTX 3090 (24GB).

```
                             ROUTED_DATASET_OVERRIDE          DATA_FILE_OVERRIDE
 Raw Dataset ──> [Routing] ─────────────────────────> [Preprocessing] ──────────────> [VQA Gen] ──> [Judge]
                 (NLP only)                            (Vision only)
                 SciSpacy + Gemma                      BBox / Attn / Seg
```

If routing is **not** selected in the orchestrator, preprocessing stages fall back to their default dataset configuration with zero changes required.

---

## 🗂 Repository Structure

```text
pipeline-vqa/
└── Thesis/
    ├── data_prep/         # Data preparation scripts for GEMEX and MIMIC-CXR
    ├── preprocessing/     # Core computer vision pipeline
    │   ├── attention_map/   # Heatmap generation from attention weights
    │   ├── bounding_box/    # BBox extraction and intersection metrics
    │   ├── medclip_routing/ # NLP query expansion middleware (SciSpacy + Gemma)
    │   └── segmentation/    # Localization and MedSAM/SAM-based segmentation
    ├── vqa/               # Generative AI components
    │   ├── src/           # VQA generation and LLM judge evaluation logic
    │   └── tests/         # Integration and E2E error handling tests
    ├── orchestrator/      # High-level pipeline controller
    └── experiments/       # Large-scale gridsearch configurations & benchmark reports

```

---

## ✨ Key Features

* **End-to-End Orchestration**: A single orchestrator script allows you to run the entire pipeline seamlessly using bash and python sub-processes.
* **Modular Preprocessing**:
* Generates robust attention heatmaps.
* Localizes regions of interest.
* Performs zero-shot and fine-tuned segmentation using **SAM / MedSAM3**.
* **Automatic VQA integration** via manifest generation (`vqa_manifest.csv`).


* **Automated Bounding Boxes**: Gridsearch configurations allow tuning of bounding box extraction thresholds, CRF integration, and region exploding/compositing.
* **Agentic Query Expansion**: Optional NLP middleware that evaluates query quality with SciSpacy and expands brief queries using Gemma-2-2B-it before visual preprocessing, improving downstream VQA quality.
* **Preprocessing→VQA Bridge**: The orchestrator automatically detects routing→preprocessing and preprocessing→VQA chains and configures data flow via environment variable injection (`ROUTED_DATASET_OVERRIDE`, `DATA_FILE_OVERRIDE`).
* **Advanced VQA Generation**: Ties extracted image features to text via state-of-the-art vision-language models (MedGemma, Qwen2-VL, etc.).
* **LLM as a Judge**: Evaluates VQA outputs systematically against ground-truth/gold-standard datasets using an LLM Judge, minimizing the need for manual grading.
* **Dockerized Environments**: Specific `Dockerfile`s (e.g., `Dockerfile.3090`, `Dockerfile.5090`, `Dockerfile.eval`) ensure dependency isolation and hardware-specific optimization.

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
```bash
git clone [https://github.com/your-username/pipeline-vqa.git](https://github.com/your-username/pipeline-vqa.git)
cd pipeline-vqa/Thesis

```


2. **Build the Docker Images:**
We recommend running modules using Docker to avoid dependency conflicts.

**Option A: Build All Images (Recommended)**
```bash
# From the Thesis/ root directory:
./build_all_images.sh
# Automatically discovers and builds all Docker images in the project

```

**Option B: Build Individual Modules**
Navigate to the module you want to run and execute its build script:
```bash
cd preprocessing/segmentation
./scripts/build_image.sh
# Or for specific GPUs: ./scripts/build_image_MedSAM3.sh

```


3. **Install Local Requirements (if running locally):**
```bash
pip install -r preprocessing/attention_map/docker/requirements.txt
# Repeat for other submodules as needed

```



---

## 🚀 Usage

### Running the Orchestrator

To run the full end-to-end pipeline automatically:

```bash
cd orchestrator
./run_orchestrator.sh

```

*Note: Ensure your configuration paths inside the orchestrator match your local dataset mounts.*

### Running Individual Modules

You can easily trigger individual components using the provided Slurm/Bash submit scripts:

* **Data Preparation**: `python data_prep/prepare_gemex.py`
* **Heatmaps**: `./preprocessing/attention_map/submit_heatmap_gen.sh`
* **Segmentation**: `./preprocessing/segmentation/submit_segmentation.sh`
* **Bounding Boxes**: `./preprocessing/bounding_box/submit_bbox_preprocessing.sh`
* **VQA & LLM Judge**: `./vqa/submit_generation.sh` && `./vqa/scripts/run_judge.sh`

---

## 📊 Experiments & Benchmarks

The repository includes a comprehensive `experiments/01_bbox_gridsearch` directory capable of generating massive amounts of benchmark configurations (varying thresholds, loose/smart padding, CRF integration).

You can find the aggregated markdown reports here:

* `BENCHMARK_V1_SUMMARY.md`
* `BENCHMARK_V2_SUMMARY.md`

To generate a new grid of configs:

```bash
cd experiments/01_bbox_gridsearch
python generate_grid_configs.py
./submit_benchmark.sh

```

---

## 📜 License

This project is licensed under the [MIT License](https://www.google.com/search?q=LICENSE). Please see the https://www.google.com/search?q=LICENSE file for details.
