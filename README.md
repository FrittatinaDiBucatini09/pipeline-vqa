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

---

## 🗂 Repository Structure

```text
pipeline-vqa/
└── Thesis/
    ├── data_prep/         # Data preparation scripts for GEMEX and MIMIC-CXR
    ├── preprocessing/     # Core computer vision pipeline
    │   ├── attention_map/ # Heatmap generation from attention weights
    │   ├── bounding_box/  # BBox extraction and intersection metrics
    │   └── segmentation/  # Localization and MedSAM/SAM-based segmentation
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


* **Automated Bounding Boxes**: Gridsearch configurations allow tuning of bounding box extraction thresholds, CRF integration, and region exploding/compositing.
* **Advanced VQA Generation**: Ties extracted image features to text via state-of-the-art vision-language models.
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
We recommend running modules using Docker to avoid dependency conflicts. Navigate to the module you want to run and execute its build script:
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
