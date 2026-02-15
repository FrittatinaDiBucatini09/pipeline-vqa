# 🏥 **Medical VQA Inference Pipeline**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-Apache_2.0-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c?logo=pytorch&logoColor=white)
<br>
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)
![NVIDIA](https://img.shields.io/badge/NVIDIA-RTX_3090_%2F_5090-76B900?logo=nvidia&logoColor=white)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-orange)


This repository contains a robust pipeline for running inference and evaluation on Medical Visual Question Answering (VQA) tasks using Multimodal Large Language Models (e.g., Google MedGemma, LLaVA).

The project is designed to automate the process of generating answers from a medical dataset (specifically optimized for `VQA-RAD`) and evaluating them against ground truth using standard NLP metrics (ROUGE, Exact Match) and semantic evaluation.

## 📂 Project Structure

This sub-repository is organized as follows:

```text
General-Inference
├── build/                  # Containerization files
│   ├── Dockerfile.3090     # Docker config for RTX 3090 environment
│   ├── Dockerfile.5090     # Docker config for RTX 5090 environment
│   └── requirements.txt    # Python dependencies
├── configs/                # ⚙️ Experiment Configurations
│   ├── generation/         # VQA Generation configs (.conf)
│   └── judge/              # LLM Judge configs (.conf)
├── prepare_datasets/             # Data Ingestion & Mapping Scripts
│   ├── prepare_gemex_vqa.py      # Downloader/Mapper for GEMeX
│   ├── prepare_gemex.py          # Alternative mapper
│   └── prepare_mimic_ext.py      # Adapter for MIMIC-Ext (Text-Only)
├── scripts/                # Helper scripts for modular execution
│   ├── run_generation.sh   # Wrapper for the generation step
│   └── run_judge.sh        # Wrapper for the evaluation step
├── src/                    # Core source code
│   ├── generate_vqa.py     # Model loading & Generation
│   └── llm_judge.py        # Evaluation logic (Metrics calculation)
└── submit_generation.sh    # Main entry point (Orchestrator)
```

## 🛠️ Installation & Setup

You can run this project locally or within a Docker container (recommended for reproducibility).

### Option A: Docker (Recommended)

Build the image corresponding to your hardware availability:

```bash
# For NVIDIA RTX 3090
docker build -t med_vqa_project:3090 -f build/Dockerfile.3090 .

# For NVIDIA RTX 5090 (Future Proofing/HPC)
docker build -t med_vqa_project:5090 -f build/Dockerfile.3090 .
```

### Option B: Local Installation

Ensure you have Python 3.10+ and CUDA drivers installed.

```bash
pip install -r build/requirements.txt
```


## 🔑 Environment Configuration

Before running the pipeline, you must configure the necessary API keys and tokens.
Create a `.env` file in the root directory or export them in your shell:

```bash
# Required for downloading gated models (like MedGemma)
HF_TOKEN=your_huggingface_token_here

# Required ONLY if running the LLM Judge (Evaluation)
OPENAI_API_KEY=sk-your_openai_api_key_here

```

**Note:** The `submit_generation.sh` script is configured to automatically load these variables from the `.env` file if they are not set in the system environment.

## 🚀 Usage

The pipeline consists of two main stages:

1.  **Generation**: The model processes images and questions to predict answers.
2.  **Evaluation (Judge)**: Predictions are compared against the Ground Truth.

### Quick Start

To run the full pipeline (Generation + Evaluation) with default settings, execute the main submission script:

```bash
bash submit_generation.sh
```

This script orchestrates the execution of `src/generate_vqa.py` followed immediately by `src/llm_judge.py`.

-----

## 📊 Data Configuration

### Using a Local Dataset (CSV/JSON)

To use a local dataset (e.g., `gemex_VQA_mimic_mapped.csv`), configure `scripts/run_generation.sh` as follows:

1. **Disable HuggingFace download:** Set `DATASET_NAME=""`.
2. **Set Local File:** Point `DATA_FILE` to your CSV/JSON.
3. **Map Columns:** Ensure column names match your CSV headers.

**Example `scripts/run_generation.sh`:**

```bash
DATASET_NAME=""
DATA_FILE="gemex_VQA_mimic_mapped.csv" 

# Column Mapping
IMAGE_COLUMN="original_hf_path"  # Use relative paths (e.g., p10/p102...)
QUESTION_COLUMN="question"
ANSWER_COLUMN="answer"

```

### Image Paths & External Datasets

* **Relative Paths (Recommended):** If your CSV contains paths like `images/001.jpg`, place the `images` folder in the project root.
* **External Paths (e.g., `/datasets`):** If your images are stored outside the project (e.g., `/datasets/MIMIC-CXR`), you must mount the volume in `submit_generation.sh`:

```bash
# inside submit_generation.sh
docker run \
    ...
    -v "/datasets:/datasets" \  # Mount external dataset folder
    ...

```

*Note:* You might need to update `src/generate_vqa.py` to allow reading from root (`allowed_media_path = "/"`).

-----

## ⚙️ Configuration & Parameters

The core logic resides in `src/generate_vqa.py`. Below is a detailed explanation of the arguments you can modify in `submit_generation.sh` or pass directly to the python script.

### Model & Data Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--model_name` | `str` | `google/medgemma-4b-it` | The HuggingFace Hub path or local path to the multimodal model. |
| `--dataset_name` | `str` | `flaviagiammarino/vqa-rad` | The path to the VQA dataset (HuggingFace ID). |
| `--dataset_split` | `str` | `test` | The dataset split to use for inference (e.g., `train`, `test`, `validation`). |
| `--image_folder` | `str` | `images/` | Directory where images are stored (if loading locally). |

### Inference Parameters

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--batch_size` | `int` | `16` | Number of samples processed per batch. Adjust based on VRAM availability. |
| `--max_new_tokens` | `int` | `100` | Maximum length of the generated answer. |
| `--temperature` | `float` | `0.0` | Sampling temperature. `0.0` ensures deterministic (greedy) decoding. |
| `--use_images` | `bool` | `True` | **Crucial**: Enables multimodal input. If `False`, the model only sees the text question. |

### Prompting Strategy

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--few_shot` | `bool` | `True` | Enables In-Context Learning (providing examples in the prompt). |
| `--num_few_shot` | `int` | `5` | Number of examples to include in the context window. |
| `--cot` | `bool` | `False` | Enables **Chain-of-Thought**. If true, prompts the model to "think step-by-step". |
| `--thinking` | `bool` | `False` | Experimental flag for internal reasoning traces (if supported by model). |

## 📊 Outputs

After a successful run, the results are saved in the `results/` directory (or the path defined in your script) with the following structure:

1.  **`generations_DATE.json`**: A detailed log containing:
      * Image path
      * Question
      * Reference Answer (Gold)
      * Model Prediction
      * Correctness boolean
2.  **`metrics.json`**: Aggregated quantitative results:
      * Exact Match Accuracy
      * ROUGE-1, ROUGE-2, ROUGE-L scores.

## 📝 Example Output

An entry in the `generations.json` file looks like this:

```json
{
  "image_file": "images/synpic54610.jpg",
  "question": "Is there evidence of acute infarct?",
  "reference_answer": "no",
  "predicted_answer": "No, there is no evidence of an acute infarct.",
  "correct": true
}
```
## 🚨 Medical Disclaimer
**Important:** This software is for research and experimental purposes only.

The Artificial Intelligence models and algorithms (e.g., MedGemma) used in this repository are not approved medical devices. The results, predictions, and answers generated by this pipeline:

* Must **NOT** be used for clinical diagnosis, treatment decisions, or patient management.

* Must **NOT** replace professional medical advice from a qualified healthcare provider.

* May contain hallucinations, factual errors, or misinterpretations of medical imaging data.

The authors and contributors assume no liability for any direct or indirect consequences arising from the use of this code or the generated outputs.