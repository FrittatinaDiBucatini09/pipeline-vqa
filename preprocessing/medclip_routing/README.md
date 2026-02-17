# MedCLIP Agentic Routing

Query-aware preprocessing stage for the Medical VQA pipeline. Evaluates text query quality and dynamically expands vague queries before generating bounding boxes.

## Pipeline

1. **SciSpacy** (`en_core_sci_sm`, CPU): Extracts clinical entities from the text query
2. **Routing Decision**: If the query has < 2 entities OR < 5 words, it is flagged as "brief"
3. **Gemma-2-2B-it** (GPU, ~5GB VRAM): Expands brief queries into detailed clinical prompts
4. **BiomedCLIP + GradCAM** (GPU, ~1GB VRAM): Generates bounding boxes from the final query + image

## VRAM Budget

| Component | VRAM |
|-----------|------|
| SciSpacy en_core_sci_sm | 0 GB (CPU) |
| Gemma-2-2B-it (float16) | ~5-6 GB |
| BiomedCLIP ViT-B/16 | ~1 GB |
| GradCAM overhead | ~1-2 GB |
| **Total** | **~8-9 GB** (fits RTX 3090, 24GB) |

## Output

- `results/predictions.jsonl`: One JSON record per image with bboxes, routing metadata, and original CSV fields
- `results/vqa_manifest.csv`: Bridge file for the VQA Generation stage (image_path, question, answer)

## Usage

### Local Test
```bash
# Edit configs/default.conf to set STOP_AFTER="5"
./submit_routing.sh configs/default.conf
```

### SLURM Submission
```bash
sbatch submit_routing.sh configs/default.conf
```

### Via Orchestrator
The stage is registered in the pipeline orchestrator as "Preprocessing: MedCLIP Agentic Routing".

## Configuration

All parameters are controlled via `.conf` files in `configs/`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CAM_THRESHOLD` | 0.50 | GradCAM activation threshold |
| `ENTITY_THRESHOLD` | 2 | Min clinical entities for "detailed" |
| `WORD_THRESHOLD` | 5 | Min word count for "detailed" |
| `GEMMA_MAX_NEW_TOKENS` | 128 | Max tokens for query expansion |
| `STOP_AFTER` | "" | Limit rows for debugging |
