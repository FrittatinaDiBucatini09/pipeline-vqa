# MedCLIP Agentic Routing (NLP Query Expansion Middleware)

NLP middleware stage for the Medical VQA pipeline. Despite the folder name (`medclip_routing`), this stage no longer uses any vision models (BiomedCLIP/GradCAM have been removed). It evaluates text query quality and dynamically expands vague queries using an LLM before handing off to downstream visual preprocessing stages.

## Pipeline

1. **SciSpacy** (`en_core_sci_sm`, CPU): Extracts clinical entities from the text query
2. **Routing Decision**: If the query has < `ENTITY_THRESHOLD` entities OR < `WORD_THRESHOLD` words, it is flagged as "brief"
3. **Gemma-2-2B-it** (GPU, ~5GB VRAM): Expands brief queries into detailed clinical prompts via a radiology-specific prompt template

No images are loaded or processed. The stage operates purely on text data.

## VRAM Budget

| Component | VRAM |
|-----------|------|
| SciSpacy en_core_sci_sm | 0 GB (CPU) |
| Gemma-2-2B-it (float16) | ~5-6 GB |
| **Total** | **~5-6 GB** (fits RTX 3090, 24GB) |

## Output

- `results/expanded_queries.jsonl`: One JSON record per row containing **all original dataset columns** plus:
  - `original_question`: The unmodified query text
  - `question` (or configured `--text_col`): Overwritten with expanded text (if expansion was triggered) or left unchanged
  - `was_expanded` (bool): Whether Gemma expansion was applied
  - `entities_detected` (list): Clinical entities extracted by SciSpacy

## Orchestrator Integration

This stage acts as **optional middleware** in the pipeline. When selected via the orchestrator alongside a downstream preprocessing stage (bounding box, attention map, or segmentation):

1. The routing stage runs first, producing `expanded_queries.jsonl`
2. The orchestrator injects a **bridge** that exports `ROUTED_DATASET_OVERRIDE` pointing to the JSONL
3. Downstream preprocessing scripts detect this variable, copy the JSONL into their metadata directory, and consume it as their input dataset
4. The downstream stage then produces its own `vqa_manifest.csv` for the VQA generation stage (via `DATA_FILE_OVERRIDE`)

```
medclip_routing -> ROUTED_DATASET_OVERRIDE -> bbox/attn/seg -> DATA_FILE_OVERRIDE -> vqa_gen
```

If routing is **not** enabled, downstream stages fall back to their default dataset from configuration. Zero changes needed.

## Usage

### Local Test
```bash
# Edit configs/test_e2e.conf (STOP_AFTER="3")
./submit_routing.sh configs/test_e2e.conf
```

### SLURM Submission
```bash
sbatch submit_routing.sh configs/default.conf
```

### Via Orchestrator
The stage is registered as "Routing: NLP Query Expansion" in the pipeline orchestrator.

## Configuration

All parameters are controlled via `.conf` files in `configs/`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ENTITY_THRESHOLD` | 2 | Min clinical entities for "detailed" |
| `WORD_THRESHOLD` | 5 | Min word count for "detailed" |
| `GEMMA_MAX_NEW_TOKENS` | 128 | Max tokens for query expansion |
| `STOP_AFTER` | "" | Limit rows for debugging |
| `METADATA_FILENAME` | gemex_VQA_mimic_mapped.csv | Input dataset filename |
