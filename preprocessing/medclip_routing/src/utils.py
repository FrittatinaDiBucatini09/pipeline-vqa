"""
Utility functions for the MedCLIP Agentic Routing preprocessing stage.

Handles:
- Model loading (SciSpacy, Gemma-2-2B-it)
- Query quality evaluation via SciSpacy entity extraction
- Query expansion via Gemma-2-2B-it
"""

from typing import Any, List, Tuple

import torch


# ==============================================================================
# 1. MODEL LOADERS
# ==============================================================================

def load_scispacy():
    """Load the SciSpacy en_core_sci_sm model on CPU."""
    import spacy

    nlp = spacy.load("en_core_sci_sm")
    print("[INFO] SciSpacy en_core_sci_sm loaded (CPU)")
    return nlp


def load_gemma(model_name: str = "google/gemma-2-2b-it") -> Tuple[Any, Any]:
    """
    Load Gemma-2-2B-it for query expansion.

    Returns:
        (model, tokenizer) tuple.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()
    print(f"[INFO] Gemma model loaded: {model_name} (float16, CUDA)")
    return model, tokenizer


# ==============================================================================
# 2. QUERY QUALITY EVALUATION
# ==============================================================================

def evaluate_query_quality(
    nlp,
    query: str,
    entity_threshold: int = 2,
    word_threshold: int = 5,
) -> Tuple[bool, List[str]]:
    """
    Evaluate whether a text query is detailed enough for visual grounding.

    Heuristic:
    - The query must contain at least `entity_threshold` clinical entities.
    - The query must be at least `word_threshold` words long.

    Args:
        nlp: SciSpacy Language model.
        query: The text query to evaluate.
        entity_threshold: Minimum number of clinical entities required.
        word_threshold: Minimum word count required.

    Returns:
        (is_detailed, entities) tuple.
    """
    doc = nlp(query)
    entities = [ent.text for ent in doc.ents]
    word_count = len(query.split())

    is_detailed = len(entities) >= entity_threshold and word_count >= word_threshold
    return is_detailed, entities


# ==============================================================================
# 3. QUERY EXPANSION
# ==============================================================================

EXPANSION_PROMPT_TEMPLATE = (
    "You are a radiology assistant. A user submitted a brief medical image query "
    "that lacks clinical detail. Rewrite and expand it into a highly descriptive "
    "clinical prompt suitable for a medical vision model. Keep it under 50 words.\n\n"
    "Original query: \"{query}\"\n\n"
    "Expanded clinical prompt:"
)


def expand_query(
    model,
    tokenizer,
    query: str,
    max_new_tokens: int = 128,
) -> str:
    """
    Expand a brief query into a detailed clinical prompt using Gemma-2-2B-it.

    Args:
        model: The loaded Gemma model.
        tokenizer: The corresponding tokenizer.
        query: The original brief query.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        The expanded query string.
    """
    prompt = EXPANSION_PROMPT_TEMPLATE.format(query=query)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Decode only the newly generated tokens
    generated_ids = outputs[0, inputs["input_ids"].shape[1] :]
    expanded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Take only the first sentence/line to avoid runaway generation
    expanded = expanded.split("\n")[0].strip()
    if not expanded:
        return query  # Fallback to original if generation is empty

    return expanded
