
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import pandas as pd
import numpy as np
from datetime import datetime

from vllm import LLM, SamplingParams


JUDGE_PROMPT = """You are an expert medical evaluator tasked with determining if a predicted answer is semantically equivalent to the reference answer for a medical Visual Question Answering (VQA) task.

**Question:** {question}

**Reference Answer (Ground Truth):** {reference_answer}

**Predicted Answer:** {predicted_answer}

**Task:** Determine if the predicted answer is correct by comparing it to the reference answer. The predicted answer does not need to match the reference answer word-for-word, but it must convey the same medical meaning and factual content.

**Evaluation Criteria:**
1. **Semantic Equivalence**: Does the predicted answer have the same meaning as the reference answer?
2. **Medical Accuracy**: Is the predicted answer medically correct according to the reference?
3. **Completeness**: Does the predicted answer address all key points in the reference answer?
4. **Paraphrasing Tolerance**: Accept valid paraphrases, synonyms, or different phrasings that preserve meaning.

**Examples of Correct Matches:**
- Reference: "Yes" | Predicted: "Yes, it is visible" → CORRECT
- Reference: "The image shows pneumonia" | Predicted: "Pneumonia is present in the image" → CORRECT
- Reference: "No fracture" | Predicted: "There is no fracture" → CORRECT
- Reference: "Chest X-ray" | Predicted: "CXR" or "Chest radiograph" → CORRECT

**Examples of Incorrect Matches:**
- Reference: "Yes" | Predicted: "No" → INCORRECT
- Reference: "Pneumonia" | Predicted: "Normal lungs" → INCORRECT
- Reference: "Frontal view" | Predicted: "Lateral view" → INCORRECT

**Your Response:**
Provide your evaluation in the following format:

**VERDICT:** [CORRECT or INCORRECT]
**CONFIDENCE:** [HIGH, MEDIUM, or LOW]
**EXPLANATION:** [Brief explanation of your decision in 1-2 sentences]

Respond now:"""


JUDGE_PROMPT_COT = """You are an expert medical evaluator tasked with determining if a predicted answer is semantically equivalent to the reference answer for a medical Visual Question Answering (VQA) task.

**Question:** {question}

**Reference Answer (Ground Truth):** {reference_answer}

**Predicted Answer:** {predicted_answer}

**Task:** Determine if the predicted answer is correct by comparing it to the reference answer. The predicted answer does not need to match the reference answer word-for-word, but it must convey the same medical meaning and factual content.

**Evaluation Criteria:**
1. **Semantic Equivalence**: Does the predicted answer have the same meaning as the reference answer?
2. **Medical Accuracy**: Is the predicted answer medically correct according to the reference?
3. **Completeness**: Does the predicted answer address all key points in the reference answer?
4. **Paraphrasing Tolerance**: Accept valid paraphrases, synonyms, or different phrasings that preserve meaning.

**Examples of Correct Matches:**
- Reference: "Yes" | Predicted: "Yes, it is visible" → CORRECT
- Reference: "The image shows pneumonia" | Predicted: "Pneumonia is present in the image" → CORRECT
- Reference: "No fracture" | Predicted: "There is no fracture" → CORRECT
- Reference: "Chest X-ray" | Predicted: "CXR" or "Chest radiograph" → CORRECT

**Examples of Incorrect Matches:**
- Reference: "Yes" | Predicted: "No" → INCORRECT
- Reference: "Pneumonia" | Predicted: "Normal lungs" → INCORRECT
- Reference: "Frontal view" | Predicted: "Lateral view" → INCORRECT

**Instructions:**
Follow these steps to evaluate:

**STEP 1 - ANALYZE REFERENCE:**
Identify the key factual points and medical concepts in the reference answer.

**STEP 2 - ANALYZE PREDICTED:**
Identify the key factual points and medical concepts in the predicted answer.

**STEP 3 - COMPARE:**
Compare the two answers for semantic equivalence, considering paraphrasing and synonyms.

**STEP 4 - VERDICT:**
Provide your final verdict in this exact format:

**VERDICT:** [CORRECT or INCORRECT]
**CONFIDENCE:** [HIGH, MEDIUM, or LOW]
**EXPLANATION:** [Brief explanation of your decision in 1-2 sentences]

Respond now:"""


def load_generations(generations_file: str) -> List[Dict]:
    """Load model generations from JSON file.

    Args:
        generations_file: Path to JSON file containing model generations

    Returns:
        List of generation dictionaries

    Raises:
        FileNotFoundError: If the generations file doesn't exist
        ValueError: If the file is not valid JSON or has wrong format
    """
    try:
        with open(generations_file, 'r', encoding='utf-8') as f:
            generations = json.load(f)

        if not isinstance(generations, list):
            raise ValueError(f"Expected list of generations, got {type(generations).__name__}")

        if len(generations) == 0:
            raise ValueError("Generations file is empty")

        return generations
    except FileNotFoundError:
        raise FileNotFoundError(f"Generations file not found: {generations_file}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in generations file: {e}")


def prepare_judge_prompts(generations: List[Dict], use_cot: bool = False) -> List[str]:
    """Prepare prompts for the LLM judge.

    Args:
        generations: List of generation dictionaries with question, reference_answer, predicted_answer
        use_cot: Whether to use Chain-of-Thought prompting for the judge

    Returns:
        List of formatted prompts for the judge
    """
    prompts = []
    template = JUDGE_PROMPT_COT if use_cot else JUDGE_PROMPT

    for gen in generations:
        question = gen.get('question', 'N/A')
        reference_answer = gen.get('reference_answer', 'N/A')
        predicted_answer = gen.get('predicted_answer', 'N/A')

        prompt = template.format(
            question=question,
            reference_answer=reference_answer,
            predicted_answer=predicted_answer
        )
        prompts.append(prompt)

    return prompts


def run_judge_inference(model: LLM, prompts: List[str], sampling_params: SamplingParams,
                       batch_size: int = 32) -> List[str]:
    """Run batch inference for judge evaluations."""
    print(f"Running LLM judge on {len(prompts)} samples with batch size {batch_size}")

    all_responses = []

    # Process in batches
    for i in tqdm(range(0, len(prompts), batch_size), desc="Running judge inference"):
        batch_prompts = prompts[i:i+batch_size]

        # Generate responses for the batch
        outputs = model.generate(batch_prompts, sampling_params)

        # Extract generated text
        batch_responses = []
        for output in outputs:
            generated_text = output.outputs[0].text
            batch_responses.append(generated_text)

        all_responses.extend(batch_responses)

    return all_responses


def parse_judge_response(response: str, verbose: bool = False) -> Dict:
    """Parse the judge's response to extract verdict, confidence, and explanation.

    Args:
        response: Raw judge response
        verbose: Whether to print parsing details

    Returns:
        Dictionary with verdict, confidence, explanation, and raw_response
    """
    import re

    # Default values
    verdict = "UNKNOWN"
    confidence = "UNKNOWN"
    explanation = ""

    # Try to extract VERDICT
    verdict_match = re.search(r'\*\*VERDICT:\*\*\s*(CORRECT|INCORRECT)', response, re.IGNORECASE)
    if verdict_match:
        verdict = verdict_match.group(1).upper()
    else:
        # Fallback: look for verdict without markdown
        verdict_match = re.search(r'VERDICT:\s*(CORRECT|INCORRECT)', response, re.IGNORECASE)
        if verdict_match:
            verdict = verdict_match.group(1).upper()

    # Try to extract CONFIDENCE
    confidence_match = re.search(r'\*\*CONFIDENCE:\*\*\s*(HIGH|MEDIUM|LOW)', response, re.IGNORECASE)
    if confidence_match:
        confidence = confidence_match.group(1).upper()
    else:
        # Fallback: look for confidence without markdown
        confidence_match = re.search(r'CONFIDENCE:\s*(HIGH|MEDIUM|LOW)', response, re.IGNORECASE)
        if confidence_match:
            confidence = confidence_match.group(1).upper()

    # Try to extract EXPLANATION
    explanation_match = re.search(r'\*\*EXPLANATION:\*\*\s*(.+?)(?=\n\*\*|\n\n|$)', response, re.IGNORECASE | re.DOTALL)
    if explanation_match:
        explanation = explanation_match.group(1).strip()
    else:
        # Fallback: look for explanation without markdown
        explanation_match = re.search(r'EXPLANATION:\s*(.+?)(?=\n\*\*|\n\n|$)', response, re.IGNORECASE | re.DOTALL)
        if explanation_match:
            explanation = explanation_match.group(1).strip()

    if verbose:
        print(f"[Judge Parse] Verdict: {verdict}, Confidence: {confidence}")
        print(f"[Judge Parse] Explanation: {explanation[:100]}...")

    return {
        'verdict': verdict,
        'confidence': confidence,
        'explanation': explanation,
        'raw_response': response
    }


def calculate_judge_metrics(judge_results: List[Dict], exact_match_results: List[bool]) -> Dict:
    """Calculate metrics from judge evaluations.

    Args:
        judge_results: List of parsed judge responses
        exact_match_results: List of boolean exact match results for comparison

    Returns:
        Dictionary with metrics
    """
    total = len(judge_results)

    # Count verdicts
    correct_count = sum(1 for r in judge_results if r['verdict'] == 'CORRECT')
    incorrect_count = sum(1 for r in judge_results if r['verdict'] == 'INCORRECT')
    unknown_count = sum(1 for r in judge_results if r['verdict'] == 'UNKNOWN')

    # Calculate accuracy
    judge_accuracy = correct_count / total if total > 0 else 0.0

    # Confidence distribution
    confidence_dist = {
        'HIGH': sum(1 for r in judge_results if r['confidence'] == 'HIGH'),
        'MEDIUM': sum(1 for r in judge_results if r['confidence'] == 'MEDIUM'),
        'LOW': sum(1 for r in judge_results if r['confidence'] == 'LOW'),
        'UNKNOWN': sum(1 for r in judge_results if r['confidence'] == 'UNKNOWN')
    }

    # Compare with exact match
    exact_match_count = sum(exact_match_results)
    exact_match_accuracy = exact_match_count / total if total > 0 else 0.0

    # Agreement between judge and exact match
    agreement_count = sum(
        1 for i, (judge_res, exact_res) in enumerate(zip(judge_results, exact_match_results))
        if (judge_res['verdict'] == 'CORRECT') == exact_res
    )
    agreement_rate = agreement_count / total if total > 0 else 0.0

    # Judge found correct but exact match missed (potential paraphrases)
    judge_correct_exact_wrong = sum(
        1 for judge_res, exact_res in zip(judge_results, exact_match_results)
        if judge_res['verdict'] == 'CORRECT' and not exact_res
    )

    return {
        'total_samples': total,
        'judge_correct': correct_count,
        'judge_incorrect': incorrect_count,
        'judge_unknown': unknown_count,
        'judge_accuracy': judge_accuracy,
        'exact_match_accuracy': exact_match_accuracy,
        'agreement_rate': agreement_rate,
        'judge_correct_exact_wrong': judge_correct_exact_wrong,
        'confidence_distribution': confidence_dist
    }


def save_judge_results(judge_results: List[Dict], metrics: Dict, output_file: str):
    """Save judge evaluation results to JSON file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'total_samples': metrics['total_samples']
        },
        'metrics': metrics,
        'evaluations': judge_results
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nJudge results saved to: {output_file}")


def save_judge_stats_to_csv(metrics: Dict, args: argparse.Namespace, output_dir: str,
                            original_model_name: str):
    """Save judge statistics to CSV file."""
    csv_path = Path(output_dir) / "llm_judge_stats.csv"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row_data = {
        'timestamp': timestamp,
        'original_model': original_model_name,
        'judge_model': args.judge_model,
        'judge_use_cot': args.judge_use_cot,
        'batch_size': args.batch_size,
        'max_tokens': args.max_tokens,
        'temperature': args.temperature,
        'top_k': args.top_k,
        'top_p': args.top_p,
        'min_p': args.min_p,
        'total_samples': metrics['total_samples'],
        'judge_accuracy': metrics['judge_accuracy'],
        'exact_match_accuracy': metrics['exact_match_accuracy'],
        'agreement_rate': metrics['agreement_rate'],
        'judge_correct_exact_wrong': metrics['judge_correct_exact_wrong'],
        'judge_correct': metrics['judge_correct'],
        'judge_incorrect': metrics['judge_incorrect'],
        'judge_unknown': metrics['judge_unknown'],
        'confidence_high': metrics['confidence_distribution']['HIGH'],
        'confidence_medium': metrics['confidence_distribution']['MEDIUM'],
        'confidence_low': metrics['confidence_distribution']['LOW'],
        'confidence_unknown': metrics['confidence_distribution']['UNKNOWN']
    }

    new_row_df = pd.DataFrame([row_data])

    if csv_path.exists():
        new_row_df.to_csv(csv_path, mode='a', header=False, index=False)
        print(f"Appended judge stats to: {csv_path}")
    else:
        new_row_df.to_csv(csv_path, mode='w', header=True, index=False)
        print(f"Created new judge stats CSV: {csv_path}")


def print_judge_metrics(metrics: Dict):
    """Print judge evaluation metrics."""
    print(f"\n{'='*70}")
    print(f"LLM JUDGE EVALUATION RESULTS")
    print(f"{'='*70}")

    total = metrics['total_samples']
    if total == 0:
        print("⚠️  No samples to evaluate!")
        print(f"{'='*70}")
        return

    print(f"Total samples: {total}")
    print(f"\nVERDICTS:")
    print(f"  Correct:   {metrics['judge_correct']:4d} ({metrics['judge_correct']/total*100:.2f}%)")
    print(f"  Incorrect: {metrics['judge_incorrect']:4d} ({metrics['judge_incorrect']/total*100:.2f}%)")
    print(f"  Unknown:   {metrics['judge_unknown']:4d} ({metrics['judge_unknown']/total*100:.2f}%)")

    print(f"\nACCURACY COMPARISON:")
    print(f"  Judge Accuracy:       {metrics['judge_accuracy']:.4f} ({metrics['judge_accuracy']*100:.2f}%)")
    print(f"  Exact Match Accuracy: {metrics['exact_match_accuracy']:.4f} ({metrics['exact_match_accuracy']*100:.2f}%)")
    print(f"  Agreement Rate:       {metrics['agreement_rate']:.4f} ({metrics['agreement_rate']*100:.2f}%)")
    print(f"  Judge Correct but Exact Match Wrong: {metrics['judge_correct_exact_wrong']} (potential paraphrases)")

    print(f"\nCONFIDENCE DISTRIBUTION:")
    conf_dist = metrics['confidence_distribution']
    print(f"  High:    {conf_dist['HIGH']:4d} ({conf_dist['HIGH']/total*100:.2f}%)")
    print(f"  Medium:  {conf_dist['MEDIUM']:4d} ({conf_dist['MEDIUM']/total*100:.2f}%)")
    print(f"  Low:     {conf_dist['LOW']:4d} ({conf_dist['LOW']/total*100:.2f}%)")
    print(f"  Unknown: {conf_dist['UNKNOWN']:4d} ({conf_dist['UNKNOWN']/total*100:.2f}%)")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description='LLM-as-a-Judge for VQA Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input arguments
    parser.add_argument('--generations_file', type=str, required=True,
                       help='Path to model generations JSON file')

    # Judge model arguments
    parser.add_argument('--judge_model', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                       help='Judge model name for vLLM')
    parser.add_argument('--judge_use_cot', action='store_true',
                       help='Enable Chain-of-Thought prompting for judge')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='results/judge_results',
                       help='Directory to save judge results')

    # Inference arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for judge inference')
    parser.add_argument('--max_tokens', type=int, default=200,
                       help='Maximum tokens for judge response')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Temperature for judge sampling (0.0 = greedy, higher = more random)')
    parser.add_argument('--top_k', type=int, default=-1,
                       help='Top-k sampling parameter (-1 = disabled)')
    parser.add_argument('--top_p', type=float, default=1.0,
                       help='Top-p (nucleus) sampling parameter (1.0 = disabled)')
    parser.add_argument('--min_p', type=float, default=0.0,
                       help='Min-p sampling parameter (0.0 = disabled)')

    # Processing arguments
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to evaluate (for testing)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output for debugging')

    args = parser.parse_args()

    print("********** LLM JUDGE CONFIGURATION **********")
    print(f"Generations File: {args.generations_file}")
    print(f"Judge Model: {args.judge_model}")
    print(f"Use Chain-of-Thought: {'Enabled' if args.judge_use_cot else 'Disabled'}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Max Tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-K: {args.top_k}")
    print(f"Top-P: {args.top_p}")
    print(f"Min-P: {args.min_p}")
    print(f"Max Samples: {args.max_samples if args.max_samples else 'No limit'}")
    print("*********************************************\n")

    # Adjust max_tokens if using CoT
    if args.judge_use_cot and args.max_tokens < 300:
        args.max_tokens = 400
        print(f"Increased max_tokens to {args.max_tokens} for Chain-of-Thought judging")

    # Load generations
    print(f"Loading generations from: {args.generations_file}")
    generations = load_generations(args.generations_file)
    print(f"Loaded {len(generations)} generations")

    # Limit samples if specified
    if args.max_samples and args.max_samples < len(generations):
        generations = generations[:args.max_samples]
        print(f"Limited to {len(generations)} samples for evaluation")

    # Extract original model name from filename
    # Expected format: vqa_{model}_{N}samples_generations_{timestamp}.json
    original_model_name = "unknown"
    filename = Path(args.generations_file).stem

    # Try pattern: vqa_{model}_{N}samples_generations_{timestamp}
    import re
    match = re.match(r'vqa_(.+?)_\d+samples_generations', filename)
    if match:
        original_model_name = match.group(1)
    elif '_generations_' in filename:
        # Fallback: split on '_generations_' and take everything before
        original_model_name = filename.split('_generations_')[0]
    else:
        # Last resort: use entire filename stem
        original_model_name = filename

    # Calculate exact match results for comparison
    exact_match_results = []
    for gen in generations:
        ref = gen.get('reference_answer', '').lower().strip()
        pred = gen.get('predicted_answer', '').lower().strip()
        exact_match_results.append(ref == pred)

    # Initialize judge model
    print(f"\nLoading judge model: {args.judge_model}")
    try:
        judge_model = LLM(
            model=args.judge_model,
            max_model_len=8192
        )
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"\n❌ Failed to load judge model: {e}")
        print(f"   Make sure HF_HOME is set correctly and the model exists in the cache.")
        print(f"   Current HF_HOME: {os.getenv('HF_HOME', 'not set')}")
        import sys
        sys.exit(1)

    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p
    )

    # Prepare judge prompts
    print("\nPreparing judge prompts...")
    judge_prompts = prepare_judge_prompts(generations, use_cot=args.judge_use_cot)

    # Print sample prompt
    if args.verbose:
        print(f"\n{'='*70}")
        print("SAMPLE JUDGE PROMPT:")
        print(f"{'='*70}")
        print(judge_prompts[0])
        print(f"{'='*70}\n")

    # Run judge inference
    judge_responses = run_judge_inference(judge_model, judge_prompts, sampling_params, args.batch_size)

    # Validate response count
    if len(judge_responses) != len(judge_prompts):
        print(f"\n❌ ERROR: Judge returned {len(judge_responses)} responses but expected {len(judge_prompts)}")
        print(f"   vLLM inference may have failed mid-batch. Check GPU memory and model configuration.")
        import sys
        sys.exit(1)

    # Parse judge responses
    print("\nParsing judge responses...")
    judge_results = []
    for i, (gen, response) in enumerate(zip(generations, judge_responses)):
        verbose = args.verbose and (i < 3)  # Show details for first 3 if verbose
        parsed = parse_judge_response(response, verbose=verbose)

        # Add original data to result
        result = {
            'question': gen.get('question', 'N/A'),
            'reference_answer': gen.get('reference_answer', 'N/A'),
            'predicted_answer': gen.get('predicted_answer', 'N/A'),
            'image_file': gen.get('image_file', 'N/A'),
            'exact_match': exact_match_results[i],
            **parsed
        }
        judge_results.append(result)

    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_judge_metrics(judge_results, exact_match_results)

    # Print results
    print_judge_metrics(metrics)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    # Create output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    judge_model_clean = args.judge_model.replace('/', '_').replace('\\', '_')
    output_file = os.path.join(
        args.output_dir,
        f"judge_results_{judge_model_clean}_{len(generations)}samples_{timestamp}.json"
    )

    save_judge_results(judge_results, metrics, output_file)

    # Save stats to CSV
    save_judge_stats_to_csv(metrics, args, args.output_dir, original_model_name)

    # Print summary of disagreements (judge correct but exact match wrong)
    if metrics['judge_correct_exact_wrong'] > 0 and args.verbose:
        print(f"\n{'='*70}")
        print(f"EXAMPLES WHERE JUDGE FOUND CORRECT BUT EXACT MATCH FAILED:")
        print(f"(Potential valid paraphrases)")
        print(f"{'='*70}")

        disagreement_examples = [
            r for r in judge_results
            if r['verdict'] == 'CORRECT' and not r['exact_match']
        ]

        for i, ex in enumerate(disagreement_examples[:5], 1):  # Show first 5
            print(f"\n[Example {i}]")
            print(f"Question: {ex['question']}")
            print(f"Reference: {ex['reference_answer']}")
            print(f"Predicted: {ex['predicted_answer']}")
            print(f"Confidence: {ex['confidence']}")
            print(f"Explanation: {ex['explanation']}")

        if len(disagreement_examples) > 5:
            print(f"\n... and {len(disagreement_examples) - 5} more")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
