
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple, Union
from tqdm import tqdm
import re
from datetime import datetime
import os
import glob
from PIL import Image
from io import BytesIO
import requests

from vllm import LLM, SamplingParams
from datasets import load_dataset
from rouge_score import rouge_scorer


# Base path can be configured via environment if needed
BASE_IMAGE_PATH = os.getenv("VQA_IMAGE_PATH", ".")

PROMPT = """
You are a medical expert analyzing medical images. Given a medical image and a question about it, provide a clear and accurate answer.

{few_shot_examples}

**Question:**
{question}

**Task:** Carefully examine the provided medical image and answer the question based on what you observe. Provide a concise, direct answer.

**IMPORTANT FORMATTING REQUIREMENTS:**
- Provide a direct answer to the question
- Be concise and specific
- Do not add unnecessary explanations unless the question requires it
- Base your answer only on what is visible in the image
"""

PROMPT_COT = """
You are a medical expert analyzing medical images. Given a medical image and a question about it, provide a clear and accurate answer using step-by-step reasoning.

{few_shot_examples}

**Question:**
{question}

**Task:** Carefully examine the provided medical image and answer the question step-by-step based on what you observe.

**Instructions:**
1. First, carefully examine the medical image and identify relevant features, anatomical structures, or abnormalities.
2. Consider what aspects of the image are relevant to answering the question.
3. Reason through how the visual findings relate to the question being asked.
4. Formulate your final answer based on your analysis.

**Format your response exactly as follows:**

**STEP 1 - IMAGE ANALYSIS:**
[Describe what you observe in the medical image]

**STEP 2 - RELEVANT FINDINGS:**
[Identify the specific findings relevant to answering the question]

**STEP 3 - REASONING:**
[Explain how the findings lead to your answer]

**STEP 4 - FINAL ANSWER:**
[Provide your concise answer to the question]
"""

FEW_SHOT_TEMPLATE = """
**Example {example_num}:**
Question: {question}
Answer: {answer}
"""

def load_image(source: Union[str, Image.Image, dict]) -> Image.Image:
    """
    Load an image from various sources and convert to RGB.

    Args:
        source: Image source (PIL Image, dict with bytes, URL string, or file path)

    Returns:
        PIL Image in RGB format
    """
    try:
        if isinstance(source, Image.Image):
            return source.convert("RGB")

        if isinstance(source, dict) and "bytes" in source:
            return Image.open(BytesIO(source["bytes"])).convert("RGB")

        if isinstance(source, str):
            if source.lower().startswith(("http://", "https://")):
                response = requests.get(source, timeout=30)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content))
            else:
                img = Image.open(source)
            return img.convert("RGB")

        raise ValueError(f"Unrecognized image format: {type(source)}")

    except Exception as e:
        print(f"Failed to load image from {source}: {str(e)}")
        raise


def validate_dataset_columns(dataset, image_column: str, question_column: str, answer_column: str) -> bool:
    """
    Validate that the dataset has required columns.

    Args:
        dataset: HuggingFace dataset
        image_column: Name of image column
        question_column: Name of question column
        answer_column: Name of answer column

    Returns:
        True if all columns exist

    Raises:
        ValueError: If required columns are missing
    """
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")

    sample = dataset[0]
    missing_columns = []

    if image_column not in sample:
        missing_columns.append(image_column)
    if question_column not in sample:
        missing_columns.append(question_column)
    if answer_column not in sample:
        missing_columns.append(answer_column)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return True


def load_vqa_dataset(dataset_name: str = None, split: str = "test",
                     data_file: str = None, max_samples: int = None) -> pd.DataFrame:
    """
    Load VQA dataset from HuggingFace or local JSON file.

    Args:
        dataset_name: HuggingFace dataset name (e.g., 'flaviagiammarino/vqa-rad')
        split: Dataset split (train/test/validation)
        data_file: Path to local JSON file (alternative to HuggingFace)
        max_samples: Maximum number of samples to load

    Returns:
        DataFrame containing the VQA dataset
    """
    if data_file:
        # Load from local file (JSON or CSV)
        print(f"Loading dataset from local file: {data_file}")
        if data_file.endswith('.csv'):
            df = pd.read_csv(data_file)
        else:
            with open(data_file, 'r', encoding='utf-8') as f:
                vqa_data = json.load(f)
            df = pd.DataFrame(vqa_data)
        print(f"Loaded {len(df)} records from local file")
    elif dataset_name:
        # Load from HuggingFace
        print(f"Loading dataset from HuggingFace: {dataset_name} (split: {split})")
        dataset = load_dataset(dataset_name, split=split)

        # Limit samples if specified
        if max_samples and max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))

        # Convert to DataFrame
        df = pd.DataFrame(dataset)
        print(f"Loaded {len(df)} records from HuggingFace")
    else:
        raise ValueError("Either dataset_name or data_file must be provided")

    return df

def get_image_path(image_file: str) -> str:
    """
    Get the full path to an image file.

    Args:
        image_file: Image filename or path

    Returns:
        Full path to the image file
    """
    # If it's already an absolute path, return it
    if os.path.isabs(image_file):
        return image_file

    # Otherwise, construct relative to BASE_IMAGE_PATH
    full_path = os.path.join(BASE_IMAGE_PATH, image_file)

    if not os.path.exists(full_path):
        print(f"Warning: Image file does not exist: {full_path}")

    return full_path

def format_question_text(question_text: str) -> str:
    """Format question text for better readability.

    Args:
        question_text: The raw question text
    """
    if pd.isna(question_text) or question_text.strip() == '':
        return "No question text available."

    return question_text.strip()

def create_few_shot_examples(df: pd.DataFrame, num_examples: int = 5, seed: int = 42) -> str:
    """Create few-shot examples from the dataset.

    Best practices implemented:
    - Random sampling with fixed seed for reproducibility
    - Diverse examples to cover different question types
    - Clear formatting with numbered examples
    - Separation from the main query

    Args:
        df: DataFrame containing the VQA dataset
        num_examples: Number of few-shot examples to include
        seed: Random seed for reproducibility

    Returns:
        Formatted string containing few-shot examples
    """
    if num_examples == 0 or len(df) == 0:
        return ""

    # Ensure we don't request more examples than available
    num_examples = min(num_examples, len(df))

    # Sample diverse examples
    np.random.seed(seed)
    sampled_indices = np.random.choice(len(df), size=num_examples, replace=False)
    sampled_df = df.iloc[sampled_indices]

    # Build few-shot examples section
    examples_list = []
    examples_list.append("**Here are some examples of how to answer medical image questions:**\n")

    for idx, (_, row) in enumerate(sampled_df.iterrows(), 1):
        # Try to get question and answer from common column names
        question_col = 'question' if 'question' in row else 'Question'
        answer_col = 'answer' if 'answer' in row else ('reference_answer' if 'reference_answer' in row else 'Answer')

        question = format_question_text(row.get(question_col, ''))
        answer = parse_gold_answer(row.get(answer_col, ''))

        example_text = FEW_SHOT_TEMPLATE.format(
            example_num=idx,
            question=question,
            answer=answer
        )
        examples_list.append(example_text)

    examples_list.append("**Now, answer the following question in the same manner:**\n")

    return "\n".join(examples_list)

def prepare_prompts(df: pd.DataFrame, use_few_shot: bool = False,
                    num_few_shot: int = 5, few_shot_seed: int = 42) -> List[str]:
    """Prepare prompts for batch inference.

    Args:
        df: DataFrame containing the VQA dataset
        use_few_shot: Whether to include few-shot examples
        num_few_shot: Number of few-shot examples to include
        few_shot_seed: Random seed for few-shot sampling
    """
    prompts = []

    # Create few-shot examples once if enabled
    few_shot_examples = ""
    if use_few_shot:
        few_shot_examples = create_few_shot_examples(df, num_few_shot, few_shot_seed)

    for _, row in df.iterrows():
        question_text = row['question']
        formatted_question = format_question_text(question_text)

        prompt = PROMPT.format(
            question=formatted_question,
            few_shot_examples=few_shot_examples
        )
        prompts.append(prompt)

    return prompts

def parse_answer(response: str, use_cot: bool = False, verbose: bool = False) -> str:
    """
    Parse answer from model response, robustly handling CoT responses.

    Args:
        response: Raw model response
        use_cot: Whether Chain-of-Thought was used
        verbose: Whether to print parsing details for debugging

    Returns:
        Extracted final answer string
    """
    response = response.strip()
    original_response = response  # Keep for debugging

    if use_cot:
        # For CoT responses, look for the "STEP 4 - FINAL ANSWER:" section
        # Pattern 1: Strict match with markdown formatting
        final_answer_match = re.search(
            r'(?i)\*\*STEP 4 - FINAL ANSWER:\*\*\s*\n?\s*(.*?)(?=\n\s*\*\*STEP \d|\n\s*\*\*|$)',
            response,
            re.DOTALL
        )

        if final_answer_match:
            answer = final_answer_match.group(1).strip()
            if verbose:
                print(f"[CoT Parse] Found STEP 4 with regex (strict)")
        else:
            # Pattern 2: More flexible match without markdown
            flexible_match = re.search(
                r'(?i)STEP 4[:\-\s]*FINAL ANSWER[:\-\s]*\n?\s*(.*?)(?=\n\s*STEP \d|\n\s*\*\*|$)',
                response,
                re.DOTALL
            )

            if flexible_match:
                answer = flexible_match.group(1).strip()
                if verbose:
                    print(f"[CoT Parse] Found STEP 4 with regex (flexible)")
            else:
                # Fallback: look for any line after "STEP 4"
                lines = response.split('\n')
                answer = None

                for i, line in enumerate(lines):
                    if 'STEP 4' in line.upper() and 'FINAL' in line.upper() and 'ANSWER' in line.upper():
                        # Take the next few lines after this header
                        remaining_lines = lines[i+1:]
                        # Skip empty lines
                        non_empty_lines = [l.strip() for l in remaining_lines if l.strip()]
                        answer = '\n'.join(non_empty_lines[:3]).strip() if non_empty_lines else response
                        if verbose:
                            print(f"[CoT Parse] Found STEP 4 with line search")
                        break

                if answer is None:
                    # Last resort: take everything after "STEP 4" if it exists
                    if 'STEP 4' in response.upper():
                        # Find the position case-insensitively
                        match = re.search(r'(?i)STEP 4', response)
                        if match:
                            answer = response[match.end():].strip()
                            if verbose:
                                print(f"[CoT Parse] Using text after STEP 4 (last resort)")
                    else:
                        answer = response
                        if verbose:
                            print(f"[CoT Parse] WARNING: No STEP 4 found, using full response")
    else:
        # For non-CoT, the entire response is the answer
        answer = response

    # Clean up common prefixes and formatting
    answer = re.sub(r'^(Answer:?\s*)', '', answer, flags=re.IGNORECASE).strip()
    answer = re.sub(r'^(Final Answer:?\s*)', '', answer, flags=re.IGNORECASE).strip()
    answer = re.sub(r'^\*\*', '', answer).strip()  # Remove leading markdown
    answer = re.sub(r'\*\*$', '', answer).strip()  # Remove trailing markdown

    # Remove any remaining step headers that might have been captured
    answer = re.sub(r'(?i)^STEP \d+.*?:\s*', '', answer).strip()

    # Clean up extra whitespace
    answer = ' '.join(answer.split())

    return answer


def parse_gold_answer(gold_answer: str) -> str:
    """Parse gold standard answer from the dataset."""
    if pd.isna(gold_answer) or gold_answer.strip() == '':
        return ""

    return gold_answer.strip()

def run_inference(model: LLM, prompts: List[str], sampling_params: SamplingParams, batch_size: int = 32) -> List[str]:
    """Run batch inference using vLLM."""
    print(f"Running inference on {len(prompts)} prompts with batch size {batch_size}")
    
    all_responses = []
    
    # Process in batches
    for i in tqdm(range(0, len(prompts), batch_size), desc="Running inference"):
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

def calculate_metrics(y_true_answers: List[str], y_pred_answers: List[str]) -> Dict:
    """Calculate VQA evaluation metrics including exact match and ROUGE scores."""
    exact_matches = 0

    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for true_answer, pred_answer in zip(y_true_answers, y_pred_answers):
        # Normalize answers for comparison (lowercase, strip whitespace)
        true_normalized = true_answer.lower().strip()
        pred_normalized = pred_answer.lower().strip()

        # Exact match
        if true_normalized == pred_normalized:
            exact_matches += 1

        # Calculate ROUGE scores
        if true_normalized and pred_normalized:  # Only if both are non-empty
            scores = scorer.score(true_normalized, pred_normalized)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)

    # Exact match accuracy
    exact_match_accuracy = exact_matches / len(y_true_answers) if len(y_true_answers) > 0 else 0.0

    # Average ROUGE scores
    avg_rouge1 = np.mean(rouge1_scores) if rouge1_scores else 0.0
    avg_rouge2 = np.mean(rouge2_scores) if rouge2_scores else 0.0
    avg_rougeL = np.mean(rougeL_scores) if rougeL_scores else 0.0

    return {
        'exact_match_accuracy': exact_match_accuracy,
        'total_samples': len(y_true_answers),
        'exact_matches': exact_matches,
        'rouge1': avg_rouge1,
        'rouge2': avg_rouge2,
        'rougeL': avg_rougeL
    }

def save_results(results: Dict, predictions: List, output_dir: str, split_name: str, model_name: str, num_samples: int):
    """Save evaluation results and predictions with model name and sample count in filename."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Clean model name for filename (replace slashes and special characters)
    clean_model_name = model_name.replace('/', '_').replace('\\', '_').replace(':', '_')

    # Create filename with model name and sample count
    base_filename = f"{split_name}_{clean_model_name}_{num_samples}samples"

    # Save metrics
    metrics_file = output_path / f"{base_filename}_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(results, f, indent=2)

    # # Save predictions
    # predictions_file = output_path / f"{base_filename}_predictions.json"
    # with open(predictions_file, 'w') as f:
    #     json.dump(predictions, f, indent=2)

    print(f"Results saved to {output_path}")
    print(f"  Metrics: {metrics_file.name}")
    # print(f"  Predictions: {predictions_file.name}")

def save_generations(predictions: List, output_dir: str, model_name: str, num_samples: int):
    """Save model generations to file with metadata."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Clean model name for filename (replace slashes and special characters)
    clean_model_name = model_name.replace('/', '_').replace('\\', '_').replace(':', '_')

    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    generations_file = output_path / f"vqa_{clean_model_name}_{num_samples}samples_generations_{timestamp}.json"

    with open(generations_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(f"Generations saved to: {generations_file.name}")

def save_model_stats_to_csv(results: Dict, args: argparse.Namespace, split_name: str, num_samples: int, output_dir: str):
    """Save model statistics to a CSV file, creating it if it doesn't exist or appending if it does."""
    csv_path = Path(output_dir) / "model_evaluation_stats.csv"

    # Prepare the row data
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create model name suffix for identification
    model_name_suffix = f"{args.model_name}"
    if args.use_cot:
        model_name_suffix += "_cot"
    if args.use_images:
        model_name_suffix += "_multimodal"
    if args.use_few_shot:
        model_name_suffix += f"_fewshot{args.num_few_shot}"

    row_data = {
        'timestamp': timestamp,
        'model_name': args.model_name,
        'model_config': model_name_suffix,
        'split': split_name,
        'num_samples': num_samples,
        'use_cot': args.use_cot,
        'enable_thinking': args.enable_thinking,
        'use_images': args.use_images,
        'use_few_shot': args.use_few_shot,
        'num_few_shot': args.num_few_shot if args.use_few_shot else 0,
        'few_shot_seed': args.few_shot_seed if args.use_few_shot else None,
        'batch_size': args.batch_size,
        'max_tokens': args.max_tokens,
        'temperature': args.temperature,
        'top_k': args.top_k,
        'top_p': args.top_p,
        'min_p': args.min_p,
        'exact_match_accuracy': results['exact_match_accuracy'],
        'exact_matches': results['exact_matches'],
        'rouge1': results['rouge1'],
        'rouge2': results['rouge2'],
        'rougeL': results['rougeL']
    }

    # Convert to DataFrame
    new_row_df = pd.DataFrame([row_data])

    # Check if CSV exists
    if csv_path.exists():
        # Append to existing CSV
        new_row_df.to_csv(csv_path, mode='a', header=False, index=False)
        print(f"Appended model stats to existing CSV: {csv_path}")
    else:
        # Create new CSV
        new_row_df.to_csv(csv_path, mode='w', header=True, index=False)
        print(f"Created new CSV with model stats: {csv_path}")

def print_sample_prompts(prompts: List[str], num_samples: int = 3):
    """Print sample prompts to inspect their quality."""
    print(f"\n{'='*80}")
    print("SAMPLE PROMPTS FOR QUALITY INSPECTION")
    print(f"{'='*80}")
    
    for i in range(min(num_samples, len(prompts))):
        print(f"\n--- PROMPT {i+1} ---")
        print(prompts[i])
        print(f"--- END PROMPT {i+1} ---")
        if i < min(num_samples, len(prompts)) - 1:
            print("\n" + "-"*40)
    
    print(f"\n{'='*80}")

def print_sample_conversations(conversations: List[List[Dict]], num_samples: int = 3):
    """Print sample conversations to inspect their quality."""
    print(f"\n{'='*80}")
    print("SAMPLE CONVERSATIONS FOR QUALITY INSPECTION")
    print(f"{'='*80}")
    
    for i in range(min(num_samples, len(conversations))):
        print(f"\n--- CONVERSATION {i+1} ---")
        conversation = conversations[i]
        
        for j, message in enumerate(conversation):
            role = message["role"].upper()
            content = message["content"]
            
            print(f"\n{role} MESSAGE:")
            # Handle both string content and list content (for multimodal)
            if isinstance(content, str):
                # Truncate very long content for readability
                if len(content) > 500:
                    print(content[:500] + "... [TRUNCATED]")
                else:
                    print(content)
            elif isinstance(content, list):
                # Handle multimodal content
                for item in content:
                    if item["type"] == "text":
                        text_content = item["text"]
                        if len(text_content) > 500:
                            print(f"TEXT: {text_content[:500]}... [TRUNCATED]")
                        else:
                            print(f"TEXT: {text_content}")
                    elif item["type"] == "image":
                        print(f"IMAGE: {item['url']}")
        
        print(f"--- END CONVERSATION {i+1} ---")
        if i < min(num_samples, len(conversations)) - 1:
            print("\n" + "-"*40)
    
    print(f"\n{'='*80}")

def print_metrics(results: Dict, split_name: str):
    """Print evaluation metrics in a formatted way."""
    print(f"\n{'='*60}")
    print(f"VQA EVALUATION RESULTS - {split_name.upper()}")
    print(f"{'='*60}")
    print(f"Total samples: {results['total_samples']}")
    print(f"Exact matches: {results['exact_matches']}")
    print(f"\nEXACT MATCH ACCURACY:")
    print(f"  Accuracy:  {results['exact_match_accuracy']:.4f} ({results['exact_match_accuracy']*100:.2f}%)")
    print(f"\nROUGE SCORES:")
    print(f"  ROUGE-1:   {results['rouge1']:.4f}")
    print(f"  ROUGE-2:   {results['rouge2']:.4f}")
    print(f"  ROUGE-L:   {results['rougeL']:.4f}")
    print(f"{'='*60}")

def prepare_chat_conversations(df: pd.DataFrame, use_cot: bool = False, use_images: bool = False,
                              use_few_shot: bool = False, num_few_shot: int = 5,
                              few_shot_seed: int = 42, test_df: pd.DataFrame = None,
                              image_column: str = 'image', question_column: str = 'question',
                              images_dir: str = None) -> List[List[Dict]]:
    """Prepare chat conversations for batch inference using chat templates.

    Args:
        df: DataFrame containing the samples to process
        use_cot: Whether to use Chain-of-Thought prompting
        use_images: Whether to include images in the conversation
        use_few_shot: Whether to include few-shot examples
        num_few_shot: Number of few-shot examples to include
        few_shot_seed: Random seed for few-shot sampling
        test_df: Separate DataFrame to sample few-shot examples from (to avoid data leakage)
                 If None, samples from df itself
        image_column: Name of the image column in the dataset
        question_column: Name of the question column in the dataset
        images_dir: Directory to save images (if processing from HuggingFace dataset)
    """
    conversations = []

    # Choose the appropriate prompt template
    prompt_template = PROMPT_COT if use_cot else PROMPT

    # Create few-shot examples once if enabled
    # Best practice: Sample from a separate set to avoid data leakage
    few_shot_examples = ""
    if use_few_shot:
        source_df = test_df if test_df is not None else df
        few_shot_examples = create_few_shot_examples(source_df, num_few_shot, few_shot_seed)

    # Create images directory if needed
    if images_dir:
        os.makedirs(images_dir, exist_ok=True)

    for idx, row in df.iterrows():
        question_text = row[question_column]
        image_source = row[image_column]

        # Format the question
        formatted_question = format_question_text(question_text)

        # Use the selected prompt template and format it with the question and few-shot examples
        user_message_text = prompt_template.format(
            question=formatted_question,
            few_shot_examples=few_shot_examples
        )

        if use_images:
            # Handle image loading and saving
            if images_dir:
                # Load image from source (HuggingFace dataset, URL, or file)
                try:
                    image = load_image(image_source)
                    # Save to local directory
                    image_filename = f"{idx:06d}.png"
                    image_path = os.path.join(images_dir, image_filename)
                    image.save(image_path, "PNG")

                    # Convert to absolute path for vLLM
                    image_path = os.path.abspath(image_path)
                except Exception as e:
                    print(f"Warning: Failed to load/save image for sample {idx}: {e}")
                    # Skip this sample or use text-only
                    image_path = None
            else:
                # Use image path directly (assume it's a local file path)
                if isinstance(image_source, str):
                    image_path = os.path.abspath(image_source)
                else:
                    print(f"Warning: Image source is not a string for sample {idx}, skipping image")
                    image_path = None

            if image_path:
                # Create multimodal content using vLLM's expected format
                # vLLM expects "image_url" instead of "image"
                content = []

                # Add the image using the correct vLLM format
                # Must use file:// prefix with absolute path
                file_url = f"file://{image_path}"
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": file_url
                    }
                })

                # Add the text prompt
                content.append({
                    "type": "text",
                    "text": user_message_text
                })

                conversation = [
                    {
                        "role": "user",
                        "content": content
                    }
                ]
            else:
                # Fallback to text-only if image loading failed
                conversation = [
                    {
                        "role": "user",
                        "content": user_message_text
                    }
                ]
        else:
            # Text-only conversation (just the question)
            conversation = [
                {
                    "role": "user",
                    "content": user_message_text
                }
            ]

        conversations.append(conversation)

    return conversations

def run_chat_inference(model: LLM, conversations: List[List[Dict]], sampling_params: SamplingParams, batch_size: int = 32, enable_thinking : bool = False) -> List[str]:
    """Run batch inference using chat templates with vLLM."""
    print(f"Running chat inference on {len(conversations)} conversations with batch size {batch_size}")
    
    all_responses = []
    
    # Process in batches
    for i in tqdm(range(0, len(conversations), batch_size), desc="Running chat inference"):
        batch_conversations = conversations[i:i+batch_size]
        
        # Generate responses for the batch using chat method
        outputs = model.chat(
            batch_conversations,
            sampling_params=sampling_params,
            use_tqdm=False,
            chat_template_kwargs={"enable_thinking":enable_thinking}
        )
        
        # Extract generated text
        batch_responses = []
        for output in outputs:
            generated_text = output.outputs[0].text
            batch_responses.append(generated_text)
        
        all_responses.extend(batch_responses)
    
    return all_responses

def main():
    parser = argparse.ArgumentParser(
        description='Visual Question Answering (VQA) for Medical Images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data source arguments (HuggingFace or local file)
    parser.add_argument('--dataset_name', type=str, default=None,
                       help='HuggingFace dataset name (e.g., flaviagiammarino/vqa-rad)')
    parser.add_argument('--split', type=str, default='test',
                       help='Dataset split (train/test/validation)')
    parser.add_argument('--data_file', type=str, default=None,
                       help='Path to local VQA dataset JSON file (alternative to --dataset_name)')

    # Dataset column names
    parser.add_argument('--image_column', type=str, default='image',
                       help='Name of the image column in the dataset')
    parser.add_argument('--question_column', type=str, default='question',
                       help='Name of the question column in the dataset')
    parser.add_argument('--answer_column', type=str, default='answer',
                       help='Name of the answer column in the dataset')

    # Model arguments
    parser.add_argument('--model_name', type=str, required=True,
                       help='Model name for vLLM (e.g., Qwen/Qwen2-VL-7B-Instruct)')
    parser.add_argument('--trust_remote_code', action='store_true',
                       help='Allow execution of remote code (required for some models like MedGemma)')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='results/vqa_results',
                       help='Directory to save results')
    parser.add_argument('--images_dir', type=str, default='images',
                       help='Directory to save images when loading from HuggingFace')

    # Inference arguments
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for inference')
    parser.add_argument('--max_tokens', type=int, default=100,
                       help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Temperature for sampling (0.0 = greedy, higher = more random)')
    parser.add_argument('--top_k', type=int, default=-1,
                       help='Top-k sampling parameter (-1 = disabled)')
    parser.add_argument('--top_p', type=float, default=1.0,
                       help='Top-p (nucleus) sampling parameter (1.0 = disabled)')
    parser.add_argument('--min_p', type=float, default=0.0,
                       help='Min-p sampling parameter (0.0 = disabled)')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.6,
                       help='Fraction of GPU memory for vLLM (0.0-1.0, default: 0.6)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to evaluate (for testing)')

    # Prompting strategy arguments
    parser.add_argument('--use_cot', action='store_true',
                       help='Enable Chain-of-Thought prompting for more detailed reasoning')
    parser.add_argument('--enable_thinking', action='store_true',
                       help='Enable thinking mode for the model to reason through steps')
    parser.add_argument('--use_images', action='store_true', default=True,
                       help='Enable multimodal input with medical images (default: True)')

    # Few-shot learning arguments
    parser.add_argument('--use_few_shot', action='store_true',
                       help='Enable few-shot prompting with example question-answer pairs')
    parser.add_argument('--num_few_shot', type=int, default=5,
                       help='Number of few-shot examples to include in the prompt (default: 5)')
    parser.add_argument('--few_shot_seed', type=int, default=42,
                       help='Random seed for few-shot example sampling (default: 42)')
    parser.add_argument('--few_shot_source', type=str, default=None,
                       help='Optional separate data source for few-shot examples (dataset name or JSON file)')

    # Output control arguments
    parser.add_argument('--save_generations', action='store_true',
                       help='Save model generations to file (default: False)')

    args = parser.parse_args()

    # Validate that either dataset_name or data_file is provided
    if not args.dataset_name and not args.data_file:
        parser.error("Either --dataset_name or --data_file must be provided")

    print("********** VQA CONFIGURATION **********")
    print(f"Model Name: {args.model_name}")
    if args.dataset_name:
        print(f"Data Source: HuggingFace - {args.dataset_name} (split: {args.split})")
    else:
        print(f"Data Source: Local File - {args.data_file}")
    print(f"Dataset Columns: image={args.image_column}, question={args.question_column}, answer={args.answer_column}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Images Directory: {args.images_dir}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Max Tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-K: {args.top_k}")
    print(f"Top-P: {args.top_p}")
    print(f"Min-P: {args.min_p}")
    print(f"GPU Memory Utilization: {args.gpu_memory_utilization}")
    print(f"Max Samples: {args.max_samples if args.max_samples else 'No limit'}")
    print(f"Use Chain-of-Thought: {'Enabled' if args.use_cot else 'Disabled'}")
    print(f"Enable Thinking Mode: {'Enabled' if args.enable_thinking else 'Disabled'}")
    print(f"Use Images (Multimodal): {'Enabled' if args.use_images else 'Disabled'}")
    print(f"Use Few-Shot Prompting: {'Enabled' if args.use_few_shot else 'Disabled'}")
    if args.use_few_shot:
        print(f"  Number of Examples: {args.num_few_shot}")
        print(f"  Random Seed: {args.few_shot_seed}")
        print(f"  Example Source: {args.few_shot_source if args.few_shot_source else 'Same as test data'}")
    print("***************************************\n")

    # Adjust max_tokens if using CoT (needs more tokens for reasoning)
    if args.use_cot and args.max_tokens < 200:
        args.max_tokens = 300
        print(f"Increased max_tokens to {args.max_tokens} for Chain-of-Thought prompting")

    # Initialize vLLM model
    print(f"Loading model: {args.model_name}")

    # Build model initialization parameters
    model_init_kwargs = {
        "model": args.model_name,
        "max_model_len": 8192,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": args.trust_remote_code,
    }

    # Set up allowed local media paths for image loading
    if args.use_images:
        # Restrict vLLM file access to the base image path (dataset root or working dir)
        # This is safer than allowing "/" while still covering all image locations
        allowed_media_path = os.path.abspath(BASE_IMAGE_PATH)

        # Add allowed paths to model kwargs
        # vLLM only accepts a single path, not multiple paths
        model_init_kwargs["allowed_local_media_path"] = allowed_media_path
        model_init_kwargs["limit_mm_per_prompt"] = {"image": 10}

        print(f"Allowed local media path: {allowed_media_path}")

    model = None
    try:
        model = LLM(**model_init_kwargs)

        # Set up sampling parameters
        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_k=args.top_k,
            top_p=args.top_p,
            min_p=args.min_p,
            stop=["\n\n"] if not args.use_cot else ["**STEP 5:"]
        )

        # Load VQA dataset
        df = load_vqa_dataset(
            dataset_name=args.dataset_name,
            split=args.split,
            data_file=args.data_file,
            max_samples=args.max_samples
        )

        # Validate dataset has required columns
        if args.dataset_name:
            # For HuggingFace datasets, validate column names
            sample_keys = df.columns.tolist()
            print(f"Dataset columns: {sample_keys}")

            missing_cols = []
            if args.image_column not in sample_keys:
                missing_cols.append(args.image_column)
            if args.question_column not in sample_keys:
                missing_cols.append(args.question_column)
            if args.answer_column not in sample_keys:
                missing_cols.append(args.answer_column)

            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            print("Dataset validation passed.")

        # Load few-shot source data if specified (best practice: avoid data leakage)
        few_shot_df = None
        if args.use_few_shot and args.few_shot_source:
            print(f"\nLoading few-shot examples from: {args.few_shot_source}")
            # Check if it's a HuggingFace dataset or local file
            if args.few_shot_source.endswith('.json'):
                few_shot_df = load_vqa_dataset(data_file=args.few_shot_source)
            else:
                few_shot_df = load_vqa_dataset(dataset_name=args.few_shot_source, split=args.split)
            print(f"Loaded {len(few_shot_df)} samples for few-shot examples")

        # Store actual number of samples used
        num_samples_used = len(df)

        # Determine if we need to save images (HuggingFace datasets require this)
        images_save_dir = args.images_dir if args.dataset_name and args.use_images else None

        # Prepare chat conversations
        conversations = prepare_chat_conversations(
            df,
            use_cot=args.use_cot,
            use_images=args.use_images,
            use_few_shot=args.use_few_shot,
            num_few_shot=args.num_few_shot,
            few_shot_seed=args.few_shot_seed,
            test_df=few_shot_df,
            image_column=args.image_column,
            question_column=args.question_column,
            images_dir=images_save_dir
        )

        # Print sample conversations for quality inspection
        print_sample_conversations(conversations, num_samples=2 if args.use_cot else 3)

        # Run chat inference
        responses = run_chat_inference(model, conversations, sampling_params, args.batch_size, args.enable_thinking)

        # Parse predictions and gold standard answers
        # Enable verbose parsing for first few samples to verify CoT parsing
        predicted_answers = []
        for i, response in enumerate(responses):
            verbose = (i < 3)  # Show parsing details for first 3 samples
            parsed = parse_answer(response, use_cot=args.use_cot, verbose=verbose)
            predicted_answers.append(parsed)

            if verbose and args.use_cot:
                print(f"\n[Sample {i}] CoT Response Parsing:")
                print(f"  Raw response length: {len(response)} chars")
                print(f"  Parsed answer: '{parsed}'")

        gold_answers = [parse_gold_answer(row[args.answer_column]) for _, row in df.iterrows()]

        # Verify parsing quality for CoT
        if args.use_cot:
            print(f"\n{'='*60}")
            print("CoT PARSING QUALITY CHECK")
            print(f"{'='*60}")
            avg_answer_length = np.mean([len(ans) for ans in predicted_answers])
            avg_response_length = np.mean([len(resp) for resp in responses])
            print(f"Average raw response length: {avg_response_length:.1f} chars")
            print(f"Average parsed answer length: {avg_answer_length:.1f} chars")
            print(f"Compression ratio: {(avg_answer_length/avg_response_length)*100:.1f}%")
            print(f"{'='*60}\n")

        # Calculate metrics
        results = calculate_metrics(gold_answers, predicted_answers)

        # Print results
        print_metrics(results, "VQA")

        # Prepare detailed predictions for saving
        detailed_predictions = []
        for i, (_, row) in enumerate(df.iterrows()):
            # Get image reference (local file or saved file)
            if images_save_dir:
                image_ref = os.path.join(images_save_dir, f"{i:06d}.png")
            else:
                image_ref = row[args.image_column] if isinstance(row[args.image_column], str) else f"sample_{i}"

            # Calculate individual metrics for this sample
            correct = gold_answers[i].lower().strip() == predicted_answers[i].lower().strip()

            detailed_predictions.append({
                'image_file': image_ref,
                'question': row[args.question_column],
                'reference_answer': gold_answers[i],
                'predicted_answer': predicted_answers[i],
                'raw_response': responses[i] if not args.use_cot else responses[i][:200] + "..." if len(responses[i]) > 200 else responses[i],  # Truncate CoT for space
                'full_response': responses[i],  # Keep full response for CoT analysis
                'correct': correct,
                'use_cot': args.use_cot,
                'use_images': args.use_images,
                'use_few_shot': args.use_few_shot,
                'num_few_shot': args.num_few_shot if args.use_few_shot else 0,
                'answer_length': len(predicted_answers[i]),
                'response_length': len(responses[i])
            })

        # Create model name suffix
        model_name_suffix = f"{args.model_name}"
        if args.use_cot:
            model_name_suffix += "_cot"
        if args.use_images:
            model_name_suffix += "_multimodal"
        if args.use_few_shot:
            model_name_suffix += f"_fewshot{args.num_few_shot}"

        # Save results with model name and sample count
        save_results(results, detailed_predictions, args.output_dir, "vqa", model_name_suffix, num_samples_used)

        # Save model statistics to CSV
        save_model_stats_to_csv(results, args, "vqa", num_samples_used, args.output_dir)

        # Save generations if requested
        if args.save_generations:
            save_generations(detailed_predictions, args.output_dir, model_name_suffix, num_samples_used)

    finally:
        # Graceful GPU cleanup to prevent zombie processes on the cluster
        import gc
        del model
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

if __name__ == "__main__":
    main()