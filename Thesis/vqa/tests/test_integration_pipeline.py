#!/usr/bin/env python3
"""
Integration Test: End-to-End VQA Pipeline (Generation → Judge)

This test validates the complete pipeline:
1. Generate synthetic VQA data
2. Run judge evaluation
3. Validate outputs and metrics

Note: This is a LIGHTWEIGHT test that doesn't load real models.
For full testing with real models, use the cluster deployment scripts.
"""

import sys
import os
import json
import tempfile
from pathlib import Path
from datetime import datetime

# Mock heavy dependencies
from unittest.mock import MagicMock, Mock, patch
sys.modules['dotenv'] = MagicMock()
sys.modules['vllm'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['numpy'] = MagicMock()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from llm_judge import (
    load_generations,
    prepare_judge_prompts,
    run_judge_inference,
    parse_judge_response,
    calculate_judge_metrics,
    save_judge_results,
    save_judge_stats_to_csv,
    print_judge_metrics
)


class IntegrationTestPipeline:
    """Simulates the full VQA pipeline without loading real models."""

    def __init__(self, test_dir: str):
        self.test_dir = Path(test_dir)
        self.test_dir.mkdir(parents=True, exist_ok=True)

        self.generation_dir = self.test_dir / "vqa_results"
        self.judge_dir = self.test_dir / "judge_results"

        self.generation_dir.mkdir(exist_ok=True)
        self.judge_dir.mkdir(exist_ok=True)

    def create_synthetic_vqa_data(self, num_samples: int = 20) -> str:
        """
        Simulate Phase 1: VQA Generation

        Creates synthetic generation data that mimics real VQA output.
        In production, this would come from generate_vqa.py with a real VLM.
        """
        print(f"\n{'='*70}")
        print(f"PHASE 1: VQA GENERATION (Synthetic)")
        print(f"{'='*70}")
        print(f"Generating {num_samples} synthetic VQA samples...")

        # Simulate realistic VQA questions and answers
        question_templates = [
            ("Is there evidence of {condition}?", "yes", "Yes, there is evidence of {condition}."),
            ("Is there evidence of {condition}?", "no", "No, there is no evidence of {condition}."),
            ("What type of view is shown?", "frontal", "This is a frontal view."),
            ("What type of view is shown?", "lateral", "lateral"),
            ("Is a {device} present?", "yes", "Yes, a {device} is visible."),
            ("Is a {device} present?", "no", "No {device} is present."),
            ("What organ is visible?", "lung", "The lungs are visible."),
            ("What organ is visible?", "heart", "heart"),
        ]

        conditions = ["pneumonia", "fracture", "effusion", "consolidation"]
        devices = ["pacemaker", "catheter", "tube", "implant"]

        generations = []
        for i in range(num_samples):
            # Pick a random question template
            template_idx = i % len(question_templates)
            question_template, reference, predicted_template = question_templates[template_idx]

            # Fill in the template
            if "{condition}" in question_template:
                condition = conditions[i % len(conditions)]
                question = question_template.format(condition=condition)
                predicted = predicted_template.format(condition=condition) if "{condition}" in predicted_template else predicted_template
            elif "{device}" in question_template:
                device = devices[i % len(devices)]
                question = question_template.format(device=device)
                predicted = predicted_template.format(device=device) if "{device}" in predicted_template else predicted_template
            else:
                question = question_template
                predicted = predicted_template

            # Introduce some variation (some exact matches, some paraphrases, some errors)
            if i % 5 == 0:
                # Exact match
                predicted = reference
            elif i % 7 == 0:
                # Wrong answer
                predicted = "incorrect answer" if reference == "yes" else "yes"

            generations.append({
                "image_file": f"synthetic_image_{i:04d}.jpg",
                "question": question,
                "reference_answer": reference,
                "predicted_answer": predicted,
                "raw_response": predicted  # In real pipeline, this would be longer
            })

        # Save in the format expected by the judge
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        generations_file = self.generation_dir / f"vqa_test-model_mock_{num_samples}samples_generations_{timestamp}.json"

        with open(generations_file, 'w', encoding='utf-8') as f:
            json.dump(generations, f, indent=2, ensure_ascii=False)

        print(f"✅ Generated {len(generations)} VQA samples")
        print(f"✅ Saved to: {generations_file.name}")

        return str(generations_file)

    def simulate_judge_responses(self, prompts: list) -> list:
        """
        Simulate judge model responses without loading a real LLM.

        In production, this would be handled by run_judge_inference() with vLLM.
        """
        print(f"\n{'='*70}")
        print(f"SIMULATING JUDGE INFERENCE")
        print(f"{'='*70}")
        print(f"Processing {len(prompts)} judge prompts...")

        responses = []
        for i, prompt in enumerate(prompts):
            # Extract reference and predicted from the prompt
            import re

            ref_match = re.search(r'\*\*Reference Answer.*?:\*\*\s*(.+?)\n', prompt)
            pred_match = re.search(r'\*\*Predicted Answer.*?:\*\*\s*(.+?)\n', prompt)

            if ref_match and pred_match:
                ref = ref_match.group(1).strip().lower()
                pred = pred_match.group(1).strip().lower()

                # Simulate judge decision
                if ref in pred or pred in ref:
                    verdict = "CORRECT"
                    confidence = "HIGH"
                    explanation = f"The predicted answer '{pred}' is semantically equivalent to '{ref}'."
                else:
                    verdict = "INCORRECT"
                    confidence = "HIGH"
                    explanation = f"The predicted answer '{pred}' does not match the reference '{ref}'."
            else:
                verdict = "UNKNOWN"
                confidence = "LOW"
                explanation = "Could not parse the question properly."

            # Format response like a real judge model would
            response = f"""**VERDICT:** {verdict}
**CONFIDENCE:** {confidence}
**EXPLANATION:** {explanation}"""

            responses.append(response)

        print(f"✅ Generated {len(responses)} judge responses")
        return responses

    def run_integration_test(self, num_samples: int = 20):
        """
        Execute the full integration test:
        1. Generate synthetic VQA data
        2. Run judge evaluation
        3. Validate outputs
        """
        print(f"\n{'#'*70}")
        print(f"# INTEGRATION TEST: VQA PIPELINE (Generation → Judge)")
        print(f"#"*70)

        # PHASE 1: Generate VQA data
        generations_file = self.create_synthetic_vqa_data(num_samples)

        # PHASE 2: Load generations (validates Phase 1 output format)
        print(f"\n{'='*70}")
        print(f"PHASE 2: JUDGE EVALUATION")
        print(f"{'='*70}")

        try:
            generations = load_generations(generations_file)
            print(f"✅ Loaded {len(generations)} generations from Phase 1")
        except Exception as e:
            print(f"❌ Failed to load generations: {e}")
            return False

        # Validate generation format
        required_fields = ['image_file', 'question', 'reference_answer', 'predicted_answer']
        for i, gen in enumerate(generations[:3]):  # Check first 3
            for field in required_fields:
                if field not in gen:
                    print(f"❌ Generation {i} missing required field: {field}")
                    return False

        print(f"✅ Generation format validated")

        # Prepare judge prompts
        print(f"\nPreparing judge prompts...")
        judge_prompts = prepare_judge_prompts(generations, use_cot=False)
        print(f"✅ Prepared {len(judge_prompts)} judge prompts")

        # Simulate judge inference (in production, this would use vLLM)
        judge_responses = self.simulate_judge_responses(judge_prompts)

        # Validate response count
        if len(judge_responses) != len(judge_prompts):
            print(f"❌ Response count mismatch: {len(judge_responses)} != {len(judge_prompts)}")
            return False

        print(f"✅ Response count validated")

        # Parse judge responses
        print(f"\nParsing judge responses...")
        judge_results = []
        exact_match_results = []

        for i, (gen, response) in enumerate(zip(generations, judge_responses)):
            parsed = parse_judge_response(response, verbose=False)

            # Calculate exact match
            ref = gen.get('reference_answer', '').lower().strip()
            pred = gen.get('predicted_answer', '').lower().strip()
            exact_match = (ref == pred)

            result = {
                'question': gen.get('question', 'N/A'),
                'reference_answer': gen.get('reference_answer', 'N/A'),
                'predicted_answer': gen.get('predicted_answer', 'N/A'),
                'image_file': gen.get('image_file', 'N/A'),
                'exact_match': exact_match,
                **parsed
            }
            judge_results.append(result)
            exact_match_results.append(exact_match)

        print(f"✅ Parsed {len(judge_results)} judge responses")

        # Calculate metrics
        print(f"\nCalculating metrics...")
        metrics = calculate_judge_metrics(judge_results, exact_match_results)

        # Print results
        print_judge_metrics(metrics)

        # Save results
        print(f"\n{'='*70}")
        print(f"SAVING RESULTS")
        print(f"{'='*70}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.judge_dir / f"integration_test_results_{timestamp}.json"

        save_judge_results(judge_results, metrics, str(output_file))

        # Validate saved file
        if not output_file.exists():
            print(f"❌ Results file not created")
            return False

        with open(output_file, 'r') as f:
            saved_data = json.load(f)

        required_keys = ['metadata', 'metrics', 'evaluations']
        for key in required_keys:
            if key not in saved_data:
                print(f"❌ Saved results missing key: {key}")
                return False

        print(f"✅ Results file validated")

        # Summary
        print(f"\n{'='*70}")
        print(f"INTEGRATION TEST SUMMARY")
        print(f"{'='*70}")
        print(f"✅ Phase 1 (Generation): Simulated {num_samples} samples")
        print(f"✅ Phase 2 (Judge): Evaluated {len(judge_results)} samples")
        print(f"✅ Metrics calculated: {metrics['total_samples']} samples")
        print(f"✅ Results saved: {output_file.name}")
        print(f"\nKey Metrics:")
        print(f"  - Judge Accuracy: {metrics['judge_accuracy']:.2%}")
        print(f"  - Exact Match Accuracy: {metrics['exact_match_accuracy']:.2%}")
        print(f"  - Agreement Rate: {metrics['agreement_rate']:.2%}")
        print(f"{'='*70}")

        return True


def main():
    """Run the integration test."""
    import argparse

    parser = argparse.ArgumentParser(description='Integration test for VQA pipeline')
    parser.add_argument('--num_samples', type=int, default=20,
                       help='Number of synthetic samples to generate')
    parser.add_argument('--test_dir', type=str, default='/tmp/vqa_integration_test',
                       help='Directory for test outputs')

    args = parser.parse_args()

    # Create test pipeline
    pipeline = IntegrationTestPipeline(args.test_dir)

    # Run integration test
    success = pipeline.run_integration_test(args.num_samples)

    if success:
        print(f"\n🎉 INTEGRATION TEST PASSED!")
        print(f"\nTest artifacts saved to: {args.test_dir}")
        return 0
    else:
        print(f"\n❌ INTEGRATION TEST FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
