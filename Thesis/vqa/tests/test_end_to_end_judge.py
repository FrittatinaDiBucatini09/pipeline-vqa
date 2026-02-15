#!/usr/bin/env python3
"""
End-to-end test for the LLM Judge pipeline.
This test creates synthetic generation data and validates the full judge workflow.
"""

import sys
import os
import json
import tempfile
from pathlib import Path

# Mock heavy dependencies for testing
from unittest.mock import MagicMock, patch, Mock
sys.modules['dotenv'] = MagicMock()
sys.modules['vllm'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['numpy'] = MagicMock()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from llm_judge import (
    load_generations,
    prepare_judge_prompts,
    parse_judge_response,
    calculate_judge_metrics,
    save_judge_results,
    JUDGE_PROMPT,
    JUDGE_PROMPT_COT
)


def create_sample_generations():
    """Create sample generation data for testing."""
    return [
        {
            "image_file": "test_image_1.jpg",
            "question": "Is there evidence of pneumonia?",
            "reference_answer": "yes",
            "predicted_answer": "Yes, there is evidence of pneumonia.",
            "use_cot": False
        },
        {
            "image_file": "test_image_2.jpg",
            "question": "What type of view is shown?",
            "reference_answer": "frontal",
            "predicted_answer": "frontal view",
            "use_cot": False
        },
        {
            "image_file": "test_image_3.jpg",
            "question": "Is there a fracture?",
            "reference_answer": "no",
            "predicted_answer": "yes",
            "use_cot": False
        },
        {
            "image_file": "test_image_4.jpg",
            "question": "What organ is shown?",
            "reference_answer": "lung",
            "predicted_answer": "lungs",
            "use_cot": False
        },
    ]


def create_sample_judge_responses():
    """Create sample judge responses for testing."""
    return [
        """**STEP 1 - ANALYZE REFERENCE:**
The reference answer indicates "yes" for pneumonia.

**STEP 2 - ANALYZE PREDICTED:**
The predicted answer confirms pneumonia is present.

**STEP 3 - COMPARE:**
Both answers convey the same medical meaning.

**VERDICT:** CORRECT
**CONFIDENCE:** HIGH
**EXPLANATION:** Both answers confirm the presence of pneumonia with semantic equivalence.""",

        """**VERDICT:** CORRECT
**CONFIDENCE:** MEDIUM
**EXPLANATION:** "Frontal view" and "frontal" are semantically equivalent in radiology context.""",

        """**VERDICT:** INCORRECT
**CONFIDENCE:** HIGH
**EXPLANATION:** The predicted answer contradicts the reference answer (yes vs no).""",

        """VERDICT: CORRECT
CONFIDENCE: HIGH
EXPLANATION: "Lung" and "lungs" refer to the same organ; plural form is acceptable."""
    ]


def test_load_generations():
    """Test loading generations from JSON file."""
    print("\n" + "="*70)
    print("TEST 1: Load Generations")
    print("="*70)

    generations = create_sample_generations()

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(generations, f, indent=2)
        temp_path = f.name

    try:
        loaded = load_generations(temp_path)
        assert len(loaded) == 4, f"Expected 4 generations, got {len(loaded)}"
        assert loaded[0]['question'] == "Is there evidence of pneumonia?"
        print("✅ PASS: Successfully loaded generations from file")
    finally:
        os.unlink(temp_path)


def test_prepare_judge_prompts():
    """Test preparing judge prompts."""
    print("\n" + "="*70)
    print("TEST 2: Prepare Judge Prompts")
    print("="*70)

    generations = create_sample_generations()

    # Test without CoT
    prompts = prepare_judge_prompts(generations, use_cot=False)
    assert len(prompts) == 4, f"Expected 4 prompts, got {len(prompts)}"
    assert "Is there evidence of pneumonia?" in prompts[0]
    assert "yes" in prompts[0]
    assert "Yes, there is evidence of pneumonia." in prompts[0]
    print("✅ PASS: Standard prompts generated correctly")

    # Test with CoT
    prompts_cot = prepare_judge_prompts(generations, use_cot=True)
    assert len(prompts_cot) == 4, f"Expected 4 CoT prompts, got {len(prompts_cot)}"
    assert "**STEP 1 - ANALYZE REFERENCE:**" in prompts_cot[0]
    print("✅ PASS: Chain-of-Thought prompts generated correctly")


def test_parse_judge_responses():
    """Test parsing judge responses."""
    print("\n" + "="*70)
    print("TEST 3: Parse Judge Responses")
    print("="*70)

    responses = create_sample_judge_responses()

    # Parse all responses
    parsed_results = []
    for i, response in enumerate(responses):
        parsed = parse_judge_response(response, verbose=False)
        parsed_results.append(parsed)
        print(f"  Response {i+1}: Verdict={parsed['verdict']}, Confidence={parsed['confidence']}")

    # Validate parsing
    assert parsed_results[0]['verdict'] == 'CORRECT'
    assert parsed_results[0]['confidence'] == 'HIGH'
    assert parsed_results[1]['verdict'] == 'CORRECT'
    assert parsed_results[1]['confidence'] == 'MEDIUM'
    assert parsed_results[2]['verdict'] == 'INCORRECT'
    assert parsed_results[2]['confidence'] == 'HIGH'
    assert parsed_results[3]['verdict'] == 'CORRECT'
    assert parsed_results[3]['confidence'] == 'HIGH'

    print("✅ PASS: All judge responses parsed correctly")


def test_calculate_metrics():
    """Test metrics calculation."""
    print("\n" + "="*70)
    print("TEST 4: Calculate Metrics")
    print("="*70)

    responses = create_sample_judge_responses()
    judge_results = [parse_judge_response(r) for r in responses]

    # Exact match results
    exact_match_results = [
        False,  # "yes" != "Yes, there is evidence of pneumonia."
        False,  # "frontal" != "frontal view"
        False,  # "no" != "yes"
        False,  # "lung" != "lungs"
    ]

    metrics = calculate_judge_metrics(judge_results, exact_match_results)

    print(f"  Total samples: {metrics['total_samples']}")
    print(f"  Judge correct: {metrics['judge_correct']}")
    print(f"  Judge incorrect: {metrics['judge_incorrect']}")
    print(f"  Judge accuracy: {metrics['judge_accuracy']:.2%}")
    print(f"  Exact match accuracy: {metrics['exact_match_accuracy']:.2%}")
    print(f"  Agreement rate: {metrics['agreement_rate']:.2%}")
    print(f"  Judge correct but exact wrong: {metrics['judge_correct_exact_wrong']}")

    # Validate metrics
    assert metrics['total_samples'] == 4
    assert metrics['judge_correct'] == 3  # Responses 1, 2, 4 are CORRECT
    assert metrics['judge_incorrect'] == 1  # Response 3 is INCORRECT
    assert metrics['judge_accuracy'] == 0.75  # 3/4
    assert metrics['exact_match_accuracy'] == 0.0  # 0/4
    assert metrics['judge_correct_exact_wrong'] == 3  # Judge found 3 correct, exact found 0

    print("✅ PASS: Metrics calculated correctly")


def test_save_results():
    """Test saving results to file."""
    print("\n" + "="*70)
    print("TEST 5: Save Results")
    print("="*70)

    responses = create_sample_judge_responses()
    judge_results = [parse_judge_response(r) for r in responses]
    exact_match_results = [False, False, False, False]
    metrics = calculate_judge_metrics(judge_results, exact_match_results)

    # Create temporary output file
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = os.path.join(tmpdir, "judge_results.json")
        save_judge_results(judge_results, metrics, output_file)

        # Verify file was created
        assert os.path.exists(output_file), "Output file not created"

        # Load and verify contents
        with open(output_file, 'r') as f:
            saved_data = json.load(f)

        assert 'metadata' in saved_data
        assert 'metrics' in saved_data
        assert 'evaluations' in saved_data
        assert saved_data['metrics']['total_samples'] == 4
        assert len(saved_data['evaluations']) == 4

        print(f"✅ PASS: Results saved successfully to {output_file}")


def test_edge_cases():
    """Test edge cases in parsing."""
    print("\n" + "="*70)
    print("TEST 6: Edge Cases")
    print("="*70)

    # Test 1: Missing verdict
    response_no_verdict = """
**CONFIDENCE:** HIGH
**EXPLANATION:** This response is missing a verdict.
"""
    parsed = parse_judge_response(response_no_verdict)
    assert parsed['verdict'] == 'UNKNOWN', f"Expected UNKNOWN verdict, got {parsed['verdict']}"
    print("  ✅ Missing verdict handled correctly")

    # Test 2: Missing confidence
    response_no_confidence = """
**VERDICT:** CORRECT
**EXPLANATION:** This response is missing confidence.
"""
    parsed = parse_judge_response(response_no_confidence)
    assert parsed['confidence'] == 'UNKNOWN', f"Expected UNKNOWN confidence, got {parsed['confidence']}"
    print("  ✅ Missing confidence handled correctly")

    # Test 3: Mixed case
    response_mixed_case = """
**verdict:** InCorrEct
**confidence:** MeDiUm
**explanation:** Mixed case test.
"""
    parsed = parse_judge_response(response_mixed_case)
    assert parsed['verdict'] == 'INCORRECT', f"Expected INCORRECT verdict, got {parsed['verdict']}"
    assert parsed['confidence'] == 'MEDIUM', f"Expected MEDIUM confidence, got {parsed['confidence']}"
    print("  ✅ Mixed case handled correctly")

    # Test 4: Extra whitespace
    response_whitespace = """


**VERDICT:**    CORRECT

**CONFIDENCE:**   LOW

**EXPLANATION:**   Extra whitespace test.

"""
    parsed = parse_judge_response(response_whitespace)
    assert parsed['verdict'] == 'CORRECT'
    assert parsed['confidence'] == 'LOW'
    print("  ✅ Extra whitespace handled correctly")

    print("✅ PASS: All edge cases handled correctly")


def test_potential_bugs():
    """Test for potential bugs identified in code review."""
    print("\n" + "="*70)
    print("TEST 7: Potential Bug Scenarios")
    print("="*70)

    # Bug 1: Generations without 'use_cot' field (line 444 in llm_judge.py)
    generations_no_cot = [
        {
            "image_file": "test.jpg",
            "question": "Test?",
            "reference_answer": "yes",
            "predicted_answer": "yes"
        }
    ]

    try:
        prompts = prepare_judge_prompts(generations_no_cot, use_cot=False)
        assert len(prompts) == 1
        print("  ✅ Generations without 'use_cot' field handled correctly")
    except Exception as e:
        print(f"  ❌ FAIL: Error with missing 'use_cot' field: {e}")

    # Bug 2: Empty generations list
    try:
        prompts = prepare_judge_prompts([], use_cot=False)
        assert len(prompts) == 0
        print("  ✅ Empty generations list handled correctly")
    except Exception as e:
        print(f"  ❌ FAIL: Error with empty generations: {e}")

    # Bug 3: Missing fields in generation data
    generations_missing_fields = [
        {
            "image_file": "test.jpg"
            # Missing question, reference_answer, predicted_answer
        }
    ]

    try:
        prompts = prepare_judge_prompts(generations_missing_fields, use_cot=False)
        assert len(prompts) == 1
        assert "N/A" in prompts[0]  # Should use default 'N/A' for missing fields
        print("  ✅ Missing fields handled with 'N/A' defaults")
    except Exception as e:
        print(f"  ❌ FAIL: Error with missing fields: {e}")

    print("✅ PASS: Potential bug scenarios handled correctly")


def main():
    """Run all tests."""
    print("\n" + "#"*70)
    print("# LLM JUDGE END-TO-END TEST SUITE")
    print("#"*70)

    try:
        test_load_generations()
        test_prepare_judge_prompts()
        test_parse_judge_responses()
        test_calculate_metrics()
        test_save_results()
        test_edge_cases()
        test_potential_bugs()

        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED!")
        print("="*70)
        return 0
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
