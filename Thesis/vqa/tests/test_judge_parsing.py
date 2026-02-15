import sys
import os
import unittest
from unittest.mock import MagicMock

# Mock dependencies not available or broken in host environment
sys.modules['dotenv'] = MagicMock()
sys.modules['vllm'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['numpy'] = MagicMock()

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from llm_judge import parse_judge_response

class TestJudgeParsing(unittest.TestCase):
    def test_standard_format(self):
        response = """
**VERDICT:** CORRECT
**CONFIDENCE:** HIGH
**EXPLANATION:** The predicted answer matches the reference exactly.
"""
        result = parse_judge_response(response)
        self.assertEqual(result['verdict'], 'CORRECT')
        self.assertEqual(result['confidence'], 'HIGH')
        self.assertIn("matches the reference", result['explanation'])

    def test_no_markdown_bold(self):
        response = """
VERDICT: INCORRECT
CONFIDENCE: MEDIUM
EXPLANATION: The predicted answer is partially correct but misses key details.
"""
        result = parse_judge_response(response)
        self.assertEqual(result['verdict'], 'INCORRECT')
        self.assertEqual(result['confidence'], 'MEDIUM')
        self.assertIn("partially correct", result['explanation'])

    def test_qwen_style_output(self):
        # reliable extraction from typical Qwen/Llama chat output
        response = """Based on the evaluation criteria:

1. Semantic Equivalence: Yes.
2. Medical Accuracy: Yes.

**VERDICT:** CORRECT
**CONFIDENCE:** HIGH
**EXPLANATION:** Both answers confirm the presence of pneumonia.
"""
        result = parse_judge_response(response)
        self.assertEqual(result['verdict'], 'CORRECT')
        self.assertEqual(result['confidence'], 'HIGH')
        self.assertEqual(result['explanation'], "Both answers confirm the presence of pneumonia.")

    def test_lowercase_handling(self):
        response = """
**Verdict:** Correct
**Confidence:** High
**Explanation:** meaningful match.
"""
        # The regex in original code uses re.IGNORECASE for the keys, but we should check if it capitalizes the value
        result = parse_judge_response(response)
        self.assertEqual(result['verdict'], 'CORRECT')
        self.assertEqual(result['confidence'], 'HIGH')

if __name__ == '__main__':
    unittest.main()
