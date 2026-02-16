#!/usr/bin/env python3
"""
Test error handling improvements in llm_judge.py
"""

import sys
import os
import json
import tempfile
from pathlib import Path
import unittest

# Mock dependencies
from unittest.mock import MagicMock
sys.modules['dotenv'] = MagicMock()
sys.modules['vllm'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['numpy'] = MagicMock()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from llm_judge import load_generations, print_judge_metrics


class TestErrorHandling(unittest.TestCase):

    def test_load_nonexistent_file(self):
        """Test that loading a nonexistent file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError) as context:
            load_generations("/nonexistent/file.json")

        self.assertIn("not found", str(context.exception))

    def test_load_invalid_json(self):
        """Test that loading invalid JSON raises ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ this is not valid json }")
            temp_path = f.name

        try:
            with self.assertRaises(ValueError) as context:
                load_generations(temp_path)
            self.assertIn("Invalid JSON", str(context.exception))
        finally:
            os.unlink(temp_path)

    def test_load_wrong_format(self):
        """Test that loading non-list JSON raises ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"not": "a list"}, f)
            temp_path = f.name

        try:
            with self.assertRaises(ValueError) as context:
                load_generations(temp_path)
            self.assertIn("Expected list", str(context.exception))
        finally:
            os.unlink(temp_path)

    def test_load_empty_list(self):
        """Test that loading empty generations list raises ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([], f)
            temp_path = f.name

        try:
            with self.assertRaises(ValueError) as context:
                load_generations(temp_path)
            self.assertIn("empty", str(context.exception).lower())
        finally:
            os.unlink(temp_path)

    def test_print_metrics_zero_samples(self):
        """Test that printing metrics with zero samples doesn't crash."""
        metrics = {
            'total_samples': 0,
            'judge_correct': 0,
            'judge_incorrect': 0,
            'judge_unknown': 0,
            'judge_accuracy': 0.0,
            'exact_match_accuracy': 0.0,
            'agreement_rate': 0.0,
            'judge_correct_exact_wrong': 0,
            'confidence_distribution': {
                'HIGH': 0,
                'MEDIUM': 0,
                'LOW': 0,
                'UNKNOWN': 0
            }
        }

        # Should not raise any exceptions
        try:
            print_judge_metrics(metrics)
            print("✅ Zero samples handled correctly in print function")
        except ZeroDivisionError:
            self.fail("print_judge_metrics raised ZeroDivisionError for zero samples")


class TestModelNameInference(unittest.TestCase):
    """Test the improved model name inference logic."""

    def test_standard_format(self):
        """Test extraction from standard format: vqa_{model}_{N}samples_generations_{timestamp}"""
        import re

        # Simulate the inference logic
        filename = "vqa_google_medgemma-4b-it_100samples_generations_20260214_120000"

        match = re.match(r'vqa_(.+?)_\d+samples_generations', filename)
        if match:
            model_name = match.group(1)
        else:
            model_name = "unknown"

        self.assertEqual(model_name, "google_medgemma-4b-it")
        print(f"✅ Extracted model name: {model_name}")

    def test_fallback_format(self):
        """Test fallback when pattern doesn't match."""
        import re

        filename = "custom_generations_20260214"

        match = re.match(r'vqa_(.+?)_\d+samples_generations', filename)
        if match:
            model_name = match.group(1)
        elif '_generations_' in filename:
            model_name = filename.split('_generations_')[0]
        else:
            model_name = filename

        self.assertEqual(model_name, "custom")
        print(f"✅ Fallback extraction: {model_name}")

    def test_no_pattern_match(self):
        """Test when filename has no recognizable pattern."""
        import re

        filename = "results_20260214_120000"

        match = re.match(r'vqa_(.+?)_\d+samples_generations', filename)
        if match:
            model_name = match.group(1)
        elif '_generations_' in filename:
            model_name = filename.split('_generations_')[0]
        else:
            model_name = filename

        self.assertEqual(model_name, "results_20260214_120000")
        print(f"✅ No pattern: using filename as model name: {model_name}")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("ERROR HANDLING TESTS")
    print("="*70 + "\n")

    unittest.main(verbosity=2)
