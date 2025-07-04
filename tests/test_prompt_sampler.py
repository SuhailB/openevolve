"""
Tests for PromptSampler in openevolve.prompt.sampler
"""

import unittest
from openevolve.config import Config
from openevolve.prompt.sampler import PromptSampler


class TestPromptSampler(unittest.TestCase):
    """Tests for prompt sampler"""

    def setUp(self):
        """Set up test prompt sampler"""
        config = Config()
        self.prompt_sampler = PromptSampler(config.prompt)

    def test_build_prompt(self):
        """Test building a prompt"""
        current_program = "def test(): pass"
        parent_program = "def test(): pass"
        program_metrics = {"score": 0.5}
        previous_programs = [
            {
                "id": "prev1",
                "code": "def prev1(): pass",
                "metrics": {"score": 0.4},
            }
        ]
        top_programs = [
            {
                "id": "top1",
                "code": "def top1(): pass",
                "metrics": {"score": 0.6},
            }
        ]

        prompt = self.prompt_sampler.build_prompt(
            current_program=current_program,
            parent_program=parent_program,
            program_metrics=program_metrics,
            previous_programs=previous_programs,
            top_programs=top_programs,
        )

        self.assertIn("system", prompt)
        self.assertIn("user", prompt)
        self.assertIn("def test(): pass", prompt["user"])
        self.assertIn("score: 0.5", prompt["user"])

    def test_metric_minimization_feature(self):
        """Test that metrics starting with '-' are handled correctly for minimization"""
        current_program = "def test(): pass"
        parent_program = "def test(): pass"
        
        # Test with both regular and minimization metrics
        program_metrics = {
            "score": 0.8,           # Regular metric (maximize)
            "-error_rate": 0.6,     # Minimization metric
            "-latency": 0.05,        # Another minimization metric
            "accuracy": 0.9         # Regular metric (maximize)
        }
        
        # Create previous programs with different metric values to test comparison logic
        previous_programs = [
            {
                "id": "prev1",
                "code": "def prev1(): pass",
                "metrics": {
                    "score": 0.4,           # Worse than current (0.5)
                    "-error_rate": 0.5,     # Worse than current (0.3) - should be higher for minimization
                    "-latency": 0.9,        # Worse than current (0.8) - should be higher for minimization
                    "accuracy": 0.91        # Better than current (0.8)
                },
            },
            {
                "id": "prev2", 
                "code": "def prev2(): pass",
                "metrics": {
                    "score": 0.6,           # Better than current (0.5)
                    "-error_rate": 0.2,     # Better than current (0.3) - should be lower for minimization
                    "-latency": 0.7,        # Better than current (0.8) - should be lower for minimization
                    "accuracy": 0.95        # Better than current (0.9)
                },
            }
        ]

        response = self.prompt_sampler._identify_improvement_areas(
            current_program=current_program,
            parent_program=parent_program,
            metrics=program_metrics,
            previous_programs=previous_programs
        )
        print(f"=======Response=========\n{response}")

if __name__ == "__main__":
    unittest.main()
