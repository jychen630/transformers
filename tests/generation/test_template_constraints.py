# Template Constraints

import unittest
from typing import List, Union

from transformers import is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch
    # This would be your new class
    from transformers.generation import TemplateConstraint


@require_torch
class TemplateConstraintUnitTest(unittest.TestCase):
    def test_fixed_template(self):
        # fixed 
        template = [1,2,3]
        constraint = TemplateConstraint(template)
        
        # initially
        self.assertFalse(constraint.completed)
        self.assertEqual(constraint.remaining(), 3)
        
        # walking through the template
        result = constraint.update(1)
        self.assertTrue(result.stepped)
        self.assertFalse(result.completed)
        self.assertFalse(result.reset)
        self.assertEqual(constraint.remaining(), 2)
        
        result = constraint.update(2)
        self.assertTrue(result.stepped)
        self.assertFalse(result.completed)
        self.assertFalse(result.reset)
        self.assertEqual(constraint.remaining(), 1)
        
        result = constraint.update(3)
        self.assertTrue(result.stepped)
        self.assertTrue(result.completed)
        self.assertFalse(result.reset)
        self.assertEqual(constraint.remaining(), 0)
        self.assertTrue(constraint.completed)

    def test_mixed_template(self):
        # flexi + fixed
        template = [1, "", 3]
        tc = TemplateConstraint(template)
        
        result = tc.update(1)
        self.assertTrue(result.stepped and not result.completed and not result.reset)
        self.assertEqual(tc.current_position, 1)
        self.assertEqual(tc.remaining(), 2)
        
        result = tc.update(44)
        self.assertTrue(result.stepped and not result.completed and not result.reset)
        self.assertEqual(tc.current_position, 2)
        self.assertEqual(tc.remaining(), 1)
        
        result = tc.update(3)
        self.assertTrue(result.stepped and not result.completed and not result.reset)
        self.assertTrue(tc.completed)
        self.assertEqual(tc.current_position, 3)
        self.assertEqual(tc.remaining(), 0)

    def test_wrong_token_resets(self):
        # wrong case, reset the constraint
        template = [1, 2]
        constraint = TemplateConstraint(template)
        
        constraint.update(1)  # "the"
        self.assertEqual(constraint.current_position, 1)
        
        # Wrong second token should reset
        result = constraint.update(99)  # Not "woman"
        self.assertTrue(result.reset)
        self.assertEqual(constraint.current_position, 0)
        
        # Should accept correct sequence after reset
        constraint.update(1)  # "the"
        constraint.update(2)  # "woman"
        self.assertTrue(constraint.completed)

    def test_nested_options(self):
        # nested
        template = ["Report", [["on", ""], ["about", ""]], "submitted"]
        constraint = TemplateConstraint(template)
        
        # Path 1: "Report on <any> submitted"
        constraint.update(1)  # "Report"
        constraint.update(2)  # "on"
        constraint.update(99)  # any token
        constraint.update(3)  # "submitted"
        self.assertTrue(constraint.completed)
        
        constraint.reset()
        
        # Path 2: "Report about <any> submitted"
        constraint.update(1)  # "Report"
        constraint.update(4)  # "about"
        constraint.update(100)  # any token
        constraint.update(3)  # "submitted"
        self.assertTrue(constraint.completed)

    def test_remaining_tokens(self):
        """Test accurate counting of remaining tokens"""
        template = ["Start", "", "middle", "", "end"]
        constraint = TemplateConstraint(template)
        
        self.assertEqual(constraint.remaining(), 5)
        constraint.update(1)  # "Start"
        self.assertEqual(constraint.remaining(), 4)
        constraint.update(99)  # any token
        self.assertEqual(constraint.remaining(), 3)
        constraint.update(2)  # "middle"
        self.assertEqual(constraint.remaining(), 2)
        constraint.update(100)  # any token
        self.assertEqual(constraint.remaining(), 1)
        constraint.update(3)  # "end"
        self.assertEqual(constraint.remaining(), 0)

    def test_blank_slots(self):
        """Test completely flexible slots (empty strings)"""
        template = ["", "fixed", ""]
        constraint = TemplateConstraint(template)
        
        # First blank slot
        result = constraint.update(99)  # any token
        self.assertTrue(result.stepped)
        
        # Fixed part
        result = constraint.update(1)  # "fixed"
        self.assertTrue(result.stepped)
        
        # Last blank slot
        result = constraint.update(100)  # any token
        self.assertTrue(result.stepped)
        self.assertTrue(result.completed)

    def test_reset_behavior(self):
        """Test that reset returns to initial state"""
        template = ["first", "second"]
        constraint = TemplateConstraint(template)
        
        constraint.update(1)  # "first"
        self.assertEqual(constraint.current_position, 1)
        
        constraint.reset()
        self.assertEqual(constraint.current_position, 0)
        self.assertFalse(constraint.completed)
        
        # Should accept sequence after reset
        constraint.update(1)  # "first"
        constraint.update(2)  # "second"
        self.assertTrue(constraint.completed)


@require_torch
class TemplateConstraintIntegrationTest(unittest.TestCase):
    def test_with_mock_beam_scorer(self):
        """Example of how you might test integration with a mock beam scorer"""
        
        class MockBeamScorer:
            def __init__(self, constraints):
                self.constraints = constraints
            
            def apply_constraints(self, token_ids):
                for constraint in self.constraints:
                    result = constraint.update(token_ids[-1])
                    if result.reset:
                        return False
                return True
        
        template = ["the", "woman"]
        constraint = TemplateConstraint(template)
        scorer = MockBeamScorer([constraint])
        
        # Test good sequence
        self.assertTrue(scorer.apply_constraints([1]))  # "the"
        self.assertTrue(scorer.apply_constraints([1, 2]))  # "woman"
        
        # Test bad sequence
        self.assertTrue(scorer.apply_constraints([1]))  # "the"
        self.assertFalse(scorer.apply_constraints([1, 99]))  # not "woman"