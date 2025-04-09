# coding=utf-8
# Copyright 2020 The HuggingFace Team Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a clone of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch

    from transformers.generation import DisjunctiveConstraint, TemplateConstraint


@require_torch
class ConstraintTest(unittest.TestCase):
    def test_input_types(self):
        # For consistency across different places the DisjunctiveConstraint is called,
        # dc.token_ids is a list of integers. It is also initialized only by integers.

        cset = [[1, 2, 4], [1, 2, 3, 4]]
        dc = DisjunctiveConstraint(cset)
        self.assertTrue(isinstance(dc.token_ids, list))

        with self.assertRaises(ValueError):
            DisjunctiveConstraint(torch.LongTensor([[1, 2, 4], [1, 2, 3]]))

        with self.assertRaises(ValueError):
            DisjunctiveConstraint([torch.LongTensor([1, 2, 4]), torch.LongTensor([1, 2, 3, 4, 5])])

    def test_check_illegal_input(self):
        # We can't have constraints that are complete subsets of another. This leads to a perverse
        # interpretation of "constraint fulfillment": does generating [1,2,3] fulfill the constraint?
        # It would mean that it generated [1,2] which fulfills it, but it's in the middle of potentially
        # fulfilling [1,2,3,4]. If we believe that [1,2,3] does fulfill the constraint, then the algorithm
        # will necessarily never reach [1,2,3,4], giving users a false sense of control (better to just not allow it).
        cset = [[1, 2], [1, 2, 3, 4]]

        with self.assertRaises(ValueError):
            DisjunctiveConstraint(cset)  # fails here

    def test_example_progression(self):
        cset = [[1, 2, 3], [1, 2, 4]]

        dc = DisjunctiveConstraint(cset)

        stepped, completed, reset = dc.update(1)
        desired = stepped is True and completed is False and reset is False
        self.assertTrue(desired)
        self.assertTrue(not dc.completed)
        self.assertTrue(dc.current_seq == [1])

        stepped, completed, reset = dc.update(2)
        desired = stepped is True and completed is False and reset is False
        self.assertTrue(desired)
        self.assertTrue(not dc.completed)
        self.assertTrue(dc.current_seq == [1, 2])

        stepped, completed, reset = dc.update(3)
        desired = stepped is True and completed is True and reset is False
        self.assertTrue(desired)
        self.assertTrue(dc.completed)  # Completed!
        self.assertTrue(dc.current_seq == [1, 2, 3])

    def test_example_progression_unequal_three_mid_and_reset(self):
        cset = [[1, 2, 3], [1, 2, 4, 5], [1, 2, 5]]

        dc = DisjunctiveConstraint(cset)

        stepped, completed, reset = dc.update(1)
        self.assertTrue(not dc.completed)
        self.assertTrue(dc.current_seq == [1])

        stepped, completed, reset = dc.update(2)
        self.assertTrue(not dc.completed)
        self.assertTrue(dc.current_seq == [1, 2])

        stepped, completed, reset = dc.update(4)
        self.assertTrue(not dc.completed)
        self.assertTrue(dc.current_seq == [1, 2, 4])

        stepped, completed, reset = dc.update(5)
        self.assertTrue(dc.completed)  # Completed!
        self.assertTrue(dc.current_seq == [1, 2, 4, 5])

        dc.reset()

        stepped, completed, reset = dc.update(1)
        self.assertTrue(not dc.completed)
        self.assertTrue(dc.remaining() == 3)
        self.assertTrue(dc.current_seq == [1])

        stepped, completed, reset = dc.update(2)
        self.assertTrue(not dc.completed)
        self.assertTrue(dc.remaining() == 2)
        self.assertTrue(dc.current_seq == [1, 2])

        stepped, completed, reset = dc.update(5)
        self.assertTrue(dc.completed)  # Completed!
        self.assertTrue(dc.remaining() == 0)
        self.assertTrue(dc.current_seq == [1, 2, 5])


@require_torch
class TemplateConstraintTest(unittest.TestCase):
    def test_input_types(self):
        # Test valid template: ["the", "", "School of", "", "in"]
        template = [5, -1, 10, 20, -1, 15]
        tc = TemplateConstraint(template)
        self.assertTrue(isinstance(tc.token_ids, list))

        # Test invalid inputs
        with self.assertRaises(ValueError):
            TemplateConstraint(torch.tensor([5, -1, 10]))

        with self.assertRaises(ValueError):
            TemplateConstraint([])

        with self.assertRaises(ValueError):
            TemplateConstraint([5, -2, 10])

        with self.assertRaises(ValueError):
            TemplateConstraint([5, "word", 10])

    def test_example_progression_with_wildcards(self):
        # Test template: ["the", "", "school"]
        # converted to token IDs: [5, -1, 10]
        template = [5, -1, 10]
        tc = TemplateConstraint(template)

        #  "the"
        stepped, completed, reset = tc.update(5)
        self.assertTrue(stepped)
        self.assertFalse(completed)
        self.assertFalse(reset)
        self.assertFalse(tc.completed)
        self.assertEqual(tc.fulfilled_idx, 0)

        # wildcard, any token (e.g 100)
        stepped, completed, reset = tc.update(100)
        self.assertTrue(stepped)
        self.assertFalse(completed)
        self.assertFalse(reset)
        self.assertFalse(tc.completed)
        self.assertEqual(tc.fulfilled_idx, 1)

        #  "school"
        stepped, completed, reset = tc.update(10)
        self.assertTrue(stepped)
        self.assertTrue(completed)
        self.assertFalse(reset)
        self.assertTrue(tc.completed)
        self.assertEqual(tc.fulfilled_idx, 2)

    def test_reset_and_remaining(self):
        # Test template: ["the", "", "school", "", "in"]
        template = [5, -1, 10, -1, 15]
        tc = TemplateConstraint(template)

        # Match "the"
        stepped, completed, reset = tc.update(5)
        self.assertTrue(stepped)
        self.assertEqual(tc.remaining(), 4)

        stepped, completed, reset = tc.update(5)
        self.assertFalse(stepped)
        self.assertTrue(reset)
        self.assertEqual(tc.fulfilled_idx, 0)
        self.assertEqual(tc.remaining(), 5)

        # Test complete sequence
        tc.update(5)  # "the"
        tc.update(100)  # wildcard
        tc.update(10)  # "school"
        tc.update(200)  # wildcard
        stepped, completed, reset = tc.update(15)  # "in"
        self.assertTrue(completed)
        self.assertEqual(tc.remaining(), 0)

    def test_advance_and_does_advance(self):
        template = [5, -1, 10]  # ["the", "", "school"]
        tc = TemplateConstraint(template)

        self.assertEqual(
            .advance(), 5)
        self.assertTrue(tc.does_advance(5))
        self.assertFalse(tc.does_advance(6))

        tc.update(5)

        # this is wildcard position and so it should advance with any token
        self.assertIsNone(tc.advance())
        self.assertTrue(tc.does_advance(100))
        self.assertTrue(tc.does_advance(200))

        tc.update(100)

        self.assertEqual(tc.advance(), 10)
        self.assertTrue(tc.does_advance(10))
        self.assertFalse(tc.does_advance(11))

    def test_copy(self):
        template = [5, -1, 10]
        tc = TemplateConstraint(template)

        tc.update(5)
        tc.update(100)

        copied = tc.copy(stateful=False)
        self.assertEqual(copied.token_ids, tc.token_ids)
        self.assertEqual(copied.fulfilled_idx, -1)  # Reset state
        self.assertFalse(copied.completed)

        stateful_copied = tc.copy(stateful=True)
        self.assertEqual(stateful_copied.token_ids, tc.token_ids)
        self.assertEqual(stateful_copied.fulfilled_idx, tc.fulfilled_idx)
        self.assertEqual(stateful_copied.completed, tc.completed)
