import sys
sys.path.insert(0, "/local/nlp/junyao/transformers/src/transformers/generation")  

from transformers import AutoTokenizer
from transformers.generation.beam_constraints import TemplateConstraint, ConstraintListState
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.generation.utils import BeamSearchScorer
import unittest
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "gpt2"
class TestTemplateConstraint(unittest.TestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.config.pad_token_id = self.model.config.eos_token_id


    def test_template_generation(self):
        vocab = list(self.tokenizer.get_vocab().values())
        template = [
            self.tokenizer.encode(" the", add_special_tokens=False),
            [],
            [],
            self.tokenizer.encode(" University", add_special_tokens=False),
            [],
            self.tokenizer.encode(" in", add_special_tokens=False)
        ]
        
        constraint = TemplateConstraint(template, vocab)
        
        input_text = "The woman attended"
        inputs = self.tokenizer(input_text, return_tensors="pt")
        
        print("Before generation")
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=20,
            constraints=[constraint],
            num_beams=3,
            early_stopping=True,
            num_return_sequences=1
        )
        print("After generation")
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nGenerated: {generated}")
        

        self.assertTrue(" the" in generated.lower())
        self.assertTrue(" university" in generated.lower())
        self.assertTrue(" in" in generated.lower())
        
        
        parts = [" the", " university", " in"]
        indices = [generated.lower().find(part) for part in parts]
        self.assertTrue(all(idx != -1 for idx in indices), "Missing template parts")
        self.assertTrue(indices == sorted(indices), "Template parts out of order")

if __name__ == '__main__':
    unittest.main()