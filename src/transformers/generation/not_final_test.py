import sys
sys.path.insert(0, "/Users/npranitha/Downloads/HPML/Final Project/transformers/src")  

from transformers import AutoTokenizer
from transformers.generation.beam_constraints import TemplateConstraint
import torch
from transformers import AutoModelForCausalLM
import unittest
import time
import cProfile
import pstats
import wandb

MODELS = {
    "distilgpt2": "distilgpt2",
    "gpt2": "gpt2",
}

TEMPLATE_TEST_CASES = [
    {
        "name": "test1", 
        "input_text": "The woman attended",
        "template": [[" the"], [], [], [" University"], [], [" in"]],
    }
]

class TestTemplateConstraint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize WandB
        wandb.init(project="template-constraints-test", reinit=True)
        
    def setUp(self):
        self.start_time = time.time()
        self.profiler = cProfile.Profile()
        self.profiler.enable()

    def tearDown(self):
        self.profiler.disable()
        stats = pstats.Stats(self.profiler)
        stats.sort_stats('cumtime').print_stats(10)
        
        test_time = time.time() - self.start_time
        print(f"\nTest execution time: {test_time:.2f}s")
        wandb.log({"test_execution_time": test_time})

    def run_generation_test(self, model_name, test_case, num_return_sequences=1, num_beams=3):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            model.config.pad_token_id = model.config.eos_token_id
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            template = []
            for part in test_case["template"]:
                if part:
                    encoded = tokenizer.encode(part[0], add_special_tokens=False)
                    if not encoded:
                        raise ValueError(f"Invalid token: {part[0]}")
                    template.append(encoded)
                else:
                    template.append([])

            constraint = TemplateConstraint(template, list(tokenizer.get_vocab().values()))

            inputs = tokenizer(test_case["input_text"], return_tensors="pt").to(device)

            gen_kwargs = {
                "input_ids": inputs.input_ids,
                "max_length": len(inputs.input_ids[0]) + 20,
                "constraints": [constraint] if test_case["template"] else None,
                "num_beams": num_beams,
                "early_stopping": True,
                "num_return_sequences": num_return_sequences,
            }

            with torch.no_grad():
                outputs = model.generate(**gen_kwargs)

            
            # All returned sequences
            generated_sequences = [tokenizer.decode(output, skip_special_tokens=True) 
                                 for output in outputs]
            
            validation_results = []
            for seq in generated_sequences:
                validation_results.append(self.validate_output(seq, test_case, tokenizer))
            
            return {
                "model": model_name,
                "test_case": test_case["name"],
                "generated_sequences": generated_sequences,
                "validation_results": validation_results,
                "num_beams": num_beams,
                "num_return_sequences": num_return_sequences,
                "execution_success": True
            }
            
        except Exception as e:
            return {
                "model": model_name,
                "test_case": test_case["name"],
                "error": str(e),
                "execution_success": False
            }

    def validate_output(self, generated, test_case, tokenizer):
        required_parts = [part[0] for part in test_case["template"] if part]
        flexible_slots = sum(1 for part in test_case["template"] if not part)
        
        indices = [generated.lower().find(part.lower()) for part in required_parts]
        
        return {
            "template_parts_present": all(idx != -1 for idx in indices), # missing
            "template_order_correct": indices == sorted(indices), # out of order
            "slots_filled": flexible_slots == 0 or len(generated) > len(test_case["input_text"]) + sum(len(p[0]) for p in test_case["template"] if p)
        }

    def test_all_combinations(self):
        """Diff models, beams and return sequences"""
        beam_options = [2, 3, 4]
        return_seq_options = [1, 2, 3]
        
        for model_name, model_path in MODELS.items():
            print(f"\n\n===== Testing Model: {model_name} =====")
            
            for test_case in TEMPLATE_TEST_CASES:
                print(f"\n-- Test Case: {test_case['name']} --")
                print(f"Input: '{test_case['input_text']}'")
                print(f"Template: {test_case['template']}")
                
                for num_beams in beam_options:
                    for num_return_sequences in return_seq_options:
                        if num_return_sequences > num_beams:
                            continue
                            
                        print(f"\nTesting with beams={num_beams}, return_seq={num_return_sequences}")
                        result = self.run_generation_test(
                            model_path, 
                            test_case,
                            num_beams=num_beams,
                            num_return_sequences=num_return_sequences
                        )
                        
                        if result["execution_success"]:
                            print("\nGenerated Sequences:")
                            for i, (seq, validation) in enumerate(zip(
                                result["generated_sequences"], 
                                result["validation_results"]
                            )):
                                print(f"\nSequence {i+1}:")
                                print(f"  Text: {seq}")
                                print(f"  Validation:")
                                print(f"    Parts present: {validation['template_parts_present']}")
                                print(f"    Order correct: {validation['template_order_correct']}")
                                print(f"    Slots filled: {validation['slots_filled']}")
                            
                            wandb.log({
                                "model": model_name,
                                "test_case": test_case["name"],
                                "num_beams": num_beams,
                                "num_return_sequences": num_return_sequences,
                                "generated_text": result["generated_sequences"][0],
                                **result["validation_results"][0]
                            })
                        else:
                            print(f"ERROR: {result['error']}")
                            wandb.log({
                                "model": model_name,
                                "test_case": test_case["name"],
                                "error": result["error"],
                                "execution_success": False
                            })

if __name__ == '__main__':
    unittest.main()