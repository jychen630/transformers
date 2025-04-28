from transformers import AutoTokenizer
from transformers.generation.beam_constraints import TemplateConstraint
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import time

from transformers import LogitsProcessor
# we shoul mask out all non-allowed tokens at generation step.
#we must at least add a logits processor to mask invalid tokens at each generation step.
#Otherwise, it is mathematically impossible to enforce the template with the current Huggingface beam search.
class TemplateConstraintLogitsProcessor(LogitsProcessor):
    def __init__(self, template, vocab_size):
        self.template = template
        self.vocab_size = vocab_size
        self.position = 0

    def __call__(self, input_ids, scores):
        if self.position >= len(self.template):
            return scores 

        expected = self.template[self.position]
        self.position += 1

        if expected is None:
            return scores
        else:
            mask = torch.full_like(scores, -float('inf'))  # mask everything
            mask[..., expected] = 0 
            return scores + mask

model_name = "gpt2-xl"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id
vocab = list(tokenizer.get_vocab().values())

if torch.cuda.is_available():
    print("CUDA is available")
    model.cuda()

template_tokens = []
constraint_template = [" the", "", " School", " of", "", " in"]
#constraint_template = [" the", "", " University",  "", " in"]
for segment in constraint_template:
    if segment == "":
        template_tokens.append(None)
    else:
        ids = tokenizer.encode(segment, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Segment '{segment}' tokenized into multiple tokens: {ids}")
        template_tokens.append(ids[0])
print(template_tokens)

constraint = TemplateConstraint(template_tokens, vocab_length=len(vocab))

start = time.time()
input_text = "The woman attended"
inputs = tokenizer(input_text, return_tensors="pt")
if torch.cuda.is_available():
    inputs.input_ids = inputs.input_ids.cuda()
num_beams = 5
num_return_sequences = 5
assert num_beams >= num_return_sequences
outputs = model.generate(
    inputs.input_ids,
    constraints=[constraint],
    logits_processor=[TemplateConstraintLogitsProcessor(constraint.template, len(vocab))],
    max_length=20,
    num_beams=num_beams,
    early_stopping=True,
    num_return_sequences=num_return_sequences,
    no_repeat_ngram_size=2,
)

print(outputs)
for output in outputs:
    generated = tokenizer.decode(output, skip_special_tokens=True)
    end = time.time()
    print(f"Time taken:  {(end - start)}sec")
    print(f"Generated (in the bracket): [{generated}]")


indices = [generated.lower().find(part.lower()) if part != "" else -999 for part in constraint_template]
print(indices)
assert all(idx != -1 for idx in indices), "Missing template parts"



###########################################################################
### For dashboard
import json

data = {
    "model_name": model_name,
    "constraint": constraint_template,
    "outputs": [
        tokenizer.decode(out, skip_special_tokens=True) for out in outputs
    ]
}

with open("dashboard_data.json", "w") as f:
    json.dump(data, f, indent=2)