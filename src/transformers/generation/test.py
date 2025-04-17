from transformers import AutoTokenizer
from transformers.generation.beam_constraints import TemplateConstraint
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import time


model_name = "gpt2-xl"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id
vocab = list(tokenizer.get_vocab().values())

if torch.cuda.is_available():
    print("CUDA is available")
    model.cuda()

template = [
    tokenizer.encode(" the", add_special_tokens=False),
    [],
    [],
    tokenizer.encode(" University", add_special_tokens=False),
    [],
    tokenizer.encode(" in", add_special_tokens=False)
]

constraint = TemplateConstraint(template, vocab)
start = time.time()
input_text = "The woman attended"
inputs = tokenizer(input_text, return_tensors="pt")
if torch.cuda.is_available():
    inputs.input_ids = inputs.input_ids.cuda()
outputs = model.generate(
    inputs.input_ids,
    max_length=10,
    constraints=[constraint],
    num_beams=2,
    early_stopping=True,
    num_return_sequences=1
)
generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
end = time.time()
print(f"Time taken: {(end - start) / 60:.2f}min {(end - start) % 60:.2f}sec")
print(f"\nGenerated: {generated}")
        
parts = [" the", " university", " in"]
#parts = [" the", "School of", " in"]

indices = [generated.lower().find(part) for part in parts]
assert all(idx != -1 for idx in indices), "Missing template parts"
assert indices == sorted(indices), "Template parts out of order"



###########################################################################
### For dashboard
import json

data = {
    "model_name": model_name,
    "constraint": template,
    "outputs": [
        tokenizer.decode(out, skip_special_tokens=True) for out in outputs
    ]
}

with open("dashboard_data.json", "w") as f:
    json.dump(data, f, indent=2)