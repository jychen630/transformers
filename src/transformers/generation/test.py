from transformers import GPT2LMHeadModel, GPT2Tokenizer
from beam_constraints import TemplateConstraint
import torch

# Load model and tokenizer
model_name = "gpt2-xl"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Template example: "the", [wildcard], "School", "of", [wildcard], "in"
template = ["the", "", "School", "of", "", "in"]

# Convert template to token IDs: string → token ID; "" → None
template_token_ids = []
for segment in template:
    if segment == "":
        template_token_ids.append(None)
    else:
        ids = tokenizer.encode(segment, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Segment '{segment}' tokenized to more than one token: {ids}")
        template_token_ids.append(ids[0])

# Build the constraint using DisjunctiveTrie-based TemplateConstraint
constraint = TemplateConstraint(template_token_ids)

# Prepare input
prompt = "The woman"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

# Attempt generation
# NOTE: This will only work if your model's generation loop or beam search has been extended to support `TemplateConstraint`
outputs = model.generate(
    input_ids=input_ids,
    num_beams=2,
    num_return_sequences=2,
    no_repeat_ngram_size=1,
    constraints=[constraint],  # Will only work if supported by your generation code
    max_length=10,
    remove_invalid_values=True,
    pad_token_id=tokenizer.eos_token_id
)

# Decode
print("\nGenerated outputs:")
for i, output in enumerate(outputs):
    decoded = tokenizer.decode(output, skip_special_tokens=True)
    print(f"{i + 1}: {decoded}")








# token_segments = [
#     tokenizer("the", add_special_tokens=False).input_ids,
#     [-1],
#     tokenizer("School of", add_special_tokens=False).input_ids,
#     [-1],
#     tokenizer("in", add_special_tokens=False).input_ids,
# ]

# constraint = TemplateConstraint(token_segments)

# if hasattr(constraint, "token_ids"):
#     assert all(
#         token is None or (isinstance(token, int) and token >= 0)
#         for segment in constraint.token_ids
#         for token in segment if isinstance(segment, list)
#     )

# starting_text = ["The woman"]

# input_ids = tokenizer(starting_text, return_tensors="pt").input_ids

# outputs = model.generate(
#     input_ids,
#     num_beams=3,
#     constraints=[constraint],
#     num_return_sequences=2,
#     no_repeat_ngram_size=1,
#     remove_invalid_values=True,
#     pad_token_id=tokenizer.eos_token_id, 
# )


# print("Output:\n" + 100 * '-')
# for out in outputs:
#     print(tokenizer.decode(out, skip_special_tokens=True))





###########################################################################
### For dashboard
# cd src/transformers/generation/

# # Start Python HTTP server
# python -m http.server 8000

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