from transformers import AutoTokenizer
from transformers.generation.beam_constraints import TemplateConstraint
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2-xl"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id

template = ["the", "", "School of", "", "in"]
token_segments = [
    tokenizer("the", add_special_tokens=False).input_ids,
    [-1],
    tokenizer("School of", add_special_tokens=False).input_ids,
    [-1],
    tokenizer("in", add_special_tokens=False).input_ids,
]

constraint = TemplateConstraint(token_segments, max_gap_len=5)

if hasattr(constraint, "token_ids"):
    assert all(
        token is None or (isinstance(token, int) and token >= 0)
        for segment in constraint.token_ids
        for token in segment if isinstance(segment, list)
    )

starting_text = ["The woman"]

input_ids = tokenizer(starting_text, return_tensors="pt").input_ids

outputs = model.generate(
    input_ids,
    num_beams=3,
    constraints=[constraint],
    num_return_sequences=2,
    no_repeat_ngram_size=1,
    remove_invalid_values=True,
    pad_token_id=tokenizer.eos_token_id, 
)


print("Output:\n" + 100 * '-')
for out in outputs:
    print(tokenizer.decode(out, skip_special_tokens=True))
