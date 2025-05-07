from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.generation.beam_constraints import OrderedConstraint
from transformers import LogitsProcessorList
from transformers import LogitsProcessor
import torch
import time



model_name = "gpt2-xl"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# model_name = "EleutherAI/gpt-j-6b"
# model = GPT2LMHeadModel.from_pretrained(model_name)
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id
vocab = list(tokenizer.get_vocab().values())

if torch.cuda.is_available():
    print("CUDA is available")
    model.cuda()


input_text = "Healthier life "
# input_text = "Irish workers were subject"

ordered_phrases = [
#[" weather", " freaking"],
#    [" holiday", " um", " depressing", " intricate", " tariff"]
   [" be", " will", " and", " well"]
#    [" large", " amounts", " many"]
]

ordered_constraints = []
for phrase in ordered_phrases:
    token_ids = []
    for segment in phrase:
        ids = tokenizer.encode(segment, add_special_tokens=False)
        if len(ids) != 1:
            #ValueError: Segment ' halloween' tokenized into multiple tokens: [6899, 322, 6429]
            raise ValueError(f"Segment '{segment}' tokenized into multiple tokens: {ids}")
        token_ids.append(ids[0])
    constraint = OrderedConstraint(token_ids, vocab_length=len(vocab))
    ordered_constraints.append(constraint)

inputs = tokenizer(input_text, return_tensors="pt")
if torch.cuda.is_available():
    inputs.input_ids = inputs.input_ids.cuda()

num_beams = 4
num_return_sequences = 3
assert num_beams >= num_return_sequences

logits_processors = LogitsProcessorList()

start = time.time()
outputs = model.generate(
    inputs.input_ids,
    constraints=ordered_constraints,
    max_length=100,
    num_beams=num_beams,
    early_stopping=True,
    num_return_sequences=num_return_sequences,
    no_repeat_ngram_size=2,
)

decoded_outputs = []
for output in outputs:
    decoded_outputs.append(tokenizer.decode(output, skip_special_tokens=True))

end = time.time()
print(f"Time taken:  {(end - start)} sec")
for output in decoded_outputs:
    # todo output is a sentence. check if the words in ordered_phrases appear in order in the output
    print("*"*100)
    print(f"Generated output: {output}")
