from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.generation.beam_constraints import OrderedConstraint
from transformers import LogitsProcessorList
# from transformers import SimpleOrderedConstraintLogitsProcessor
# from transformers import LogitsProcessorList
from transformers import LogitsProcessor
import torch
import time

class OrderedConstraintLogitsProcessor(LogitsProcessor):
    def __init__(self, ordered_token_ids: list, penalty: float = -10000.0):
        """
        A logits processor to enforce ordered token constraints in text generation.

        :param ordered_token_ids: List of token IDs that should appear in the sequence in the given order.
        :param penalty: Penalty applied to non-expected tokens to push them down in the probability distribution.
        """
        self.ordered_token_ids = ordered_token_ids
        self.penalty = penalty 
        self.position = 0 

    def __call__(self, input_ids, logits):
        """
        Modify the logits to apply constraints based on the ordered tokens.

        :param input_ids: Current sequence of generated tokens.
        :param logits: The logits for the next token.
        :return: Modified logits.
        """
        current_position = len(input_ids[0]) - 1

        # within bounds
        if current_position < len(self.ordered_token_ids):
            expected_token_id = self.ordered_token_ids[self.position]

            # Penalize all tokens except the expected token
            logits[:, :] += self.penalty
            logits[:, expected_token_id] = 0
            
            predicted_token_ids = torch.argmax(logits, dim=-1)

            if (predicted_token_ids == expected_token_id).any():
                self.position += 1
        return logits


model_name = "gpt2-xl"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id
vocab = list(tokenizer.get_vocab().values())

if torch.cuda.is_available():
    print("CUDA is available")
    model.cuda()

ordered_phrases = [
    [" University", " California", " science"],
    [" doctor", " PhD"]
]

ordered_constraints = []
for phrase in ordered_phrases:
    token_ids = []
    for segment in phrase:
        ids = tokenizer.encode(segment, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Segment '{segment}' tokenized into multiple tokens: {ids}")
        token_ids.append(ids[0])
    constraint = OrderedConstraint(token_ids, vocab_length=len(tokenizer))
    ordered_constraints.append(constraint)

input_text = "She studies at "
inputs = tokenizer(input_text, return_tensors="pt")
if torch.cuda.is_available():
    inputs.input_ids = inputs.input_ids.cuda()

num_beams = 5
num_return_sequences = 3
assert num_beams >= num_return_sequences

logits_processors = LogitsProcessorList()

for constraint in ordered_constraints:
    logits_processors.append(OrderedConstraintLogitsProcessor(
        ordered_token_ids=constraint.ordered_token_ids    ))

start = time.time()
outputs = model.generate(
    inputs.input_ids,
    max_length=50,
    num_beams=num_beams,
    early_stopping=True,
    num_return_sequences=num_return_sequences,
    no_repeat_ngram_size=2,
    logits_processor=logits_processors,
)

decoded_outputs = []
for output in outputs:
    decoded_outputs.append(tokenizer.decode(output, skip_special_tokens=True))

end = time.time()
print(f"Time taken:  {(end - start)} sec")
for output in decoded_outputs:
    print(f"Generated output: {output}")
