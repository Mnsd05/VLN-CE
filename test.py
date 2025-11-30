# from transformers import AutoModel, AutoTokenizer
import torch
# # Initialize the model
# model = AutoModel.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True)

# result = tokenizer("A blue cat")['input_ids']
# # convert result to tensor
# result = torch.tensor(result)
# result = result.reshape(1, -1)
# print(model.text_model.transformer(input_ids=result).last_hidden_state)

instructions = [
  [5, 7, 9, 0, 0],      # padding at positions 3,4
  [4, 1, 0, 0, 0],      # padding at positions 2,3,4
  [8, 2, 3, 6, 1],      # no padding
]

instructions = torch.tensor(instructions)
valid = (instructions != 0).float()
mask = valid.unsqueeze(2) * valid.unsqueeze(1)

print(mask)

