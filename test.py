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

# instructions = [
#   [5, 7, 9, 0, 0],      # padding at positions 3,4
#   [4, 1, 0, 0, 0],      # padding at positions 2,3,4
#   [8, 2, 3, 6, 1],      # no padding
# ]

# instructions = torch.tensor(instructions)
# valid = (instructions != 0).float()
# mask = valid.unsqueeze(2) * valid.unsqueeze(1)

# print(mask)

from PIL import Image
import torchvision.transforms as T
import torch
from torch import Tensor
from transformers import AutoModel

# Load image
image_path = '/mnt/c/Users/thang/Downloads/trump.jpg'
image = Image.open(image_path).convert('RGB')  # Ensure 3 channels

# Transform to tensor suitable for CLIP
transform = T.Compose([
    T.Resize((224, 224)),  # Resize to model input
    T.ToTensor(),           # Convert to [0,1] tensor
    T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711])
])

image_tensor = transform(image).unsqueeze(0)  # Add batch dimension [1, 3, 224, 224]
model = AutoModel.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True) 
result = model.vision_model(image_tensor)       

text = model.encode_text("Donald Trump")

# print cosine similarity
print(torch.nn.functional.cosine_similarity(result, torch.as_tensor(text), dim=1))
