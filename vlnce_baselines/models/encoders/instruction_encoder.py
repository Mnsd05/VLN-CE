from transformers import AutoModel
import torch
import torch.nn as nn
from torch import Tensor

class InstructionEncoder(nn.Module):
    def __init__(self) -> None:
        """An encoder that uses CLIP to encode instructions. The output
        is the final hidden state of all the tokens.
        """
        super().__init__()

        model = AutoModel.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True)
        self.text_model = model.text_model.transformer
        # Encoder model should be frozen
        for param in self.text_model.parameters():
            param.requires_grad = False
        
    def forward(self, tokens: Tensor) -> Tensor:
        attention_mask = (tokens != 0).long()
        outputs = self.text_model(
            input_ids=tokens,
            attention_mask=attention_mask
        )
        return outputs.last_hidden_state