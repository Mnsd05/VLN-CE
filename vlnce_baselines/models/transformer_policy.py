# Reference: https://github.com/karpathy/nanoGPT
import torch
from torch import Tensor
import math
import torch.nn as nn
import torch.nn.functional as F
from gym import Space
from habitat import Config
from habitat_baselines.common.baseline_registry import BaselineRegistry
from habitat_baselines.rl.ppo.policy import Net
from vlnce_baselines.models.policy import ILPolicy
from habitat import logger

@BaselineRegistry.register_policy
class TransformerPolicy(ILPolicy):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        model_config: Config,
    ):
        super().__init__(
            TransformerNet(
                observation_space=observation_space,
                model_config=model_config,
                num_actions=action_space.n,
            ),
            action_space.n,
        )

    @classmethod
    def from_config(
        cls, config: Config, observation_space: Space, action_space: Space
    ):
        config.defrost()
        config.MODEL.TORCH_GPU_ID = config.TORCH_GPU_ID
        config.freeze()

        return cls(
            observation_space=observation_space,
            action_space=action_space,
            model_config=config.MODEL,
        )


class TransformerNet(Net):
    """A baseline sequence to sequence network that performs single modality
    encoding of the instruction, RGB, and depth observations. These encodings
    are concatentated and fed to an RNN. Finally, a distribution over discrete
    actions (FWD, L, R, STOP) is produced.
    """

    def __init__(
        self, observation_space: Space, model_config: Config, num_actions: int):
        super().__init__()
        self.model_config = model_config
        self.instruction_down_project = nn.Linear(model_config.INSTRUCTION_ENCODER.hidden_size, model_config.INSTRUCTION_ENCODER.output_size)
        self.rgb_down_project = nn.Linear(model_config.RGB_ENCODER.hidden_size, model_config.RGB_ENCODER.output_size)
        self.depth_down_project = nn.Linear(model_config.DEPTH_ENCODER.hidden_size, model_config.DEPTH_ENCODER.output_size)
        self.transformer = CustomTransformer(model_config.TRANSFORMER.d_encoder_in,
        model_config.TRANSFORMER.d_decoder_in,
        model_config.TRANSFORMER.num_heads,
        model_config.TRANSFORMER.dropout_p,
        model_config.TRANSFORMER.num_blocks)

        self.train()

    @property
    def output_size(self):
        return self.model_config.TRANSFORMER.d_decoder_in

    @property
    def num_recurrent_layers(self):
        return 0
    @property
    def is_blind(self):
        return False

    def forward(self, instruction_embedding, rgb_embedding, depth_embedding, padding_mask_encoder, padding_mask_decoder, isCausal):
        depth_embedding = self.depth_down_project(depth_embedding)
        rgb_embedding = self.rgb_down_project(rgb_embedding)
        instruction_embedding = self.instruction_down_project(instruction_embedding)

        visual_embedding = torch.cat(
            [depth_embedding, rgb_embedding], dim=2
        )
        return self.transformer(instruction_embedding, visual_embedding, padding_mask_encoder, padding_mask_decoder, isCausal)



class DotProductAttention(nn.Module):
    def __init__(self, key_dimension: int, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.scale = 1.0 / ((key_dimension) ** 0.5)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout_p)

    def forward(
        self, Q: Tensor, K: Tensor, V: Tensor, padding_mask: Tensor,  isCausal: bool = False
    ) -> Tensor:
        """Scaled dot-product attention with an optional mask.
        Args:
            query: [Batch, H, T1, d_in]
            key: [Batch, H, T2, d_in]
            value: [Batch, H, T2, d_in],
            padding_mask: [Batch, H, T1, T2],
            isCausal: bool, whether to apply causal mask
        Returns:
            tensor of dimension [Batch, H, T1, d_in]
        """
        # Shape (B, H, T1, T2)
        QKT = torch.matmul(Q, K.permute(0, 1, 3, 2)) * self.scale
        B = QKT.shape[0]
        T1 = QKT.shape[2]
        T2 = QKT.shape[3]
        padding_mask = padding_mask.to(QKT.device)
        if isCausal:
            # Causal mask is for self attention only
            causal_mask = torch.tril(torch.ones((T1, T1), device=QKT.device))
            mask = padding_mask.unsqueeze(1) * causal_mask
        else:
            mask = padding_mask.unsqueeze(1)
        QKT = QKT.masked_fill(mask == 0, -float('1e4'))
        # Shape (B, H, T1, T2)
        attn = self.softmax(QKT)
        attn = self.dropout(attn)
        # Shape (B, H, T1, d_in)
        return torch.matmul(attn, V)

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        num_heads: int, 
        dropout_p: float = 0.0,
    ) -> None:
        """The residual connection of Vaswani et al is not used here. The
        residual makes sense if self-attention is being used.
        Args:
            d_in (int): dimension of the input vector
            num_heads (int): number of attention heads
        """ 
        super().__init__()
        self.num_heads = num_heads
        self.q_linear = nn.Linear(d_in, d_in, bias=False)
        self.k_linear = nn.Linear(d_in, d_in, bias=False)
        self.v_linear = nn.Linear(d_in, d_in, bias=False)

        self.attn = DotProductAttention(d_in // num_heads, dropout_p)

    def forward(
        self, x: Tensor, padding_mask: Tensor, 
        isCausal: bool = False
    ) -> Tensor:
        """Performs multihead scaled dot product attention for some Q, K, V.
        Args:
            x: [Batch, T1, d_in]
            padding_mask: [Batch, T1, T2]
            isCausal: bool, whether to apply causal mask
        """
        # Shape (B, T1, d_in)
        Q = self.q_linear(x)
        # Shape (B, T2, d_in)
        K = self.k_linear(x)
        # Shape (B, T2, d_in)
        V = self.v_linear(x)

        # Shape (B, H, T1, d_in / H)
        Q = Q.view(Q.shape[0], Q.shape[1], self.num_heads, Q.shape[2] // self.num_heads).transpose(1, 2)
        # Shape (B, H, T2, d_in / H)
        K = K.view(K.shape[0], K.shape[1], self.num_heads, K.shape[2] // self.num_heads).transpose(1, 2)
        # Shape (B, H, T2, d_in / H)
        V = V.view(V.shape[0], V.shape[1], self.num_heads, V.shape[2] // self.num_heads).transpose(1, 2)
        x = self.attn(Q, K, V, padding_mask, isCausal)
        # Shape (B, T1, d_in)
        x = x.transpose(1, 2)
        x = x.contiguous().view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        return x

class MultiHeadCrossAttention(nn.Module):
    def __init__(
        self,
        d_q_in: int,
        d_kv_in: int,
        num_heads: int,
        dropout_p: float = 0.0
    ) -> None:
        """
        Args:
            d_in (int): dimension of the input vector
            num_heads (int): number of attention heads
        """
        super().__init__()
        self.num_heads = num_heads
        self.q_linear = nn.Linear(d_q_in, d_q_in, bias=False)
        self.k_linear = nn.Linear(d_kv_in, d_q_in, bias=False)
        self.v_linear = nn.Linear(d_kv_in, d_q_in, bias=False)

        self.attn = DotProductAttention(d_q_in // num_heads, dropout_p)
    def forward(
        self, x: Tensor, encoder_out: Tensor, padding_mask: Tensor, 
        isCausal: bool = False
    ) -> Tensor:
        """Performs multihead scaled dot product attention for some Q, K, V.
        Args:
            x: [Batch, T1, d_q_in]
            encoder_out: [Batch, T2, d_kv_in]
            padding_mask: [Batch, T1, T2]
            isCausal: bool, whether to apply causal mask
        """
        # Shape (B, T1, d_in)
        Q = self.q_linear(x)
        # Shape (B, T, d_in)
        K = self.k_linear(encoder_out)
        # Shape (B, T, d_in)
        V = self.v_linear(encoder_out)

        # Shape (B, H, T, d_in)
        Q = Q.view(Q.shape[0], Q.shape[1], self.num_heads, Q.shape[2] // self.num_heads).transpose(1, 2)
        # Shape (B, H, T, d_in)
        K = K.view(K.shape[0], K.shape[1], self.num_heads, K.shape[2] // self.num_heads).transpose(1, 2)
        # Shape (B, H, T, d_in)
        V = V.view(V.shape[0], V.shape[1], self.num_heads, V.shape[2] // self.num_heads).transpose(1, 2)
        x = self.attn(Q, K, V, padding_mask, isCausal)
        # Shape (B, T, d_in)
        x = x.transpose(1, 2)
        x = x.contiguous().view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        return x

class MLP(nn.Module):
    def __init__(self, d_in: int, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_in * 3)
        self.fc2 = nn.Linear(d_in * 3, d_in)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, d_encoder_in, num_heads, dropout_p) -> None:
        super().__init__()
        assert d_encoder_in % num_heads == 0, "d_encoder_in must be divisible by num_heads"
        self.attn = MultiHeadSelfAttention(d_encoder_in, num_heads, dropout_p)
        self.mlp = MLP(d_encoder_in, dropout_p)
        self.layer_norm1 = nn.LayerNorm(d_encoder_in)
        self.layer_norm2 = nn.LayerNorm(d_encoder_in)

    def forward(self, x: Tensor, padding_mask: Tensor) -> Tensor:
        x = x + self.attn(self.layer_norm1(x), padding_mask, isCausal=False)
        x = x + self.mlp(self.layer_norm2(x))
        return x

class Encoder(nn.Module):
    def __init__(self, d_encoder_in, num_heads, dropout_p, num_blocks) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([EncoderBlock(d_encoder_in, num_heads, dropout_p) for _ in range(num_blocks)])
        self.layer_norm = nn.LayerNorm(d_encoder_in)

        max_len = 200
        d_model = d_encoder_in
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor, padding_mask: Tensor) -> Tensor:
        x = x + self.pe[:, :x.size(1)]
        for block in self.blocks:
            x = block(x, padding_mask)
        x = self.layer_norm(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_decoder_in, d_encoder_in, num_heads, dropout_p) -> None:
        super().__init__()
        assert d_decoder_in % num_heads == 0, "d_decoder_in must be divisible by num_heads"
        self.attn1 = MultiHeadSelfAttention(d_decoder_in, num_heads, dropout_p)
        self.attn2 = MultiHeadCrossAttention(d_decoder_in, d_encoder_in, num_heads, dropout_p)
        self.mlp = MLP(d_decoder_in, dropout_p)
        self.layer_norm1 = nn.LayerNorm(d_decoder_in)
        self.layer_norm2 = nn.LayerNorm(d_decoder_in)
        self.layer_norm3 = nn.LayerNorm(d_decoder_in)

    def forward(self, x: Tensor, encoder_out: Tensor, padding_mask_encoder: Tensor, padding_mask_decoder: Tensor, isCausal: bool = False) -> Tensor:
        x = x + self.attn1(self.layer_norm1(x), padding_mask_decoder, isCausal = True)
        B, T1, T1 = padding_mask_encoder.shape
        B, T2, T2 = padding_mask_decoder.shape
        padding_mask_encoder = padding_mask_encoder[:, 0:1, :]
        padding_mask_encoder = padding_mask_encoder.expand(B, T2, T1)
        x = x + self.attn2(self.layer_norm2(x), encoder_out, padding_mask_encoder, isCausal = False)
        x = x + self.mlp(self.layer_norm3(x))
        return x


class Decoder(nn.Module):
    def __init__(self, d_decoder_in, d_encoder_in, num_heads, dropout_p, num_blocks) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([DecoderBlock(d_decoder_in, d_encoder_in, num_heads, dropout_p) for _ in range(num_blocks)])
        self.layer_norm = nn.LayerNorm(d_decoder_in)
        max_len = 500
        d_model = d_decoder_in
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    

    def forward(self, x: Tensor, encoder_out: Tensor, padding_mask_encoder: Tensor, padding_mask_decoder: Tensor, isCausal: bool = False) -> Tensor:
        x = x + self.pe[:, :x.size(1)]
        for block in self.blocks:
            x = block(x, encoder_out, padding_mask_encoder, padding_mask_decoder, isCausal)
        x = self.layer_norm(x)
        return x

class CustomTransformer(nn.Module):
    def __init__(self, d_encoder_in, d_decoder_in, num_heads, dropout_p, num_blocks) -> None:
        super().__init__()
        self.encoder = Encoder(d_encoder_in, num_heads, dropout_p, num_blocks)
        self.decoder = Decoder(d_decoder_in, d_encoder_in, num_heads, dropout_p, num_blocks)
    
    def forward(self, instruction: Tensor, visual: Tensor, padding_mask_encoder: Tensor, padding_mask_decoder: Tensor, isCausal: bool = False) -> Tensor:
        instruction = self.encoder(instruction, padding_mask_encoder)
        features = self.decoder(visual, instruction, padding_mask_encoder, padding_mask_decoder, isCausal)
        return features
