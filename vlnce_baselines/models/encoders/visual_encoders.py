from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from gym import Space, spaces
from habitat.core.simulator import Observations
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder
from torch import Tensor
from transformers import AutoModel
import torchvision.transforms as Transformation
from vlnce_baselines.common.utils import single_frame_box_shape
import copy

class VlnResnetDepthEncoder(nn.Module):
    def __init__(
        self,
        observation_space: Space,
        output_size: int = 128,
        checkpoint: str = "NONE",
        backbone: str = "resnet50",
        resnet_baseplanes: int = 32,
        normalize_visual_inputs: bool = False,
        trainable: bool = False,
        spatial_output: bool = False,
    ) -> None:
        super().__init__()

        self.visual_encoder = ResNetEncoder(
            spaces.Dict(
                {
                    "depth": single_frame_box_shape(
                        observation_space.spaces["depth"]
                    )
                }
            ),
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
        )

        for param in self.visual_encoder.parameters():
            param.requires_grad_(trainable)

        if checkpoint != "NONE":
            ddppo_weights = torch.load(checkpoint)

            weights_dict = {}
            for k, v in ddppo_weights["state_dict"].items():
                split_layer_name = k.split(".")[2:]
                if split_layer_name[0] != "visual_encoder":
                    continue

                layer_name = ".".join(split_layer_name[1:])
                weights_dict[layer_name] = v

            del ddppo_weights
            self.visual_encoder.load_state_dict(weights_dict, strict=True)

    def forward(self, observations: Observations) -> Tensor:
        """
        Args:
            observations: [BATCH, HEIGHT, WIDTH, CHANNEL]
        Returns:
            [BATCH, OUTPUT_SIZE]
        """
        if "depth_features" in observations:
            x = observations["depth_features"]
        else:
            copy_observations =  observations.copy()
            B, T, H, W, C = copy_observations['depth'].shape
            copy_observations['depth'] = copy_observations['depth'].reshape(B * T, H, W, C)
            x = self.visual_encoder(copy_observations)
            x = x.reshape(B, T, -1)
        return x

class VlnRGBEncoder(nn.Module):
    def __init__(
        self
    ) -> None:
        super().__init__()
        model = AutoModel.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True)
        self.vision_model = model.vision_model
        # Encoder model should be frozen
        for param in self.vision_model.parameters():
            param.requires_grad = False

    def forward(self, observations: Observations) -> Tensor:
        """
        Args:
            observations: [BATCH, HEIGHT, WIDTH, CHANNEL]
        Returns:
            [BATCH, OUTPUT_SIZE]
        """

        if "rgb_features" in observations:
            x = observations["rgb_features"]
        else:
            x = observations['rgb'].float() / 255.0
            x = x.permute(0,1,4,2,3)
            B, T, C, H, W = x.shape
            x = x.reshape(B * T, C, H, W)
            transform = Transformation.Compose([
                Transformation.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711])
            ])
            x = self.vision_model(transform(x))
            x = x.reshape(B, T, -1)
        return x