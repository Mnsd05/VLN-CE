import abc
from typing import Any

from habitat_baselines.rl.ppo.policy import Policy
from habitat_baselines.utils.common import CategoricalNet
from habitat import logger
from vlnce_baselines.models.utils import CustomFixedCategorical
import torch


class ILPolicy(Policy, metaclass=abc.ABCMeta):
    def __init__(self, net, dim_actions):
        """Defines an imitation learning policy as having functions act() and
        build_distribution().
        """
        super(Policy, self).__init__()
        self.net = net
        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        instruction_embedding,
        rgb_embedding,
        depth_embedding,
        prev_actions,
        padding_mask_encoder,
        padding_mask_decoder,
        isCausal,
        lengths,
        deterministic=False,
    ):
        features = self.net(
            instruction_embedding,
            rgb_embedding,
            depth_embedding,
            prev_actions,
            padding_mask_encoder,
            padding_mask_decoder,
            isCausal,
        )
        pos = [l - 1 for l in lengths]
        features = features[torch.arange(features.size(0)), pos]
        distribution = self.action_distribution(features)
        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()
        return action

    # def get_value(self, *args: Any, **kwargs: Any):
    #     raise NotImplementedError

    # def evaluate_actions(self, *args: Any, **kwargs: Any):
    #     raise NotImplementedError

    def build_distribution(
        self, instruction_embedding, rgb_embedding, depth_embedding, prev_actions, padding_mask_encoder, padding_mask_decoder, isCausal
    ) -> CustomFixedCategorical:
        features = self.net(
            instruction_embedding,
            rgb_embedding,
            depth_embedding,
            prev_actions,
            padding_mask_encoder,
            padding_mask_decoder,
            isCausal,
        )
        return self.action_distribution(features)
