import gc
import os
import random
import warnings
from collections import defaultdict

import lmdb
import msgpack_numpy
import numpy as np
import torch
import tqdm
from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs

from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.base_il_trainer import BaseVLNCETrainer
from vlnce_baselines.common.env_utils import construct_envs
from vlnce_baselines.common.utils import extract_instruction_tokens

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf  # noqa: F401


class ObservationsDict(dict):
    def pin_memory(self):
        for k, v in self.items():
            self[k] = v.pin_memory()

        return self


def collate_fn(batch):
    """Each sample in batch: (
        obs,
        prev_actions,
        oracle_actions,
        inflec_weight,
    )
    """

    def _pad_helper(t, max_len, fill_val=0):
        pad_amount = max_len - t.size(0)
        if pad_amount == 0:
            return t

        pad = torch.full_like(t[0:1], fill_val).expand(
            pad_amount, *t.size()[1:]
        )
        return torch.cat([t, pad], dim=0)

    transposed = list(zip(*batch))

    observations_batch = list(transposed[0])
    prev_actions_batch = list(transposed[1])
    corrected_actions_batch = list(transposed[2])
    weights_batch = list(transposed[3])
    B = len(prev_actions_batch)

    new_observations_batch = defaultdict(list)
    for sensor in observations_batch[0]:
        for bid in range(B):
            new_observations_batch[sensor].append(
                observations_batch[bid][sensor]
            )

    observations_batch = new_observations_batch

    max_traj_len = max(ele.size(0) for ele in prev_actions_batch)
    for bid in range(B):
        for sensor in observations_batch:
            observations_batch[sensor][bid] = _pad_helper(
                observations_batch[sensor][bid], max_traj_len, fill_val=1.0
            )

        prev_actions_batch[bid] = _pad_helper(
            prev_actions_batch[bid], max_traj_len
        )
        corrected_actions_batch[bid] = _pad_helper(
            corrected_actions_batch[bid], max_traj_len, fill_val=-1
        )
        weights_batch[bid] = _pad_helper(weights_batch[bid], max_traj_len)

    for sensor in observations_batch:
        observations_batch[sensor] = torch.stack(
            observations_batch[sensor], dim=0
        )

    observations_batch["instruction"] = observations_batch["instruction"][:, 0, :]
    prev_actions_batch = torch.stack(prev_actions_batch, dim=0)
    corrected_actions_batch = torch.stack(corrected_actions_batch, dim=0)
    weights_batch = torch.stack(weights_batch, dim=0)
    not_done_masks = torch.ones_like(
        corrected_actions_batch, dtype=torch.uint8
    )
    not_done_masks[0] = 0

    observations_batch = ObservationsDict(observations_batch)

    instructions = observations_batch['instruction']
    valid = (instructions != 0).float()
    padding_mask_encoder = valid.unsqueeze(2) * valid.unsqueeze(1)

    valid = (corrected_actions_batch != -1).float()
    padding_mask_decoder = valid.unsqueeze(2) * valid.unsqueeze(1)
    '''
    Shape:
    observations_batch: (B, T, ...)
    prev_actions_batch: (B, T)
    not_done_masks: (B)
    corrected_actions_batch: (B, T)
    weights_batch: (B, T)
    padding_mask_encoder: (B, 200, 200)
    padding_mask_decoder: (B, T, T)
    '''
    return (
        observations_batch,
        prev_actions_batch,
        not_done_masks,
        corrected_actions_batch,
        weights_batch,
        padding_mask_encoder,
        padding_mask_decoder,
    )


def _block_shuffle(lst, block_size):
    blocks = [lst[i : i + block_size] for i in range(0, len(lst), block_size)]
    random.shuffle(blocks)

    return [ele for block in blocks for ele in block]


class IWTrajectoryDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        lmdb_features_dir,
        use_iw,
        inflection_weight_coef=1.0,
        lmdb_map_size=1e9,
        batch_size=1,
    ):
        super().__init__()
        self.lmdb_features_dir = lmdb_features_dir
        self.lmdb_map_size = lmdb_map_size
        self.preload_size = batch_size * 100
        self._preload = []
        self.batch_size = batch_size

        if use_iw:
            self.inflec_weights = torch.tensor([1.0, inflection_weight_coef])
        else:
            self.inflec_weights = torch.tensor([1.0, 1.0])

        with lmdb.open(
            self.lmdb_features_dir,
            map_size=int(self.lmdb_map_size),
            readonly=True,
            lock=False,
        ) as lmdb_env:
            self.length = lmdb_env.stat()["entries"]

    def _load_next(self):
        if len(self._preload) == 0:
            if len(self.load_ordering) == 0:
                raise StopIteration

            new_preload = []
            lengths = []
            with lmdb.open(
                self.lmdb_features_dir,
                map_size=int(self.lmdb_map_size),
                readonly=True,
                lock=False,
            ) as lmdb_env, lmdb_env.begin(buffers=True) as txn:
                for _ in range(self.preload_size):
                    if len(self.load_ordering) == 0:
                        break

                    new_preload.append(
                        msgpack_numpy.unpackb(
                            txn.get(str(self.load_ordering.pop()).encode()),
                            raw=False,
                        )
                    )

                    lengths.append(len(new_preload[-1][0]))

            sort_priority = list(range(len(lengths)))
            random.shuffle(sort_priority)

            sorted_ordering = list(range(len(lengths)))
            sorted_ordering.sort(key=lambda k: (lengths[k], sort_priority[k]))

            for idx in _block_shuffle(sorted_ordering, self.batch_size):
                self._preload.append(new_preload[idx])

        return self._preload.pop()

    def __next__(self):
        obs, prev_actions, oracle_actions = self._load_next()

        for k, v in obs.items():
            obs[k] = torch.from_numpy(np.copy(v))

        prev_actions = torch.from_numpy(np.copy(prev_actions))
        oracle_actions = torch.from_numpy(np.copy(oracle_actions))

        inflections = torch.cat(
            [
                torch.tensor([1], dtype=torch.long),
                (oracle_actions[1:] != oracle_actions[:-1]).long(),
            ]
        )

        return (
            obs,
            prev_actions,
            oracle_actions,
            self.inflec_weights[inflections],
        )

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start = 0
            end = self.length
        else:
            per_worker = int(np.ceil(self.length / worker_info.num_workers))

            start = per_worker * worker_info.id
            end = min(start + per_worker, self.length)

        # Reverse so we can use .pop()
        self.load_ordering = list(
            reversed(
                _block_shuffle(list(range(start, end)), self.preload_size)
            )
        )

        return self


@baseline_registry.register_trainer(name="dagger")
class DaggerTrainer(BaseVLNCETrainer):
    def __init__(self, config=None):
        self.lmdb_features_dir = config.IL.DAGGER.lmdb_features_dir.format(
            split=config.TASK_CONFIG.DATASET.SPLIT
        )
        super().__init__(config)

    def _make_dirs(self) -> None:
        self._make_ckpt_dir()
        os.makedirs(self.lmdb_features_dir, exist_ok=True)
        if self.config.EVAL.SAVE_RESULTS:
            self._make_results_dir()

    def _update_dataset(self, data_it):
        def pad_obs_batch(envs_obs, padding_value=0.0):
            """
            envs_obs: List[List[Dict[sensor, tensor]]]
                    Length B, each element is a sequence of length Ti
            """
            B = len(envs_obs)
            max_len = max(len(seq) for seq in envs_obs)
            sensors = list(envs_obs[0][0].keys())

            # Prepare storage
            padded = {s: [] for s in sensors}

            for b in range(B):
                seq = envs_obs[b]
                T = len(seq)

                # For each sensor, create padded tensor
                for s in sensors:
                    # stack time dimension of this env
                    obs_seq = torch.stack([step[s] for step in seq], dim=0)  # (T, ...)

                    # length difference
                    pad_len = max_len - T
                    if pad_len > 0:
                        pad_shape = (pad_len, *obs_seq.shape[1:])
                        pad = torch.full(pad_shape, padding_value, dtype=obs_seq.dtype, device=obs_seq.device)
                        obs_seq = torch.cat([obs_seq, pad], dim=0)  # → (max_len, ...)

                    padded[s].append(obs_seq)

            # Stack across batch → (B, max_len, ...)
            for s in sensors:
                padded[s] = torch.stack(padded[s], dim=0)

            return padded

        def _pad_helper(t, max_len, fill_val=0):
            pad_amount = max_len - t.size(0)
            if pad_amount == 0:
                return t

            pad = torch.full_like(t[0:1], fill_val).expand(
                pad_amount, *t.size()[1:]
            )
            return torch.cat([t, pad], dim=0)


        if torch.cuda.is_available():
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

        envs = construct_envs(self.config, get_env_class(self.config.ENV_NAME))
        expert_uuid = self.config.IL.DAGGER.expert_policy_sensor_uuid

        # Initialize previous actions
        prev_actions = torch.zeros(
            envs.num_envs,
            1,
            device=self.device,
            dtype=torch.long,
        )

        # Initialize not done masks
        not_done_masks = torch.zeros(
            envs.num_envs, 1, dtype=torch.uint8, device=self.device
        )

        # Reset environment
        observations = envs.reset()
        
        # Extract instruction tokens
        observations = extract_instruction_tokens(
            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        )
        
        # Batch observations
        batch = batch_obs(observations, self.device)
        
        # Apply observation transformations
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        # Buffer for storing episodes
        episodes = [[] for _ in range(envs.num_envs)]
        # Buffer for skipping episodes if trajectory is not valid
        skips = [False for _ in range(envs.num_envs)]
        # Populate dones with False initially
        # This is used to mark the end of an episode
        dones = [False for _ in range(envs.num_envs)]

        # https://arxiv.org/pdf/1011.0686.pdf
        # Theoretically, any beta function is fine so long as it converges to
        # zero as data_it -> inf. The paper suggests starting with beta = 1 and
        # exponential decay.
        p = self.config.IL.DAGGER.p
        # in Python 0.0 ** 0.0 == 1.0, but we want 0.0
        beta = 0.0 if p == 0.0 else p ** data_it

        # ensure_unique_episodes = beta == 1.0
        
        # Function to register forward hooks
        def hook_builder(tgt_tensor):
            def hook(m, i, o):
                tgt_tensor.set_(o.cpu())

            return hook

        # Register forward hooks for RGB and depth encoders
        rgb_features = None
        rgb_hook = None
        if not self.config.MODEL.RGB_ENCODER.trainable:
            rgb_features = torch.zeros((1,), device="cpu")
            rgb_hook = self.policy.net.rgb_encoder.down_project.register_forward_hook(
                hook_builder(rgb_features)
            )

        depth_features = None
        depth_hook = None
        if not self.config.MODEL.DEPTH_ENCODER.trainable:
            depth_features = torch.zeros((1,), device="cpu")
            depth_hook = self.policy.net.depth_encoder.visual_encoder.register_forward_hook(
                hook_builder(depth_features)
            )

        collected_eps = 0
        ep_ids_collected = None
        # if ensure_unique_episodes:
        #     ep_ids_collected = {
        #         ep.episode_id for ep in envs.current_episodes()
        #     }

        # Start collecting episodes
        with tqdm.tqdm(
            total=self.config.IL.DAGGER.update_size, dynamic_ncols=True
        ) as pbar, lmdb.open(
            self.lmdb_features_dir,
            map_size=int(self.config.IL.DAGGER.lmdb_map_size),
        ) as lmdb_env, torch.no_grad():
            start_id = lmdb_env.stat()["entries"]
            txn = lmdb_env.begin(write=True)

            while collected_eps < self.config.IL.DAGGER.update_size:
                current_episodes = None
                envs_to_pause = None
                # if ensure_unique_episodes:
                #     envs_to_pause = []
                #     current_episodes = envs.current_episodes()

                for i in range(envs.num_envs):
                    # if an episode is done and not skipped, write it to lmdb
                    if dones[i] and not skips[i]:
                        ep = episodes[i]
                        traj_obs = batch_obs(
                            [step[0] for step in ep],
                            device=torch.device("cpu"),
                        )
                        del traj_obs[expert_uuid]
                        for k, v in traj_obs.items():
                            traj_obs[k] = v.numpy()
                            if self.config.IL.DAGGER.lmdb_fp16:
                                traj_obs[k] = traj_obs[k].astype(np.float16)

                        transposed_ep = [
                            traj_obs,
                            np.array([step[1] for step in ep], dtype=np.int64),
                            np.array([step[2] for step in ep], dtype=np.int64),
                        ]
                        txn.put(
                            str(start_id + collected_eps).encode(),
                            msgpack_numpy.packb(
                                transposed_ep, use_bin_type=True
                            ),
                        )

                        pbar.update()
                        collected_eps += 1

                        if (
                            collected_eps
                            % self.config.IL.DAGGER.lmdb_commit_frequency
                        ) == 0:
                            txn.commit()
                            txn = lmdb_env.begin(write=True)

                        # if ensure_unique_episodes:
                        #     if (
                        #         current_episodes[i].episode_id
                        #         in ep_ids_collected
                        #     ):
                        #         envs_to_pause.append(i)
                        #     else:
                        #         ep_ids_collected.add(
                        #             current_episodes[i].episode_id
                        #         )

                    if dones[i]:
                        episodes[i] = []

                    envs_obs = []
                    envs_actions = []
                    max_len = 0
                    for i in range(envs.num_envs):
                        ep = episodes[i]
                        envs_obs.append([step[0] for step in ep] + [observations[i]])
                        envs_actions.append([step[1] for step in ep])
                        max_len = max(max_len, len(ep))
                    
                    sensors = list(envs_obs[0][0].keys())
                    max_len = max(len(seq) for seq in envs_obs)
                    B = len(envs_obs)
                    history_observations = pad_obs_batch(envs_obs)
                    history_observations = {
                        k: v.to(self.device) for k, v in history_observations.items()
                    }
                    history_observations["instruction"] = history_observations["instruction"][:, 0, :]
                    valid = (history_observations["instruction"]!= -1).float()
                    padding_mask_encoder = valid.unsqueeze(2) * valid.unsqueeze(1)

                    for i in range(envs.num_envs):
                        envs_actions[i] = _pad_helper(envs_actions[i], max_len, fill_val=-1)
                    actions_batch = torch.stack(envs_actions, dim=0)
                    valid = (actions_batch != -1).float()
                    padding_mask_decoder = valid.unsqueeze(2) * valid.unsqueeze(1)
                # if ensure_unique_episodes:
                #     (
                #         envs,
                #         rnn_states,
                #         not_done_masks,
                #         prev_actions,
                #         batch,
                #         _,
                #     ) = self._pause_envs(
                #         envs_to_pause,
                #         envs,
                #         rnn_states,
                #         not_done_masks,
                #         prev_actions,
                #         batch,
                #     )
                #     if envs.num_envs == 0:
                #         break

                # Perform actions
                actions = self.policy.act(
                    history_observations, padding_mask_encoder, padding_mask_decoder, True,
                    deterministic=False,
                )
                # Determine whether to use expert action or not
                actions = torch.where(
                    torch.rand_like(actions, dtype=torch.float) < beta,
                    batch[expert_uuid].long(),
                    actions,
                )

                for i in range(envs.num_envs):
                    if rgb_features is not None:
                        observations[i]["rgb_features"] = rgb_features[i]
                        del observations[i]["rgb"]

                    if depth_features is not None:
                        observations[i]["depth_features"] = depth_features[i]
                        del observations[i]["depth"]
                    # Append the observation, previous action, and expert action to the episode buffer
                    episodes[i].append(
                        (
                            observations[i],
                            prev_actions[i].item(),
                            batch[expert_uuid][i].item(),
                        )
                    )

                skips = batch[expert_uuid].long() == -1
                actions = torch.where(
                    skips, torch.zeros_like(actions), actions
                )
                skips = skips.squeeze(-1).to(device="cpu", non_blocking=True)
                prev_actions.copy_(actions)
                
                # Prepare the next observation
                outputs = envs.step([a[0].item() for a in actions])
                observations, _, dones, _ = [list(x) for x in zip(*outputs)]
                # Extract instruction tokens
                observations = extract_instruction_tokens(
                    observations,
                    self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
                )
                batch = batch_obs(observations, self.device)
                batch = apply_obs_transforms_batch(batch, self.obs_transforms)

                not_done_masks = torch.tensor(
                    [[0] if done else [1] for done in dones],
                    dtype=torch.uint8,
                    device=self.device,
                )

            txn.commit()

        envs.close()
        envs = None

        if rgb_hook is not None:
            rgb_hook.remove()
        if depth_hook is not None:
            depth_hook.remove()

    def train(self) -> None:
        """Main method for training DAgger."""
        if self.config.IL.DAGGER.preload_lmdb_features:
            try:
                lmdb.open(self.lmdb_features_dir, readonly=True)
            except lmdb.Error as err:
                logger.error(
                    "Cannot open database for teacher forcing preload."
                )
                raise err
        else:
            with lmdb.open(
                self.lmdb_features_dir,
                map_size=int(self.config.IL.DAGGER.lmdb_map_size),
            ) as lmdb_env, lmdb_env.begin(write=True) as txn:
                txn.drop(lmdb_env.open_db())

        EPS = self.config.IL.DAGGER.expert_policy_sensor
        if EPS not in self.config.TASK_CONFIG.TASK.SENSORS:
            self.config.TASK_CONFIG.TASK.SENSORS.append(EPS)

        self.config.defrost()

        # if doing teacher forcing, don't switch the scene until it is complete
        if self.config.IL.DAGGER.p == 1.0:
            self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
                -1
            )
        self.config.freeze()

        observation_space, action_space = self._get_spaces(self.config)

        self._initialize_policy(
            self.config,
            self.config.IL.load_from_ckpt,
            observation_space=observation_space,
            action_space=action_space,
        )

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR,
            flush_secs=self.flush_secs,
            purge_step=0,
        ) as writer:
            for dagger_it in range(self.config.IL.DAGGER.iterations):
                step_id = 0
                if not self.config.IL.DAGGER.preload_lmdb_features:
                    self._update_dataset(
                        dagger_it + (1 if self.config.IL.load_from_ckpt else 0)
                    )

                if torch.cuda.is_available():
                    with torch.cuda.device(self.device):
                        torch.cuda.empty_cache()
                gc.collect()

                dataset = IWTrajectoryDataset(
                    self.lmdb_features_dir,
                    self.config.IL.use_iw,
                    inflection_weight_coef=self.config.IL.inflection_weight_coef,
                    lmdb_map_size=self.config.IL.DAGGER.lmdb_map_size,
                    batch_size=self.config.IL.batch_size,
                )
                diter = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=self.config.IL.batch_size,
                    shuffle=False,
                    collate_fn=collate_fn,
                    pin_memory=False,
                    drop_last=True,  # drop last batch if smaller
                    num_workers=3,
                )

                # AuxLosses.activate()
                for epoch in tqdm.trange(
                    self.config.IL.epochs, dynamic_ncols=True
                ):
                    for batch in tqdm.tqdm(
                        diter,
                        total=dataset.length // dataset.batch_size,
                        leave=False,
                        dynamic_ncols=True,
                    ):
                        (
                            observations_batch,
                            prev_actions_batch,
                            not_done_masks,
                            corrected_actions_batch,
                            weights_batch,
                            padding_mask_encoder,
                            padding_mask_decoder,
                        ) = batch

                        observations_batch = {
                            k: v.to(
                                device=self.device,
                                dtype=torch.float32,
                                non_blocking=True,
                            )
                            for k, v in observations_batch.items()
                        }
                        padding_mask_encoder = padding_mask_encoder.to(
                            device=self.device, non_blocking=True
                        )
                        padding_mask_decoder = padding_mask_decoder.to(
                            device=self.device, non_blocking=True
                        )

                        loss = self._update_agent(
                            observations_batch,
                            padding_mask_encoder,
                            padding_mask_decoder,
                            True,
                            corrected_actions_batch.to(
                                device=self.device, non_blocking=True
                            ),
                        )

                        logger.info(f"train_loss: {loss}")
                        logger.info(f"Batches processed: {step_id}.")
                        logger.info(
                            f"On DAgger iter {dagger_it}, Epoch {epoch}."
                        )
                        writer.add_scalar(
                            f"train_loss_iter_{dagger_it}", loss, step_id
                        )
                        step_id += 1  # noqa: SIM113

                    self.save_checkpoint(
                        f"ckpt.{dagger_it * self.config.IL.epochs + epoch}.pth"
                    )