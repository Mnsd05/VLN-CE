import os

import time
import warnings
from typing import List

import torch        
import tqdm
from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.rl.ddppo.algo.ddp_utils import is_slurm_batch_job

from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.base_il_trainer import BaseVLNCETrainer
from vlnce_baselines.common.recollection_dataset import (
    TeacherRecollectionDataset,
)
from vlnce_baselines.dagger_trainer import collate_fn
from vlnce_baselines.models.encoders.visual_encoders import (
    VlnResnetDepthEncoder,
    VlnRGBEncoder,
)
from vlnce_baselines.models.encoders.instruction_encoder import (
    InstructionEncoder,
)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf  # noqa: F401


@baseline_registry.register_trainer(name="recollect_trainer")
class RecollectTrainer(BaseVLNCETrainer):
    """A Teacher Forcing trainer that re-collects episodes from simulation
    rather than saving them all to disk. Included as starter code for the
    RxR-Habitat Challenge but can also train R2R agents.
    """

    supported_tasks: List[str] = ["VLN-v0"]

    def __init__(self, config=None):
        super().__init__(config)

    def _make_dirs(self) -> None:
        self._make_ckpt_dir()
        os.makedirs(
            os.path.dirname(
                self.config.IL.RECOLLECT_TRAINER.trajectories_file
            ),
            exist_ok=True,
        )
        if self.config.EVAL.SAVE_RESULTS:
            self._make_results_dir()

    def save_checkpoint(self, epoch: int, step_id: int) -> None:
        torch.save(
            obj={
                "state_dict": self.policy.state_dict(),
                "config": self.config,
                "optim_state": self.optimizer.state_dict(),
                "epoch": epoch,
                "step_id": step_id,
            },
            f=os.path.join(self.config.CHECKPOINT_FOLDER, f"ckpt.{epoch}.pth"),
        )

    def train(self) -> None:
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        self.config.defrost()
        self.config.TASK_CONFIG.TASK.NDTW.SPLIT = split
        self.config.TASK_CONFIG.TASK.SDTW.SPLIT = split
        self.config.IL.RECOLLECT_TRAINER.gt_path = (
            self.config.TASK_CONFIG.TASK.NDTW.GT_PATH
        )
        self.config.use_pbar = True  # not is_slurm_batch_job()
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
            -1
        )
        self.config.freeze()

        dataset = TeacherRecollectionDataset(self.config)
        # Init the instruction encoder
        instruction_encoder = InstructionEncoder().to(self.device)
        # Init the depth encoderer"
        depth_encoder = VlnResnetDepthEncoder(
            dataset.observation_space,
            output_size=self.config.MODEL.DEPTH_ENCODER.output_size,
            checkpoint=self.config.MODEL.DEPTH_ENCODER.ddppo_checkpoint,
            backbone=self.config.MODEL.DEPTH_ENCODER.backbone,
            trainable=self.config.MODEL.DEPTH_ENCODER.trainable,
            spatial_output=False,
        ).to(self.device)
        # Init the RGB visual encoder
        rgb_encoder = VlnRGBEncoder().to(self.device)
        
        diter = iter(
            torch.utils.data.DataLoader(
                dataset,
                batch_size=dataset.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                pin_memory=False,
                drop_last=True,
                num_workers=1,
            )
        )

        self._initialize_policy(
            self.config,
            self.config.IL.load_from_ckpt,
            observation_space=dataset.observation_space,
            action_space=dataset.action_space,
        )

        if self.config.IL.RECOLLECT_TRAINER.effective_batch_size > 0:
            assert (
                self.config.IL.RECOLLECT_TRAINER.effective_batch_size
                % self.config.IL.batch_size
                == 0
            ), (
                "Gradient accumulation: effective_batch_size"
                " should be a multiple of batch_size."
            )

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR,
            flush_secs=self.flush_secs,
            purge_step=0,
        ) as writer:

            batches_per_epoch = dataset.length // dataset.batch_size

            for epoch in range(self.start_epoch, self.config.IL.epochs):
                epoch_time = time.time()
                epoch_str = f"{epoch + 1}/{self.config.IL.epochs}"

                t = (
                    tqdm.trange(
                        batches_per_epoch, leave=False, dynamic_ncols=True
                    )
                    if self.config.use_pbar
                    else range(batches_per_epoch)
                )

                for batch_idx in t:
                    batch_time = time.time()
                    batch_str = f"{batch_idx + 1}/{batches_per_epoch}"

                    (
                        observations_batch,
                        prev_actions_batch,
                        not_done_masks,
                        corrected_actions_batch,
                        weights_batch,
                        padding_mask_encoder,
                        padding_mask_decoder,
                    ) = next(diter)

                    observations_batch = apply_obs_transforms_batch(
                        {
                            k: v.to(device=self.device, non_blocking=True)
                            for k, v in observations_batch.items()
                        },
                        dataset.obs_transforms,
                    )

                    prev_actions_batch = prev_actions_batch.to(
                        device=self.device, non_blocking=True
                    )
                    corrected_actions_batch = corrected_actions_batch.to(
                        device=self.device, non_blocking=True
                    )
                    padding_mask_encoder = padding_mask_encoder.to(
                        device=self.device, non_blocking=True
                    )
                    padding_mask_decoder = padding_mask_decoder.to(
                        device=self.device, non_blocking=True
                    )

                    instruction = observations_batch['instruction']
                    instruction_embedding = instruction_encoder(instruction)
                    rgb_embedding = rgb_encoder(observations_batch)
                    depth_embedding = depth_encoder(observations_batch)
                    # gradient accumulation
                    if (
                        self.config.IL.RECOLLECT_TRAINER.effective_batch_size
                        > 0
                    ):
                        loss_accumulation_scalar = (
                            self.config.IL.RECOLLECT_TRAINER.effective_batch_size
                            // self.config.IL.batch_size
                        )
                        step_grad = bool(
                            self.step_id % loss_accumulation_scalar
                        )
                    else:
                        loss_accumulation_scalar = 1
                        step_grad = True

                    loss = self._update_agent(
                        instruction_embedding,
                        rgb_embedding,
                        depth_embedding,
                        padding_mask_encoder,
                        padding_mask_decoder,
                        True,
                        corrected_actions_batch,
                        step_grad=step_grad,
                        loss_accumulation_scalar=loss_accumulation_scalar,
                    )

                    if self.config.use_pbar:
                        t.set_postfix(
                            {
                                "Epoch": epoch_str,
                                "Loss": round(loss, 4),
                            }
                        )
                    else:
                        logger.info(
                            f"[Epoch: {epoch_str}] [Batch: {batch_str}]"
                            + f" [BatchTime: {round(time.time() - batch_time, 2)}s]"
                            + f" [EpochTime: {round(time.time() - epoch_time)}s]"
                            + f" [Loss: {round(loss, 4)}]"
                        )
                    writer.add_scalar("loss", loss, self.step_id)
                    self.step_id += 1

                self.save_checkpoint(epoch, self.step_id)

            dataset.close_sims()
