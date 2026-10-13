# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# 261013 hw: wandb

import os

import wandb
from torch.utils.tensorboard import SummaryWriter


class WandbSummaryWriter(SummaryWriter):
    """TensorBoard summary writer that also logs scalars to wandb."""

    def __init__(self, log_dir: str, cfg: dict, env_cfg, flush_secs: int = 10):
        super().__init__(log_dir, flush_secs=flush_secs)
        wandb.init(
            project=cfg["wandb_project"],
            name=os.path.basename(log_dir),
            dir=log_dir,
            config={"runner_cfg": cfg, "env_cfg": env_cfg.to_dict()},
        )

    def add_scalar(self, tag, scalar_value, global_step=None, *args, **kwargs):
        super().add_scalar(tag, scalar_value, global_step, *args, **kwargs)
        # wandb needs increasing iteration steps, so plots over wall-clock time stay in TensorBoard.
        if not tag.endswith("/time"):
            wandb.log({tag: scalar_value}, step=global_step)

    def close(self):
        super().close()
        wandb.finish()
