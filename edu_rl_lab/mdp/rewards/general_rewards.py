# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv




def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def action_smoothness_l2(
    env: ManagerBasedRLEnv, action_name: str = "LegJointPositionAction"
) -> torch.Tensor:
    """DEX second difference of raw actions: ||a[t] - 2*a[t-1] + a[t-2]||²."""
    action = env.action_manager.get_term(action_name)
    return torch.sum(torch.square(action.raw_actions - 2 * action.prev_actions + action.prev_prev_actions), dim=1)


def is_terminated(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.termination_manager.terminated.float()
