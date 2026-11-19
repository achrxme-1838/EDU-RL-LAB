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
    # 261119 hw: w2 실습
    # [w2 수 | 기본-5] action rate 패널티
    #
    # - 수식: sum_j (a_t,j - a_t-1,j)^2
    # - 출력: (num_envs,)
    # - 재료: env.action_manager.action (이번 action), env.action_manager.prev_action (한 step 전 action)
    # - 로봇을 가져올 필요 없음. 한 줄로 끝남
    raise NotImplementedError("w2 수 | 기본-5: action_rate_l2 미작성")


def action_smoothness_l2(
    env: ManagerBasedRLEnv, action_name: str = "LegJointPositionAction"
) -> torch.Tensor:
    """Second difference of raw actions: ||a[t] - 2*a[t-1] + a[t-2]||²."""
    # 261119 hw: w2 실습
    # [w2 수 | 중간-2] action smoothness 패널티 (2차 차분)
    #
    # - 수식: sum_j (a_t - 2 a_t-1 + a_t-2)^2
    # - 출력: (num_envs,)
    # - action term 꺼내기: env.action_manager.get_term(action_name)
    #   - 이 term(JointPositionActionWithHistory)이 raw_actions, prev_actions, prev_prev_actions를 가짐
    raise NotImplementedError("w2 수 | 중간-2: action_smoothness_l2 미작성")


def is_terminated(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.termination_manager.terminated.float()
