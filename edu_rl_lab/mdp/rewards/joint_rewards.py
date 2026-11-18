# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv




from edu_rl_lab import mdp


def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the term.
    """
    # 261119 hw: w2 실습
    # [w2 수 | 기본-4] 관절 속도 패널티 (아래 joint_acc_l2, joint_torques_l2도 같은 형태)
    #
    # - 수식: sum_j joint_vel_j^2
    # - 출력: (num_envs,)
    #
    # 1. 로봇 가져오기 (Articulation)
    # 2. 관절 속도: asset.data.joint_vel, 관절 선택 [:, asset_cfg.joint_ids]
    # 3. 제곱 → 관절끼리 더하기 (dim=1)
    raise NotImplementedError("w2 수 | 기본-4: joint_vel_l2 미작성")


def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations contribute to the term.
    """
    # 261119 hw: w2 실습
    # [w2 수 | 기본-4] 관절 가속도 패널티
    #
    # - 수식: sum_j joint_acc_j^2
    # - joint_vel_l2와 같은 형태. asset.data에서 가속도 필드 찾기
    raise NotImplementedError("w2 수 | 기본-4: joint_acc_l2 미작성")


def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    # 261119 hw: w2 실습
    # [w2 수 | 기본-4] 관절 토크 패널티
    #
    # - 수식: sum_j torque_j^2
    # - joint_vel_l2와 같은 형태. 토크 필드는 화요일 중간-1 joint_effort에서 쓴 것
    raise NotImplementedError("w2 수 | 기본-4: joint_torques_l2 미작성")


def joint_pos_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    # 261119 hw: w2 실습
    # [w2 수 | 제공] 관절 한계 패널티 (함수는 제공, 등록만 작성)
    #
    # - soft_joint_pos_limits[..., 0] = 아래 한계, [..., 1] = 위 한계
    # - 한계 안쪽이면 0, 넘어간 만큼 패널티
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


# The following project terms are licensed under Apache-2.0; see README.md and LICENCE.
def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    # 261119 hw: w2 실습
    # [w2 수 | 중간-1] 에너지 패널티
    #
    # - 수식: sum_j |joint_vel_j| * |torque_j|
    # - 출력: (num_envs,)
    # - 재료: asset.data.joint_vel, asset.data.applied_torque
    #         관절 선택은 [:, asset_cfg.joint_ids]
    raise NotImplementedError("w2 수 | 중간-1: energy 미작성")


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # 261119 hw: w2 실습
    # 2주차 실습에서 쓰지 않는 함수. 수요일 도전 joint_position_penalty_command_aware의 정답과 거의 같아 본문을 가림
    # 원래 본문은 w2_result_wed부터 돌아옴
    #
    #
    raise NotImplementedError("2주차 실습에서 쓰지 않는 함수")


def joint_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # 261119 hw: w2 실습
    # 2주차 실습에서 쓰지 않는 함수. 화요일 기본-3 joint_pos_rel의 정답과 거의 같아 본문을 가림
    # 원래 본문은 w2_result_tue부터 돌아옴
    #
    raise NotImplementedError("2주차 실습에서 쓰지 않는 함수")


def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)


def joint_position_penalty_command_aware(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float,
    command_name: str | None = "base_velocity",
) -> torch.Tensor:
    """Penalize joint deviation, with stronger scaling when standing still.

    Set command_name=None to decide standing solely from measured base XY speed.
    The default preserves the velocity-command behavior of existing RL tasks.
    """
    # 261119 hw: w2 실습
    # [w2 수 | 도전] 기본 자세 유지 패널티 (서 있을 때 더 세게)
    #
    # - 출력: (num_envs,), 0 이상
    # - 패널티 크기: 기본 자세 대비 관절 위치 벡터의 길이 (12개 관절 전체, norm)
    # - 움직이는 중 판정 (둘 중 하나라도 참이면 움직이는 중)
    #   - base xy 선속도의 크기 > velocity_threshold
    #   - command_name이 None이 아니고, command 크기 > 0
    # - 움직이는 중이면 패널티 그대로, 서 있으면 stand_still_scale배
    # - 재료: asset.data.root_lin_vel_b, asset.data.joint_pos, asset.data.default_joint_pos
    #         env.command_manager.get_command(command_name)
    #         torch.linalg.norm(..., dim=1), torch.where(조건, 참일 때, 거짓일 때)
    raise NotImplementedError("w2 수 | 도전: joint_position_penalty_command_aware 미작성")
