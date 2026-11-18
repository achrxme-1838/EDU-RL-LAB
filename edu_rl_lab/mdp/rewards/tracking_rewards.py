# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


from edu_rl_lab.utils.maths.maths import quat_apply_inverse, yaw_quat


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # 261119 hw: w2 실습
    # [w2 화 | 도전] 선속도 추종 reward (x, y 2축)
    #
    # - 수식: exp( -( (cmd_x - v_x)^2 + (cmd_y - v_y)^2 ) / std^2 )
    # - 출력: (num_envs,), 0 초과 1 이하. 명령과 같은 속도면 1
    # - 아래 track_ang_vel_z_exp(풀이 예시, 1축)를 xy 2축으로 넓히기
    #   - command의 앞 두 성분, 몸체 선속도(body frame)의 앞 두 성분
    #   - 성분별 제곱을 더할 때 축 지정 (dim=1)
    # - std가 클수록 오차를 너그럽게 봄
    # - 이 항목이 없으면 로봇이 걸을 이유가 없음
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # 261119 hw: w2 실습
    # [w2 화 | 예시] 회전 속도 추종 reward (풀이 예시)
    #
    # - 수식: exp( -(cmd_wz - w_z)^2 / std^2 )
    # - 출력: (num_envs,), 0 초과 1 이하
    #
    # 1. 로봇 가져오기
    # 2. 오차 = (command의 z 회전 성분 - 몸체 각속도의 z 성분)^2
    #    - command: env.command_manager.get_command(command_name), 모양 (num_envs, 3) = (vx, vy, wz)
    #    - 몸체 각속도: asset.data.root_ang_vel_b, 모양 (num_envs, 3)
    #    - [:, 2] = z 성분
    # 3. exp 커널: 오차 0이면 1, 오차가 커질수록 0에 가까워짐
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # 261119 hw: w2 실습
    # 2주차 실습에서 쓰지 않는 함수. 화요일 도전 track_lin_vel_xy_exp의 정답과 거의 같아 본문을 가림
    # 원래 본문은 w2_result_tue부터 돌아옴
    #
    #
    #
    raise NotImplementedError("2주차 실습에서 쓰지 않는 함수")


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)
