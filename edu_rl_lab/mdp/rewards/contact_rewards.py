# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject
from isaaclab.sensors import ContactSensor
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv




def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # w2 실습
    # [w2 목 | 예시] undesired contacts 패널티 (풀이 예시)
    # - 모양 기호: N = num_envs, H = history 3, B = body 17, K = 고른 부위 수
    # - 출력: (N,), 닿은 부위 개수
    # 1. 접촉 센서 가져오기: env.scene.sensors[sensor_cfg.name] (ContactSensor)
    # 2. 접촉력 기록: net_forces_w_history, 모양 (N, H, B, xyz 3)
    #    - 고를 부위: [:, :, sensor_cfg.body_ids] → (N, H, K, 3) (cfg에서 body_names로 지정)
    # 3. 힘의 크기 torch.norm(..., dim=-1) → (N, H, K) → H번 중 최대 torch.max(..., dim=1)[0] → (N, K)
    #    - torch.max(x, dim=축)은 (values, indices) 묶음을 돌려줌. [0] = values (최댓값)
    # 4. threshold보다 크면 닿은 것 (N, K) → 부위끼리 더하기 sum(dim=1) → (N,)
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    return torch.sum(is_contact, dim=1)


def desired_contacts(env, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """Penalize if none of the desired contacts are present."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > threshold
    )
    zero_contact = (~contacts).all(dim=1)
    return 1.0 * zero_contact


def contact_forces(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize contact forces as the amount of violations of the net contact force."""
    # w2 실습
    # [w2 목 | 기본-4] feet contact force 초과량 패널티
    #
    # - 수식: sum_발 max(0, |F| - threshold)
    # - 출력: (num_envs,), N
    # - 1–3단계는 위 undesired_contacts(풀이 예시)와 같음
    # - 바꿀 곳: "threshold보다 큰가(참/거짓)" 대신 "threshold를 넘은 양" (안 넘으면 음수)
    #   - 음수를 0으로 자르기: x.clip(min=0.0), 모양 그대로
    raise NotImplementedError("w2 목 | 기본-4: contact_forces 미작성")


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    # w2 실습
    # [w2 목 | 중간-2] foot clearance reward
    #
    # - 수식: exp( -[ sum_발 (z_발 - target_height)^2 * tanh(tanh_mult * |v_발,xy|) ] / std )
    # - 출력: (num_envs,), 0 초과 1 이하
    # - 뜻: 수평으로 움직이는 발(스윙 중인 발)만 목표 높이에 가까운지 봄. 디딘 발은 속도가 0이라 빠짐
    # - 재료: 로봇 asset = env.scene[asset_cfg.name]
    #   - asset.data.body_pos_w, asset.data.body_lin_vel_w (월드 좌표계), 모양 (num_envs, body 17, 3)
    #   - 발 선택은 [:, asset_cfg.body_ids, 성분]. z = 2, xy = :2
    # - 쓸 함수
    #   - 원소마다 계산 (모양 그대로): torch.square(x), torch.tanh(x), torch.exp(x)
    #   - 한 축을 줄임: torch.norm(x, dim=축) = 크기, torch.sum(x, dim=축) = 합
    raise NotImplementedError("w2 목 | 중간-2: foot_clearance_reward 미작성")
