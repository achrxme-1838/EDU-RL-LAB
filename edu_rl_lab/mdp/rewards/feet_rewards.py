# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv




def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # w2 실습
    # [w2 목 | 도전] feet air time reward
    #
    # - 수식: sum_발 (last_air_time - threshold) × (이번 step에 막 착지했는가 0/1)
    #         그리고 command xy 크기가 0.1 이하면 0
    # - 출력: (num_envs,). 음수가 될 수 있음 (threshold보다 짧게 뜨면)
    # - 재료 (모두 접촉 센서 env.scene.sensors[sensor_cfg.name]에 있음)
    #   - 막 착지한 발: 센서의 compute_first_contact(env.step_dt) 메서드
    #     돌려주는 값: 모양 (num_envs, 센서 body 수) bool. 이번 step 안에 막 땅에 닿은 body면 True
    #   - 떠 있던 시간: 센서 data의 last_air_time
    #     뜻: 마지막으로 떠 있던 구간의 길이(초). 착지하는 순간 정해지고 다음 착지까지 그대로
    #   - bool 텐서에 float 텐서를 곱하면 True = 1, False = 0으로 계산됨
    #   - 둘 다 발만 선택: [:, sensor_cfg.body_ids]
    # - 정지 명령 mask: env.command_manager.get_command(command_name)의 앞 두 성분 크기, 모양 (num_envs, 3)
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # w2 실습
    # [w2 목 | 중간-1] feet slide 패널티
    #
    # - 수식: sum_발 |v_발,xy| × (닿아 있는가 0/1)
    # - 출력: (num_envs,)
    # - 닿아 있는가: contact_rewards.py의 undesired_contacts(목 예시)와 같은 방법, threshold 1.0
    #   - 센서는 sensor_cfg, 발 선택은 sensor_cfg.body_ids
    # - 발 속도: env.scene[asset_cfg.name].data.body_lin_vel_w, 발 선택은 asset_cfg.body_ids, xy는 [..., :2]
    #   - 센서와 로봇은 서로 다른 객체. body_ids도 각자의 것을 씀
    # - 쓸 함수: torch.norm(x, dim=축) = 크기, torch.sum(x, dim=축) = 합 (그 축이 없어짐)
    # - True/False 텐서 × 숫자 텐서: True = 1, False = 0으로 계산됨
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


# The following project terms are licensed under Apache-2.0; see README.md and LICENCE.
def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # w2 실습
    # [w2 목 | 제공] air time variance (함수는 제공, 등록만 작성)
    #
    # - 네 발의 last_air_time, last_contact_time 분산. 0.5초에서 자름
    # - 한 발만 오래 들거나 오래 딛으면 커짐
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward
