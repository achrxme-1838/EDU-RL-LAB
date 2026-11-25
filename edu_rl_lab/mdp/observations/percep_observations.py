# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
from isaaclab.sensors import RayCaster




def height_scan_finite(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """Replace missed rays with zero before observation clipping."""
    return torch.nan_to_num(height_scan(env, sensor_cfg, offset), nan=0.0, posinf=0.0, neginf=0.0)


def height_scan(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    # w2 실습
    # [w2 목 | 기본-2] height scan (critic 전용)
    #
    # - 역할: 몸 아래 1.6 m × 1.0 m 격자(0.1 m 간격, 17 × 11 = 187점)의 지면 높이
    # - 출력: (num_envs, 187)
    # - 수식: 센서 높이 - ray가 닿은 점의 높이 - offset
    #
    # - 재료 (sensor는 아래 줄에서 가져옴)
    #   - 센서 높이: sensor.data.pos_w, 모양 (num_envs, 3)
    #   - 닿은 점: sensor.data.ray_hits_w, 모양 (num_envs, 187, 3)
    #   - z = 마지막 축의 2번. x[..., 2] ("..." = 앞의 축 모두)
    # - 모양이 (num_envs,)와 (num_envs, 187)이면 바로 못 뺌
    #   - x.unsqueeze(dim): dim 자리에 크기 1인 축 추가. (N, 1) - (N, 187) → (N, 187)
    # - 안 닿은 ray(inf)는 위 height_scan_finite가 0으로 바꿈 (제공)
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    raise NotImplementedError("w2 목 | 기본-2: height_scan 미작성")
