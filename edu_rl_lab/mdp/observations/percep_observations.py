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
    # 1. 센서는 아래 줄에서 이미 가져옴 (RayCaster)
    # 2. 센서 높이: sensor.data.pos_w 의 z 성분 (pos_w[:, 2]), 모양 (num_envs,)
    #    - pos_w = 센서가 붙은 base의 위치. cfg의 offset (0, 0, 20)은 ray 시작점에만 더해지고 여기에는 없음
    #    - (num_envs, 187)에서 빼려면 센서 높이에 둘째 축을 추가 (unsqueeze)
    # 3. 닿은 점: sensor.data.ray_hits_w, 모양 (num_envs, 187, 3). z는 마지막 축의 2번
    # 4. 위 함수 height_scan_finite가 이 함수를 부르고, 안 닿은 ray(inf)를 0으로 바꿈 (제공)
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    raise NotImplementedError("w2 목 | 기본-2: height_scan 미작성")
