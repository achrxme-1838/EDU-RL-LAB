# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # 261119 hw: w2 실습
    # [w2 수 | 예시] 몸체 상하 속도 벌점 (풀이 예시)
    #
    # - 수식: v_z^2
    # - 출력: (num_envs,), 0 이상. cfg의 음수 weight가 곱해져 벌점이 됨
    #
    # 1. 로봇 가져오기
    # 2. 몸체 선속도(body frame)의 z 성분 [:, 2]
    # 3. 제곱 (torch.square). 성분이 하나라 더할 필요 없음
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # 261119 hw: w2 실습
    # [w2 수 | 기본-2] 몸체 롤·피치 각속도 벌점
    #
    # - 수식: w_x^2 + w_y^2
    # - 출력: (num_envs,)
    #
    # 1. 로봇 가져오기
    # 2. 몸체 각속도(body frame)의 x, y 성분 → [:, :2]
    # 3. 제곱한 뒤 성분끼리 더하기 → torch.sum(..., dim=1)
    #    - dim=1을 빼면 env까지 모두 더해져 숫자 하나가 됨
    raise NotImplementedError("w2 수 | 기본-2: ang_vel_xy_l2 미작성")


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # 261119 hw: w2 실습
    # [w2 수 | 기본-3] 몸체 기울기 벌점
    #
    # - 수식: g_x^2 + g_y^2   (g = 화요일 기본-2의 projected gravity)
    # - 출력: (num_envs,)
    # - 똑바로 서 있으면 g = (0, 0, -1)이라 xy 성분이 0
    # - 형태는 기본-2와 같음. 꺼내는 필드만 다름
    raise NotImplementedError("w2 수 | 기본-3: flat_orientation_l2 미작성")


def _mean_finite_hit_height(sensor: RayCaster) -> torch.Tensor:
    """Per-env mean world-z of ray hits, ignoring missed rays.

    Missed rays return ``+inf`` (warp raycast kernel fills misses with infinity). A plain
    ``torch.mean`` over them collapses the whole env to ``inf``, which then propagates as an
    ``inf``/``nan`` reward and blows up training. We average only over finite hits and fall
    back to ``0.0`` for envs where every ray missed.
    """
    hits_z = sensor.data.ray_hits_w[..., 2]
    finite = torch.isfinite(hits_z)
    hits_z_safe = torch.where(finite, hits_z, torch.zeros_like(hits_z))
    count = finite.sum(dim=1).clamp(min=1)
    return hits_z_safe.sum(dim=1) / count


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # 261119 hw: w2 실습
    # [w2 수 | 중간-3] 몸체 높이 벌점
    #
    # - 수식: (몸체 z - 목표 높이)^2
    # - 출력: (num_envs,)
    # - 목표 높이 = target_height + 발밑 지형 높이
    #   - sensor_cfg가 있으면: 지형 높이 = _mean_finite_hit_height(sensor) (위 함수, 제공)
    #   - sensor_cfg가 None이면: 평지로 보고 target_height 그대로
    # - 재료: env.scene[sensor_cfg.name] (RayCaster), asset.data.root_pos_w (world frame 위치)
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    raise NotImplementedError("w2 수 | 중간-3: base_height_l2 미작성")
