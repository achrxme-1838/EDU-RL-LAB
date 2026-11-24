# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.utils.io_descriptors import (
    generic_io_descriptor, record_dtype, record_joint_names,
    record_joint_pos_offsets, record_joint_vel_offsets, record_shape,
)


@generic_io_descriptor(
    units="m/s", axes=["X", "Y", "Z"], observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def base_lin_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # w2 실습
    # [w2 화 | 중간-1] base lin vel (critic 전용)
    #
    # - 역할: base가 실제로 움직이는 속도. 실제 로봇에는 이 값을 바로 재는 센서가 없음
    # - 출력: (num_envs, 3), m/s, base 좌표계
    # - 기본-1 base_ang_vel과 같은 형태. 필드 이름은 asset.data에서 직접 찾기
    raise NotImplementedError("w2 화 | 중간-1: base_lin_vel 미작성")


@generic_io_descriptor(
    units="rad/s", axes=["X", "Y", "Z"], observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def base_ang_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root angular velocity in the asset's root frame."""
    # w2 실습
    # [w2 화 | 기본-1] base ang vel
    #
    # - 역할: IMU 자이로가 재는 값
    # - 출력: (num_envs, 3), rad/s, base 좌표계
    #
    # 1. 로봇 가져오기
    #    - env.scene[asset_cfg.name]
    # 2. base ang vel 꺼내기
    #    - 위치: asset.data
    #    - 이름 끝 _b = base 좌표계, _w = 월드 좌표계
    # 3. 그대로 return
    raise NotImplementedError("w2 화 | 기본-1: base_ang_vel 미작성")


@generic_io_descriptor(
    units="m/s^2", axes=["X", "Y", "Z"], observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def projected_gravity(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Gravity projection on the asset's root frame."""
    # w2 실습
    # [w2 화 | 기본-2] projected gravity (base 좌표계에서 본 중력 방향)
    #
    # - 역할: 몸이 얼마나 기울었는지. 똑바로 서 있으면 (0, 0, -1)
    # - 출력: (num_envs, 3), 단위 벡터, base 좌표계
    #
    # 1. 로봇 가져오기 (기본-1과 같음)
    # 2. 중력 방향 꺼내기
    #    - 위치: asset.data
    #    - 이름에 gravity가 들어 있고, base 좌표계이므로 끝이 _b
    # 3. 그대로 return
    raise NotImplementedError("w2 화 | 기본-2: projected_gravity 미작성")


@generic_io_descriptor(
    observation_type="JointState",
    on_inspect=[record_joint_names, record_dtype, record_shape, record_joint_pos_offsets],
    units="rad",
)
def joint_pos_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # w2 실습
    # [w2 화 | 기본-3] joint pos (기본 자세 대비)
    #
    # - 역할: 기본 자세(서 있는 자세)에서 각 관절이 얼마나 벗어났는지
    # - 출력: (num_envs, 12), rad
    #
    # 1. 로봇 가져오기 (Articulation)
    # 2. 현재 관절 위치 - 기본 관절 위치
    #    - 현재 값: asset.data.joint_pos
    #    - 기본 값: asset.data 안에서 이름이 default_로 시작하는 것
    # 3. 관절 선택
    #    - 두 텐서 모두 [:, asset_cfg.joint_ids]
    #    - 모양 (num_envs, 관절 수)에서 둘째 축이 관절
    raise NotImplementedError("w2 화 | 기본-3: joint_pos_rel 미작성")


@generic_io_descriptor(
    observation_type="JointState",
    on_inspect=[record_joint_names, record_dtype, record_shape, record_joint_vel_offsets],
    units="rad/s",
)
def joint_vel_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The joint velocities of the asset w.r.t. the default joint velocities.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # w2 실습
    # [w2 화 | 기본-4] joint vel (기본값 대비)
    #
    # - 출력: (num_envs, 12), rad/s
    # - 기본-3과 같은 형태. pos 대신 vel
    raise NotImplementedError("w2 화 | 기본-4: joint_vel_rel 미작성")


@generic_io_descriptor(
    observation_type="JointState", on_inspect=[record_joint_names, record_dtype, record_shape], units="N.m"
)
def joint_effort(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint applied effort of the robot.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their effort returned.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with this observation.

    Returns:
        The joint effort (N or N-m) for joint_names in asset_cfg, shape is [num_env,num_joints].
    """
    # w2 실습
    # [w2 화 | 중간-1] joint effort (critic 전용)
    #
    # - 역할: 모터가 실제로 낸 토크
    # - 출력: (num_envs, 12), Nm
    # - 위치: asset.data. 이름에 torque가 들어간 필드가 둘 있음
    #   - computed_torque: 모터 한계로 자르기 전 값 (쓰지 않음)
    #   - 다른 하나: 한계로 자른 뒤 관절에 실제로 가한 값 (이것을 씀)
    # - 관절 선택은 기본-3과 같음
    raise NotImplementedError("w2 화 | 중간-1: joint_effort 미작성")
