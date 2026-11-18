# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.envs.utils.io_descriptors import (
    generic_io_descriptor, record_shape,
)


@generic_io_descriptor(dtype=torch.float32, observation_type="Action", on_inspect=[record_shape])
def last_action(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
    """The last input action to the environment.

    The name of the action term for which the action is required. If None, the
    entire action tensor is returned.
    """
    # 261119 hw: w2 실습
    # [w2 화 | 예시] 직전 action (풀이 예시)
    #
    # - 역할: 정책이 바로 전에 낸 출력. 다음 출력을 부드럽게 잇는 데 씀
    # - 출력: (num_envs, 12), 관절 12개
    #
    # 1. env 안의 manager에서 값 꺼내기
    #    - action은 env.action_manager가 가지고 있음
    # 2. 그대로 return
    #
    # - 흐름: 함수 → a2_env_cfg.py의 ObsTerm 등록 → 신경망 입력 12칸 × history 5
    if action_name is None:
        return env.action_manager.action
    else:
        return env.action_manager.get_term(action_name).raw_actions


@generic_io_descriptor(dtype=torch.float32, observation_type="Command", on_inspect=[record_shape])
def generated_commands(env: ManagerBasedRLEnv, command_name: str | None = None) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    # 261119 hw: w2 실습
    # [w2 수 | 기본-1] 속도 명령
    #
    # - 역할: 조이스틱이 주는 목표 속도 (vx, vy, wz)
    # - 출력: (num_envs, 3)
    #
    # 1. command는 env.command_manager가 가지고 있음
    #    - last_action에서 env.action_manager를 쓴 것과 같은 형태
    # 2. 이름(command_name)으로 command 하나 꺼내기
    #    - command_manager의 메서드 중 get_으로 시작하는 것
    raise NotImplementedError("w2 수 | 기본-1: generated_commands 미작성")
