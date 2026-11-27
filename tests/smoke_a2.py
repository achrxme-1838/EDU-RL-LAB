"""Run with Isaac Sim: python tests/smoke_a2.py (8 environments, two PPO updates)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True).app

try:
    import gymnasium as gym
    import torch
    from pxr import UsdUtils

    from edu_rl_lab.assets.a2 import A2_USD_PATH
    from edu_rl_lab.tasks import A2_TASK_ID, TASK_ID, load_task_configs
    from edu_rl_lab.utils.vecenv_wrapper import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(A2_USD_PATH))
    # OmniPBR is an Isaac Sim built-in shader resolved by MDL, not a USD asset.
    assert set(unresolved) <= {"OmniPBR.mdl"}, unresolved
    assert layers
    for asset in [layer.realPath for layer in layers] + list(assets):
        assert Path(asset).resolve().is_relative_to(ROOT), asset

    cfg, agent_cfg = load_task_configs(A2_TASK_ID)
    play_cfg, _ = load_task_configs(A2_TASK_ID, play=True)
    go2_cfg, go2_agent = load_task_configs(TASK_ID)
    for config in (cfg, play_cfg, go2_cfg):
        config.validate()
    assert cfg.scene.num_envs == 4096
    assert play_cfg.scene.num_envs == 64
    assert cfg.observations.policy.history_length == 5
    assert cfg.observations.policy.enable_corruption
    assert len(cfg.scene.terrain.terrain_generator.sub_terrains) == 7
    assert go2_cfg.observations.policy.history_length in (None, 0)
    assert go2_agent.save_interval == 100
    assert agent_cfg.max_iterations == 8000 and agent_cfg.save_interval == 200  # w2 실습 수업용 학습 길이·저장 간격

    cfg.seed = 42
    cfg.scene.num_envs = 8
    cfg.scene.terrain.terrain_generator.num_rows = 2
    cfg.scene.terrain.terrain_generator.num_cols = 10
    cfg.scene.terrain.visual_material = None
    cfg.scene.sky_light.spawn.texture_file = None
    cfg.commands.base_velocity.debug_vis = False
    cfg.episode_length_s = 0.2
    env = RslRlVecEnvWrapper(gym.make(A2_TASK_ID, cfg=cfg), clip_actions=agent_cfg.clip_actions)
    try:
        robot = env.unwrapped.scene["robot"]
        assert robot.num_joints == 12 and robot.num_bodies == 17
        assert env.num_actions == 12
        assert len(env.unwrapped.reward_manager.active_terms) == 20
        obs = env.get_observations()
        assert obs["policy"].shape == (8, 225)
        assert obs["critic"].shape == (8, 247)

        # Raw action history must advance once per control step and clear on reset.
        action = env.unwrapped.action_manager.get_term("LegJointPositionAction")
        previous = torch.zeros(8, 12, device=env.device)
        previous_previous = previous.clone()
        saw_timeout = False
        for step in range(25):
            current = torch.full_like(previous, (step % 3 - 1) * 0.1)
            obs, rewards, dones, extras = env.step(current)
            assert torch.isfinite(rewards).all()
            assert all(torch.isfinite(obs[key]).all() for key in ("policy", "critic"))
            alive = ~dones.bool()
            torch.testing.assert_close(action.prev_actions[alive], previous[alive])
            torch.testing.assert_close(action.prev_prev_actions[alive], previous_previous[alive])
            # Last 60 actor inputs are the five 12-dimensional action samples.
            torch.testing.assert_close(obs["policy"][:, -12:], action.raw_actions)
            for buffer in (action.raw_actions, action.prev_actions, action.prev_prev_actions):
                assert torch.count_nonzero(buffer[~alive]) == 0
            saw_timeout |= bool(extras["time_outs"].any())
            previous_previous.copy_(previous)
            previous.copy_(current)
            previous[~alive] = 0
            previous_previous[~alive] = 0
        assert saw_timeout

        runner = OnPolicyRunner(env, agent_cfg.to_dict(), device=env.device)
        policy = runner.alg.policy
        assert [(m.in_features, m.out_features) for m in policy.actor if isinstance(m, torch.nn.Linear)] == [
            (225, 512),
            (512, 256),
            (256, 128),
            (128, 12),
        ]
        assert [(m.in_features, m.out_features) for m in policy.critic if isinstance(m, torch.nn.Linear)] == [
            (247, 512),
            (512, 256),
            (256, 128),
            (128, 1),
        ]
        before = policy.actor[0].weight.detach().clone()
        runner.learn(2)
        assert not torch.equal(before, policy.actor[0].weight)
        assert all(torch.isfinite(p).all() for p in policy.parameters())
        print("PASS: local USD dependencies, 225/247 observations, action resets, two PPO updates.", flush=True)
    finally:
        env.close()
except Exception:
    # Older SimulationApp.close() may exit before Python prints pending exceptions.
    import traceback

    traceback.print_exc()
    raise
finally:
    app.close()
