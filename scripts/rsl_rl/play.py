"""Evaluate a saved Go2 policy."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cli_args
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
cli_args.add_common_args(parser)
parser.add_argument("--steps", type=int, default=None, help="Stop after this many simulation steps.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
cli_args.validate_args(parser, args)
if args.video:
    args.enable_cameras = True

app = AppLauncher(args).app


def main():
    import gymnasium as gym
    import torch

    from edu_rl_lab.tasks.locomotion.agents.ppo_cfg import Go2PPORunnerCfg
    from edu_rl_lab.tasks.locomotion.go2_env_cfg import Go2PlayEnvCfg
    from edu_rl_lab.utils.checkpoints import resolve_checkpoint
    from edu_rl_lab.utils.vecenv_wrapper import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    env_cfg, agent_cfg = Go2PlayEnvCfg(), Go2PPORunnerCfg()
    env_cfg.seed = args.seed
    if args.device is not None:
        env_cfg.sim.device = agent_cfg.device = args.device
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    checkpoint = resolve_checkpoint(
        ROOT / "logs" / "rsl_rl" / agent_cfg.experiment_name, args.load_run, args.checkpoint
    )
    print(f"[INFO] Checkpoint: {checkpoint}")
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    try:
        if args.video:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=str(checkpoint.parent / "videos" / "play"),
                step_trigger=lambda step: step == 0,
                video_length=args.video_length,
                disable_logger=True,
            )
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), device=agent_cfg.device)
        runner.load(str(checkpoint), load_optimizer=False)
        policy = runner.get_inference_policy(device=env.device)
        obs = env.get_observations()
        step = 0
        limit = args.steps if args.steps is not None else (args.video_length if args.video else None)
        while app.is_running() and (limit is None or step < limit):
            with torch.inference_mode():
                obs, _, _, _ = env.step(policy(obs))
            step += 1
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
