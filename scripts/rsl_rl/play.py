"""Evaluate a saved locomotion policy."""

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

    from edu_rl_lab.tasks import load_task_configs
    from edu_rl_lab.utils.checkpoints import resolve_checkpoint
    from edu_rl_lab.utils.export_policy import export_policy_as_onnx
    from edu_rl_lab.utils.vecenv_wrapper import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    env_cfg, agent_cfg = load_task_configs(args.task, play=True)
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
        onnx_path = export_policy_as_onnx(
            runner.alg.policy, runner.alg.policy.get_actor_obs(obs), checkpoint, task=args.task
        )
        print(f"[INFO] ONNX exported: {onnx_path}", flush=True)
        step = 0
        limit = args.steps if args.steps is not None else (args.video_length if args.video else None)
        # 261022 hw: day-5 reset count per termination term (cumulative, printed every 5 s)
        tm = env.unwrapped.termination_manager
        resets = dict.fromkeys(tm.active_terms, 0)
        print_every = round(5.0 / env.unwrapped.step_dt)
        while app.is_running() and (limit is None or step < limit):
            with torch.inference_mode():
                obs, _, dones, _ = env.step(policy(obs))
            # get_term() keeps the last episode's cause -> count only envs done this step
            for name in resets:
                resets[name] += int((tm.get_term(name) & dones.bool()).sum())
            step += 1
            if step % print_every == 0:
                print(f"[RESET] t={step * env.unwrapped.step_dt:.0f}s " + " ".join(f"{k}={v}" for k, v in resets.items()))
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
