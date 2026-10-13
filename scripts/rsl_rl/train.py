"""Train Go2 with the repository's PPO implementation."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cli_args
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
cli_args.add_common_args(parser)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--run_name", default="", help="Optional suffix for the run directory.")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--video_interval", type=int, default=2000)
# 261013 hw: wandb
parser.add_argument("--logger", choices=["wandb", "tensorboard"], help="Override the logger in ppo_cfg.py.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
cli_args.validate_args(parser, args)
if args.checkpoint and not args.resume:
    parser.error("--checkpoint requires --resume for training")
if args.video:
    args.enable_cameras = True

app = AppLauncher(args).app


def main():
    # Isaac Sim must start before importing simulation modules.
    import gymnasium as gym
    import torch
    from datetime import datetime

    from isaaclab.utils.io import dump_yaml
    from edu_rl_lab.tasks.locomotion.agents.ppo_cfg import Go2PPORunnerCfg
    from edu_rl_lab.tasks.locomotion.go2_env_cfg import Go2EnvCfg
    from edu_rl_lab.utils.checkpoints import resolve_checkpoint
    from edu_rl_lab.utils.vecenv_wrapper import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    env_cfg, agent_cfg = Go2EnvCfg(), Go2PPORunnerCfg()
    env_cfg.seed = agent_cfg.seed = args.seed
    if args.device is not None:
        env_cfg.sim.device = agent_cfg.device = args.device
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    # 261013 hw: wandb
    if args.logger is not None:
        agent_cfg.logger = args.logger
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    log_root = ROOT / "logs" / "rsl_rl" / agent_cfg.experiment_name
    resume_path = resolve_checkpoint(log_root, args.load_run, args.checkpoint) if args.resume else None
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    if args.run_name:
        run_name += "_" + args.run_name.replace("/", "_").replace(chr(92), "_")
    log_dir = log_root / run_name
    print(f"[INFO] Logs: {log_dir}")
    print(f"[INFO] PPO implementation: {Path(sys.modules[OnPolicyRunner.__module__].__file__).resolve()}")
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    runner = None
    try:
        if args.video:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=str(log_dir / "videos"),
                step_trigger=lambda step: step % args.video_interval == 0,
                video_length=args.video_length,
                disable_logger=True,
            )
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
        if resume_path:
            runner.load(str(resume_path))
        dump_yaml(str(log_dir / "params" / "env.yaml"), env_cfg)
        dump_yaml(str(log_dir / "params" / "agent.yaml"), agent_cfg)
        runner.learn(agent_cfg.max_iterations, init_at_random_ep_len=True)
    finally:
        if runner is not None and runner.writer is not None:
            runner.writer.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
