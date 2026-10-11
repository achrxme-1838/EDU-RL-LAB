"""Shared command-line settings for the single Go2 task."""

import argparse

from edu_rl_lab.tasks import TASK_ID


def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument("--task", choices=[TASK_ID], default=TASK_ID)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", help="Explicit path to a model_*.pt checkpoint.")
    parser.add_argument("--load_run", default="*", help="Run directory name or glob under logs/rsl_rl/go2_velocity.")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video_length", type=int, default=200)


def validate_args(parser, args):
    for name in ("num_envs", "video_length", "max_iterations", "video_interval", "steps"):
        value = getattr(args, name, None)
        if value is not None and value <= 0:
            parser.error(f"--{name} must be positive")
