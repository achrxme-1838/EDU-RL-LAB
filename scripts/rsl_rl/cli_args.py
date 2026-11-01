"""Shared command-line settings for locomotion tasks."""

import argparse

# Task registration imports Gym, so defer it until after AppLauncher starts.
from edu_rl_lab.task_ids import TASK_ID, TASK_IDS


def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument("--task", choices=TASK_IDS, default=TASK_ID)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", help="Explicit path to a model_*.pt checkpoint.")
    parser.add_argument("--load_run", default="*", help="Run directory name or glob under the task's log directory.")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video_length", type=int, default=200)


def validate_args(parser, args):
    for name in ("num_envs", "video_length", "max_iterations", "video_interval", "steps"):
        value = getattr(args, name, None)
        if value is not None and value <= 0:
            parser.error(f"--{name} must be positive")
