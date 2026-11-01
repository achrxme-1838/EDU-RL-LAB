"""List the teaching task without starting Isaac Sim."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edu_rl_lab.tasks import TASK_IDS  # noqa: E402

if __name__ == "__main__":
    for task_id in TASK_IDS:
        print(f"{task_id}  |  velocity tracking  |  PPO")
