"""List the teaching task without starting Isaac Sim."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edu_rl_lab.tasks import TASK_ID  # noqa: E402

if __name__ == "__main__":
    print(f"{TASK_ID}  |  Go2 velocity tracking  |  PPO")
