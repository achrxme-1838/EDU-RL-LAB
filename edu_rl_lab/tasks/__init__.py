"""Register locomotion tasks without importing the simulator."""

from importlib import import_module
import gymnasium as gym

from edu_rl_lab.task_ids import TASK_ID, A2_TASK_ID, TASK_IDS

gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "edu_rl_lab.tasks.locomotion.go2_env_cfg:Go2EnvCfg",
        "play_env_cfg_entry_point": "edu_rl_lab.tasks.locomotion.go2_env_cfg:Go2PlayEnvCfg",
        "rsl_rl_cfg_entry_point": "edu_rl_lab.tasks.locomotion.agents.ppo_cfg:Go2PPORunnerCfg",
    },
)

gym.register(
    id=A2_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "edu_rl_lab.tasks.locomotion.a2_env_cfg:A2EnvCfg",
        "play_env_cfg_entry_point": "edu_rl_lab.tasks.locomotion.a2_env_cfg:A2PlayEnvCfg",
        "rsl_rl_cfg_entry_point": "edu_rl_lab.tasks.locomotion.agents.a2_ppo_cfg:A2PPORunnerCfg",
    },
)


def load_task_configs(task_id: str, *, play: bool = False):
    """Instantiate a registered task's environment and PPO configs after AppLauncher."""
    if task_id not in TASK_IDS:
        raise ValueError(f"Unknown teaching task: {task_id}")
    entries = gym.spec(task_id).kwargs

    def load(key):
        module, name = entries[key].split(":")
        return getattr(import_module(module), name)()

    return load("play_env_cfg_entry_point" if play else "env_cfg_entry_point"), load("rsl_rl_cfg_entry_point")
