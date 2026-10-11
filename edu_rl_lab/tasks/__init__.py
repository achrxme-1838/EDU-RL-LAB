"""Register the single Go2 velocity-tracking task without importing the simulator."""

import gymnasium as gym

TASK_ID = "Edu-Go2-Velocity-v0"

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
