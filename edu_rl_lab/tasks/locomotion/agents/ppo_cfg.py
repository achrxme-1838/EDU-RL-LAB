"""PPO hyperparameters for Go2. Implementations live in the local rsl_rl package."""

from isaaclab.utils import configclass


@configclass
class PolicyCfg:
    class_name: str = "ActorCritic"
    init_noise_std: float = 1.0
    noise_std_type: str = "scalar"
    actor_obs_normalization: bool = False
    critic_obs_normalization: bool = False
    actor_hidden_dims: list[int] = [512, 256, 128]
    critic_hidden_dims: list[int] = [512, 256, 128]
    activation: str = "elu"


@configclass
class PPOCfg:
    class_name: str = "PPO"
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-3
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    normalize_advantage_per_mini_batch: bool = False


@configclass
class Go2PPORunnerCfg:
    seed: int = 42
    device: str = "cuda:0"
    num_steps_per_env: int = 24
    max_iterations: int = 10000
    save_interval: int = 100
    experiment_name: str = "go2_velocity"
    # 261013 hw: wandb
    logger: str = "wandb"  # or "tensorboard"
    wandb_project: str = "edu_rl_lab"
    clip_actions: float | None = None
    obs_groups: dict[str, list[str]] = {"policy": ["policy"], "critic": ["critic"]}
    policy: PolicyCfg = PolicyCfg()
    algorithm: PPOCfg = PPOCfg()
