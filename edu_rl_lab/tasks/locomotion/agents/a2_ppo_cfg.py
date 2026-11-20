"""Feedforward PPO hyperparameters for A2 locomotion."""

from isaaclab.utils import configclass

from .ppo_cfg import Go2PPORunnerCfg, PolicyCfg, PPOCfg


@configclass
class A2PPORunnerCfg(Go2PPORunnerCfg):
    num_steps_per_env: int = 24
    max_iterations: int = 8000  # 261120 hw: w2 실습 수업용 학습 길이 (정답 기준 10001)
    save_interval: int = 500
    experiment_name: str = "a2_base_rl"
    policy: PolicyCfg = PolicyCfg(
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm: PPOCfg = PPOCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
    )
