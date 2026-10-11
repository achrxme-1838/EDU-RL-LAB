# EDU RL LAB

An educational repository for studying Go2 velocity tracking and PPO.
Read and modify the MDP functions, action/command classes, actor/critic networks,
rollout buffer, PPO updates, and training loop directly in this repository.

- Scope: **Go2 locomotion · feedforward PPO · single GPU**
- Target versions: **Isaac Lab 2.3.0 · Isaac Sim 5.1.0 · Python 3.11**
- Environment ID: `Edu-Go2-Velocity-v0`
- Algorithm: a PPO subset of RSL-RL 3.0.1 included in the local `rsl_rl/` directory

## Structure

```text
edu_rl_lab/
├── assets/go2.py
├── mdp/
│   ├── actions/
│   │   ├── cfgs/actions_cfg.py
│   │   └── joint_actions.py
│   ├── commands/
│   │   ├── cfgs/commands_cfg.py
│   │   └── velocity_command.py
│   ├── observations/           # robot_observations, mdp_observations
│   ├── events/                 # randomize, reset, step, event_utils
│   ├── rewards/                # root, joint, tracking, contact, feet, general
│   ├── terminations/           # mdp, root, contact
│   ├── spot/                   # Upstream Spot MDP in a separate namespace
│   ├── symmetry/anymal.py      # Upstream ANYmal symmetry transforms
│   └── curriculums.py
├── tasks/locomotion/
│   ├── go2_env_cfg.py
│   └── agents/ppo_cfg.py
└── utils/
    ├── maths/maths.py          # Local math without Isaac/Omniverse dependencies
    ├── vecenv_wrapper.py
    └── checkpoints.py
rsl_rl/
├── algorithms/ppo.py
├── modules/actor_critic.py
├── networks/                  # MLP, observation normalization
├── storage/rollout_storage.py
├── runners/on_policy_runner.py
├── env/vec_env.py
└── utils/
scripts/
├── list_envs.py
└── rsl_rl/{train,play,cli_args}.py
third_party/                   # Upstream licenses
```

MDP code is organized by function, with configuration classes in `cfgs/` directories.
The full `locomotion/velocity` scope of official **Isaac Lab v2.3.0** was reviewed,
including MDP terms referenced by shared configurations, Spot-specific MDP code,
and ANYmal symmetry transforms. The bodies of **71 upstream definitions** are
preserved verbatim, including the required base classes and helpers. The reference
source is [Isaac Lab v2.3.0](https://github.com/isaac-sim/IsaacLab/tree/v2.3.0).

The project-specific `energy` and `joint_position_penalty` rewards live in
`rewards/joint_rewards.py`; `air_time_variance_penalty` lives in
`rewards/feet_rewards.py`. Upstream Spot implementations with overlapping names
are kept in the `edu_rl_lab.mdp.spot` namespace. MDP reference implementations for
other robots are included, but Go2 is the only registered environment.

Math utilities follow DEX's `utils/maths/maths.py` layout and use the official
Isaac Lab 2.3.0 implementation. All 59 public functions and internal helpers were
copied; only the warning output in two deprecated functions was changed to Python
logging. This module can be imported with just PyTorch and NumPy, without Isaac
or Omniverse. All local MDP math imports point to this implementation.

Local MDP code does not import `isaaclab.envs.mdp` or `isaaclab_tasks...mdp`,
or inherit from their implementation classes. Isaac Lab still provides the
simulator, articulations, sensors, manager interfaces, configuration utilities,
and actuators. Mimic, environments and assets for other robots, hardware
deployment, recurrent policies, and distillation are excluded.

## Installation

Install in a new conda environment, separate from your existing environments.
The commands below target Linux x86_64. See also the
[official Isaac Lab 2.3.0 installation guide](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/pip_installation.html).

```bash
cd /path/to/edu_rl_lab
conda env create -f environment.yml
conda activate edu_rl_lab
export PYTHONNOUSERSITE=1

python -m pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# Use a directory separate from any existing Isaac Lab checkout.
git clone --depth 1 --branch v2.3.0 https://github.com/isaac-sim/IsaacLab.git ../IsaacLab-2.3.0
# This project only needs isaaclab core. Do not install the external rsl-rl-lib package.
python -m pip install -e ../IsaacLab-2.3.0/source/isaaclab

python -m pip install -e ".[dev]"
python scripts/list_envs.py
```

The official Go2 USD is used by default, so the first run requires access to the
asset service. To use a local USD file for the same model, set:

```bash
export EDU_GO2_USD=/absolute/path/to/go2.usd
```

## Training and Evaluation

```bash
# Check simulator integration with a small number of environments first.
python scripts/rsl_rl/train.py --headless --num_envs 32 --max_iterations 2

# Train.
python scripts/rsl_rl/train.py --headless --num_envs 4096 --max_iterations 10000

# Resume from the highest saved iteration of the latest run.
python scripts/rsl_rl/train.py --headless --resume --max_iterations 1000

# Evaluate the latest checkpoint.
python scripts/rsl_rl/play.py --num_envs 32

# Evaluate a specific checkpoint in headless mode, then exit.
python scripts/rsl_rl/play.py --headless --num_envs 32 --steps 500 \
  --checkpoint /absolute/path/to/model_1000.pt

# Record a video.
python scripts/rsl_rl/play.py --video --video_length 200

tensorboard --logdir logs/rsl_rl/go2_velocity
```

`--max_iterations` specifies the number of PPO updates to perform in the current
invocation. Resuming continues after the saved iteration. Use `--load_run` to
select a run by name or glob pattern, or `--resume --checkpoint ...` to choose
a checkpoint explicitly. Configurations and checkpoints are saved under
`logs/rsl_rl/go2_velocity/<run>/`. At startup, the training script prints the
path of the local PPO implementation it uses.

The default terrain is a flat tile. The actor receives 48-dimensional observations,
the critic receives 63-dimensional observations, and the policy controls 12 joints.
Edit `edu_rl_lab/tasks/locomotion/go2_env_cfg.py` for environment settings and
`agents/ppo_cfg.py` in the same directory for PPO settings.

## Code Checks Without the Simulator

You can also run static code checks in a separate, new environment.

```bash
conda create -n edu_rl_lab_test python=3.11 pip -y
conda activate edu_rl_lab_test
export PYTHONNOUSERSITE=1
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
ruff check edu_rl_lab rsl_rl scripts
```

Use the simulator training commands above to check actual Go2 simulation and
locomotion convergence.

## Licenses

Upstream copyright notices and licenses are retained.

- [Isaac Lab 2.3.0](https://github.com/isaac-sim/IsaacLab/tree/v2.3.0): MDP code, math utilities, Go2 asset configuration, and environment wrapper.
  Copyright (c) 2022-2025, The Isaac Lab Project Developers.
  [BSD-3-Clause](third_party/ISAACLAB_LICENSE). MDP file locations and imports were changed,
  and math warnings were redirected to Python logging. An environment variable can override the asset path.
- [RSL-RL 3.0.1](https://github.com/leggedrobotics/rsl_rl/tree/v3.0.1): local PPO implementation.
  Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION.
  [BSD-3-Clause](third_party/RSL_RL_LICENSE). The subset needed for feedforward PPO was retained,
  with changes to configuration copying, mini-batch size checks, checkpoint selection, and resuming.
- [Unitree RL Lab](https://github.com/unitreerobotics/unitree_rl_lab): the basis for the Go2 environment configuration,
  `energy`, `joint_position_penalty`, `air_time_variance_penalty`, and command curriculum.
  [Apache-2.0](LICENCE). The file structure was changed and unnecessary settings were removed;
  the robot asset and motors use the official Isaac Lab Go2 defaults.
