# EDU RL LAB

Go2의 속도 추종과 PPO를 공부하기 위한 교육용 레포입니다.
MDP 함수, action/command 클래스, actor/critic, rollout buffer, PPO 업데이트와
학습 루프를 이 레포에서 직접 읽고 수정할 수 있습니다.

- 대상: **Go2 locomotion · feedforward PPO · 단일 GPU**
- 기준: **Isaac Lab 2.3.0 · Isaac Sim 5.1.0 · Python 3.11**
- 환경 ID: `Edu-Go2-Velocity-v0`
- 알고리즘: 로컬 `rsl_rl/`에 포함한 RSL-RL 3.0.1의 PPO subset

## 구조

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
│   ├── spot/                  # 원본 Spot 전용 MDP (이름 충돌 분리)
│   ├── symmetry/anymal.py      # 원본 ANYmal 대칭 변환
│   └── curriculums.py
├── tasks/locomotion/
│   ├── go2_env_cfg.py
│   └── agents/ppo_cfg.py
└── utils/
    ├── maths/maths.py          # Isaac/Omniverse 없는 로컬 math
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
third_party/                   # 원본 라이선스
```

MDP는 기능별 폴더와 `cfgs/`로 분리했습니다. 공식 **Isaac Lab v2.3.0**의
`locomotion/velocity` 전체를 확인해 공용 설정에서 참조하는 MDP, Spot 전용 MDP,
ANYmal 대칭 변환까지 가져왔습니다. 필요한 기본 클래스·helper를 포함해
공식 정의 **71개**의 본문을 그대로 보존합니다. 기준 소스는
[Isaac Lab v2.3.0](https://github.com/isaac-sim/IsaacLab/tree/v2.3.0)입니다.

프로젝트 고유 `energy`, `joint_position_penalty`는 `rewards/joint_rewards.py`,
`air_time_variance_penalty`는 `rewards/feet_rewards.py`에 있습니다. 이름이 겹치는
공식 Spot 구현은 `edu_rl_lab.mdp.spot` 네임스페이스에 보존합니다.
로봇별 MDP 참조 구현을 포함하지만 실행 환경 등록은 Go2 하나입니다.

Math는 DEX와 같은 `utils/maths/maths.py` 구조이며 구현은 공식 Isaac Lab 2.3.0을
사용합니다. 공개 함수 59개와 내부 helper를 복사했고, deprecated 함수 두 개의
경고 출력만 Python logging으로 바꿨습니다. 이 모듈은 Isaac/Omniverse 없이
PyTorch·NumPy만으로 import할 수 있으며, 모든 로컬 MDP의 math import가 이 파일을
가리킵니다.

`isaaclab.envs.mdp`와 `isaaclab_tasks...mdp`를 가져오거나 그 구현 클래스를
상속하지 않습니다. 시뮬레이터, articulation, sensor, manager 인터페이스,
설정 유틸리티와 actuator는 Isaac Lab을 사용합니다.
Mimic, 다른 로봇의 실행 환경·asset, 실기 배포, recurrent policy, distillation은 포함하지 않습니다.

## 설치

기존 환경과 분리한 새 conda 환경에서 설치합니다. 아래는 Linux x86_64 기준입니다.
[Isaac Lab 2.3.0 공식 설치 문서](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/pip_installation.html)도 참고하세요.

```bash
cd /path/to/edu_rl_lab
conda env create -f environment.yml
conda activate edu_rl_lab
export PYTHONNOUSERSITE=1

python -m pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# 기존 Isaac Lab checkout과 별개의 디렉터리를 사용합니다.
git clone --depth 1 --branch v2.3.0 https://github.com/isaac-sim/IsaacLab.git ../IsaacLab-2.3.0
# 이 프로젝트는 isaaclab core만 사용합니다. 외부 rsl-rl-lib는 설치하지 않습니다.
python -m pip install -e ../IsaacLab-2.3.0/source/isaaclab

python -m pip install -e ".[dev]"
python scripts/list_envs.py
```

공식 Go2 USD가 기본값이며 첫 실행에는 asset 서비스 접근이 필요합니다.
로컬에 같은 모델의 USD가 있다면 아래처럼 지정할 수 있습니다.

```bash
export EDU_GO2_USD=/absolute/path/to/go2.usd
```

## 학습과 평가

```bash
# 먼저 작은 환경 수로 실제 simulator 연결 확인
python scripts/rsl_rl/train.py --headless --num_envs 32 --max_iterations 2

# 학습
python scripts/rsl_rl/train.py --headless --num_envs 4096 --max_iterations 10000

# 가장 최근 run의 가장 높은 iteration에서 재개
python scripts/rsl_rl/train.py --headless --resume --max_iterations 1000

# 가장 최근 checkpoint 평가
python scripts/rsl_rl/play.py --num_envs 32

# 특정 checkpoint 평가 및 headless 종료
python scripts/rsl_rl/play.py --headless --num_envs 32 --steps 500 \
  --checkpoint /absolute/path/to/model_1000.pt

# 동영상 저장
python scripts/rsl_rl/play.py --video --video_length 200

tensorboard --logdir logs/rsl_rl/go2_velocity
```

`--max_iterations`는 이번 실행에서 수행할 PPO 업데이트 횟수입니다.
재개 시 저장된 iteration 뒤부터 이어갑니다. `--load_run`으로 run 이름이나 glob을
지정할 수 있고, `--resume --checkpoint ...`로 재개 파일을 직접 선택할 수도 있습니다.
설정과 checkpoint는 `logs/rsl_rl/go2_velocity/<run>/`에 저장됩니다.
학습 스크립트는 시작할 때 실제로 사용하는 로컬 PPO 경로를 출력합니다.

기본 지형은 평평한 tile입니다. Actor는 48차원, critic은 63차원 관측을 받고
12개 관절을 제어합니다. 환경 설정은 `edu_rl_lab/tasks/locomotion/go2_env_cfg.py`,
PPO 설정은 같은 폴더의 `agents/ppo_cfg.py`에서 변경합니다.

## 시뮬레이터 없이 코드 검사

별도의 새 환경에서 정적 코드 검사를 실행할 수도 있습니다.

```bash
conda create -n edu_rl_lab_test python=3.11 pip -y
conda activate edu_rl_lab_test
export PYTHONNOUSERSITE=1
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
ruff check edu_rl_lab rsl_rl scripts
```

실제 Go2 시뮬레이션이나 보행 수렴 확인은 위의 simulator 학습 명령으로 수행합니다.

## 라이선스

원본 코드의 저작권 표기와 라이선스를 유지합니다.

- [Isaac Lab 2.3.0](https://github.com/isaac-sim/IsaacLab/tree/v2.3.0): MDP·math·Go2 asset 설정·환경 wrapper.
  Copyright (c) 2022-2025, The Isaac Lab Project Developers.
  [BSD-3-Clause](third_party/ISAACLAB_LICENSE). MDP의 파일 배치와 import를 변경했고,
  math의 경고 출력은 Python logging으로 바꿨습니다. Asset 경로는 환경변수로 지정할 수 있습니다.
- [RSL-RL 3.0.1](https://github.com/leggedrobotics/rsl_rl/tree/v3.0.1): 로컬 PPO 구현.
  Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION.
  [BSD-3-Clause](third_party/RSL_RL_LICENSE). Feedforward PPO에 필요한 부분을 남기고
  설정 복사, mini-batch 크기 검사, checkpoint 선택·재개 처리를 수정했습니다.
- [Unitree RL Lab](https://github.com/unitreerobotics/unitree_rl_lab): Go2 환경 구성,
  `energy`·`joint_position_penalty`·`air_time_variance_penalty` 및 command curriculum의 기반.
  [Apache-2.0](LICENCE). 파일 구조를 변경하고 불필요한 설정을 제거했으며,
  robot asset·motor는 공식 Isaac Lab Go2 기본값을 사용합니다.
