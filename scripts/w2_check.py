"""w2 실습 점검: 그날까지의 항목이 작성·등록됐는지, 함수 성질이 맞는지 확인 (headless, 8 env).

    python scripts/w2_check.py --day tue      # 화 항목
    python scripts/w2_check.py --day wed      # 화 + 수 항목
    python scripts/w2_check.py --day thur     # 화 + 수 + 목 항목

정답 코드는 담지 않음. 출력 shape, 유한값, 부호, 물리적으로 당연한 성질만 봄.
마지막 줄이 "학습 시작 가능"이면 train.py를 실행해도 됨.
"""

# 261119 hw: w2 실습 (점검 스크립트)
import argparse
import copy
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--day", choices=["tue", "wed", "thur"], required=True)
parser.add_argument("--num_envs", type=int, default=8)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

DAY = {"tue": 1, "wed": 2, "thur": 3}[args.day]
CMD = "base_velocity"

# 작성할 함수: (태그, 함수 이름, 종류, 출력 열 수). 종류 obs / pen(벌점, 0 이상) / rew(상) / event
FUNCS = [
    ("화 기본-1", "base_ang_vel", "obs", 3),
    ("화 기본-2", "projected_gravity", "obs", 3),
    ("화 기본-3", "joint_pos_rel", "obs", 12),
    ("화 기본-4", "joint_vel_rel", "obs", 12),
    ("화 중간-1", "base_lin_vel", "obs", 3),
    ("화 중간-1", "joint_effort", "obs", 12),
    ("화 도전", "track_lin_vel_xy_exp", "rew", None),
    ("수 기본-1", "generated_commands", "obs", 3),
    ("수 기본-2", "ang_vel_xy_l2", "pen", None),
    ("수 기본-3", "flat_orientation_l2", "pen", None),
    ("수 기본-4", "joint_vel_l2", "pen", None),
    ("수 기본-4", "joint_acc_l2", "pen", None),
    ("수 기본-4", "joint_torques_l2", "pen", None),
    ("수 기본-5", "action_rate_l2", "pen", None),
    ("수 중간-1", "energy", "pen", None),
    ("수 중간-2", "action_smoothness_l2", "pen", None),
    ("수 중간-3", "base_height_l2", "pen", None),
    ("수 도전", "joint_position_penalty_command_aware", "pen", None),
    ("목 기본-2", "height_scan", "obs", 187),
    ("목 기본-4", "contact_forces", "pen", None),
    ("목 중간-1", "feet_slide", "pen", None),
    ("목 중간-2", "foot_clearance_reward", "rew", None),
    ("목 도전", "feet_air_time", "rew", None),
    ("목 중간-4", "randomize_actuator_gains", "event", None),
]
# 등록 항목 (a2_env_cfg.py). policy: 이름, 태그, func, scale, 목요일 noise 범위 / critic: 이름, 태그, func, scale
POLICY = [
    ("base_ang_vel", "화 기본-5", "base_ang_vel", 0.2, 0.2),
    ("projected_gravity", "화 기본-5", "projected_gravity", None, 0.05),
    ("velocity_commands", "수 기본-1", "generated_commands", None, None),
    ("joint_pos_rel", "화 기본-5", "joint_pos_rel", None, 0.01),
    ("joint_vel_rel", "화 기본-5", "joint_vel_rel", 0.05, 1.5),
    ("last_action", "화 예시", "last_action", None, None),
]
CRITIC = [
    ("base_lin_vel", "화 중간-2", "base_lin_vel", None),
    ("base_ang_vel", "화 중간-2", "base_ang_vel", 0.2),
    ("projected_gravity", "화 중간-2", "projected_gravity", None),
    ("velocity_commands", "수 기본-1", "generated_commands", None),
    ("joint_pos_rel", "화 중간-2", "joint_pos_rel", None),
    ("joint_vel_rel", "화 중간-2", "joint_vel_rel", 0.05),
    ("joint_effort", "화 중간-2", "joint_effort", 0.01),
    ("last_action", "화 예시", "last_action", None),
    ("height_scan", "목 기본-2", "height_scan_finite", None),
]
REWARDS = [  # 이름, 태그, func, weight (힌트 표의 값)
    ("track_lin_vel_xy", "화 도전", "track_lin_vel_xy_exp", 1.0),
    ("track_ang_vel_z", "화 예시", "track_ang_vel_z_exp", 0.5),
    ("base_linear_velocity", "수 기본-6", "lin_vel_z_l2", -2.0),
    ("base_angular_velocity", "수 기본-6", "ang_vel_xy_l2", -0.05),
    ("base_height", "수 중간-4", "base_height_l2", -1.0),
    ("joint_vel", "수 기본-6", "joint_vel_l2", -1e-5),
    ("joint_acc", "수 기본-6", "joint_acc_l2", -2.5e-7 * 0.5 * 0.5),
    ("joint_torques", "수 기본-6", "joint_torques_l2", -1e-5 * 0.5),
    ("action_rate", "수 기본-6", "action_rate_l2", -0.01 * 0.8),
    ("action_smoothness", "수 중간-4", "action_smoothness_l2", -0.01 * 0.5),
    ("dof_pos_limits", "수 중간-4", "joint_pos_limits", -10.0),
    ("energy", "수 중간-4", "energy", -2e-5 * 0.5),
    ("flat_orientation_l2", "수 기본-6", "flat_orientation_l2", -0.01),
    ("joint_pos", "수 도전", "joint_position_penalty_command_aware", -0.05),
    ("feet_air_time", "목 도전", "feet_air_time", 0.5),
    ("air_time_variance", "목 중간-3", "air_time_variance_penalty", -0.2),
    ("feet_slide", "목 중간-3", "feet_slide", -0.1),
    ("feet_contact_forces", "목 기본-4", "contact_forces", -1e-5),
    ("feet_clearance", "목 중간-3", "foot_clearance_reward", 0.5),
    ("undesired_contacts", "목 예시", "undesired_contacts", -2.0),
]
EVENTS = [
    ("push_robot", "목 기본-3", "push_by_setting_velocity"),
    ("actuator_gains", "목 중간-4", "randomize_actuator_gains"),
]
DIMS = {1: (210, 57, 2), 2: (225, 60, 14), 3: (225, 247, 20)}  # policy, critic, reward 수
DIM_TAGS = {1: ("화 기본-5", "화 중간-2"), 2: ("수 기본-1", "수 기본-1"), 3: ("수 기본-1", "목 기본-2")}


def day_of(tag):
    return 0 if "예시" in tag else {"화": 1, "수": 2, "목": 3}[tag.split()[0]]


rows, blockers, warnings = [], [], []


def report(part, tag, item, result, note="", block=True):
    """result가 통과가 아니면 block=True는 남은 태그, False는 경고로 셈."""
    rows.append((part, tag, item, result, note))
    if result != "통과":
        (blockers if block else warnings).append(tag)


def fname(term):
    return getattr(term.func, "__name__", type(term.func).__name__)


def terms(group):
    return {k: v for k, v in vars(group).items() if v is not None and hasattr(v, "func")}


def check_registration(cfg):
    pol, cri = terms(cfg.observations.policy), terms(cfg.observations.critic)
    rew, evt = terms(cfg.rewards), terms(cfg.events)

    for part, have, spec in (("policy", pol, POLICY), ("critic", cri, CRITIC)):
        want = [s for s in spec if day_of(s[1]) <= DAY]
        for s in want:
            name, tag, func, scale = s[:4]
            t = have.get(name)
            if t is None:
                report(part, tag, name, "미등록", "a2_env_cfg.py에 등록 줄이 없음")
                continue
            note = []
            if fname(t) != func:
                note.append(f"func가 mdp.{func}가 아님 ({fname(t)})")
            if (t.scale or 1.0) != (scale or 1.0):
                note.append(f"scale {t.scale} (힌트: {scale or '없음'})")
            if t.clip is None or tuple(t.clip) not in ((-100, 100), (-1.0, 5.0)):
                note.append(f"clip {t.clip}")
            want_noise = s[4] if part == "policy" and DAY >= 3 else None
            if want_noise is None and t.noise is not None:
                note.append("noise가 있으면 안 됨" + (" (critic)" if part == "critic" else ""))
            elif want_noise is not None:
                lo, hi = getattr(t.noise, "n_min", None), getattr(t.noise, "n_max", None)
                if t.noise is None:
                    note.append("noise 없음")
                elif lo is None or abs(lo + want_noise) > 1e-9 or abs(hi - want_noise) > 1e-9:
                    note.append(f"noise 범위 ({lo}, {hi}), 힌트 ±{want_noise}")
                if note and note[-1].startswith("noise"):
                    tag = "목 기본-1"
            report(part, tag, name, "오류" if note else "통과", "; ".join(note))
        names = [s[0] for s in want]
        if [k for k in have if k in names] != [k for k in names if k in have]:
            tag = DIM_TAGS[DAY][0 if part == "policy" else 1]
            report(part, tag, "등록 순서", "오류", f"지금: {', '.join(k for k in have if k in names)} / 맞는 순서: {', '.join(names)}")
        extra = [k for k in have if k not in names]
        if extra:
            report(part, "-", "그날 범위 밖 항목", "경고", ", ".join(extra), block=False)

    names = [r[0] for r in REWARDS if day_of(r[1]) <= DAY]
    for name, tag, func, weight in REWARDS:
        if day_of(tag) > DAY:
            continue
        t, challenge = rew.get(name), "도전" in tag
        if t is None:
            note = "미등록: 학습은 시작되지만 이 항목이 빠짐" if challenge else "a2_env_cfg.py에 등록 줄이 없음"
            report("reward", tag, name, "미등록", note, block=not challenge)
            continue
        note = []
        if fname(t) != func:
            note.append(f"func가 mdp.{func}가 아님 ({fname(t)})")
        if t.weight * weight <= 0:
            note.append(f"weight 부호가 반대 ({t.weight})")
        elif abs(t.weight - weight) > 1e-6 * abs(weight):
            note.append(f"weight {t.weight} (힌트 표: {weight})")
        report("reward", tag, name, "오류" if note else "통과", "; ".join(note))
    extra = [k for k in rew if k not in names]
    if extra:
        report("reward", "-", "그날 범위 밖 항목", "경고", ", ".join(extra), block=False)

    for name, tag, func in EVENTS:
        t = evt.get(name)
        if day_of(tag) > DAY:
            if t is not None:
                report("event", tag, name, "경고", "그날 범위 밖 항목", block=False)
        elif t is None:
            report("event", tag, name, "미등록", "a2_env_cfg.py에 등록 줄이 없음")
        elif fname(t) != func:
            report("event", tag, name, "오류", f"func가 mdp.{func}가 아님 ({fname(t)})")
        elif t.mode != ("interval" if name == "push_robot" else "reset"):
            report("event", tag, name, "오류", f'mode "{t.mode}"')
        else:
            report("event", tag, name, "통과")
    return pol, cri, rew, evt


def main():
    import gymnasium as gym
    import torch
    import isaaclab.terrains as terrain_gen
    from isaaclab.actuators import ImplicitActuator
    from isaaclab.managers import SceneEntityCfg as SE
    from isaaclab.utils.math import quat_apply

    from edu_rl_lab import mdp
    from edu_rl_lab.tasks import A2_TASK_ID, load_task_configs

    feet = {"body_names": ".*_foot"}
    default_params = {  # 미등록 함수를 시험할 때 쓰는 인자 (힌트 표와 같은 값)
        "generated_commands": {"command_name": CMD},
        "height_scan": {"sensor_cfg": SE("height_scanner")},
        "track_lin_vel_xy_exp": {"command_name": CMD, "std": 0.5},
        "base_height_l2": {"target_height": 0.42, "sensor_cfg": SE("height_scanner")},
        "joint_position_penalty_command_aware": {
            "asset_cfg": SE("robot", joint_names=".*"), "stand_still_scale": 5.0, "velocity_threshold": 0.3},
        "contact_forces": {"threshold": 200.0, "sensor_cfg": SE("contact_forces", **feet)},
        "feet_slide": {"asset_cfg": SE("robot", **feet), "sensor_cfg": SE("contact_forces", **feet)},
        "foot_clearance_reward": {
            "asset_cfg": SE("robot", **feet), "target_height": 0.25, "std": 0.5, "tanh_mult": 2.0},
        "feet_air_time": {"sensor_cfg": SE("contact_forces", **feet), "command_name": CMD, "threshold": 0.5},
        "randomize_actuator_gains": {
            "asset_cfg": SE("robot"), "stiffness_range": (0.9, 1.1), "damping_range": (0.9, 1.1),
            "strength_range": (0.9, 1.1)},
    }

    # ---- 1. 등록 상태 (환경 없이 cfg만 읽음)
    try:
        cfg, _ = load_task_configs(A2_TASK_ID)
    except Exception as e:
        report("cfg", "a2_env_cfg.py", "cfg 읽기", "오류", f"{type(e).__name__}: {e}")
        return
    pol, cri, rew, evt = check_registration(cfg)
    used = {fname(t): t.params for t in [*pol.values(), *cri.values(), *rew.values(), *evt.values()]}
    if "height_scan_finite" in used:
        used["height_scan"] = used["height_scan_finite"]

    # ---- 2. 작성 여부: 소스에 raise 줄이 남았는지
    written = {}
    for tag, name, kind, _ in FUNCS:
        if day_of(tag) > DAY:
            continue
        written[name] = 'raise NotImplementedError("w2' not in inspect.getsource(getattr(mdp, name))
        if not written[name]:
            note = "raise NotImplementedError 줄이 남아 있음"
            if name in used:
                note += " (등록돼 있어 이대로는 학습이 멈춤)"
            report("함수", tag, name, "미작성", note, block="도전" not in tag or name in used)

    # ---- 3. 시험용 환경: 평지, 등록 항목은 제공 항목만 남기고 함수는 직접 불러 봄
    probe = copy.deepcopy(cfg)
    probe.seed = 0
    probe.scene.num_envs = args.num_envs
    gen = probe.scene.terrain.terrain_generator
    gen.sub_terrains = {"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)}
    gen.num_rows, gen.num_cols = 2, 2
    probe.scene.terrain.visual_material = None
    probe.scene.sky_light.spawn.texture_file = None
    probe.commands.base_velocity.debug_vis = False
    for group in (probe.observations.policy, probe.observations.critic):
        for k in terms(group):
            if k != "last_action":
                setattr(group, k, None)
    for k in terms(probe.rewards):
        if k != "track_ang_vel_z":
            setattr(probe.rewards, k, None)
    for k in ("push_robot", "actuator_gains"):
        if getattr(probe.events, k, None) is not None:
            setattr(probe.events, k, None)
    env = gym.make(A2_TASK_ID, cfg=probe).unwrapped
    n, dev = env.num_envs, env.device
    robot, cmd = env.scene["robot"], env.command_manager.get_term(CMD)

    def params_of(name):
        p = copy.deepcopy(used.get(name, default_params.get(name, {})))
        for v in p.values():
            if isinstance(v, SE):
                v.resolve(env.scene)
        return p

    def safe(name):
        try:
            fn = getattr(mdp, name)
            if name == "randomize_actuator_gains":
                return fn(env, torch.arange(n, device=dev), **params_of(name)), None
            return fn(env, **params_of(name)), None
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"

    def set_body_vel(v_b, w_b):  # body frame 속도를 world frame으로 바꿔 시뮬레이터에 써 넣음
        q = robot.data.root_quat_w
        robot.write_root_velocity_to_sim(torch.cat([quat_apply(q, v_b), quat_apply(q, w_b)], dim=-1))

    def vec(x, y, z):
        return torch.tensor([x, y, z], device=dev, dtype=torch.float32).repeat(n, 1)

    def with_command(value, name):
        saved = cmd.vel_command_b.clone()
        cmd.vel_command_b[:] = torch.tensor(value, device=dev)
        out = safe(name)
        cmd.vel_command_b[:] = saved
        return out

    def properties(name, standing, reset_pos):
        """물리적으로 당연한 성질. 문제 문장 목록을 돌려줌."""
        if name == "projected_gravity":
            g = standing[name][0]
            if not (g[:, 2] < -0.9).all() or not torch.allclose(g.norm(dim=1), torch.ones(n, device=dev), atol=1e-2):
                return [f"서 있을 때 z가 -1 근처가 아님 (z 평균 {g[:, 2].mean().item():.3f}). body frame(_b)인지 확인"]
        elif name == "joint_pos_rel":
            o, e = reset_pos
            if e is None and o.abs().max() > 0.05:
                return [f"reset 직후(기본 자세)인데 0 근처가 아님 (최대 {o.abs().max().item():.3f}). 기본값을 뺐는지 확인"]
        elif name in ("base_lin_vel", "base_ang_vel"):
            v_b, w_b = vec(0.3, -0.2, 0.1), vec(0.2, -0.1, 0.5)
            set_body_vel(v_b, w_b)
            o, e = safe(name)
            if e or not torch.allclose(o, v_b if name == "base_lin_vel" else w_b, atol=1e-2):
                return ["써 넣은 몸체 속도와 다름. 선속도/각속도, body frame(_b)인지 확인"]
        elif name == "generated_commands":
            o, e = safe(name)
            if e or not torch.allclose(o, cmd.command):
                return ["command manager의 값과 다름"]
        elif name == "track_lin_vel_xy_exp":
            c, z = cmd.command.clone(), torch.zeros(n, device=dev)
            set_body_vel(torch.stack([c[:, 0], c[:, 1], z], 1), torch.stack([z, z, c[:, 2]], 1))
            same, e1 = safe(name)
            set_body_vel(torch.stack([c[:, 0] + 0.5, c[:, 1] - 0.5, z], 1), torch.stack([z, z, c[:, 2]], 1))
            off, e2 = safe(name)
            if e1 or e2:
                return [e1 or e2]
            p = []
            if (same < 0.99).any():
                p.append(f"몸체 속도 = command인데 1 근처가 아님 (최소 {same.min().item():.3f})")
            if (off > 0.9).any():
                p.append("x, y 속도가 command와 0.5씩 달라도 1 근처. 오차 계산 확인")
            return p
        elif name == "joint_position_penalty_command_aware":
            set_body_vel(vec(0, 0, 0), vec(0, 0, 0))
            still, e1 = with_command([0.0, 0.0, 0.0], name)
            walk, e2 = with_command([0.5, 0.0, 0.0], name)
            scale = params_of(name).get("stand_still_scale", 5.0)
            if e1 or e2:
                return [e1 or e2]
            if (walk <= 0).all() or not torch.allclose(still, scale * walk, rtol=1e-3, atol=1e-6):
                return [f"정지(command 0, 몸체 속도 0)일 때 벌점이 움직일 때의 stand_still_scale({scale})배가 아님"]
        elif name == "feet_air_time":
            o, e = with_command([0.0, 0.0, 0.0], name)
            if e or (o != 0).any():
                return [e or "command가 0인데 0이 아님. 정지 명령 mask 확인"]
        return []

    def motor_dr():
        tag, name = "목 중간-4", "randomize_actuator_gains"
        acts = [a for a in robot.actuators.values() if not isinstance(a, ImplicitActuator)]
        kp0, kd0 = robot.data.default_joint_stiffness.clone(), robot.data.default_joint_damping.clone()
        p = []
        if not acts:
            p.append("explicit actuator가 없음")
        if (kp0 == 0).all() or (kd0 == 0).all():
            p.append("default_joint_stiffness/damping이 0. 기본 gain을 가져오는 곳을 바꿔야 함 (강의자 확인)")
        for _ in range(10):  # 여러 번 불러도 누적되지 않아야 함
            _, e = safe(name)
            if e:
                p.append(e)
                break
        lo, hi = 0.9 * 0.9 - 1e-4, 1.1 * 1.1 + 1e-4
        for a in (acts if not p else []):
            for label, now, base in (("Kp", a.stiffness, kp0), ("Kd", a.damping, kd0)):
                r = now / base[:, a.joint_indices]
                if not torch.isfinite(r).all() or (r < lo).any() or (r > hi).any():
                    p.append(f"{label} 배율이 0.81..1.21 밖 ({r.min().item():.3f}..{r.max().item():.3f}). 기본값에서 다시 곱하는지 확인")
                elif r.std() < 1e-4:
                    p.append(f"{label}가 바뀌지 않음")
        note = "; ".join(p) if p else f"Kp {acts[0].stiffness.min().item():.2f}..{acts[0].stiffness.max().item():.2f}"
        report("함수", tag, name, "오류" if p else "통과", note)

    try:
        env.reset()
        reset_pos = safe("joint_pos_rel") if written.get("joint_pos_rel") else (None, None)
        zero = torch.zeros(n, env.action_manager.total_action_dim, device=dev)
        for _ in range(50):  # 1초 동안 기본 자세
            env.step(zero)
        checked = [x for x in FUNCS if day_of(x[0]) <= DAY and written.get(x[1]) and x[2] != "event"]
        standing = {x[1]: safe(x[1]) for x in checked}
        for _ in range(20):  # 무작위 action: 벌점이 0이 아닌 상태
            env.step(torch.rand_like(zero) - 0.5)
        moving = {x[1]: safe(x[1]) for x in checked}

        cols = {"last_action": env.action_manager.total_action_dim}
        for tag, name, kind, ncol in checked:
            block = "도전" not in tag or name in used
            p = []
            shape = (n, ncol) if kind == "obs" else (n,)
            for label, (o, e) in (("정지", standing[name]), ("움직임", moving[name])):
                if e:
                    p.append(f"{label}: {e}")
                elif tuple(o.shape) != shape:
                    p.append(f"shape {tuple(o.shape)} (맞는 값 {shape})")
                elif not torch.isfinite(o.float()).all():
                    p.append(f"{label}: nan/inf 있음")
                elif kind == "pen" and (o < 0).any():
                    p.append(f"{label}: 음수 {o.min().item():.3g} (벌점 함수는 0 이상)")
                elif kind == "rew" and name != "feet_air_time" and ((o <= 0).any() or (o > 1 + 1e-5).any()):
                    p.append(f"{label}: 0 초과 1 이하가 아님 ({o.min().item():.3g}..{o.max().item():.3g})")
                if p:
                    break
            if not p:
                if kind == "obs":
                    cols[name] = ncol
                p = properties(name, standing, reset_pos)
            o = moving[name][0]
            note = "; ".join(p) if p else (f"평균 {o.float().mean().item():.3g}" if o is not None else "")
            result = "미작성" if "NotImplementedError" in note else ("오류" if p else "통과")
            report("함수", tag, name, result, note, block=block)

        if DAY >= 3 and written.get("randomize_actuator_gains"):
            motor_dr()

        # ---- 차원: 등록 항목의 열 수 × history
        cols["height_scan_finite"] = cols.get("height_scan")
        try:
            pdim = sum(cols[fname(t)] for t in pol.values()) * (cfg.observations.policy.history_length or 1)
            cdim = sum(cols[fname(t)] for t in cri.values()) * (cfg.observations.critic.history_length or 1)
        except (KeyError, TypeError):
            report("차원", "-", "policy·critic 입력", "경고", "등록 항목 중 오류·미작성이 있어 계산하지 않음", block=False)
        else:
            want_p, want_c, _ = DIMS[DAY]
            report("차원", DIM_TAGS[DAY][0], "policy 입력", "통과" if pdim == want_p else "오류", f"{pdim} (맞는 값 {want_p})")
            report("차원", DIM_TAGS[DAY][1], "critic 입력", "통과" if cdim == want_c else "오류", f"{cdim} (맞는 값 {want_c})")
        want_r = DIMS[DAY][2]
        report("차원", "-", "reward 항목 수", "통과" if len(rew) == want_r else "경고", f"{len(rew)} (맞는 값 {want_r})",
               block=False)
    finally:
        env.close()


def print_table():
    print("\n구분 | 태그 | 항목 | 결과 | 내용")
    print("-" * 80)
    for r in rows:
        print(" | ".join(str(x) for x in r))
    print("-" * 80)
    order = {"화": 1, "수": 2, "목": 3}
    left = sorted(set(blockers), key=lambda t: (order.get(t.split()[0], 0), t))
    warn = sorted(set(warnings) - set(blockers) - {"-"})
    if warn:
        print("경고 (학습은 시작됨): " + ", ".join(warn))
    print(("남은 태그: " + ", ".join(left)) if left else "학습 시작 가능")
    return not left


if __name__ == "__main__":
    ok = False
    try:
        main()
        ok = print_table()
    except Exception:
        import traceback

        traceback.print_exc()
        print_table()
        print("점검 중 오류: 위 traceback을 강의자에게 보여 주세요")
    finally:
        app.close()
    sys.exit(0 if ok else 1)
