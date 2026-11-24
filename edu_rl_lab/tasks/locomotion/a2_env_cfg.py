"""A2 locomotion training configuration."""

import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from edu_rl_lab.assets.a2 import A2_CFG as ROBOT_CFG
from edu_rl_lab import mdp


COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=COBBLESTONE_ROAD_CFG,
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Configuration for events."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.15),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # w2 실습
    # [w2 목 | 기본-3] push_robot 등록 (함수 mdp.push_by_setting_velocity는 제공)
    #
    # - 형식: 위 EventTerm들과 같음. 이름 push_robot
    # - mode: "startup" = 시작 때 한 번, "reset" = 에피소드마다, "interval" = 정해진 시간 간격마다
    # - 이 항목: mode "interval", interval_range_s=(5.0, 10.0)
    #   params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}}  (base 속도를 갑자기 바꿔 미는 효과)
    #
    # (여기에 작성)

    # w2 실습
    # [w2 목 | 중간-4] motor DR 등록 (함수는 randomize_events.py의 randomize_actuator_gains)
    #
    # - 이름 actuator_gains, mode "reset" (에피소드마다 새로 뽑음)
    # - params: asset_cfg SceneEntityCfg("robot")
    #           stiffness_range, damping_range, strength_range 모두 (0.9, 1.1)
    #
    # (여기에 작성)


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.6, 0.6), lin_vel_y=(-0.4, 0.4), ang_vel_z=(-1.0, 1.0)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    LegJointPositionAction = mdp.JointPositionActionWithHistoryCfg(
        asset_name="robot",
        joint_names=[
            ".*_hip_joint",
            ".*_thigh_joint",
            ".*_calf_joint",
        ],
        scale=0.25,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # w2 실습
        # [w2 화 | 기본-5] policy 관측 등록
        #
        # - 형식: 이름 = ObsTerm(func=mdp.함수, scale=..., clip=(-100, 100))
        # - 순서: base_ang_vel, projected_gravity, joint_pos_rel, joint_vel_rel, last_action
        #         이 순서가 곧 신경망 입력 순서. deploy.yaml과 같아야 함
        # - scale: base_ang_vel 0.2, joint_vel_rel 0.05, 나머지는 없음
        # - 아래 __post_init__의 history_length = 5: 항목마다 최근 5개를 이어 붙임 → 42 × 5 = 210
        #
        # (여기에 작성)
        #
        # [w2 목 | 기본-1] 관측 noise (policy만, 목요일에 작성)
        # - 위 4줄에 noise=Unoise(n_min=..., n_max=...) 추가. Unoise = 균등분포 noise
        #   base_ang_vel ±0.2, projected_gravity ±0.05, joint_pos_rel ±0.01, joint_vel_rel ±1.5
        # - velocity_commands, last_action에는 넣지 않음 (센서로 재는 값이 아님)
        # - noise는 __post_init__의 enable_corruption = True일 때만 적용됨
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100))
        #
        # [w2 수 | 기본-1] velocity_commands는 projected_gravity 바로 다음 자리 (수요일에 작성)
        # - func mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}, scale 없음
        # - 입력 차원 45 × 5 = 225. deploy.yaml의 observations 순서도 같이 맞출 것
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100))
        #
        # [w2 화 | 예시] last_action 등록 (풀이 예시). 함수는 mdp_observations.py
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))

        def __post_init__(self):
            # Term-major history: five samples of each term, then the next term.
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        # w2 실습
        # [w2 화 | 중간-2] critic 관측 등록
        #
        # - 순서: base_lin_vel, base_ang_vel, projected_gravity, joint_pos_rel, joint_vel_rel, joint_effort, last_action
        # - policy와 같은 항목은 같은 scale, clip. noise는 넣지 않음
        # - base_lin_vel: scale 없음, clip=(-100, 100) / joint_effort: scale 0.01, clip=(-100, 100)
        # - critic은 history 없음 (아래 __post_init__ 참고) → 3+3+3+12+12+12+12 = 57
        #
        # (여기에 작성)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100))
        #
        # [w2 수 | 기본-1] critic의 velocity_commands도 projected_gravity 바로 다음 자리 (57 → 60, 수요일에 작성)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100))
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100, 100))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))

        # [w2 목 | 기본-2] height_scan 등록 (critic 맨 끝, 60 → 247, 목요일에 작성)
        # - 이름 height_scan, func mdp.height_scan_finite (percep_observations.py, 안 닿은 ray를 0으로 바꾸는 함수, 제공)
        # - params={"sensor_cfg": SceneEntityCfg("height_scanner")}, clip=(-1.0, 5.0)
        # - 여러 줄로 써도 됨. 괄호 안에서는 줄을 바꿔도 이어진 한 줄로 읽힘

        def __post_init__(self):
            self.concatenate_terms = True

    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # w2 실습
    # [w2 화 | 도전] track_lin_vel_xy 등록 (함수는 tracking_rewards.py의 track_lin_vel_xy_exp)
    #
    # - 아래 track_ang_vel_z(풀이 예시)와 같은 형태
    # - weight 1.0, params command_name "base_velocity", std math.sqrt(0.25)
    # - 이름은 반드시 track_lin_vel_xy (curriculum lin_vel_cmd_levels가 이 이름으로 찾음)
    #
    # (여기에 작성)
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    #
    # [w2 화 | 예시] track_ang_vel_z 등록 (풀이 예시)
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # w2 실습
    # [w2 수 | 기본-6] base, joint, action 패널티 등록 (weight가 음수면 패널티)
    #
    # - 예시 (그대로 써도 됨): base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    # - 나머지는 같은 형식, params 없음
    #   | 이름                  | func                     | weight              |
    #   | base_angular_velocity | mdp.ang_vel_xy_l2        | -0.05               |
    #   | joint_vel             | mdp.joint_vel_l2         | -1e-5               |
    #   | joint_acc             | mdp.joint_acc_l2         | -2.5e-7 * 0.5 * 0.5 |
    #   | joint_torques         | mdp.joint_torques_l2     | -1e-5 * 0.5         |
    #   | action_rate           | mdp.action_rate_l2       | -0.01 * 0.8         |
    #   | flat_orientation_l2   | mdp.flat_orientation_l2  | -0.01               |
    #
    # [w2 수 | 중간-4] params가 있는 패널티와 나머지 등록
    #   | 이름              | func                      | weight      | params                                              |
    #   | base_height       | mdp.base_height_l2        | -1.0        | target_height 0.42, sensor_cfg = 센서 height_scanner |
    #   | action_smoothness | mdp.action_smoothness_l2  | -0.01 * 0.5 | 없음                                                |
    #   | dof_pos_limits    | mdp.joint_pos_limits      | -10.0       | 없음                                                |
    #   | energy            | mdp.energy                | -2e-5 * 0.5 | 없음                                                |
    # - params 형식: params={"키": 값, ...}. 센서, 로봇 지정은 SceneEntityCfg("이름")
    #
    # [w2 수 | 도전] joint_pos 등록 (함수 joint_position_penalty_command_aware도 직접 작성)
    # - weight -0.05
    # - params: asset_cfg = 로봇 전체 관절 SceneEntityCfg("robot", joint_names=".*")
    #           stand_still_scale 5.0, velocity_threshold 0.3
    #
    # (여기에 작성)
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-1.0,
        params={
            "target_height": 0.42,
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-5)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7 * 0.5 * 0.5)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5 * 0.5)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01 * 0.8)
    action_smoothness = RewTerm(func=mdp.action_smoothness_l2, weight=-0.01 * 0.5)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5 * 0.5)

    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.01)

    joint_pos = RewTerm(
        func=mdp.joint_position_penalty_command_aware,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.3,
        },
    )

    # w2 실습
    # [w2 목 | 예시] undesired_contacts 등록 (풀이 예시, 그대로 써도 됨)
    #     undesired_contacts = RewTerm(
    #         func=mdp.undesired_contacts,
    #         weight=-2.0,
    #         params={
    #             "threshold": 1,
    #             "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_hip", ".*_thigh", ".*_calf"]),
    #         },
    #     )
    # - body_names는 정규식. ".*_hip" = 이름이 _hip으로 끝나는 모든 body (FL_hip, FR_hip, RL_hip, RR_hip)
    #
    # [w2 목 | 기본-4] feet_contact_forces 등록
    # - func mdp.contact_forces, weight -1e-5
    # - params: threshold 200.0, sensor_cfg = contact_forces 센서의 네 발
    #   (발 body 이름: FL_foot, FR_foot, RL_foot, RR_foot)
    #
    # [w2 목 | 중간-3] feet reward 등록 (발 = 위 예시처럼 정규식 하나로)
    #   | 이름              | func                          | weight | params                                                 |
    #   | air_time_variance | mdp.air_time_variance_penalty | -0.2   | sensor_cfg (접촉 센서, 발)                             |
    #   | feet_slide        | mdp.feet_slide                | -0.1   | asset_cfg (로봇, 발), sensor_cfg (접촉 센서, 발)       |
    #   | feet_clearance    | mdp.foot_clearance_reward     | 0.5    | asset_cfg (로봇, 발), target_height 0.25, std 0.5, tanh_mult 2.0 |
    # - 로봇의 body를 고를 때도 SceneEntityCfg("robot", body_names=...)
    #
    # [w2 목 | 도전] feet_air_time 등록 (함수 feet_air_time도 직접 작성)
    # - weight 0.5 (양수 = 상)
    # - params: sensor_cfg (접촉 센서, 발), command_name "base_velocity", threshold 0.5
    #
    # (여기에 작성)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link"]), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels, params={"delta_lin": (-0.1, 0.1)})


@configclass
class A2EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

        self.reward_type_dict = {
            "task": ["track_lin_vel_xy", "track_ang_vel_z"],
            "joint": [
                "joint_pos",
                "joint_vel",
                "joint_acc",
                "joint_torques",
                "action_rate",
                "dof_pos_limits",
                "energy",
            ],
        }

        self.obs_type_dict = {
            "partial_prop": ["base_ang_vel", "projected_gravity", "joint_pos_rel", "joint_vel_rel", "last_action"],
            "estimation": ["base_lin_vel"],
        }


@configclass
class A2PlayEnvCfg(A2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.terrain.terrain_generator.num_rows = 10
        self.scene.terrain.terrain_generator.num_cols = 6
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges.copy()
        # w2 실습 재생 카메라
        # - env 0 로봇을 따라감 (기본값은 world 원점 고정이라 --video 영상에 로봇이 안 보임)
        # - eye, lookat은 로봇 몸통 기준 위치 (m)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (2.5, 2.5, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.0)
