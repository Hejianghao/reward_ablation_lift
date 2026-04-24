from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

import reward_ablation_lift.tasks.manager_based.lift.mdp as lift_mdp
from reward_ablation_lift.tasks.manager_based.lift.lift_env_cfg import ObjectTableSceneCfg, ObservationsCfg

from .baseline_lift_env_cfg import BaselineLiftEnvCfg
from isaaclab.utils.noise.noise_cfg import GaussianNoiseCfg


@configclass
class RGBCameraLiftSceneCfg(ObjectTableSceneCfg):
    """Scene with an added overhead tiled camera."""
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, -0.05),       # 手腕后方 5cm
            rot=(1.0, 0.0, 0.0, 0.0),   # 与 panda_hand 朝向一致
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,           # 缩短焦距，扩大 FOV
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.02, 2.0),  # 近端截断改小，避免近距裁剪
        ),
        width=120,
        height=160,
    )

    # tiled_camera: TiledCameraCfg = TiledCameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Camera",
    #     offset=TiledCameraCfg.OffsetCfg(
    #         pos=(0.5, 0.0, 2.2),
    #         rot=(0.7071068, 0.0, 0.7071068, 0.0),
    #         convention="world",
    #         # rot=(0.0, 0.0, 1.0, 0.0),
    #         # convention="opengl",
    #     ),
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
    #         intrinsic_matrix=[
    #             617.7935 / 2,
    #             0.0,
    #             326.6838 / 2,
    #             0.0,
    #             617.8885 / 2,
    #             240.0364 / 2,
    #             0.0,
    #             0.0,
    #             1.0,
    #         ],
    #         height=120,
    #         width=160,
    #         clipping_range=(0.01, 10.0),
    #     ),
    #     height=120,
    #     width=160,
    # )


# @configclass
# class RGBCameraLiftObjectTableSceneCfg(CameraObjectTableSceneCfg):
#     """Camera scene with contact sensors on Franka finger tips."""

#     contact_sensor_finger_1: ContactSensorCfg = ContactSensorCfg(
#         prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
#         update_period=0.0,
#         history_length=6,
#         debug_vis=False,
#         filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
#     )

#     contact_sensor_finger_2: ContactSensorCfg = ContactSensorCfg(
#         prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
#         update_period=0.0,
#         history_length=6,
#         debug_vis=False,
#         filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
#     )


@configclass
class RGBCameraLiftObservationsCfg(ObservationsCfg):
    """Observation specifications for the visuomotor lift policy."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Return image and proprio observations as separate dictionary entries."""

        image = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("tiled_camera"), 
                "data_type": "rgb", 
                "normalize": True,
            },
            noise=GaussianNoiseCfg(mean=0.0, std=0.04),
        )

        proprio = ObsTerm(func=lift_mdp.proprio_observations)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    


@configclass
class RGBCameraLiftEnvCfg(BaselineLiftEnvCfg):
    # Replace scene with camera-enabled version
    scene: RGBCameraLiftSceneCfg = RGBCameraLiftSceneCfg(num_envs=512, env_spacing=2.5)
    observations: RGBCameraLiftObservationsCfg = RGBCameraLiftObservationsCfg()

    

    def __post_init__(self):
        # parent sets scene.robot, scene.object, scene.ee_frame
        super().__post_init__()


@configclass
class RGBCameraLiftEnvCfg_PLAY(RGBCameraLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
