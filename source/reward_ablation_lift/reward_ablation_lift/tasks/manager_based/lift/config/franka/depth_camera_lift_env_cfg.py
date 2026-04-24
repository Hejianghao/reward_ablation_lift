from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.utils.noise.noise_cfg import GaussianNoiseCfg

import reward_ablation_lift.tasks.manager_based.lift.mdp as lift_mdp
from reward_ablation_lift.tasks.manager_based.lift.lift_env_cfg import ObjectTableSceneCfg, ObservationsCfg

from .baseline_lift_env_cfg import BaselineLiftEnvCfg


@configclass
class DepthCameraLiftSceneCfg(ObjectTableSceneCfg):
    """Scene with depth camera instead of RGB."""
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.55, 0.0, 1.7),
            rot=(0.7071068, 0.0, 0.7071068, 0.0),
            convention="world",
        ),
        data_types=["depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=48.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
        width=120,
        height=160,
    )


@configclass
class DepthCameraLiftObservationsCfg(ObservationsCfg):
    """Observations using depth image instead of RGB."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Depth image + proprio."""

        depth_image = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("tiled_camera"),
                "data_type": "depth",
                "normalize": True,
            },
            noise=GaussianNoiseCfg(mean=0.0, std=0.003),
        )

        proprio = ObsTerm(func=lift_mdp.proprio_observations)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class DepthCameraLiftEnvCfg(BaselineLiftEnvCfg):
    scene: DepthCameraLiftSceneCfg = DepthCameraLiftSceneCfg(num_envs=512, env_spacing=2.5)
    observations: DepthCameraLiftObservationsCfg = DepthCameraLiftObservationsCfg()

    def __post_init__(self):
        super().__post_init__()


@configclass
class DepthCameraLiftEnvCfg_PLAY(DepthCameraLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
