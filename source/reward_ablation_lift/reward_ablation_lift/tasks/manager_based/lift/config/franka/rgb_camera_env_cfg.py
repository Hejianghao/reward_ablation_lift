from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

import reward_ablation_lift.tasks.manager_based.lift.mdp as lift_mdp
from reward_ablation_lift.tasks.manager_based.lift.lift_env_cfg import ObjectTableSceneCfg, ObservationsCfg

from .joint_pos_env_cfg import FrankaCubeLiftEnvCfg


@configclass
class CameraObjectTableSceneCfg(ObjectTableSceneCfg):
    """Scene with an added overhead tiled camera."""

    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.5, 0.0, 2.0),
            rot=(0.7071068, 0.7071068, 0.0, 0.0),
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
        width=128,
        height=128,
    )


@configclass
class RGBCameraObservationsCfg(ObservationsCfg):
    """Observation specifications for the visuomotor lift policy."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Return image and proprio observations as separate dictionary entries."""

        image = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("tiled_camera"), "data_type": "rgb", "normalize": True},
        )
        proprio = ObsTerm(func=lift_mdp.proprio_observations)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class FrankaCubeLiftRGBCameraEnvCfg(FrankaCubeLiftEnvCfg):
    # Replace scene with camera-enabled version
    scene: CameraObjectTableSceneCfg = CameraObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: RGBCameraObservationsCfg = RGBCameraObservationsCfg()

    def __post_init__(self):
        # parent sets scene.robot, scene.object, scene.ee_frame
        super().__post_init__()


@configclass
class FrankaCubeLiftRGBCameraEnvCfg_PLAY(FrankaCubeLiftRGBCameraEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
