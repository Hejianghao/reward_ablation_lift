# eval_envs.py（或者分散到几个文件里）

from isaaclab.utils import configclass

from .baseline_lift_env_cfg import BaselineLiftEnvCfg
from .rgb_camera_lift_env_cfg import RGBCameraLiftEnvCfg
from .depth_camera_lift_env_cfg import DepthCameraLiftEnvCfg
from .rgbd_camera_lift_env_cfg import RGBDCameraLiftEnvCfg

# === ID Baseline（其实就是 PLAY 配置）===
@configclass
class BaselineLiftEnvCfg_EVAL_ID(BaselineLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 256
        self.observations.policy.enable_corruption = False

# === OOD: Color ===
@configclass
class BaselineLiftEnvCfg_EVAL_COLOR(BaselineLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 256
        self.observations.policy.enable_corruption = False
        # 改 cube 颜色为蓝色
        self.scene.object.init_state.rot = (0.7071, 0.0, 0.0, 0.7071)

# === OOD: Size ===
@configclass
class BaselineLiftEnvCfg_EVAL_SIZE(BaselineLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 256
        self.observations.policy.enable_corruption = False
        # 改大一点
        self.scene.object.spawn.scale = (1.5, 1.5, 1.5)