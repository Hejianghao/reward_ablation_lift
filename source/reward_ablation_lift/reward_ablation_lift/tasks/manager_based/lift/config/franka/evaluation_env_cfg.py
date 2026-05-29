from isaaclab.utils import configclass

from .baseline_lift_env_cfg import BaselineLiftEnvCfg
from .rgb_camera_lift_env_cfg import RGBCameraLiftEnvCfg
from .depth_camera_lift_env_cfg import DepthCameraLiftEnvCfg
from .rgbd_camera_lift_env_cfg import RGBDCameraLiftEnvCfg


# --- 公共 eval 设置 ---

def _eval_defaults(cfg):
    cfg.scene.num_envs = 256
    cfg.observations.policy.enable_corruption = False

def _apply_color_ood(cfg):
    cfg.scene.object.init_state.rot = (0.7071, 0.0, -0.7071, 0.0)

def _apply_size_ood(cfg):
    cfg.scene.object.spawn.scale = (1.0, 1.0, 1.0)


# === Baseline Eval ===

@configclass
class BaselineLiftEnvCfg_EVAL_ID(BaselineLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _eval_defaults(self)

@configclass
class BaselineLiftEnvCfg_EVAL_COLOR(BaselineLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _eval_defaults(self)
        _apply_color_ood(self)

@configclass
class BaselineLiftEnvCfg_EVAL_SIZE(BaselineLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _eval_defaults(self)
        _apply_size_ood(self)


@configclass
class BaselineLiftEnvCfg_EVAL_COLOR_SIZE(BaselineLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _eval_defaults(self)
        _apply_color_ood(self)
        _apply_size_ood(self)


# === RGB Camera Eval ===

@configclass
class RGBCameraLiftEnvCfg_EVAL_ID(RGBCameraLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _eval_defaults(self)

@configclass
class RGBCameraLiftEnvCfg_EVAL_COLOR(RGBCameraLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _eval_defaults(self)
        _apply_color_ood(self)

@configclass
class RGBCameraLiftEnvCfg_EVAL_SIZE(RGBCameraLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _eval_defaults(self)
        _apply_size_ood(self)

@configclass
class RGBCameraLiftEnvCfg_EVAL_COLOR_SIZE(RGBCameraLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _eval_defaults(self)
        _apply_color_ood(self)
        _apply_size_ood(self)


# === Depth Camera Eval ===
@configclass
class DepthCameraLiftEnvCfg_EVAL_ID(DepthCameraLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _eval_defaults(self)

@configclass
class DepthCameraLiftEnvCfg_EVAL_COLOR(DepthCameraLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _eval_defaults(self)
        _apply_color_ood(self)

@configclass
class DepthCameraLiftEnvCfg_EVAL_SIZE(DepthCameraLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _eval_defaults(self)
        _apply_size_ood(self)

@configclass
class DepthCameraLiftEnvCfg_EVAL_COLOR_SIZE(DepthCameraLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _eval_defaults(self)
        _apply_color_ood(self)
        _apply_size_ood(self)

# === RGBD Camera Eval ===
@configclass
class RGBDCameraLiftEnvCfg_EVAL_ID(RGBDCameraLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _eval_defaults(self)

@configclass
class RGBDCameraLiftEnvCfg_EVAL_COLOR(RGBDCameraLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _eval_defaults(self)
        _apply_color_ood(self)

@configclass
class RGBDCameraLiftEnvCfg_EVAL_SIZE(RGBDCameraLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _eval_defaults(self)
        _apply_size_ood(self)

@configclass
class RGBDCameraLiftEnvCfg_EVAL_COLOR_SIZE(RGBDCameraLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _eval_defaults(self)
        _apply_color_ood(self)
        _apply_size_ood(self)