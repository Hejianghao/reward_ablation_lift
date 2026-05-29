# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/My-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Baseline-Franka-Lift",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.baseline_lift_env_cfg:BaselineLiftEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Baseline-Franka-Lift-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.baseline_lift_env_cfg:BaselineLiftEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="RGB-Franka-Lift",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rgb_camera_lift_env_cfg:RGBCameraLiftEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:rgb_skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)


gym.register(
    id="RGB-Franka-Lift-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rgb_camera_lift_env_cfg:RGBCameraLiftEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:rgb_skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)


gym.register(
    id="Depth-Franka-Lift",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.depth_camera_lift_env_cfg:DepthCameraLiftEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:depth_skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Depth-Franka-Lift-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.depth_camera_lift_env_cfg:DepthCameraLiftEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:depth_skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="RGBD-Franka-Lift",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rgbd_camera_lift_env_cfg:RGBDCameraLiftEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:rgbd_skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)


gym.register(
    id="RGBD-Franka-Lift-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rgbd_camera_lift_env_cfg:RGBDCameraLiftEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:rgbd_skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

# Evaluation environmens
gym.register(
    id="Baseline-Franka-Lift-Eval-ID",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.evaluation_env_cfg:BaselineLiftEnvCfg_EVAL_ID",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Baseline-Franka-Lift-Eval-Color",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.evaluation_env_cfg:BaselineLiftEnvCfg_EVAL_COLOR",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Baseline-Franka-Lift-Eval-Size",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.evaluation_env_cfg:BaselineLiftEnvCfg_EVAL_SIZE",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Baseline-Franka-Lift-Eval-Color-Size",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.evaluation_env_cfg:BaselineLiftEnvCfg_EVAL_COLOR_SIZE",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)



gym.register(
    id="RGB-Franka-Lift-Eval-ID",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.evaluation_env_cfg:RGBCameraLiftEnvCfg_EVAL_ID",
        "skrl_cfg_entry_point": f"{agents.__name__}:rgb_skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="RGB-Franka-Lift-Eval-Color",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.evaluation_env_cfg:RGBCameraLiftEnvCfg_EVAL_COLOR",
        "skrl_cfg_entry_point": f"{agents.__name__}:rgb_skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="RGB-Franka-Lift-Eval-Size",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.evaluation_env_cfg:RGBCameraLiftEnvCfg_EVAL_SIZE",
        "skrl_cfg_entry_point": f"{agents.__name__}:rgb_skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="RGB-Franka-Lift-Eval-Color-Size",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.evaluation_env_cfg:RGBCameraLiftEnvCfg_EVAL_COLOR_SIZE",
        "skrl_cfg_entry_point": f"{agents.__name__}:rgb_skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Depth-Franka-Lift-Eval-ID",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.evaluation_env_cfg:DepthCameraLiftEnvCfg_EVAL_ID",
        "skrl_cfg_entry_point": f"{agents.__name__}:depth_skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Depth-Franka-Lift-Eval-Color",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.evaluation_env_cfg:DepthCameraLiftEnvCfg_EVAL_COLOR",
        "skrl_cfg_entry_point": f"{agents.__name__}:depth_skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Depth-Franka-Lift-Eval-Size",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.evaluation_env_cfg:DepthCameraLiftEnvCfg_EVAL_SIZE",
        "skrl_cfg_entry_point": f"{agents.__name__}:depth_skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Depth-Franka-Lift-Eval-Color-Size",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.evaluation_env_cfg:DepthCameraLiftEnvCfg_EVAL_COLOR_SIZE",
        "skrl_cfg_entry_point": f"{agents.__name__}:depth_skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="RGBD-Franka-Lift-Eval-ID",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.evaluation_env_cfg:RGBDCameraLiftEnvCfg_EVAL_ID",
        "skrl_cfg_entry_point": f"{agents.__name__}:rgbd_skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="RGBD-Franka-Lift-Eval-Color",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.evaluation_env_cfg:RGBDCameraLiftEnvCfg_EVAL_COLOR",
        "skrl_cfg_entry_point": f"{agents.__name__}:rgbd_skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="RGBD-Franka-Lift-Eval-Size",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.evaluation_env_cfg:RGBDCameraLiftEnvCfg_EVAL_SIZE",
        "skrl_cfg_entry_point": f"{agents.__name__}:rgbd_skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="RGBD-Franka-Lift-Eval-Color-Size",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.evaluation_env_cfg:RGBDCameraLiftEnvCfg_EVAL_COLOR_SIZE",
        "skrl_cfg_entry_point": f"{agents.__name__}:rgbd_skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)
