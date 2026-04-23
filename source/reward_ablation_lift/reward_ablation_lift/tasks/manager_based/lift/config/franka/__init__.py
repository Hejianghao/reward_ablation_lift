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
    id="IsaacLab-Lift-Baseline",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaCubeLiftEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="IsaacLab-Lift-Baseline-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaCubeLiftEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="RGB-Franka-Lift",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rgb_camera_env_cfg:FrankaCubeLiftRGBCameraEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:rgb_skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Depth-Franka-Lift",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.depth_camera_env_cfg:FrankaCubeLiftDepthCameraEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:depth_skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Depth-Franka-Lift-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.depth_camera_env_cfg:FrankaCubeLiftDepthCameraEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:depth_skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="My-Lift-Ablation-No-Fine-Grained",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.first_ablation_env_cfg:FrankaLiftNoFineGrainedEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="My-Lift-Ablation-No-Fine-Grained-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.first_ablation_env_cfg:FrankaLiftNoFineGrainedEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)
