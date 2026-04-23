# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.envs.mdp as mdp
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w)
    return object_pos_b


def ee_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """The position of the end-effector (panda_hand + 0.1034 offset) relative to the robot base."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # target_pos_source: (num_envs, num_targets, 3), relative to source frame (panda_link0)
    return ee_frame.data.target_pos_source[:, 0, :]


def proprio_observations(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "object_pose",
    action_name: str | None = None,
) -> torch.Tensor:
    """Robot proprioception plus task command, excluding any direct object-state observation."""
    joint_pos = mdp.joint_pos_rel(env, asset_cfg=robot_cfg)
    joint_vel = mdp.joint_vel_rel(env, asset_cfg=robot_cfg)

    ee_pos = ee_position_in_robot_root_frame(env)

    target_command = mdp.generated_commands(env, command_name=command_name)
    actions = mdp.last_action(env, action_name=action_name)
    return torch.cat([joint_pos, joint_vel, ee_pos, target_command, actions], dim=-1)
