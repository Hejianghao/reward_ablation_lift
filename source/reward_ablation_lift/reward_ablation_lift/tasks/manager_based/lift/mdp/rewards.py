# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer, ContactSensor
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv



def object_is_grasped(
    env: ManagerBasedRLEnv,
    normal_force_threshold: float = 3.0,
    contact_sensor_finger_1_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor_finger_1"),
    contact_sensor_finger_2_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor_finger_2"),
) -> torch.Tensor:
    """Return True for envs where both finger tips have contact force > threshold with the object."""
    sensor_1: ContactSensor = env.scene[contact_sensor_finger_1_cfg.name]
    sensor_2: ContactSensor = env.scene[contact_sensor_finger_2_cfg.name]

    # force_matrix_w: (num_envs, num_bodies, num_filtered_shapes, 3)
    # Use norm across xyz so result is direction-agnostic
    force_1 = torch.norm(sensor_1.data.force_matrix_w[:, 0, 0, :], dim=-1)
    force_2 = torch.norm(sensor_2.data.force_matrix_w[:, 0, 0, :], dim=-1)

    return (force_1 > normal_force_threshold) & (force_2 > normal_force_threshold)

def object_lift_height(
    env: ManagerBasedRLEnv,
    resting_z: float = 0.03,
    target_height: float = 0.3,
    std: float = 0.1,
    normal_force_threshold: float = 3.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Dense reward peaking at target_height above resting position.
    Uses 1 - tanh(|Δz - target| / std): peaks at 1.0 when Δz == target_height,
    decreases symmetrically on both sides. resting_z is the cube's world-frame
    z coordinate when at rest on the table.
    """
    object: RigidObject = env.scene[object_cfg.name]
    delta_z = torch.clamp(object.data.root_pos_w[:, 2] - resting_z, min=0.0)
    grasped = object_is_grasped(env, normal_force_threshold)
    reward = 1.0 - torch.tanh(torch.abs(delta_z - target_height) / std)
    return grasped.float() * reward


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))
