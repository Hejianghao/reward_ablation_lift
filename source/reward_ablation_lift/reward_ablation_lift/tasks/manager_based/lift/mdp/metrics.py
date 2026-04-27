from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms
from isaaclab_tasks.manager_based.manipulation.lift.mdp.rewards import object_is_lifted

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _ensure_log_dict(env):
    if "log" not in env.extras:
        env.extras["log"] = {}

def lift_episode_success_rate(env, minimal_height: float = 0.1, sustained_steps: int = 50):
    is_lifted = object_is_lifted(env, minimal_height=minimal_height).bool()

    if not hasattr(env, "_lift_ever_succeeded"):
        env._lift_ever_succeeded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._lift_consecutive_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    # report and reset for envs whose episode just ended
    just_reset = (env.episode_length_buf == 0)
    if just_reset.any():
        _ensure_log_dict(env)
        env.extras["log"]["lift_episode_success_rate"] = env._lift_ever_succeeded[just_reset].float().mean()
        env._lift_ever_succeeded[just_reset] = False
        env._lift_consecutive_steps[just_reset] = 0

    # increment counter if lifted this step, reset to 0 otherwise
    env._lift_consecutive_steps = torch.where(
        is_lifted,
        env._lift_consecutive_steps + 1,
        torch.zeros_like(env._lift_consecutive_steps),
    )

    # success once sustained for enough consecutive steps (stays True for rest of episode)
    env._lift_ever_succeeded |= (env._lift_consecutive_steps >= sustained_steps)

    return env._lift_ever_succeeded.float()

def success_rate(
    env,
    distance_threshold: float = 0.02,
    sustained_steps: int = 50,
    command_name: str = "target_pos",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    _ensure_log_dict(env)

    robot: RigidObject = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])

    cube_pos_w = env.scene["object"].data.root_pos_w
    distance = torch.norm(cube_pos_w - des_pos_w, dim=-1)
    is_at_height = (distance < distance_threshold).bool()

    if not hasattr(env, "_goal_ever_reached"):
        env._goal_ever_reached = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._goal_consecutive_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not hasattr(env, "_last_distance"):
        env._last_distance = torch.zeros(env.num_envs, device=env.device)

    just_reset = (env.episode_length_buf == 0)
    if just_reset.any():
        env.extras["log"]["goal_reached_success_rate"] = env._goal_ever_reached[just_reset].float().mean()
        env.extras["log"]["distance_rmse"] = torch.sqrt(torch.mean(env._last_distance[just_reset] ** 2))
        env._goal_ever_reached[just_reset] = False
        env._goal_consecutive_steps[just_reset] = 0

    env._last_distance = distance

    env._goal_consecutive_steps = torch.where(
        is_at_height,
        env._goal_consecutive_steps + 1,
        torch.zeros_like(env._goal_consecutive_steps),
    )

    env._goal_ever_reached |= (env._goal_consecutive_steps >= sustained_steps)

    return env._goal_ever_reached.float()