from __future__ import annotations

from typing import TYPE_CHECKING

import torch

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
    object_rest_z: float = 0.03,
    height_threshold: float = 0.02,
    sustained_steps: int = 50,
):
    _ensure_log_dict(env)

    target_height: float = env.cfg.target_height
    cube_z = env.scene["object"].data.root_pos_w[:, 2]
    delta_z = cube_z - object_rest_z
    height_error = (delta_z - target_height).abs()
    is_at_target = (height_error < height_threshold).bool()

    if not hasattr(env, "_goal_ever_reached"):
        env._goal_ever_reached = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._goal_consecutive_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not hasattr(env, "_last_height_error"):
        env._last_height_error = torch.zeros(env.num_envs, device=env.device)

    just_reset = (env.episode_length_buf == 0)
    if just_reset.any():
        env.extras["log"]["goal_reached_success_rate"] = env._goal_ever_reached[just_reset].float().mean()
        env.extras["log"]["height_error_rmse"] = torch.sqrt(torch.mean(env._last_height_error[just_reset] ** 2))
        env._goal_ever_reached[just_reset] = False
        env._goal_consecutive_steps[just_reset] = 0

    env._last_height_error = height_error

    env._goal_consecutive_steps = torch.where(
        is_at_target,
        env._goal_consecutive_steps + 1,
        torch.zeros_like(env._goal_consecutive_steps),
    )

    env._goal_ever_reached |= (env._goal_consecutive_steps >= sustained_steps)

    return env._goal_ever_reached.float()