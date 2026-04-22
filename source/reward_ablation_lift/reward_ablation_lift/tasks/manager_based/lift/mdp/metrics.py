from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab_tasks.manager_based.manipulation.lift.mdp.rewards import object_is_lifted

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _ensure_log_dict(env):
    if "log" not in env.extras:
        env.extras["log"] = {}

def lift_episode_success_rate(env, minimal_height: float = 0.04):
    is_lifted = object_is_lifted(env, minimal_height=minimal_height).bool()

    if not hasattr(env, "_lift_ever_succeeded"):
        env._lift_ever_succeeded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # report success rate for envs whose episode just ended
    just_reset = (env.episode_length_buf == 0)
    if just_reset.any():
        _ensure_log_dict(env)
        env.extras["log"]["lift_episode_success_rate"] = env._lift_ever_succeeded[just_reset].float().mean()
        env._lift_ever_succeeded[just_reset] = False

    # accumulate across timesteps within the episode
    env._lift_ever_succeeded = is_lifted

    return is_lifted.float()

def placement_success_rate(env, distance_threshold: float = 0.05):
    _ensure_log_dict(env)
    cube_pos = env.scene["object"].data.root_pos_w[:, :2]
    target_pos = env.scene["target"].data.root_pos_w[:, :2]
    is_placed = ((cube_pos - target_pos).norm(dim=1) < distance_threshold).float()
    env.extras["log"]["placement_success_rate"] = is_placed.mean()
    return is_placed  # weight=0 to record only