from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def _ensure_log_dict(env):
    if "log" not in env.extras:
        env.extras["log"] = {}

def reach_rmse_position_error(env, command_name: str, asset_cfg):
    _ensure_log_dict(env)
    # extract the asset (to enable type hinting)
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    position_error = torch.norm(curr_pos_w - des_pos_w, dim=1)
    env.extras["log"]["reach_rmse_position_error"] = position_error.mean()
    return position_error.unsqueeze(-1)