# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from reward_ablation_lift.tasks.manager_based.reach.config.franka.joint_pos_env_cfg import FrankaReachEnvCfg


##
# Environment configuration
##


@configclass
class ReachAblationEnvCfg(FrankaReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # ablation: remove tanh-based fine-grained position reward
        self.rewards.end_effector_position_tracking_fine_grained = None


@configclass
class ReachAblationEnvCfg_PLAY(ReachAblationEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
