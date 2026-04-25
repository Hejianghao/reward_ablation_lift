# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

import reward_ablation_lift.tasks.manager_based.lift.mdp as my_mdp

from .baseline_lift_env_cfg import BaselineLiftEnvCfg


@configclass
class VelocityLiftEnvCfg(BaselineLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.rewards.ee_velocity_near_object = RewTerm(
            func=my_mdp.ee_velocity_near_object,
            params={"proximity_threshold": 0.05, "minimal_height": 0.04},
            weight=0.5,
        )


@configclass
class VelocityLiftEnvCfg_PLAY(VelocityLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
