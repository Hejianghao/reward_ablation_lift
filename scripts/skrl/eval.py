# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to evaluate a checkpoint of an RL agent from skrl and log environment extras to TensorBoard.

This is intentionally close to ``play.py``: it loads a trained checkpoint, runs the policy in eval mode,
and records all scalar values from ``extras["log"]`` returned by the environment at each step.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--eval_episodes",
    type=int,
    default=None,
    help=(
        "Number of episodes to evaluate per environment. "
        "For example, with 516 environments and --eval_episodes 1, evaluation stops after 516 completed episodes."
    ),
)
parser.add_argument(
    "--eval_log_name",
    type=str,
    default=None,
    help="Name of the evaluation TensorBoard run directory. Defaults to a timestamped directory.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import random
import time
from collections.abc import Mapping
from datetime import datetime

import gymnasium as gym
import skrl
import torch
from packaging import version
from torch.utils.tensorboard import SummaryWriter

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import reward_ablation_lift.tasks  # noqa: F401

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


def _flatten_log_dict(log_dict: Mapping, prefix: str = "") -> dict[str, object]:
    """Flatten nested log dictionaries into TensorBoard tag names."""
    flattened = {}
    for key, value in log_dict.items():
        tag = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(_flatten_log_dict(value, tag))
        else:
            flattened[tag] = value
    return flattened


def _as_scalar(value) -> float | None:
    """Convert common tensor/array scalar values to a Python float."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return value.detach().float().mean().item()
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except (TypeError, ValueError):
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _write_extra_logs(writer: SummaryWriter, extras: Mapping | None, step: int) -> int:
    """Write every numeric item in extras['log'] to TensorBoard."""
    if not isinstance(extras, Mapping):
        return 0

    log_dict = extras.get("log", {})
    if not isinstance(log_dict, Mapping):
        return 0

    logged_count = 0
    for key, value in _flatten_log_dict(log_dict).items():
        scalar = _as_scalar(value)
        if scalar is None:
            continue
        writer.add_scalar(f"Eval/{key}", scalar, step)
        logged_count += 1
    return logged_count


def _done_mask(dones) -> torch.Tensor:
    """Convert done signals from single-agent or multi-agent envs into a per-env boolean mask."""
    if isinstance(dones, Mapping):
        masks = [_done_mask(value) for value in dones.values()]
        if not masks:
            raise ValueError("Done mapping is empty.")
        combined_mask = masks[0]
        for mask in masks[1:]:
            combined_mask = combined_mask | mask
        return combined_mask

    if isinstance(dones, torch.Tensor):
        dones = dones.to(dtype=torch.bool)
        if dones.ndim == 0:
            return dones.reshape(1)
        if dones.ndim == 1:
            return dones
        return dones.reshape(dones.shape[0], -1).any(dim=-1)

    raise TypeError(f"Unsupported done signal type: {type(dones)}")


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    """Evaluate with skrl agent and log environment extras."""
    if args_cli.eval_episodes is not None and args_cli.eval_episodes < 1:
        raise ValueError("--eval_episodes must be greater than or equal to 1.")

    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    train_log_dir = os.path.dirname(os.path.dirname(resume_path))

    eval_log_name = args_cli.eval_log_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S_eval")
    eval_log_dir = os.path.join(train_log_dir, "eval", eval_log_name)
    print(f"[INFO] Logging evaluation TensorBoard data to: {eval_log_dir}")

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = eval_log_dir

    # save the evaluation configuration next to the TensorBoard event file
    dump_yaml(os.path.join(eval_log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(eval_log_dir, "params", "agent.yaml"), experiment_cfg)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(eval_log_dir, "videos", "eval"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during evaluation.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # eval.py writes extras["log"] manually
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints during eval
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    # set agent to evaluation mode
    runner.agent.set_running_mode("eval")

    writer = SummaryWriter(log_dir=eval_log_dir)
    logged_keys: set[str] = set()
    start_time = time.time()

    # reset environment
    obs, _ = env.reset()
    num_envs = getattr(env, "num_envs", env.unwrapped.num_envs)
    completed_episodes = torch.zeros(num_envs, dtype=torch.long, device=env.unwrapped.device)
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        step_start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            # - multi-agent (deterministic) actions
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            # - single-agent (deterministic) actions
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            # env stepping
            obs, _, terminated, truncated, extras = env.step(actions)

        completed_episodes += (_done_mask(terminated) | _done_mask(truncated)).to(dtype=torch.long)

        if isinstance(extras, Mapping) and isinstance(extras.get("log", None), Mapping):
            logged_keys.update(_flatten_log_dict(extras["log"]).keys())
        _write_extra_logs(writer, extras, timestep)

        timestep += 1
        if args_cli.video and args_cli.eval_episodes is None and timestep == args_cli.video_length:
            break
        if args_cli.eval_episodes is not None and torch.all(completed_episodes >= args_cli.eval_episodes).item():
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - step_start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    writer.flush()
    writer.close()

    print(f"Evaluation time: {round(time.time() - start_time, 2)} seconds")
    print(f"[INFO] Evaluation steps: {timestep}")
    print(f"[INFO] Completed episodes: {int(completed_episodes.sum().item())}")
    print(f"[INFO] Minimum completed episodes per environment: {int(completed_episodes.min().item())}")
    print(f"[INFO] Logged {len(logged_keys)} extras['log'] keys to TensorBoard.")
    if logged_keys:
        print("[INFO] Logged keys:")
        for key in sorted(logged_keys):
            print(f"    Eval/{key}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
