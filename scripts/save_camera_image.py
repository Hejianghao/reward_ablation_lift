"""Save a single camera frame from the RGB camera environment to disk."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Save a camera image from the RGB lift environment."
)
parser.add_argument("--task", type=str, default="RGB-Franka-Lift", help="Task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--output", type=str, default="camera_frame.png", help="Output path.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import reward_ablation_lift.tasks  # noqa: F401, E402


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    env.reset()

    # Raw sensor tensor: [num_envs, H, W, C], uint8 in [0, 255]
    camera_data = env.unwrapped.scene["tiled_camera"].data.output["rgb"]
    frame = camera_data[0]  # first env

    if frame.is_floating_point():
        frame = (frame.clamp(0.0, 1.0) * 255).byte()

    img = Image.fromarray(frame.cpu().numpy(), mode="RGB")
    img.save(args_cli.output)
    print(f"[INFO] Camera image saved to: {args_cli.output}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
