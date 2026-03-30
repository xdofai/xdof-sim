"""Replay episode data directly in the MuJoCo sim (no ZMQ required).

Loads states from a .npy file or states_actions.bin and replays them,
rendering camera views at each timestep and saving a video.

Usage:
    # From a states .npy file
    MUJOCO_GL=egl python -m xdof_sim.examples.replay_data \
        --states-npy /path/to/states.npy \
        --output /tmp/replay.mp4

    # From an episode directory (expects parsed/states.npy inside)
    MUJOCO_GL=egl python -m xdof_sim.examples.replay_data \
        --episode-dir /path/to/episode_NNNNNN \
        --output /tmp/replay.mp4

    # From a raw states_actions.bin file (14 state + 14 action per row)
    MUJOCO_GL=egl python -m xdof_sim.examples.replay_data \
        --bin-file /path/to/states_actions.bin --n-states 14 --n-actions 14 \
        --output /tmp/replay.mp4
"""

from __future__ import annotations

import argparse
import os
import struct

import numpy as np


def load_states_from_bin(bin_path: str, n_states: int = 14, n_actions: int = 14) -> np.ndarray:
    """Load states from a states_actions.bin file."""
    row_dim = n_states + n_actions
    with open(bin_path, "rb") as f:
        data = f.read()
    n_rows = len(data) // (row_dim * 4)  # float32
    arr = np.frombuffer(data, dtype=np.float32).reshape(n_rows, row_dim)
    return arr[:, :n_states]  # return just the states


def main():
    parser = argparse.ArgumentParser(description="Replay episode data in MuJoCo sim")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--states-npy", type=str, help="Path to states .npy file")
    group.add_argument("--episode-dir", type=str, help="Path to episode directory")
    group.add_argument("--bin-file", type=str, help="Path to states_actions.bin")
    parser.add_argument("--n-states", type=int, default=14, help="State dimension")
    parser.add_argument("--n-actions", type=int, default=14, help="Action dimension")
    parser.add_argument("--output", type=str, default="/tmp/xdof_replay.mp4")
    parser.add_argument("--scene", type=str, default="hybrid",
                       choices=["eval", "training", "hybrid"])
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=None,
                       help="Max timesteps to replay")
    args = parser.parse_args()

    # Load states
    if args.states_npy:
        states = np.load(args.states_npy).astype(np.float32)
    elif args.episode_dir:
        states_path = os.path.join(args.episode_dir, "parsed", "states.npy")
        if not os.path.exists(states_path):
            # Try states_actions.bin
            bin_path = os.path.join(args.episode_dir, "states_actions.bin")
            if os.path.exists(bin_path):
                states = load_states_from_bin(bin_path, args.n_states, args.n_actions)
            else:
                raise FileNotFoundError(
                    f"Neither {states_path} nor {bin_path} found"
                )
        else:
            states = np.load(states_path).astype(np.float32)
    else:
        states = load_states_from_bin(args.bin_file, args.n_states, args.n_actions)

    if args.max_steps:
        states = states[:args.max_steps]

    print(f"Loaded {len(states)} timesteps, state dim={states.shape[-1]}")

    # Create environment
    import xdof_sim
    from xdof_sim.viewer import render_cameras

    env = xdof_sim.make_env(scene=args.scene)
    env.reset()

    # Set initial state
    env._set_qpos_from_state(states[0])
    import mujoco
    mujoco.mj_forward(env.model, env.data)

    # Replay and record
    frames = []
    for t in range(len(states)):
        env._set_qpos_from_state(states[t])
        mujoco.mj_forward(env.model, env.data)
        images_hwc = render_cameras(env)
        grid = np.concatenate(
            [images_hwc[name] for name in env.camera_names], axis=1
        )
        frames.append(grid)
        if t % 100 == 0:
            print(f"  Frame {t}/{len(states)}")

    # Save
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        import imageio.v3 as iio
        iio.imwrite(args.output, np.stack(frames), fps=args.fps)
        print(f"Saved video ({len(frames)} frames) to {args.output}")
    except ImportError:
        from PIL import Image
        frames_dir = args.output.replace(".mp4", "_frames")
        os.makedirs(frames_dir, exist_ok=True)
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(os.path.join(frames_dir, f"frame_{i:05d}.png"))
        print(f"Saved {len(frames)} frames to {frames_dir}/")

    env.close()


if __name__ == "__main__":
    main()
