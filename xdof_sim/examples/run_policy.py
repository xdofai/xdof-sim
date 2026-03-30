"""Run a trained LBM policy in the MuJoCo YAM simulation.

Supports train-time RTC (ttRTC) with action prefix conditioning between chunks.

Usage:
    MUJOCO_GL=egl python -m xdof_sim.examples.run_policy \
        --checkpoint-path /path/to/checkpoint.pt \
        --output /tmp/policy_rollout.mp4

    # Without ttRTC
    MUJOCO_GL=egl python -m xdof_sim.examples.run_policy \
        --checkpoint-path /path/to/checkpoint.pt \
        --no-ttrtc --output /tmp/rollout.mp4
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import mujoco
import numpy as np
import torch


def jpeg_compress_image(img_chw: np.ndarray, quality: int = 75) -> np.ndarray:
    """Apply JPEG encode/decode to match training data artifacts."""
    img_hwc = img_chw.transpose(1, 2, 0)
    img_bgr = cv2.cvtColor(img_hwc, cv2.COLOR_RGB2BGR)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, compressed = cv2.imencode(".jpg", img_bgr, encode_param)
    decoded_bgr = cv2.imdecode(compressed, cv2.IMREAD_COLOR)
    decoded_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
    return decoded_rgb.transpose(2, 0, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Run LBM policy in MuJoCo sim")
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="throw plastic bottles in bin")
    parser.add_argument("--output", type=str, default="/tmp/xdof_policy_rollout.mp4")
    parser.add_argument("--num-chunks", type=int, default=10)
    parser.add_argument("--execute-chunk-dim", type=int, default=20)
    parser.add_argument("--prefix-length", type=int, default=3)
    parser.add_argument("--diffusion-steps", type=int, default=3)
    parser.add_argument("--model-size", type=str, default="dit_xL")
    parser.add_argument("--gpu", type=int, default=None,
                       help="GPU index. Ignored if CUDA_VISIBLE_DEVICES is already set.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-ttrtc", action="store_true")
    parser.add_argument("--jpeg-quality", type=int, default=0)
    parser.add_argument("--scene", type=str, default="hybrid",
                       choices=["eval", "training", "hybrid"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--bottle-mass", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.gpu is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    import xdof_sim
    from xdof_sim.viewer import render_cameras
    from xdof_sim.policy.lbm_policy import LBMPolicy, LBMPolicyConfig

    # Create policy
    policy = LBMPolicy(
        LBMPolicyConfig(
            ckpt_path=args.checkpoint_path,
            policy_type="lbm",
            model_size=args.model_size,
            diffusion_steps=args.diffusion_steps,
        )
    )

    # Create environment
    env = xdof_sim.make_env(scene=args.scene, prompt=args.prompt)

    # Override bottle mass
    if args.bottle_mass is not None:
        for i in range(1, 7):
            body_name = f"bottle_{i}"
            body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id >= 0:
                old_mass = env.model.body_mass[body_id]
                if old_mass > 0:
                    env.model.body_inertia[body_id] *= args.bottle_mass / old_mass
                env.model.body_mass[body_id] = args.bottle_mass

    # ttRTC settings
    use_ttrtc = not args.no_ttrtc
    execute_dim = args.execute_chunk_dim if use_ttrtc else 30
    prefix_len = args.prefix_length if use_ttrtc else 0

    print(f"Rollout: {args.num_chunks} chunks, execute {execute_dim}/30 per chunk")

    # Reset
    obs, _ = env.reset()
    init_q = env.get_init_q()

    frames = []
    all_actions = []
    previous_chunk_actions = None

    # Record initial frame
    images_hwc = render_cameras(env)
    grid = np.concatenate([images_hwc[name] for name in env.camera_names], axis=1)
    frames.append(grid)

    for chunk_idx in range(args.num_chunks):
        obs_images = {}
        for name in env.camera_names:
            img = obs["images"][name]
            if args.jpeg_quality > 0:
                img = jpeg_compress_image(img, quality=args.jpeg_quality)
            obs_images[name] = img

        policy_obs = {
            "state": obs["state"],
            "images": obs_images,
            "prompt": args.prompt,
        }

        # RTC prefix
        action_prefix = None
        prefix_length = None
        if use_ttrtc:
            if previous_chunk_actions is not None:
                prefix_length = min(prefix_len, len(previous_chunk_actions))
                action_prefix = previous_chunk_actions[-prefix_length:]
            else:
                prefix_length = prefix_len
                action_prefix = np.tile(init_q, (prefix_length, 1))

        result = policy.infer(
            policy_obs, action_prefix=action_prefix, prefix_length=prefix_length
        )

        predicted_actions = result["actions"]
        if use_ttrtc and prefix_length is not None:
            execute_actions = predicted_actions[prefix_length:prefix_length + execute_dim]
        else:
            execute_actions = predicted_actions[:execute_dim]
        all_actions.append(execute_actions)

        for t in range(len(execute_actions)):
            env._step_single(execute_actions[t])
            images_hwc = render_cameras(env)
            grid = np.concatenate([images_hwc[name] for name in env.camera_names], axis=1)
            frames.append(grid)

        obs = env.get_obs()
        previous_chunk_actions = execute_actions
        print(f"  Chunk {chunk_idx}: actions range=[{predicted_actions.min():.3f}, {predicted_actions.max():.3f}]")

    # Save video
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        import imageio.v3 as iio
        iio.imwrite(args.output, np.stack(frames), fps=args.fps)
        print(f"Saved video ({len(frames)} frames) to {args.output}")
    except ImportError:
        from PIL import Image
        frames_dir = Path(args.output).with_suffix("")
        frames_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(frames_dir / f"frame_{i:05d}.png")
        print(f"Saved {len(frames)} frames to {frames_dir}/")

    # Save actions
    actions_path = Path(args.output).with_suffix(".npy")
    np.save(str(actions_path), np.concatenate(all_actions, axis=0))
    print(f"Saved actions to {actions_path}")

    env.close()


if __name__ == "__main__":
    main()
