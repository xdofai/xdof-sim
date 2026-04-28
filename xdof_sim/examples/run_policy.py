"""Run a trained LBM policy in xdof-sim.

The default rollout path uses the batched ``mjwarp`` renderer because it works
headless on machines without EGL/OSMesa. The legacy single-world MuJoCo camera
path is still available via ``--render-backend mujoco``.

Examples:
    MUJOCO_GL=glfw python -m xdof_sim.examples.run_policy \
        --checkpoint-path /path/to/checkpoint.pt \
        --task bottles \
        --render-backend mjwarp \
        --output /tmp/policy_rollout.mp4

    MUJOCO_GL=egl python -m xdof_sim.examples.run_policy \
        --checkpoint-path /path/to/checkpoint.pt \
        --render-backend mujoco \
        --output /tmp/policy_rollout.mp4
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


def _maybe_compress_image_batch(image: np.ndarray, quality: int) -> np.ndarray:
    """JPEG-compress a single CHW image or a batch of BCHW images."""
    if quality <= 0:
        return image
    image = np.asarray(image)
    if image.ndim == 3:
        return jpeg_compress_image(image, quality=quality)
    if image.ndim == 4:
        return np.stack(
            [jpeg_compress_image(frame, quality=quality) for frame in image],
            axis=0,
        )
    raise ValueError(f"Unsupported image shape for JPEG compression: {image.shape}")


def _prepare_policy_obs(obs: dict, *, prompt: str, jpeg_quality: int) -> dict:
    state = np.asarray(obs["state"], dtype=np.float32)
    batch_size = int(state.shape[0]) if state.ndim == 2 else 1
    images = {}
    for name, image in obs.get("images", {}).items():
        images[name] = _maybe_compress_image_batch(np.asarray(image), jpeg_quality)

    return {
        "state": state,
        "images": images,
        "prompt": prompt if batch_size == 1 else [prompt] * batch_size,
    }


def _grid_from_obs(obs: dict, camera_names: list[str]) -> np.ndarray:
    per_camera_batches = []
    for name in camera_names:
        image = np.asarray(obs["images"][name])
        if image.ndim == 3:
            image = image[None, ...]
        per_camera_batches.append(image.transpose(0, 2, 3, 1))

    num_worlds = int(per_camera_batches[0].shape[0])
    world_rows = []
    for world_idx in range(num_worlds):
        world_rows.append(
            np.concatenate(
                [camera_batch[world_idx] for camera_batch in per_camera_batches],
                axis=1,
            )
        )
    return np.concatenate(world_rows, axis=0)


def _task_eval_arrays(env) -> tuple[np.ndarray | None, np.ndarray | None]:
    result = env.evaluate_task()
    if result is None:
        return None, None
    reward = np.asarray(result.reward, dtype=np.float32).reshape(-1)
    success = np.asarray(result.success, dtype=np.float32).reshape(-1)
    return reward, success


def parse_args():
    parser = argparse.ArgumentParser(description="Run an LBM policy in xdof-sim")
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--task", type=str, default="bottles")
    parser.add_argument("--prompt", type=str, default="throw plastic bottles in bin")
    parser.add_argument("--output", type=str, default="/tmp/xdof_policy_rollout.mp4")
    parser.add_argument("--num-worlds", type=int, default=1)
    parser.add_argument("--num-chunks", type=int, default=10)
    parser.add_argument("--execute-chunk-dim", type=int, default=20)
    parser.add_argument("--prefix-length", type=int, default=3)
    parser.add_argument("--diffusion-steps", type=int, default=3)
    parser.add_argument("--model-size", type=str, default="dit_xL")
    parser.add_argument(
        "--render-backend",
        type=str,
        default="mjwarp",
        choices=["mjwarp", "mujoco"],
        help="Use mjwarp for headless rollout/video or mujoco for the legacy renderer path.",
    )
    parser.add_argument(
        "--camera-gpu-id",
        type=int,
        default=None,
        help="GPU index for mjwarp rendering.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Policy GPU index. Ignored if CUDA_VISIBLE_DEVICES is already set.",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-ttrtc", action="store_true")
    parser.add_argument("--jpeg-quality", type=int, default=0)
    parser.add_argument(
        "--scene",
        type=str,
        default="hybrid",
        choices=["eval", "training", "hybrid"],
    )
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
    from xdof_sim.policy.lbm_policy import LBMPolicy, LBMPolicyConfig
    from xdof_sim.task_specs import maybe_get_task_spec

    task_spec = maybe_get_task_spec(args.task)
    env_task = task_spec.env_task if task_spec is not None else args.task
    prompt = task_spec.prompt if task_spec is not None else args.prompt

    policy = LBMPolicy(
        LBMPolicyConfig(
            ckpt_path=args.checkpoint_path,
            policy_type="lbm",
            model_size=args.model_size,
            diffusion_steps=args.diffusion_steps,
        )
    )

    if args.num_worlds < 1:
        raise ValueError(f"--num-worlds must be >= 1, got {args.num_worlds}")
    if args.render_backend == "mujoco" and args.num_worlds != 1:
        raise ValueError("--render-backend mujoco only supports --num-worlds 1.")

    if args.render_backend == "mjwarp":
        env = xdof_sim.make_batched_env(
            scene=args.scene,
            task=env_task,
            prompt=prompt,
            num_worlds=args.num_worlds,
            camera_backend="mjwarp",
            camera_gpu_id=args.camera_gpu_id,
        )
    else:
        env = xdof_sim.make_env(
            scene=args.scene,
            task=env_task,
            prompt=prompt,
        )

    if args.bottle_mass is not None:
        for i in range(1, 7):
            body_name = f"bottle_{i}"
            body_id = mujoco.mj_name2id(
                env.model, mujoco.mjtObj.mjOBJ_BODY, body_name
            )
            if body_id >= 0:
                old_mass = env.model.body_mass[body_id]
                if old_mass > 0:
                    env.model.body_inertia[body_id] *= args.bottle_mass / old_mass
                env.model.body_mass[body_id] = args.bottle_mass

    use_ttrtc = not args.no_ttrtc
    execute_dim = args.execute_chunk_dim if use_ttrtc else 30
    prefix_len = args.prefix_length if use_ttrtc else 0

    print(
        f"Rollout: backend={args.render_backend}, worlds={args.num_worlds}, "
        f"{args.num_chunks} chunks, "
        f"execute {execute_dim}/30 per chunk"
    )

    obs, _ = env.reset(seed=args.seed)
    init_q = np.asarray(obs["state"], dtype=np.float32)
    is_batched = init_q.ndim == 2

    frames = [_grid_from_obs(obs, env.camera_names)]
    all_actions = []
    previous_chunk_actions = None

    for chunk_idx in range(args.num_chunks):
        policy_obs = _prepare_policy_obs(
            obs,
            prompt=prompt,
            jpeg_quality=args.jpeg_quality,
        )

        action_prefix = None
        prefix_length = None
        if use_ttrtc:
            if previous_chunk_actions is not None:
                if is_batched:
                    prefix_length = min(prefix_len, previous_chunk_actions.shape[1])
                    action_prefix = previous_chunk_actions[:, -prefix_length:, :]
                else:
                    prefix_length = min(prefix_len, len(previous_chunk_actions))
                    action_prefix = previous_chunk_actions[-prefix_length:]
            else:
                prefix_length = prefix_len
                if is_batched:
                    action_prefix = np.repeat(init_q[:, None, :], prefix_length, axis=1)
                else:
                    action_prefix = np.tile(init_q, (prefix_length, 1))

        result = policy.infer(
            policy_obs,
            action_prefix=action_prefix,
            prefix_length=prefix_length,
        )

        predicted_actions = np.asarray(result["actions"], dtype=np.float32)
        if is_batched and predicted_actions.ndim == 2:
            predicted_actions = predicted_actions[None, ...]
        if use_ttrtc and prefix_length is not None:
            if is_batched:
                execute_actions = predicted_actions[
                    :, prefix_length : prefix_length + execute_dim, :
                ]
            else:
                execute_actions = predicted_actions[
                    prefix_length : prefix_length + execute_dim
                ]
        else:
            if is_batched:
                execute_actions = predicted_actions[:, :execute_dim, :]
            else:
                execute_actions = predicted_actions[:execute_dim]
        all_actions.append(execute_actions)

        if is_batched:
            action_iter = (execute_actions[:, step_idx, :] for step_idx in range(execute_actions.shape[1]))
        else:
            action_iter = iter(execute_actions)

        for action in action_iter:
            if args.render_backend == "mjwarp":
                if is_batched:
                    obs, _, _, _, _ = env.step(action)
                else:
                    obs, _, _, _, _ = env.step(action[None, :])
            else:
                env._step_single(action)
                obs = env.get_obs()
            frames.append(_grid_from_obs(obs, env.camera_names))

        previous_chunk_actions = execute_actions
        reward, success = _task_eval_arrays(env)
        reward_mean = None if reward is None else float(reward.mean())
        success_mean = None if success is None else float(success.mean())
        success_list = None if success is None else success.astype(int).tolist()
        range_min = float(predicted_actions.min())
        range_max = float(predicted_actions.max())
        print(
            f"  Chunk {chunk_idx}: actions range=[{range_min:.3f}, {range_max:.3f}] "
            f"reward_mean={reward_mean} success_mean={success_mean} success={success_list}"
        )

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
        for idx, frame in enumerate(frames):
            Image.fromarray(frame).save(frames_dir / f"frame_{idx:05d}.png")
        print(f"Saved {len(frames)} frames to {frames_dir}/")

    actions_path = Path(args.output).with_suffix(".npy")
    action_axis = 1 if is_batched else 0
    actions = np.concatenate(all_actions, axis=action_axis)
    np.save(str(actions_path), actions)
    print(f"Saved actions to {actions_path}")

    final_reward, final_success = _task_eval_arrays(env)
    final_reward_list = None if final_reward is None else final_reward.tolist()
    final_success_list = None if final_success is None else final_success.astype(int).tolist()
    print(
        "Final task eval: "
        f"reward={final_reward_list} "
        f"success={final_success_list}"
    )
    env.close()


if __name__ == "__main__":
    main()
