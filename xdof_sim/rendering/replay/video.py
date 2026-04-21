"""Headless replay-to-video helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from typing import Sequence

import numpy as np
from PIL import Image


def tile_camera_frames(frames: dict[str, np.ndarray], camera_names: Sequence[str]) -> np.ndarray:
    """Tile camera frames horizontally in a stable order."""
    ordered = [frames[name] for name in camera_names if name in frames]
    if not ordered:
        raise ValueError("No camera frames available to tile")
    if len(ordered) == 1:
        return ordered[0]
    return np.concatenate(ordered, axis=1)


def _open_video_writer(output_path: Path, fps: float):
    """Prefer the imageio-ffmpeg writer over pyav in headless environments."""
    import imageio.v2 as iio

    try:
        import imageio_ffmpeg  # noqa: F401
    except ImportError:
        return iio.get_writer(output_path, fps=fps)
    return iio.get_writer(output_path, fps=fps, format="FFMPEG")


def write_video(output_path: Path, frames: list[np.ndarray], fps: float) -> None:
    """Write frames to an MP4, or fall back to PNGs if imageio is unavailable."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as iio  # noqa: F401
    except ImportError:
        frames_dir = output_path.with_suffix("")
        frames_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(frames_dir / f"frame_{i:05d}.png")
        print(f"imageio not available. Saved {len(frames)} PNG frames to {frames_dir}/")
        return

    with _open_video_writer(output_path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)


def _tile_render_batch(img, camera_indices: Sequence[int]):
    """Tile a GPU render batch into (B, H, ncam*W, 3) frames."""
    selected = img[:, camera_indices]
    actual, ncam, height, width, channels = selected.shape
    return (
        selected.permute(0, 2, 1, 3, 4)
        .contiguous()
        .view(actual, height, ncam * width, channels)
    )


def export_batched_qpos_sim_video(
    session,
    *,
    output_path: Path,
    fps: float,
    sim_backend: Literal["mjwarp", "madrona"],
    render_width: int,
    render_height: int,
    batch_size: int,
    gpu_id: int | None = None,
    include_initial_frame: bool = True,
    progress_every: int = 100,
    max_frames: int | None = None,
) -> int:
    """Render exact-qpos replay in batches with MJWarp or Madrona."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if session.mode != "qpos" or session.sim_states is None:
        raise RuntimeError("Batched sim export requires qpos replay mode with aligned sim states")

    from xdof_sim.rendering.replay.renderer import RendererWrapper, WarpReplayRuntime

    session.reset()
    qpos_frames = session.sim_states if include_initial_frame else session.sim_states[1:]
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError("max_frames must be positive when provided")
        qpos_frames = qpos_frames[:max_frames]
    if len(qpos_frames) == 0:
        raise RuntimeError("No qpos frames available for batched export")

    camera_names = tuple(session.env.camera_names)
    available_names = tuple(session.model.cam(i).name for i in range(session.model.ncam))
    camera_indices = [available_names.index(name) for name in camera_names if name in available_names]
    if len(camera_indices) != len(camera_names):
        missing = [name for name in camera_names if name not in available_names]
        raise RuntimeError(f"Requested sim cameras missing from model: {missing}")

    actual_batch_size = min(batch_size, len(qpos_frames))
    runtime = WarpReplayRuntime(session.model, session.data, nworld=actual_batch_size, gpu_id=gpu_id)
    qpos_frames_device = runtime.upload_qpos_frames(qpos_frames)
    renderer = RendererWrapper(
        backend=sim_backend,
        runtime=runtime,
        cam_res=(render_width, render_height),
        gpu_id=gpu_id,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as iio  # noqa: F401
    except ImportError:
        iio = None

    frames_dir = None
    writer = None
    if iio is None:
        frames_dir = output_path.with_suffix("")
        frames_dir.mkdir(parents=True, exist_ok=True)
    else:
        writer = _open_video_writer(output_path, fps=fps)

    written = 0
    next_progress = progress_every if progress_every > 0 else None
    first_batch = True

    try:
        for start in range(0, len(qpos_frames), actual_batch_size):
            stop = min(start + actual_batch_size, len(qpos_frames))
            actual = stop - start

            runtime.load_qpos_batch_from_device(qpos_frames_device, start=start, stop=stop)
            runtime.forward()
            img = (
                renderer.reset(actual_batch=actual)
                if first_batch
                else renderer.render(actual_batch=actual)
            )
            first_batch = False

            tiled = _tile_render_batch(img, camera_indices).cpu().numpy()
            if writer is not None:
                for frame in tiled:
                    writer.append_data(frame)
            else:
                for idx, frame in enumerate(tiled, start=written):
                    Image.fromarray(frame).save(frames_dir / f"frame_{idx:05d}.png")

            written += actual
            if next_progress is not None:
                while written >= next_progress:
                    print(f"  Step {min(next_progress, len(qpos_frames))}/{len(qpos_frames)}")
                    next_progress += progress_every
    finally:
        if writer is not None:
            writer.close()

    if frames_dir is not None:
        print(f"imageio not available. Saved {written} PNG frames to {frames_dir}/")

    return written


def export_replay_video(
    session,
    camera_provider,
    *,
    output_path: Path,
    fps: float,
    include_initial_frame: bool = True,
    progress_every: int = 100,
    max_frames: int | None = None,
) -> int:
    """Step a replay session and write a tiled camera video."""
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive when provided")

    session.reset()
    camera_provider.reset()

    frames: list[np.ndarray] = []
    if include_initial_frame and (max_frames is None or len(frames) < max_frames):
        initial_frames = camera_provider.initial_frames()
        if not initial_frames:
            raise RuntimeError("No camera frames available for the initial replay frame")
        frames.append(tile_camera_frames(initial_frames, camera_provider.camera_names))

    steps = 0
    while (max_frames is None or len(frames) < max_frames) and session.step():
        frame_dict = camera_provider.frames_for_step(session.current_frame_idx, session.current_timestamp)
        if not frame_dict:
            raise RuntimeError("Camera provider returned no frames during replay")
        frames.append(tile_camera_frames(frame_dict, camera_provider.camera_names))
        steps += 1
        if progress_every > 0 and steps % progress_every == 0:
            print(f"  Step {steps}/{session.total_steps}")

    write_video(output_path, frames, fps=fps)
    return len(frames)
