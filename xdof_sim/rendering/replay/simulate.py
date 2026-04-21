"""Headless replay exporter that steps the env and writes a video."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

# Must be set before mujoco / xdof_sim are imported.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

from xdof_sim.rendering.replay.camera_providers import create_camera_provider
from xdof_sim.rendering.replay.episode import load_episode_context
from xdof_sim.rendering.replay.runtime import create_replay_session
from xdof_sim.rendering.replay.video import export_batched_qpos_sim_video, export_replay_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Headless replay exporter for raw or delivered market42 episodes")
    parser.add_argument("episode_dir", help="Path to a raw episode dir or delivered episode dir")
    parser.add_argument("output", help="Output MP4 path")
    parser.add_argument(
        "--replay-mode",
        choices=("auto", "physics", "qpos"),
        default="auto",
        help="Physics re-step or exact qpos replay",
    )
    parser.add_argument(
        "--camera-source",
        choices=("sim", "recorded"),
        default="sim",
        help="Use live simulated cameras or the recorded MP4 frames",
    )
    parser.add_argument(
        "--sim-render-backend",
        choices=("mujoco", "mjwarp", "madrona"),
        default="mjwarp",
        help="Backend for --camera-source sim",
    )
    parser.add_argument("--render-width", type=int, default=640, help="Sim camera render width")
    parser.add_argument("--render-height", type=int, default=480, help="Sim camera render height")
    parser.add_argument("--gpu-id", type=int, default=None, help="GPU for MJWarp / Madrona rendering")
    parser.add_argument(
        "--sim-batch-size",
        type=int,
        default=1,
        help="Batch size for exact-qpos sim rendering with MJWarp/Madrona",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Output video FPS (default: 30.0)")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit the number of exported frames for benchmarking or quick checks",
    )
    parser.add_argument(
        "--timings-json",
        default=None,
        help="Optional path to write structured timing data as JSON",
    )
    parser.add_argument(
        "--no-initial-frame",
        action="store_true",
        help="Do not include the reset state as the first video frame",
    )
    args = parser.parse_args()

    timings: dict[str, float | int | str | None] = {
        "camera_source": args.camera_source,
        "sim_render_backend": args.sim_render_backend,
        "sim_batch_size": args.sim_batch_size,
        "replay_mode": args.replay_mode,
        "fps": args.fps,
        "max_frames": args.max_frames,
    }
    t_total0 = time.perf_counter()
    episode_dir = Path(args.episode_dir)
    output_path = Path(args.output)
    t0 = time.perf_counter()
    context = load_episode_context(
        episode_dir,
        load_recorded_cameras=(args.camera_source == "recorded"),
    )
    timings["load_episode_s"] = time.perf_counter() - t0

    print(f"\nCreating replay session (scene={context.scene}, task={context.task}, mode={args.replay_mode}) ...")
    if context.instruction:
        print(f"Instruction: {context.instruction!r}")
    t0 = time.perf_counter()
    session, control_hz = create_replay_session(context, mode=args.replay_mode)
    timings["create_session_s"] = time.perf_counter() - t0
    print(f"Control Hz: {control_hz:.1f}")
    print(f"Timeline: {len(session.actions)} action steps, {session.duration_s:.1f}s")
    if session.sim_states is not None:
        print(f"Sim states: {len(session.sim_states)} frames (nq={session.sim_states.shape[1]})")

    fps = args.fps
    use_batched_sim_export = (
        args.sim_batch_size > 1
        and args.camera_source == "sim"
        and args.sim_render_backend in {"mjwarp", "madrona"}
        and session.mode == "qpos"
    )

    if use_batched_sim_export:
        print(
            f"Exporting batched sim video with {args.sim_render_backend} "
            f"(batch_size={args.sim_batch_size}) at {fps:.1f} FPS ..."
        )
        timings["camera_setup_s"] = 0.0
        t0 = time.perf_counter()
        frame_count = export_batched_qpos_sim_video(
            session,
            output_path=output_path,
            fps=fps,
            sim_backend=args.sim_render_backend,
            render_width=args.render_width,
            render_height=args.render_height,
            batch_size=args.sim_batch_size,
            gpu_id=args.gpu_id,
            include_initial_frame=not args.no_initial_frame,
            max_frames=args.max_frames,
        )
        timings["export_s"] = time.perf_counter() - t0
    else:
        if args.sim_batch_size > 1:
            print(
                "Batched sim export requires --camera-source sim, "
                "--sim-render-backend mjwarp|madrona, and qpos replay mode; "
                "falling back to step-by-step export."
            )

        t0 = time.perf_counter()
        camera_provider = create_camera_provider(
            source=args.camera_source,
            sim_backend=args.sim_render_backend,
            episode_camera_frames=context.streams.camera_frames,
            episode_camera_ts=context.streams.camera_ts,
            env=session.env,
            width=args.render_width,
            height=args.render_height,
            gpu_id=args.gpu_id,
        )
        timings["camera_setup_s"] = time.perf_counter() - t0
        if not camera_provider.camera_names:
            raise RuntimeError("No cameras are available for video export")

        print(f"Exporting video from cameras {camera_provider.camera_names} at {fps:.1f} FPS ...")
        t0 = time.perf_counter()
        frame_count = export_replay_video(
            session,
            camera_provider,
            output_path=output_path,
            fps=fps,
            include_initial_frame=not args.no_initial_frame,
            max_frames=args.max_frames,
        )
        timings["export_s"] = time.perf_counter() - t0
        camera_provider.close()

    timings["frame_count"] = frame_count
    timings["total_s"] = time.perf_counter() - t_total0
    print(f"Saved {frame_count} frames to {output_path}")
    print(
        "Timings: "
        f"load={timings['load_episode_s']:.3f}s, "
        f"session={timings['create_session_s']:.3f}s, "
        f"camera_setup={timings['camera_setup_s']:.3f}s, "
        f"export={timings['export_s']:.3f}s, "
        f"total={timings['total_s']:.3f}s"
    )
    if args.timings_json is not None:
        timings_path = Path(args.timings_json)
        timings_path.parent.mkdir(parents=True, exist_ok=True)
        timings_path.write_text(json.dumps(timings, indent=2) + "\n")


if __name__ == "__main__":
    main()
