#!/usr/bin/env python3
"""Benchmark replay rendering backends and batch sizes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import subprocess
import sys
import time

import imageio.v2 as iio
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class RenderConfig:
    backend: str
    batch_size: int

    @property
    def uses_batched_export(self) -> bool:
        return self.backend in {"mjwarp", "madrona"} and self.batch_size > 1

    @property
    def slug(self) -> str:
        if self.batch_size == 1:
            return f"{self.backend}_step"
        return f"{self.backend}_bs{self.batch_size}"

    @property
    def label(self) -> str:
        if self.batch_size == 1:
            return f"{self.backend} step"
        return f"{self.backend} bs={self.batch_size}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark replay renderers on a delivered episode")
    parser.add_argument("episode_dir", help="Delivered episode directory")
    parser.add_argument(
        "--output-dir",
        default="/tmp/xdof_render_benchmark_20260411",
        help="Directory for benchmark artifacts",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=512,
        help="Number of frames to render for each configuration",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Output FPS")
    parser.add_argument("--render-width", type=int, default=640, help="Render width")
    parser.add_argument("--render-height", type=int, default=480, help="Render height")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256],
        help="Batched qpos sizes to benchmark for MJWarp and Madrona",
    )
    return parser.parse_args()


def build_configs(batch_sizes: list[int]) -> list[RenderConfig]:
    configs = [
        RenderConfig("mujoco", 1),
        RenderConfig("mjwarp", 1),
        RenderConfig("madrona", 1),
    ]
    for batch_size in batch_sizes:
        configs.append(RenderConfig("mjwarp", batch_size))
        configs.append(RenderConfig("madrona", batch_size))
    return configs


def run_config(
    cfg: RenderConfig,
    *,
    episode_dir: Path,
    output_dir: Path,
    max_frames: int,
    fps: float,
    render_width: int,
    render_height: int,
) -> dict[str, object]:
    video_path = output_dir / "videos" / f"{cfg.slug}.mp4"
    timings_path = output_dir / "timings" / f"{cfg.slug}.json"
    log_path = output_dir / "logs" / f"{cfg.slug}.log"
    for path in (video_path.parent, timings_path.parent, log_path.parent):
        path.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "xdof_sim.rendering.replay.simulate",
        str(episode_dir),
        str(video_path),
        "--camera-source",
        "sim",
        "--sim-render-backend",
        cfg.backend,
        "--replay-mode",
        "auto",
        "--fps",
        str(fps),
        "--render-width",
        str(render_width),
        "--render-height",
        str(render_height),
        "--max-frames",
        str(max_frames),
        "--timings-json",
        str(timings_path),
    ]
    if cfg.batch_size > 1:
        cmd.extend(["--sim-batch-size", str(cfg.batch_size)])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    wall_s = time.perf_counter() - t0

    log_path.write_text(
        f"$ {' '.join(cmd)}\n\n"
        f"exit_code={proc.returncode}\n"
        f"wall_s={wall_s:.6f}\n\n"
        "STDOUT\n"
        f"{proc.stdout}\n"
        "STDERR\n"
        f"{proc.stderr}\n"
    )

    if proc.returncode != 0:
        raise RuntimeError(f"{cfg.label} failed. See {log_path}")

    timings = json.loads(timings_path.read_text())
    video_meta = inspect_video(video_path)
    result: dict[str, object] = {
        "backend": cfg.backend,
        "batch_size": cfg.batch_size,
        "mode": "batched-qpos" if cfg.uses_batched_export else "step",
        "label": cfg.label,
        "slug": cfg.slug,
        "video_path": str(video_path),
        "timings_path": str(timings_path),
        "log_path": str(log_path),
        "wall_s": wall_s,
        "file_size_mb": video_path.stat().st_size / (1024 * 1024),
    }
    result.update(video_meta)
    result.update(timings)
    export_s = float(result["export_s"])
    total_s = float(result["total_s"])
    frame_count = int(result["frame_count"])
    result["export_fps"] = frame_count / export_s if export_s > 0 else math.nan
    result["end_to_end_fps"] = frame_count / total_s if total_s > 0 else math.nan
    return result


def inspect_video(video_path: Path) -> dict[str, object]:
    reader = iio.get_reader(video_path, format="FFMPEG")
    try:
        frame0 = reader.get_data(0)
        meta = reader.get_meta_data()
        try:
            frame_count = reader.count_frames()
        except Exception:
            frame_count = None
    finally:
        reader.close()
    return {
        "video_frames": frame_count,
        "video_width": int(frame0.shape[1]),
        "video_height": int(frame0.shape[0]),
        "video_fps": float(meta.get("fps", 0.0)),
    }


def write_results(results: list[dict[str, object]], output_dir: Path) -> tuple[Path, Path]:
    json_path = output_dir / "benchmark_results.json"
    csv_path = output_dir / "benchmark_results.csv"
    json_path.write_text(json.dumps(results, indent=2) + "\n")

    fieldnames = [
        "backend",
        "batch_size",
        "mode",
        "label",
        "slug",
        "frame_count",
        "video_frames",
        "video_width",
        "video_height",
        "video_fps",
        "file_size_mb",
        "load_episode_s",
        "create_session_s",
        "camera_setup_s",
        "export_s",
        "total_s",
        "wall_s",
        "export_fps",
        "end_to_end_fps",
        "video_path",
        "timings_path",
        "log_path",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return json_path, csv_path


def generate_charts(results: list[dict[str, object]], output_dir: Path) -> dict[str, Path]:
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    labels = [str(row["label"]) for row in results]
    x = np.arange(len(results))
    load = np.array([float(row["load_episode_s"]) for row in results])
    session = np.array([float(row["create_session_s"]) for row in results])
    setup = np.array([float(row["camera_setup_s"]) for row in results])
    export = np.array([float(row["export_s"]) for row in results])

    plt.figure(figsize=(14, 6))
    plt.bar(x, load, label="load episode")
    plt.bar(x, session, bottom=load, label="create session")
    plt.bar(x, setup, bottom=load + session, label="camera/setup")
    plt.bar(x, export, bottom=load + session + setup, label="export")
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("Seconds")
    plt.title("Replay Render Wall-Time Breakdown")
    plt.legend()
    plt.tight_layout()
    breakdown_path = charts_dir / "wall_time_breakdown.png"
    plt.savefig(breakdown_path, dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    for backend, marker in (("mjwarp", "o"), ("madrona", "s")):
        backend_rows = [row for row in results if row["backend"] == backend]
        batches = [int(row["batch_size"]) for row in backend_rows]
        fps = [float(row["export_fps"]) for row in backend_rows]
        plt.plot(batches, fps, marker=marker, linewidth=2, label=backend)
    mujoco_rows = [row for row in results if row["backend"] == "mujoco"]
    if mujoco_rows:
        plt.scatter(
            [1],
            [float(mujoco_rows[0]["export_fps"])],
            marker="^",
            s=120,
            label="mujoco step",
        )
    plt.xscale("symlog", linthresh=1)
    plt.xticks([1, 32, 64, 128, 256], ["1", "32", "64", "128", "256"])
    plt.xlabel("Batch size")
    plt.ylabel("Export FPS")
    plt.title("Replay Export Throughput by Renderer")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    fps_path = charts_dir / "export_fps_by_batch.png"
    plt.savefig(fps_path, dpi=180)
    plt.close()

    return {
        "breakdown": breakdown_path,
        "export_fps": fps_path,
    }


def create_comparison_videos(results: list[dict[str, object]], output_dir: Path) -> dict[str, Path]:
    comparison_dir = output_dir / "comparison_videos"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    result_map = {str(row["slug"]): row for row in results}
    baseline_specs = [
        ("MuJoCo step", Path(str(result_map["mujoco_step"]["video_path"]))),
        ("MJWarp step", Path(str(result_map["mjwarp_step"]["video_path"]))),
        ("Madrona step", Path(str(result_map["madrona_step"]["video_path"]))),
    ]
    baseline_path = comparison_dir / "compare_step_renderers.mp4"
    make_grid_video(
        baseline_specs,
        baseline_path,
        columns=3,
        tile_size=(640, 174),
    )

    batch_specs = []
    for backend in ("mjwarp", "madrona"):
        backend_rows = sorted(
            (row for row in results if row["backend"] == backend and int(row["batch_size"]) > 1),
            key=lambda row: int(row["batch_size"]),
        )
        for row in backend_rows:
            batch_specs.append((f"{backend} bs={int(row['batch_size'])}", Path(str(row["video_path"]))))

    outputs = {
        "step_compare": baseline_path,
    }
    if batch_specs:
        batched_path = comparison_dir / "compare_batched_renderers.mp4"
        make_grid_video(
            batch_specs,
            batched_path,
            columns=max(1, min(4, len(batch_specs))),
            tile_size=(480, 142),
        )
        outputs["batched_compare"] = batched_path

    return outputs


def make_grid_video(
    specs: list[tuple[str, Path]],
    output_path: Path,
    *,
    columns: int,
    tile_size: tuple[int, int],
) -> None:
    font = ImageFont.load_default()
    tile_width, tile_height = tile_size
    label_height = 18
    readers = [iio.get_reader(path, format="FFMPEG") for _, path in specs]
    try:
        frame_counts = [reader.count_frames() for reader in readers]
        frame_total = min(frame_counts)
        rows = math.ceil(len(specs) / columns)
        canvas_width = columns * tile_width
        canvas_height = rows * (tile_height + label_height)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with iio.get_writer(output_path, fps=30.0, format="FFMPEG") as writer:
            for frame_idx in range(frame_total):
                canvas = Image.new("RGB", (canvas_width, canvas_height), color=(16, 16, 16))
                for slot, ((label, _), reader) in enumerate(zip(specs, readers)):
                    frame = reader.get_data(frame_idx)
                    tile = Image.fromarray(frame).resize((tile_width, tile_height), Image.Resampling.BILINEAR)
                    draw = ImageDraw.Draw(tile)
                    draw.rectangle((0, 0, tile_width, label_height), fill=(0, 0, 0))
                    draw.text((6, 3), label, fill=(255, 255, 255), font=font)
                    row = slot // columns
                    col = slot % columns
                    x = col * tile_width
                    y = row * (tile_height + label_height)
                    canvas.paste(tile, (x, y))
                writer.append_data(np.asarray(canvas))
    finally:
        for reader in readers:
            reader.close()


def print_summary(results: list[dict[str, object]]) -> None:
    print("\nBenchmark summary:")
    for row in results:
        print(
            f"  {row['label']:<18} "
            f"frames={int(row['frame_count']):>4} "
            f"total={float(row['total_s']):>7.2f}s "
            f"export={float(row['export_s']):>7.2f}s "
            f"fps={float(row['export_fps']):>7.1f}"
        )


def main() -> None:
    args = parse_args()
    episode_dir = Path(args.episode_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "episode_dir": str(episode_dir),
        "output_dir": str(output_dir),
        "max_frames": args.max_frames,
        "fps": args.fps,
        "render_width": args.render_width,
        "render_height": args.render_height,
        "batch_sizes": args.batch_sizes,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    configs = build_configs(args.batch_sizes)
    results: list[dict[str, object]] = []
    for cfg in configs:
        print(f"\n=== {cfg.label} ===")
        result = run_config(
            cfg,
            episode_dir=episode_dir,
            output_dir=output_dir,
            max_frames=args.max_frames,
            fps=args.fps,
            render_width=args.render_width,
            render_height=args.render_height,
        )
        results.append(result)
        print(
            f"saved {result['video_path']} "
            f"total={float(result['total_s']):.2f}s export={float(result['export_s']):.2f}s "
            f"fps={float(result['export_fps']):.1f}"
        )

    json_path, csv_path = write_results(results, output_dir)
    charts = generate_charts(results, output_dir)
    comparison_videos = create_comparison_videos(results, output_dir)
    print_summary(results)
    print(f"\nResults JSON: {json_path}")
    print(f"Results CSV:  {csv_path}")
    for name, path in charts.items():
        print(f"Chart {name}: {path}")
    for name, path in comparison_videos.items():
        print(f"Comparison video {name}: {path}")


if __name__ == "__main__":
    main()
