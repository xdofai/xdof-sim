# Replay Render Benchmark: `sim_spell_cat` at `224x224`

Date: 2026-04-11

This run uses the same delivered episode as the earlier benchmark, but changes the render resolution to `224x224` and includes the qpos preload optimization in the batched path.

- Episode: `s3://xdof-bair-abc/data/deliveries/sim_tasks_20260409/sim_spell_cat/episode_019d71d5-0c61-7777-8e84-bee67c8b9391`
- Local copy: `/tmp/xdof_s3_samples/sim_spell_cat/episode_019d71d5-0c61-7777-8e84-bee67c8b9391`
- Harness: [benchmark_replay_rendering.py](/home/adamrasb/src/sim/xdof-sim/scripts/benchmark_replay_rendering.py:1)

## Setup

- Frame window: first `512` replay frames
- Cameras: `top`, `left`, `right`
- Per-camera resolution: `224x224`
- Output video shape: `672x224`
- Replay mode: step replay for batch size `1`; exact `qpos` batched replay for MJWarp and Madrona at `32`, `64`, `128`, `256`
- GPU: `NVIDIA GeForce RTX 4090`, driver `570.86.10`
- Python: `3.12.12`
- MuJoCo: `3.5.0`
- Warp: `1.12.1`
- Torch: `2.8.0+cu128`
- Batched qpos change: replay `sim_states` are uploaded to Warp device memory once, then sliced device-side into `d_warp.qpos` instead of rebuilding a CPU numpy batch every iteration.
- Note: MuJoCo is still only included as the step-render baseline because there is no batched qpos export path for the MuJoCo renderer in this repo.

## Summary

- `224x224` materially improves both GPU renderers and removes the Madrona square-crop penalty.
- `madrona` is now faster than `mjwarp` in step mode: `38.03 FPS` vs `32.12 FPS`.
- `mjwarp` remains the fastest batched path in this replay exporter: best result was `69.10 FPS` at `bs=32`.
- `madrona` batching is now much stronger than in the `640x480` run, but still trails `mjwarp` in the batched export path.
- Relative to MuJoCo step, `madrona step` is `13.66x` faster, `mjwarp step` is `11.54x` faster, and `mjwarp bs=32` is `24.82x` faster.

## Charts

Breakdown chart: [wall_time_breakdown.png](/tmp/xdof_render_benchmark_20260411_224/charts/wall_time_breakdown.png)

Throughput chart: [export_fps_by_batch.png](/tmp/xdof_render_benchmark_20260411_224/charts/export_fps_by_batch.png)

## Results

| Backend | Batch | Mode | Total (s) | Export (s) | Export FPS | Speedup vs MuJoCo |
|---|---:|---|---:|---:|---:|---:|
| MuJoCo | 1 | step | 186.63 | 183.95 | 2.78 | 1.00x |
| MJWarp | 1 | step | 20.19 | 15.94 | 32.12 | 11.54x |
| Madrona | 1 | step | 20.16 | 13.46 | 38.03 | 13.66x |
| MJWarp | 32 | batched qpos | 9.70 | 7.41 | 69.10 | 24.82x |
| Madrona | 32 | batched qpos | 12.03 | 9.74 | 52.55 | 18.88x |
| MJWarp | 64 | batched qpos | 9.81 | 7.54 | 67.92 | 24.40x |
| Madrona | 64 | batched qpos | 12.06 | 9.77 | 52.41 | 18.83x |
| MJWarp | 128 | batched qpos | 9.89 | 7.60 | 67.39 | 24.21x |
| Madrona | 128 | batched qpos | 12.14 | 9.86 | 51.94 | 18.66x |
| MJWarp | 256 | batched qpos | 9.87 | 7.59 | 67.49 | 24.25x |
| Madrona | 256 | batched qpos | 13.25 | 10.95 | 46.78 | 16.81x |

## Interpretation

- The qpos preload change clearly helped the batched exporter. The hot path is no longer rebuilding a padded CPU numpy batch for every render window.
- The remaining gap to VisualDexterity is mostly not about basic renderer setup anymore. That repo measures a tighter render pipeline and writes video with a raw-ffmpeg batch writer; this repo still pays more replay/export overhead.
- The `224x224` result is much closer to the VisualDexterity pattern: Madrona step rendering is ahead of MJWarp step rendering, and both GPU backends are far ahead of MuJoCo.
- In this exporter, `mjwarp` still wins once batching is enabled. That suggests the remaining bottleneck is likely outside pure ray tracing: transfer, composition, and encode behavior still matter.

## Comparison Videos

Step-render comparison: [compare_step_renderers.mp4](/tmp/xdof_render_benchmark_20260411_224/comparison_videos/compare_step_renderers.mp4)

Batched comparison: [compare_batched_renderers.mp4](/tmp/xdof_render_benchmark_20260411_224/comparison_videos/compare_batched_renderers.mp4)

Per-config videos:

- [mujoco_step.mp4](/tmp/xdof_render_benchmark_20260411_224/videos/mujoco_step.mp4)
- [mjwarp_step.mp4](/tmp/xdof_render_benchmark_20260411_224/videos/mjwarp_step.mp4)
- [madrona_step.mp4](/tmp/xdof_render_benchmark_20260411_224/videos/madrona_step.mp4)
- [mjwarp_bs32.mp4](/tmp/xdof_render_benchmark_20260411_224/videos/mjwarp_bs32.mp4)
- [madrona_bs32.mp4](/tmp/xdof_render_benchmark_20260411_224/videos/madrona_bs32.mp4)
- [mjwarp_bs64.mp4](/tmp/xdof_render_benchmark_20260411_224/videos/mjwarp_bs64.mp4)
- [madrona_bs64.mp4](/tmp/xdof_render_benchmark_20260411_224/videos/madrona_bs64.mp4)
- [mjwarp_bs128.mp4](/tmp/xdof_render_benchmark_20260411_224/videos/mjwarp_bs128.mp4)
- [madrona_bs128.mp4](/tmp/xdof_render_benchmark_20260411_224/videos/madrona_bs128.mp4)
- [mjwarp_bs256.mp4](/tmp/xdof_render_benchmark_20260411_224/videos/mjwarp_bs256.mp4)
- [madrona_bs256.mp4](/tmp/xdof_render_benchmark_20260411_224/videos/madrona_bs256.mp4)

## Raw Artifacts

- Results CSV: [benchmark_results.csv](/tmp/xdof_render_benchmark_20260411_224/benchmark_results.csv)
- Results JSON: [benchmark_results.json](/tmp/xdof_render_benchmark_20260411_224/benchmark_results.json)
- Manifest: [manifest.json](/tmp/xdof_render_benchmark_20260411_224/manifest.json)
- Timings directory: [/tmp/xdof_render_benchmark_20260411_224/timings](/tmp/xdof_render_benchmark_20260411_224/timings)
- Logs directory: [/tmp/xdof_render_benchmark_20260411_224/logs](/tmp/xdof_render_benchmark_20260411_224/logs)

## Reproduce

```bash
uv run python scripts/benchmark_replay_rendering.py \
  /tmp/xdof_s3_samples/sim_spell_cat/episode_019d71d5-0c61-7777-8e84-bee67c8b9391 \
  --output-dir /tmp/xdof_render_benchmark_20260411_224 \
  --max-frames 512 \
  --render-width 224 \
  --render-height 224 \
  --batch-sizes 32 64 128 256
```
