# Replay Render Benchmark: `sim_spell_cat`

Date: 2026-04-11

This benchmark was run against the delivered episode:

- `s3://xdof-bair-abc/data/deliveries/sim_tasks_20260409/sim_spell_cat/episode_019d71d5-0c61-7777-8e84-bee67c8b9391`
- Local copy: `/tmp/xdof_s3_samples/sim_spell_cat/episode_019d71d5-0c61-7777-8e84-bee67c8b9391`

The benchmark harness is [benchmark_replay_rendering.py](/home/adamrasb/src/sim/xdof-sim/scripts/benchmark_replay_rendering.py:1).

## Setup

- Frame window: first `512` replay frames
- Cameras: `top`, `left`, `right`
- Per-camera resolution: `640x480`
- Output video shape: `1920x480` tiled RGB
- Replay mode: step replay for batch size `1`; exact `qpos` batched replay for MJWarp and Madrona at batch sizes `32`, `64`, `128`, `256`
- GPU: `NVIDIA GeForce RTX 4090`, driver `570.86.10`
- Python: `3.12.12`
- MuJoCo: `3.5.0`
- Warp: `1.12.1`
- Torch: `2.8.0+cu128`
- Note: MuJoCo has no batched `qpos` export path in this repo, so it is only included as the step-render baseline.
- Note: These are warm-cache numbers. Warp kernels were loaded from `/home/adamrasb/.cache/warp/1.12.1`, and Madrona used the local editable `madrona_mjwarp` build from `/home/adamrasb/src/sim/madrona_mjwarp`.

## Summary

- `mjwarp` is the fastest renderer in this pipeline on this task.
- Best measured throughput was `mjwarp bs=32` at `22.47 FPS`, with `bs=64/128/256` effectively flat.
- `madrona` step rendering was close to `mjwarp` step rendering, but batching did not help much here and regressed at larger batch sizes.
- Relative to the MuJoCo step baseline, `mjwarp` step was `6.28x` faster and `mjwarp bs=32` was `8.55x` faster.
- Relative to `madrona` step, `madrona bs=32` was only `1.04x` faster, and `bs=256` dropped to `0.80x`.
- Visual output from `mjwarp` and `madrona` is effectively identical in the comparison videos. MuJoCo is close, but the background/shading differs slightly.

## Charts

Breakdown chart: [wall_time_breakdown.png](/tmp/xdof_render_benchmark_20260411/charts/wall_time_breakdown.png)

Throughput chart: [export_fps_by_batch.png](/tmp/xdof_render_benchmark_20260411/charts/export_fps_by_batch.png)

## Results

| Backend | Batch | Mode | Total (s) | Export (s) | Export FPS | Speedup vs MuJoCo |
|---|---:|---|---:|---:|---:|---:|
| MuJoCo | 1 | step | 197.48 | 194.82 | 2.63 | 1.00x |
| MJWarp | 1 | step | 35.24 | 31.03 | 16.50 | 6.28x |
| Madrona | 1 | step | 38.55 | 31.91 | 16.05 | 6.11x |
| MJWarp | 32 | batched qpos | 25.05 | 22.79 | 22.47 | 8.55x |
| Madrona | 32 | batched qpos | 32.95 | 30.67 | 16.69 | 6.35x |
| MJWarp | 64 | batched qpos | 25.21 | 22.94 | 22.32 | 8.49x |
| Madrona | 64 | batched qpos | 34.06 | 31.78 | 16.11 | 6.13x |
| MJWarp | 128 | batched qpos | 25.20 | 22.92 | 22.34 | 8.50x |
| Madrona | 128 | batched qpos | 37.08 | 34.81 | 14.71 | 5.60x |
| MJWarp | 256 | batched qpos | 25.38 | 23.11 | 22.15 | 8.43x |
| Madrona | 256 | batched qpos | 42.29 | 40.00 | 12.80 | 4.87x |

The stacked breakdown chart makes the main bottleneck obvious: almost all of the wall time is in the export stage, not episode load or session setup. On this task, batching helps `mjwarp` materially, while `madrona` stays near its step baseline at `bs=32/64` and then gets worse as batch size increases.

## Comparison Videos

Step-render comparison: [compare_step_renderers.mp4](/tmp/xdof_render_benchmark_20260411/comparison_videos/compare_step_renderers.mp4)

Batched comparison: [compare_batched_renderers.mp4](/tmp/xdof_render_benchmark_20260411/comparison_videos/compare_batched_renderers.mp4)

Per-config output videos:

- [mujoco_step.mp4](/tmp/xdof_render_benchmark_20260411/videos/mujoco_step.mp4)
- [mjwarp_step.mp4](/tmp/xdof_render_benchmark_20260411/videos/mjwarp_step.mp4)
- [madrona_step.mp4](/tmp/xdof_render_benchmark_20260411/videos/madrona_step.mp4)
- [mjwarp_bs32.mp4](/tmp/xdof_render_benchmark_20260411/videos/mjwarp_bs32.mp4)
- [madrona_bs32.mp4](/tmp/xdof_render_benchmark_20260411/videos/madrona_bs32.mp4)
- [mjwarp_bs64.mp4](/tmp/xdof_render_benchmark_20260411/videos/mjwarp_bs64.mp4)
- [madrona_bs64.mp4](/tmp/xdof_render_benchmark_20260411/videos/madrona_bs64.mp4)
- [mjwarp_bs128.mp4](/tmp/xdof_render_benchmark_20260411/videos/mjwarp_bs128.mp4)
- [madrona_bs128.mp4](/tmp/xdof_render_benchmark_20260411/videos/madrona_bs128.mp4)
- [mjwarp_bs256.mp4](/tmp/xdof_render_benchmark_20260411/videos/mjwarp_bs256.mp4)
- [madrona_bs256.mp4](/tmp/xdof_render_benchmark_20260411/videos/madrona_bs256.mp4)

## Raw Artifacts

- Results CSV: [benchmark_results.csv](/tmp/xdof_render_benchmark_20260411/benchmark_results.csv)
- Results JSON: [benchmark_results.json](/tmp/xdof_render_benchmark_20260411/benchmark_results.json)
- Manifest: [manifest.json](/tmp/xdof_render_benchmark_20260411/manifest.json)
- Timings directory: [/tmp/xdof_render_benchmark_20260411/timings](/tmp/xdof_render_benchmark_20260411/timings)
- Logs directory: [/tmp/xdof_render_benchmark_20260411/logs](/tmp/xdof_render_benchmark_20260411/logs)

## Reproduce

```bash
uv run python scripts/benchmark_replay_rendering.py \
  /tmp/xdof_s3_samples/sim_spell_cat/episode_019d71d5-0c61-7777-8e84-bee67c8b9391 \
  --output-dir /tmp/xdof_render_benchmark_20260411 \
  --max-frames 512 \
  --batch-sizes 32 64 128 256
```
