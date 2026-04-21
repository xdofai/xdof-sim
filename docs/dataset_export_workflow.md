# Sim Dataset Export Workflow

Date: 2026-04-11

This document describes the operational workflow for exporting delivered sim episodes from S3 into the trainable xdof dataset format used by the BC pipeline.

The exporter lives under `xdof_sim.dataset_export` and supports all three sim renderers:

- `mujoco`
- `mjwarp`
- `madrona`

The production flow is two-phase:

1. `s3-export`: list source episodes from S3, download this shard to local scratch, render and export locally, upload episode outputs plus shard metadata.
2. `s3-finalize`: merge shard metadata from the destination dataset root, redownload only `states_actions.npy`, compute `norm_stats.json`, generate train/val splits, and upload final metadata.

## Source Assumptions

The source prefix should contain delivered sim episodes with per-episode files like:

```text
s3://<source-bucket>/<delivery>/<task>/episode_<uuid>/
  output.mcap
  sim_state.mcap
  randomization.json   # optional
```

The exporter currently assumes exact-qpos replay data is present. In practice that means `sim_state.mcap` needs to exist for the delivered episode.

Nested duplicate layouts are handled. If the source contains both:

```text
episode_<uuid>/output.mcap
episode_<uuid>/episode_<uuid>/output.mcap
```

the exporter deduplicates by episode directory name and keeps the shorter path.

## Output Layout

The destination S3 root is treated as a dataset root:

```text
s3://<output-bucket>/<dataset-root>/
  data/
    episode_<uuid>/
      combined_camera-images-rgb.mp4
      combined_camera-images-rgb_frame_mappings.json
      top_camera-images-rgb.mp4
      top_camera-images-rgb_frame_mappings.json
      left_camera-images-rgb.mp4
      left_camera-images-rgb_frame_mappings.json
      right_camera-images-rgb.mp4
      right_camera-images-rgb_frame_mappings.json
      states_actions.npy
      states_actions.bin
      episode_metadata.json
  metadata/
    collected_shard_00.json
    camera_profiles_shard_00.json
    ...
    collected.json
    camera_profiles.json
    norm_stats.json
    train_episodes.json
    val_episodes.json
```

Each `states_actions.npy` row is `14D state + 14D action`, matching the xdof training contract.

## Prerequisites

- `uv` environment set up for this repo
- `ffmpeg` and `ffprobe`
- AWS CLI available in the shell
- AWS credentials for the source bucket and destination bucket
- Enough local scratch space for:
  - one staged episode
  - one locally rendered episode output
  - temporary finalize metadata and downloaded `states_actions.npy`

For GPU-backed rendering, the local environment also needs a working renderer stack:

- `mjwarp` path: Warp + CUDA
- `madrona` path: local `madrona_mjwarp` build available
- `mujoco` path: headless EGL-capable MuJoCo setup

## Phase 1: Shard Export

Run one `s3-export` process per shard. Each shard independently:

1. lists the source prefix
2. computes its round-robin shard assignment
3. downloads one assigned episode at a time into local scratch
4. renders and exports that episode locally
5. uploads `data/episode_<uuid>/...`
6. uploads `metadata/collected_shard_XX.json`
7. uploads `metadata/camera_profiles_shard_XX.json`

Example:

```bash
uv run python -m xdof_sim.dataset_export.cli s3-export \
  --s3-input-prefix s3://xdof-bair-abc/data/deliveries/sim_tasks_20260409/ \
  --s3-output-root s3://far-research-internal/datasets/robot_path/sim_tasks_20260409_224/ \
  --scratch-dir /mnt/nvme/xdof_export \
  --shard-index 0 \
  --num-shards 8 \
  --source-aws-profile xdof-bair-abc \
  --output-aws-profile far-compute \
  --batch-name sim_tasks_20260409_224 \
  --render-backend madrona \
  --img-width 224 \
  --img-height 224 \
  --fps 30 \
  --sim-batch-size 32
```

For `mjwarp` or `mujoco`, change `--render-backend`.

Useful knobs:

- `--max-episodes 1`: smoke test one episode
- `--gpu-id <n>`: choose a GPU for `mjwarp` or `madrona`
- `--source-region` / `--output-region`: override AWS regions if needed

## Phase 2: Finalize Metadata

After all shard exports finish, run `s3-finalize` once against the destination dataset root.

This step:

1. downloads all `collected_shard_XX.json`
2. downloads all `camera_profiles_shard_XX.json`
3. merges them into `collected.json` and `camera_profiles.json`
4. downloads `data/<episode_id>/states_actions.npy` for every exported episode
5. computes `norm_stats.json`
6. creates `train_episodes.json` and `val_episodes.json`
7. uploads the final merged metadata back to `metadata/`

Example:

```bash
uv run python -m xdof_sim.dataset_export.cli s3-finalize \
  --s3-output-root s3://far-research-internal/datasets/robot_path/sim_tasks_20260409_224/ \
  --scratch-dir /mnt/nvme/xdof_finalize \
  --aws-profile far-compute
```

Useful knobs:

- `--val-ratio 0.05`
- `--seed 123`
- `--n-states 14`
- `--n-actions 14`
- `--state-pad-size 32`
- `--action-pad-size 32`

## Local Scratch Layout

During `s3-export`, one shard uses a scratch tree like:

```text
<scratch-dir>/
  shard_00/
    source/
      <task>/episode_<uuid>/
        output.mcap
        sim_state.mcap
        randomization.json
    dataset/
      data/
        episode_<uuid>/
          ...
    metadata/
      collected_shard_00.json
      camera_profiles_shard_00.json
```

During `s3-finalize`, scratch looks like:

```text
<scratch-dir>/
  finalize/
    remote_metadata/
      collected_shard_00.json
      camera_profiles_shard_00.json
      ...
    dataset/
      data/
        episode_<uuid>/states_actions.npy
      metadata/
        collected.json
        camera_profiles.json
        norm_stats.json
        train_episodes.json
        val_episodes.json
```

The exporter cleans up per-episode staged source data and per-episode rendered output after successful upload. Finalize scratch is rebuilt from scratch on each run.

## Local Debug Workflow

For local debugging without S3 orchestration, the CLI still supports:

```bash
uv run python -m xdof_sim.dataset_export.cli run \
  --input-root /tmp/xdof_raw \
  --output-root /tmp/xdof_trainable \
  --batch-name sim_tasks_20260409_224 \
  --render-backend mjwarp \
  --img-width 224 \
  --img-height 224 \
  --fps 30 \
  --sim-batch-size 32 \
  --s3-prefix s3://far-research-internal/datasets/robot_path/sim_tasks_20260409_224/ \
  --aws-profile far-compute
```

That path is useful when iterating on one local delivery directory before running the sharded S3 flow.

## Operational Notes

- The exporter stages episodes locally. It does not stream MCAP directly from S3 into the renderer.
- The production S3 flow is intentionally episode-at-a-time. It does not require syncing the full raw source bucket to disk first.
- Finalize downloads only `states_actions.npy`, not videos.
- `mjwarp` and `madrona` use the batched exact-qpos render path. `mujoco` uses the exact-qpos CPU/offscreen path.
- `mujoco` may emit EGL warnings on some hosts even when headless export succeeds.
- There is currently no skip-existing or resumable per-episode upload logic in `s3-export`. Re-running a shard is safe, but it will re-export and re-upload that shard’s assigned episodes.
- Shard assignment is deterministic for a fixed source prefix and `num_shards`.

## Smoke Test Pattern

Before launching a full export, run a one-episode smoke test:

```bash
uv run python -m xdof_sim.dataset_export.cli s3-export \
  --s3-input-prefix s3://xdof-bair-abc/data/deliveries/sim_tasks_20260409/ \
  --s3-output-root s3://far-research-internal/datasets/robot_path/sim_tasks_20260409_224_smoke/ \
  --scratch-dir /tmp/xdof_export_smoke \
  --shard-index 0 \
  --num-shards 1 \
  --source-aws-profile xdof-bair-abc \
  --output-aws-profile far-compute \
  --batch-name sim_tasks_20260409_224_smoke \
  --render-backend mjwarp \
  --img-width 224 \
  --img-height 224 \
  --fps 30 \
  --sim-batch-size 32 \
  --max-episodes 1
```

Then finalize that smoke dataset root and inspect the resulting episode directory plus metadata before scaling out.

## Registry Integration

After upload and finalize, add the dataset to the ABC registry in the same way as a real exported xdof dataset:

- `s3_dataset_path=<dataset-root>/data`
- `metadata/collected.json`
- `metadata/norm_stats.json`
- `metadata/train_episodes.json`
- `metadata/val_episodes.json`
- `n_states=14`
- `n_actions=14`
- `source_cameras=("top", "left", "right")`
- `output_cameras=("top", "left", "right")`

At that point the same xdof-format loaders and dataset viewers can consume the sim dataset.
