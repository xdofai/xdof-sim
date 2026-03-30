# xdof-sim

Standalone MuJoCo simulation environment for the YAM bimanual robot, for use in evaluation, data replay, and teleoperation.

<p align="center">
  <img src="xdof_sim/assets/replay_highlight.gif" alt="Bimanual bottle-in-bin replay" width="720">
</p>

## Installation

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
./setup.sh          # default: video extras
./setup.sh all      # everything (policy, teleop, video)
./setup.sh policy   # policy inference (torch)
./setup.sh teleop   # ZMQ teleoperation
source .venv/bin/activate
```

Use a different Python: `PYTHON=python3.11 ./setup.sh`

## Quick Start

```python
import xdof_sim

# Create environment (one line)
env = xdof_sim.make_env(scene="hybrid")

# Reset and get observation
obs, info = env.reset()
print(obs["state"])       # 14D joint state [left_j1..6, left_grip, right_j1..6, right_grip]
print(obs["images"].keys())  # dict_keys(['top', 'left', 'right'])

# Step with an action chunk (30 timesteps x 14D)
import numpy as np
init_q = env.get_init_q()
action = np.tile(init_q, (30, 1))  # repeat init pose
obs, chunk_history, reward, terminated, truncated, info = env.step(action)

env.close()
```

## Features

### Environment

The `MuJoCoYAMEnv` simulates the bimanual YAM robot with:
- **14D state/action space**: 6 joints + 1 gripper per arm (left + right)
- **3 cameras**: top, left, right (480x640 RGB)
- **30Hz control**: physics at 500Hz with 17x decimation
- **Chunked actions**: 30-step action chunks matching the real robot interface

### Scene Variants

Three scene configurations via the `scene` parameter:
- `eval` — Cage walls + blue-grey bin
- `training` — No walls + orange bucket
- `hybrid` — Cage walls + orange bucket (recommended for evaluation)

```python
env = xdof_sim.make_env(scene="hybrid")
```

### Replay Demo

Replay a bundled episode in the MuJoCo viewer (no checkpoint needed):

```bash
python -m xdof_sim.examples.replay_demo --view --highlight
```

### Viser 3D Web Viewer

Interactive browser-based 3D replay viewer with full scene rendering, playback
controls, frame scrubbing, and camera picture-in-picture views:

```bash
pip install xdof-sim[viser]  # or: uv pip install xdof-sim[viser]

# Watch bottle pickups (auto-trimmed to bottle-in-bin events)
uv run python -m xdof_sim.examples.viser_replay \
    --episode-dir xdof_sim/examples/episode_6ib_seed0 --highlight

# Replay bundled 2-IB demo episode (default, full)
python -m xdof_sim.examples.viser_replay

# Replay bundled 6-bottle demo episode
python -m xdof_sim.examples.viser_replay \
    --episode-dir xdof_sim/examples/episode_6ib_seed0

# Auto-trim to bottle-in-bin events
python -m xdof_sim.examples.viser_replay --highlight

# Replay custom actions file
python -m xdof_sim.examples.viser_replay \
    --actions /path/to/actions.npy --scene hybrid

# Custom port
python -m xdof_sim.examples.viser_replay --port 9090
```

Open `http://localhost:8080` in a browser to view the 3D scene with:
- **Play/Pause/Step/Reset** controls
- **Speed** slider (0.1x–5x)
- **Frame scrubber** to jump to any recorded frame
- **Camera PiP** views (top, left, right wrist)
- **Collision geom** toggle

### Interactive Viewer

Inspect the scene or watch random joint motions:

```bash
python -m xdof_sim.examples.view
python -m xdof_sim.examples.view --animate
```

### Data Replay

Replay recorded episode data in the simulator:

```bash
# From a .npy file of joint states
MUJOCO_GL=egl python -m xdof_sim.examples.replay_data \
    --states-npy /path/to/states.npy \
    --output /tmp/replay.mp4

# From an episode directory
MUJOCO_GL=egl python -m xdof_sim.examples.replay_data \
    --episode-dir /path/to/episode_000000 \
    --output /tmp/replay.mp4
```

### Policy Inference

Run a trained LBM policy in simulation:

```bash
MUJOCO_GL=egl python -m xdof_sim.examples.run_policy \
    --checkpoint-path /path/to/checkpoint.pt \
    --scene hybrid --diffusion-steps 3 \
    --output /tmp/rollout.mp4
```

### GELLO Teleoperation

Drive the sim with physical GELLO leader arms via ZMQ, with a live 3D Viser
viewer in the browser:

```bash
# Terminal 1: Start the live viser viewer (sim follower + 3D view)
uv run python -m xdof_sim.examples.viser_teleop --scene hybrid

# Terminal 2: Start left GELLO leader
uv run python -m xdof_sim.teleop.gello_leader --name left --device /dev/ttyUSB0

# Terminal 3: Start right GELLO leader
uv run python -m xdof_sim.teleop.gello_leader --name right --device /dev/ttyUSB1
```

Open `http://localhost:8080` in a browser to see the live 3D scene. The viewer
includes a **Record** button to capture teleop sessions as `.npy` files for
replay later with `viser_replay`.

You can also run the headless sim follower without the Viser viewer:

```bash
uv run python -m xdof_sim.teleop.sim_follower --scene hybrid
```

> **Note:** GELLO leaders require physical hardware (Dynamixel servos connected
> via USB). Install with `./setup.sh all` or `uv pip install -e ".[gello,viser]"`.

### VR Teleop

Stream the MuJoCo scene to a VR headset (Quest, Vision Pro, etc.) via Three.js
WebXR. Physics runs server-side; the headset only renders transforms.

```bash
# Terminal 1: Start the VR streaming server
uv run python -m xdof_sim.examples.vr_streamer --task blocks

# Terminal 2: Start GELLO leader(s)
uv run python -m xdof_sim.teleop.gello_leader --name right --device /dev/ttyUSB0

# Open in VR headset browser: http://<your-ip>:8012
```

**VR viewpoint:** Use `--vr-pos` and `--vr-target` to control where you spawn in VR.

`--vr-pos X Y Z` sets your position (default: `0 0 0`):
- **X** — left/right relative to the table
- **Y** — height adjustment (0 = scene floor, your physical height is added automatically)
- **-Z** — moves you closer to the table, **+Z** moves you away

`--vr-target X Y Z` sets the point you face toward (default: `0 0.75 0`).

```bash
# Example: stand further back and slightly right, looking at the table
uv run python -m xdof_sim.examples.vr_streamer --task chess \
    --vr-pos 0.2 0 0.5 \
    --vr-target 0 0.75 0
```

### Interactive Viewer

Open the MuJoCo GUI to inspect the scene locally (macOS/Linux):

```bash
python -m xdof_sim.examples.view                # static scene
python -m xdof_sim.examples.view --animate       # arms move with sinusoidal motions
python -m xdof_sim.examples.view --scene eval    # different scene variant
```

### Rendering Test

Quick sanity check that the environment renders correctly (headless):

```bash
MUJOCO_GL=egl python -m xdof_sim.examples.test_env
# Saves images to /tmp/xdof_sim_test/
```

## API Reference

### `xdof_sim.make_env(scene, render_cameras, prompt, chunk_dim, **kwargs)`

Creates a configured `MuJoCoYAMEnv` with scene variant applied.

| Parameter | Default | Description |
|---|---|---|
| `scene` | `"hybrid"` | Scene variant: `"eval"`, `"training"`, `"hybrid"` |
| `render_cameras` | `True` | Render camera images in observations |
| `prompt` | `"fold the towel"` | Task prompt string |
| `chunk_dim` | `30` | Timesteps per action chunk |

### `MuJoCoYAMEnv`

| Method | Description |
|---|---|
| `reset()` | Reset to init_q pose, returns `(obs, info)` |
| `step(action)` | Execute action chunk `(chunk_dim, 14)`, returns `(obs, chunk_history, reward, terminated, truncated, info)` |
| `get_obs()` | Get current observation dict |
| `get_init_q()` | Get 14D initial joint positions |
| `_step_single(action_14d)` | Step a single 14D action (low-level) |
| `_set_qpos_from_state(state)` | Set joint positions from 14D state |
| `close()` | Clean up renderer |

### Scene Variant Functions

```python
from xdof_sim.scene_variants import (
    apply_scene_variant,
    apply_bottle_rgba,
    apply_bottle_opacity,
    apply_bin_position,
    apply_table_color,
    apply_wall_color,
)
```

## Architecture

```
xdof_sim/
├── __init__.py           # make_env() convenience API
├── env.py                # MuJoCoYAMEnv (Gymnasium interface)
├── config.py             # Robot/camera config dataclasses
├── scene_variants.py     # Runtime scene configuration
├── viewer.py             # Camera rendering utilities
├── transforms.py         # Normalize/Unnormalize for state/actions
├── models/               # MuJoCo XML + mesh files
│   ├── yam_bimanual_scene.xml
│   └── i2rt_yam/
├── policy/               # Policy inference (optional)
│   ├── base.py           # BasePolicy, PolicyConfig
│   └── lbm_policy.py     # LBM diffusion policy
├── teleop/               # ZMQ teleoperation (optional)
│   ├── node.py           # Base Node (pub/sub)
│   ├── communication.py  # ZMQ serialization
│   ├── sim_follower.py   # Sim follower node
│   ├── gello_leader.py   # GELLO hardware leader node
│   ├── dynamixel.py      # Dynamixel servo communication
│   └── leader_robot.py   # Batch servo read/write
└── examples/
    ├── test_env.py       # Quick render test
    ├── replay_demo.py    # Replay bundled demo episode
    ├── replay_data.py    # Replay episode data
    ├── run_policy.py     # Run trained policy
    ├── teleop_sim.py     # GELLO teleop launcher
    ├── viser_replay.py   # Viser 3D web replay viewer
    ├── viser_teleop.py   # Live teleop + Viser 3D viewer
    └── vr_streamer.py    # VR streaming teleop (Three.js WebXR)
```

## Notes

- **Headless rendering**: Set `MUJOCO_GL=egl` for GPU-accelerated headless rendering.
- **JPEG compression**: Use `--jpeg-quality 75` in policy rollouts to match training data distribution.
- **Gripper mapping**: Gripper values are in [0, 1] policy space, mapped to MuJoCo joint space via `_GRIPPER_CTRL_MAX = 0.037524`.
- **Policy dependencies**: The LBM policy requires the `DiTPolicy` model architecture (from `models.lbm`) to be available on `PYTHONPATH`. This is part of the training framework, not included in this package.
