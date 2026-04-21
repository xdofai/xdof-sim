"""Episode loading helpers for raw and delivered market42 recordings."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from xdof_sim.rendering.replay.types import EpisodeContext, EpisodeFormat, EpisodeStreams
from xdof_sim.rendering.replay.timeline import sample_hold_align
from xdof_sim.task_specs import maybe_get_task_spec


def detect_episode_format(episode_dir: Path) -> EpisodeFormat:
    """Detect whether an episode uses the raw or delivered layout."""
    if (episode_dir / "output.mcap").exists():
        return "delivered"
    return "raw"


def _load_direct_protobuf_positions(mcap_file: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load timestamped `position` arrays from a protobuf MCAP file."""
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    positions: list[list[float]] = []
    timestamps: list[float] = []
    with open(mcap_file, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _schema, _channel, _message, decoded in reader.iter_decoded_messages():
            if not hasattr(decoded, "position") or not hasattr(decoded, "timestamp"):
                continue
            positions.append(list(decoded.position))
            timestamps.append(decoded.timestamp.seconds + decoded.timestamp.nanos / 1e9)
    if not positions:
        raise ValueError(f"No protobuf position data found in {mcap_file}")
    return np.array(positions, dtype=np.float64), np.array(timestamps, dtype=np.float64)


def _load_direct_actions(episode_dir: Path, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Load 7D actions by combining 6D action joints with observed gripper state."""
    action_pos, action_ts = _load_direct_protobuf_positions(episode_dir / f"action-{side}.mcap")
    if action_pos.shape[1] != 6:
        raise ValueError(f"Expected 6 arm joints in action-{side}.mcap, got shape {action_pos.shape}")

    grip_file = episode_dir / f"{side}.mcap"
    if grip_file.exists():
        obs_pos, obs_ts = _load_direct_protobuf_positions(grip_file)
        if obs_pos.shape[1] >= 7:
            grip = sample_hold_align(obs_pos[:, 6:7], obs_ts, action_ts)
        else:
            grip = np.zeros((len(action_pos), 1), dtype=np.float64)
    else:
        grip = np.zeros((len(action_pos), 1), dtype=np.float64)

    return np.concatenate([action_pos, grip], axis=1), action_ts


def _load_direct_recorded_cameras(episode_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load recorded MP4 frames and timestamps directly from the episode directory."""
    import imageio.v3 as iio

    camera_frames: dict[str, np.ndarray] = {}
    camera_ts: dict[str, np.ndarray] = {}
    path_map = {
        "top": ("top_camera-images-rgb.mp4", "top_camera-timestamp.npy"),
        "left": ("left_camera-images-rgb.mp4", "left_camera-timestamp.npy"),
        "right": ("right_camera-images-rgb.mp4", "right_camera-timestamp.npy"),
    }
    for cam_name, (video_name, ts_name) in path_map.items():
        video_path = episode_dir / video_name
        ts_path = episode_dir / ts_name
        if not video_path.exists() or not ts_path.exists():
            continue
        frames = iio.imread(str(video_path), plugin="pyav")
        ts_arr = np.load(ts_path)
        camera_frames[cam_name] = frames
        camera_ts[cam_name] = ts_arr
        h, w = frames.shape[1], frames.shape[2]
        print(f"  camera '{cam_name}': {len(frames)} frames ({w}x{h}), {ts_arr[-1] - ts_arr[0]:.1f}s")
    return camera_frames, camera_ts


def _load_episode_streams_direct(episode_dir: Path, *, load_recorded_cameras: bool) -> EpisodeStreams:
    """Load an episode directly from the directory without xdof_sdk."""
    print(f"Loading episode directly: {episode_dir}")

    actions_left, ts_left = _load_direct_actions(episode_dir, "left")
    hz_left = (len(ts_left) - 1) / (ts_left[-1] - ts_left[0]) if len(ts_left) > 1 else 0.0
    print(f"  action-left: {len(actions_left)} frames, {ts_left[-1] - ts_left[0]:.1f}s at ~{hz_left:.0f} Hz")

    actions_right, ts_right = _load_direct_actions(episode_dir, "right")
    hz_right = (len(ts_right) - 1) / (ts_right[-1] - ts_right[0]) if len(ts_right) > 1 else 0.0
    print(f"  action-right: {len(actions_right)} frames, {ts_right[-1] - ts_right[0]:.1f}s at ~{hz_right:.0f} Hz")

    camera_frames: dict[str, np.ndarray] = {}
    camera_ts: dict[str, np.ndarray] = {}
    if load_recorded_cameras:
        camera_frames, camera_ts = _load_direct_recorded_cameras(episode_dir)

    return EpisodeStreams(
        episode_dir=episode_dir,
        actions_left=actions_left,
        ts_left=ts_left,
        actions_right=actions_right,
        ts_right=ts_right,
        camera_frames=camera_frames,
        camera_ts=camera_ts,
    )


def _load_delivered_episode_streams(
    episode_dir: Path,
    *,
    load_recorded_cameras: bool,
) -> tuple[EpisodeStreams, str]:
    """Load a delivered BAIR-format episode from output.mcap + sim_state.mcap."""
    import io

    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    print(f"Loading delivered episode: {episode_dir}")

    left_leader_pos: list[list[float]] = []
    left_leader_ts: list[float] = []
    right_leader_pos: list[list[float]] = []
    right_leader_ts: list[float] = []
    left_proprio_pos: list[list[float]] = []
    left_proprio_ts: list[float] = []
    right_proprio_pos: list[list[float]] = []
    right_proprio_ts: list[float] = []
    cam_data: dict[str, list[bytes]] = {}
    cam_ts_raw: dict[str, list[float]] = {}
    instruction = ""

    cam_topic_to_name = {
        "/top-camera/image-raw": "top",
        "/left-wrist-camera/image-raw": "left",
        "/right-wrist-camera/image-raw": "right",
    }

    with open(episode_dir / "output.mcap", "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _schema, channel, _message, decoded in reader.iter_decoded_messages():
            topic = channel.topic
            if topic == "/instruction":
                instruction = decoded.data
                continue

            ts = getattr(decoded, "timestamp", None)
            if ts is None:
                continue
            timestamp = ts.seconds + ts.nanos / 1e9

            if topic == "/left-arm-leader":
                left_leader_ts.append(timestamp)
                left_leader_pos.append(list(decoded.position))
            elif topic == "/right-arm-leader":
                right_leader_ts.append(timestamp)
                right_leader_pos.append(list(decoded.position))
            elif topic == "/left-arm-proprio":
                left_proprio_ts.append(timestamp)
                left_proprio_pos.append(list(decoded.position))
            elif topic == "/right-arm-proprio":
                right_proprio_ts.append(timestamp)
                right_proprio_pos.append(list(decoded.position))
            elif load_recorded_cameras and topic in cam_topic_to_name:
                cam_name = cam_topic_to_name[topic]
                cam_data.setdefault(cam_name, []).append(bytes(decoded.data))
                cam_ts_raw.setdefault(cam_name, []).append(timestamp)

    if not left_leader_pos or not right_leader_pos:
        raise ValueError(f"Delivered episode is missing leader action streams in {episode_dir}")
    if not left_proprio_pos or not right_proprio_pos:
        raise ValueError(f"Delivered episode is missing proprio streams in {episode_dir}")

    left_leader = np.array(left_leader_pos, dtype=np.float64)
    right_leader = np.array(right_leader_pos, dtype=np.float64)
    left_proprio = np.array(left_proprio_pos, dtype=np.float64)
    right_proprio = np.array(right_proprio_pos, dtype=np.float64)
    ts_left_leader = np.array(left_leader_ts, dtype=np.float64)
    ts_right_leader = np.array(right_leader_ts, dtype=np.float64)
    ts_left_proprio = np.array(left_proprio_ts, dtype=np.float64)
    ts_right_proprio = np.array(right_proprio_ts, dtype=np.float64)

    left_gripper = sample_hold_align(left_proprio[:, 6:7], ts_left_proprio, ts_left_leader)
    right_gripper = sample_hold_align(right_proprio[:, 6:7], ts_right_proprio, ts_right_leader)
    actions_left = np.concatenate([left_leader, left_gripper], axis=1)
    actions_right = np.concatenate([right_leader, right_gripper], axis=1)

    duration_left = ts_left_leader[-1] - ts_left_leader[0]
    duration_right = ts_right_leader[-1] - ts_right_leader[0]
    hz_left = (len(ts_left_leader) - 1) / duration_left if duration_left > 0 else 0.0
    hz_right = (len(ts_right_leader) - 1) / duration_right if duration_right > 0 else 0.0
    print(f"  Instruction: {instruction!r}")
    print(f"  action-left: {len(actions_left)} frames, {duration_left:.1f}s at ~{hz_left:.0f} Hz")
    print(f"  action-right: {len(actions_right)} frames, {duration_right:.1f}s at ~{hz_right:.0f} Hz")

    camera_frames: dict[str, np.ndarray] = {}
    camera_ts: dict[str, np.ndarray] = {}
    if load_recorded_cameras:
        import av

        for cam_name, packets in cam_data.items():
            try:
                container = av.open(io.BytesIO(b"".join(packets)), format="h264")
                frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
                ts_arr = np.array(cam_ts_raw[cam_name], dtype=np.float64)
                n = min(len(frames), len(ts_arr))
                if n == 0:
                    continue
                camera_frames[cam_name] = np.array(frames[:n])
                camera_ts[cam_name] = ts_arr[:n]
                h, w = camera_frames[cam_name][0].shape[:2]
                print(f"  camera '{cam_name}': {n} frames ({w}x{h}), {ts_arr[n - 1] - ts_arr[0]:.1f}s")
            except Exception as exc:
                print(f"  Warning: could not decode camera '{cam_name}': {exc}")

    streams = EpisodeStreams(
        episode_dir=episode_dir,
        actions_left=actions_left,
        ts_left=ts_left_leader,
        actions_right=actions_right,
        ts_right=ts_right_leader,
        camera_frames=camera_frames,
        camera_ts=camera_ts,
    )
    return streams, instruction


def load_episode_streams(episode_dir: Path, *, load_recorded_cameras: bool = True) -> EpisodeStreams:
    """Load action streams and optionally recorded camera frames via xdof_sdk."""
    if detect_episode_format(episode_dir) == "delivered":
        streams, _instruction = _load_delivered_episode_streams(
            episode_dir,
            load_recorded_cameras=load_recorded_cameras,
        )
        return streams

    try:
        import json as _json
        from collections import defaultdict

        from mcap.decoder import DecoderFactory as _McapDecoderFactory
        from mcap.reader import make_reader as _make_reader
        from mcap_protobuf.decoder import DecoderFactory as _ProtobufDecoderFactory
        from mcap_ros2.decoder import DecoderFactory as _Ros2DecoderFactory
        from xdof_sdk.data.schema.keys import DataKeys, MCAP_TOPIC_FIELD_TO_KEY
        from xdof_sdk.data.trajectory import ArmSide, ArmTrajectory, Trajectory, load_trajectory
        import numpy as _np
    except ImportError:
        return _load_episode_streams_direct(episode_dir, load_recorded_cameras=load_recorded_cameras)

    class _JsonDecoderFactory(_McapDecoderFactory):
        def decoder_for(self, message_encoding, schema):
            if message_encoding == "json":
                return lambda data: _json.loads(data)
            return None

    def _patched_load_mcap_file(self, file):
        data = defaultdict(list)
        data_timestamps = defaultdict(list)
        with open(file, "rb") as f:
            reader = _make_reader(
                f,
                decoder_factories=[
                    _ProtobufDecoderFactory(),
                    _Ros2DecoderFactory(),
                    _JsonDecoderFactory(),
                ],
            )
            for _schema, channel, _message, proto_msg in reader.iter_decoded_messages():
                topic = channel.topic
                if topic not in MCAP_TOPIC_FIELD_TO_KEY:
                    continue
                field_mappings = MCAP_TOPIC_FIELD_TO_KEY[topic]
                for field_name, data_key in field_mappings.items():
                    if not hasattr(proto_msg, field_name):
                        continue
                    data[data_key].append(getattr(proto_msg, field_name))
                    data_timestamps[data_key].append(
                        proto_msg.timestamp.seconds + proto_msg.timestamp.nanos / 1e9
                    )
        for data_key, timestamps in data_timestamps.items():
            if len(timestamps) > 1 and _np.any(_np.diff(timestamps) < 0):
                raise ValueError(f"Timestamp is not monotonically increasing for topic {data_key}")
        return data, data_timestamps

    original_load_mcap_file = Trajectory._load_mcap_file
    original_get_joint_pos_obs = ArmTrajectory.get_joint_pos_obs

    def _patched_get_joint_pos_obs(self, arm_side):
        key = f"{arm_side.value}-joint_pos"
        if key not in self._trajectory_data:
            return self._trajectory_data[f"action-{arm_side.value}-pos"][:, : self._n_dof_arm]
        return original_get_joint_pos_obs(self, arm_side)

    print(f"Loading episode: {episode_dir}")
    Trajectory._load_mcap_file = _patched_load_mcap_file
    ArmTrajectory.get_joint_pos_obs = _patched_get_joint_pos_obs
    try:
        traj = load_trajectory(episode_dir)
    finally:
        Trajectory._load_mcap_file = original_load_mcap_file
        ArmTrajectory.get_joint_pos_obs = original_get_joint_pos_obs

    actions_left = None
    ts_left = None
    actions_right = None
    ts_right = None
    for side_label, side in [("left", ArmSide.LEFT), ("right", ArmSide.RIGHT)]:
        joints = traj.get_joint_pos_action(side)
        gripper = traj.get_gripper_pos_action(side)
        pos = np.concatenate([joints, gripper], axis=1)
        ts = traj.get_data_key_timestamp(
            DataKeys.ACTION.JOINT.POS.LEFT if side == ArmSide.LEFT else DataKeys.ACTION.JOINT.POS.RIGHT
        )
        hz = (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 else 0.0
        print(f"  action-{side_label}: {len(pos)} frames, {ts[-1] - ts[0]:.1f}s at ~{hz:.0f} Hz")
        if side == ArmSide.LEFT:
            actions_left, ts_left = pos, ts
        else:
            actions_right, ts_right = pos, ts

    if actions_left is None or actions_right is None or ts_left is None or ts_right is None:
        raise ValueError("Episode is missing one or both action streams")

    camera_frames: dict[str, np.ndarray] = {}
    camera_ts: dict[str, np.ndarray] = {}
    if load_recorded_cameras:
        from xdof_sdk.data.trajectory import CameraPerspective

        for perspective, ts_key in [
            (CameraPerspective.TOP, DataKeys.CAMERA.TIMESTAMP.TOP),
            (CameraPerspective.LEFT, DataKeys.CAMERA.TIMESTAMP.LEFT),
            (CameraPerspective.RIGHT, DataKeys.CAMERA.TIMESTAMP.RIGHT),
        ]:
            cam_name = perspective.value
            try:
                mp4_file = traj.get_video_path(perspective)
                ts_arr = traj.get_data_key_timestamp(ts_key)
                import imageio.v3 as iio

                frames = iio.imread(str(mp4_file), plugin="pyav")
                camera_frames[cam_name] = frames
                camera_ts[cam_name] = ts_arr
                h, w = frames.shape[1], frames.shape[2]
                print(f"  camera '{cam_name}': {len(frames)} frames ({w}x{h}), {ts_arr[-1] - ts_arr[0]:.1f}s")
            except FileNotFoundError:
                continue
            except Exception as exc:
                print(f"  Warning: could not load camera '{cam_name}': {exc}")

    return EpisodeStreams(
        episode_dir=episode_dir,
        actions_left=actions_left,
        ts_left=ts_left,
        actions_right=actions_right,
        ts_right=ts_right,
        camera_frames=camera_frames,
        camera_ts=camera_ts,
    )


def read_sim_config(episode_dir: Path) -> dict:
    """Read scene/task from the robot node config in session_meta.json."""
    import json

    meta_file = episode_dir / "session_meta.json"
    if not meta_file.exists():
        return {}
    with open(meta_file) as f:
        meta = json.load(f)
    for node in meta.get("nodes", []):
        if node.get("role") == "robot" and "config" in node:
            cfg = node["config"]
            if "task" in cfg or "scene" in cfg:
                return cfg
    return {}


def load_sim_states(episode_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Load the raw /sim_state/qpos timeline from sim_state.mcap."""
    import json

    from mcap.reader import make_reader

    mcap_file = episode_dir / "sim_state.mcap"
    if not mcap_file.exists():
        return None

    qposes: list[list[float]] = []
    timestamps: list[float] = []
    with open(mcap_file, "rb") as f:
        reader = make_reader(f)
        for _schema, channel, message in reader.iter_messages(topics=["/sim_state/qpos"]):
            payload = json.loads(message.data)
            qposes.append(payload["qpos"])
            timestamps.append(payload["timestamp"])

    if not qposes:
        return None
    return np.array(qposes, dtype=np.float64), np.array(timestamps, dtype=np.float64)


def load_randomization(episode_dir: Path):
    """Load randomization state from randomization.json if present."""
    import json

    from xdof_sim.randomization import RandomizationState

    rand_file = episode_dir / "randomization.json"
    if not rand_file.exists():
        return None
    with open(rand_file) as f:
        return RandomizationState.from_dict(json.load(f))


def resolve_delivered_task(instruction: str, episode_dir: Path) -> tuple[str, str]:
    """Resolve xdof-sim scene/task from a delivered episode instruction."""
    spec = maybe_get_task_spec(instruction) or maybe_get_task_spec(episode_dir.parent.name)
    if spec is None:
        key = instruction.strip().replace(" ", "_")
        print(
            f"  Warning: unknown delivered task '{key or episode_dir.parent.name}', "
            "falling back to 'empty'"
        )
        return "hybrid", "empty"
    return spec.scene, spec.env_task


def load_episode_context(episode_dir: Path, *, load_recorded_cameras: bool = True) -> EpisodeContext:
    """Load a full replayable episode context."""
    episode_format = detect_episode_format(episode_dir)
    instruction: str | None = None

    if episode_format == "delivered":
        streams, instruction = _load_delivered_episode_streams(
            episode_dir,
            load_recorded_cameras=load_recorded_cameras,
        )
        scene, task = resolve_delivered_task(instruction, episode_dir)
    else:
        streams = load_episode_streams(episode_dir, load_recorded_cameras=load_recorded_cameras)
        sim_cfg = read_sim_config(episode_dir)
        scene = sim_cfg.get("scene", "hybrid")
        task = sim_cfg.get("task", "empty")

    raw = load_sim_states(episode_dir)
    raw_states, raw_timestamps = (raw if raw is not None else (None, None))
    return EpisodeContext(
        streams=streams,
        episode_format=episode_format,
        scene=scene,
        task=task,
        instruction=instruction,
        rand_state=load_randomization(episode_dir),
        raw_sim_states=raw_states,
        raw_sim_timestamps=raw_timestamps,
    )
