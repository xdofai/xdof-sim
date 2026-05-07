"""Replay runtime construction helpers."""

from __future__ import annotations

from xdof_sim.rendering.replay.timeline import build_replay_timeline
from xdof_sim.rendering.replay.session import ReplayMode, ReplaySession


def create_replay_env(
    context,
    *,
    render_cameras: bool = False,
    camera_width: int = 640,
    camera_height: int = 480,
):
    """Create an xdof-sim environment configured for the episode."""
    import xdof_sim

    make_env_kwargs = dict(
        scene=context.scene,
        task=context.task,
        render_cameras=render_cameras,
        camera_width=camera_width,
        camera_height=camera_height,
        scene_xml_string=context.scene_xml_string,
        **dict(context.physics_overrides or {}),
    )
    using_scene_xml = context.scene_xml_string is not None
    try:
        env = xdof_sim.make_env(**make_env_kwargs)
    except ValueError as exc:
        if context.scene_xml_string is None or context.rand_state is None:
            raise
        episode_dir = getattr(getattr(context, "streams", None), "episode_dir", "<unknown>")
        print(
            f"  Warning: failed to load scene_assembled.xml for {episode_dir}: {exc}. "
            "Falling back to recorded randomization.json."
        )
        make_env_kwargs["scene_xml_string"] = None
        env = xdof_sim.make_env(**make_env_kwargs)
        using_scene_xml = False

    env.reset(randomize=False)

    if not using_scene_xml and context.rand_state is not None:
        randomizer = getattr(env, "_task_randomizer", None)
        if randomizer is not None:
            randomizer.apply(env.model, env.data, context.rand_state)

    env._replay_scene_source = (
        "scene_assembled.xml"
        if using_scene_xml
        else ("randomization.json" if context.rand_state is not None else context.scene_source)
    )

    return env


def create_replay_session(context, *, mode: ReplayMode = "auto") -> tuple[ReplaySession, float]:
    """Create the env, align replay data, and return a shared replay session."""
    env = create_replay_env(context)
    control_hz = 1.0 / (env.model.opt.timestep * env._control_decimation)
    timeline = build_replay_timeline(context, control_hz=control_hz)
    session = ReplaySession(
        env,
        timeline.actions,
        timeline.grid_ts,
        replay_ctrls=context.replay_ctrls,
        sim_states=timeline.sim_states,
        sim_integration_states=timeline.sim_integration_states,
        sim_state_spec=context.raw_sim_state_spec,
        sim_qvels=context.raw_sim_qvels,
        sim_acts=context.raw_sim_acts,
        sim_mocap_pos=context.raw_sim_mocap_pos,
        sim_mocap_quat=context.raw_sim_mocap_quat,
        initial_scene_qpos=context.initial_scene_qpos,
        initial_scene_integration_state=context.initial_scene_integration_state,
        initial_scene_qvel=context.initial_scene_qvel,
        initial_scene_act=context.initial_scene_act,
        initial_scene_mocap_pos=context.initial_scene_mocap_pos,
        initial_scene_mocap_quat=context.initial_scene_mocap_quat,
        sim_state_kind=timeline.sim_state_kind,
        sim_state_alignment=timeline.sim_state_alignment,
        rand_state=context.rand_state,
        task=context.task,
        mode=mode,
    )
    return session, control_hz
