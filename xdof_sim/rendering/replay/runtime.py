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

    env = xdof_sim.make_env(
        scene=context.scene,
        task=context.task,
        render_cameras=render_cameras,
        camera_width=camera_width,
        camera_height=camera_height,
    )
    env.reset(randomize=False)

    if context.rand_state is not None:
        randomizer = getattr(env, "_task_randomizer", None)
        if randomizer is not None:
            randomizer.apply(env.model, env.data, context.rand_state)

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
        sim_states=timeline.sim_states,
        rand_state=context.rand_state,
        task=context.task,
        mode=mode,
    )
    return session, control_hz
