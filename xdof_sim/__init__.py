"""xdof-sim: Standalone MuJoCo simulation for the YAM bimanual robot."""

from xdof_sim.env import MuJoCoYAMEnv
from xdof_sim.config import get_i2rt_sim_config, get_i2rt_config, RobotSystemConfig
from xdof_sim.randomization import RandomizationState, TASK_RANDOMIZERS

__version__ = "0.1.0"


# Per-task physics overrides — contact-heavy scenes benefit from a coarser
# timestep with proportionally reduced decimation to keep ~30 Hz control.
_TASK_PHYSICS_DEFAULTS: dict[str, dict] = {
}


def make_env(
    scene: str = "hybrid",
    task: str = "bottles",
    render_cameras: bool = True,
    prompt: str = "fold the towel",
    chunk_dim: int = 30,
    wrist_fov: float = 58.0,
    **kwargs,
) -> MuJoCoYAMEnv:
    """Create a MuJoCo YAM bimanual environment.

    Args:
        scene: Scene variant — "eval", "training", or "hybrid".
        task: Task scene — "bottles" (default) or "marker".
        render_cameras: Whether to render camera images in observations.
        prompt: Task prompt string included in observations.
        chunk_dim: Number of timesteps per action chunk.
        wrist_fov: Vertical field-of-view in degrees for the wrist cameras
            ("left" and "right").  Default 58 matches the real D405 mounting.
        **kwargs: Additional kwargs passed to MuJoCoYAMEnv.

    Returns:
        Configured MuJoCoYAMEnv instance with scene variant applied.
    """
    import mujoco
    from xdof_sim.env import _SCENE_XMLS

    scene_xml = _SCENE_XMLS.get(task)
    if scene_xml is None:
        raise ValueError(f"Unknown task '{task}'. Available: {list(_SCENE_XMLS.keys())}")

    config = get_i2rt_sim_config()
    physics_kwargs = {**kwargs, **_TASK_PHYSICS_DEFAULTS.get(task, {})}
    env = MuJoCoYAMEnv(
        config=config,
        render_cameras=render_cameras,
        prompt=prompt,
        chunk_dim=chunk_dim,
        scene_xml=scene_xml,
        **physics_kwargs,
    )
    from xdof_sim.scene_variants import apply_scene_variant

    apply_scene_variant(env.model, scene)
    env._task = task  # tell the env which task it is for randomization

    if wrist_fov != 58.0:
        import math
        # Wider FOV causes the frustum edges to clip into the gripper geometry.
        tan_baseline = math.tan(math.radians(58.0 / 2))
        tan_new = math.tan(math.radians(wrist_fov / 2))
        clearance_offset = max(0.0, (tan_new - tan_baseline) / tan_baseline) * 0.025
        for cam_name in ("left", "right"):
            cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
            if cam_id >= 0:
                env.model.cam_fovy[cam_id] = wrist_fov
                env.model.cam_pos[cam_id][1] -= clearance_offset

    return env


__all__ = [
    "MuJoCoYAMEnv",
    "make_env",
    "get_i2rt_sim_config",
    "get_i2rt_config",
    "RobotSystemConfig",
    "RandomizationState",
    "TASK_RANDOMIZERS",
]
