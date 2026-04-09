"""xdof-sim: Standalone MuJoCo simulation for the YAM bimanual robot."""

from xdof_sim.env import MuJoCoYAMEnv
from xdof_sim.config import get_i2rt_sim_config, get_i2rt_config, RobotSystemConfig
from xdof_sim.randomization import RandomizationState, TASK_RANDOMIZERS

__version__ = "0.1.0"


# Per-task physics overrides — contact-heavy scenes benefit from a coarser
# timestep with proportionally reduced decimation to keep ~30 Hz control.
_TASK_PHYSICS_DEFAULTS: dict[str, dict] = {
    "pour": {"physics_dt": 0.0005, "control_decimation": 11},
    "drawer": {"physics_dt": 0.0002, "control_decimation": 11},
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
        task: Task scene — "bottles" (default), "inhand_transfer", etc.
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
    from xdof_sim.scene_variants import apply_scene_variant

    if task == "inhand_transfer":
        from xdof_sim.randomization import InHandTransferRandomizer
        from xdof_sim.randomization import _inhand_build_xml, _inhand_get_variants, _INHAND_CATEGORIES
        import numpy as _np

        config = get_i2rt_sim_config()
        seed = kwargs.pop("seed", None)
        rng = _np.random.default_rng(seed)

        # Pick initial object and generate scene XML.
        categories = _INHAND_CATEGORIES
        category = categories[int(rng.integers(0, len(categories)))]
        variants = _inhand_get_variants(category)
        variant_dir = variants[int(rng.integers(0, len(variants)))]
        from xdof_sim.randomization import _X_MIN, _X_MAX, _Y_LEFT_MIN, _Y_LEFT_MAX, _OBJ_Z
        x = float(rng.uniform(_X_MIN, _X_MAX))
        y = float(rng.uniform(_Y_LEFT_MIN, _Y_LEFT_MAX))
        yaw = float(rng.uniform(-_np.pi, _np.pi))
        xml = _inhand_build_xml(category, variant_dir, x, y, _OBJ_Z, yaw)

        physics_kwargs = {**kwargs, **_TASK_PHYSICS_DEFAULTS.get(task, {})}
        env = MuJoCoYAMEnv(
            config=config,
            render_cameras=render_cameras,
            prompt=prompt,
            chunk_dim=chunk_dim,
            scene_xml=None,  # will be overridden by _scene_xml_string below
            **physics_kwargs,
        )
        # Reload with the object-injected XML now that env is constructed.
        env.reload_from_xml(xml)
        env._task = task
        env._inhand_category = category
        env._inhand_variant = variant_dir.name

        # Bind a randomizer so XdofSimNode can call it on every reset.
        randomizer = InHandTransferRandomizer()
        randomizer.bind_env(env)
        randomizer._rng = rng
        env._task_randomizer = randomizer

        apply_scene_variant(env.model, scene)
        return env

    scene_xml = _SCENE_XMLS.get(task)
    if scene_xml is None:
        raise ValueError(
            f"Unknown task '{task}'. Available: {list(_SCENE_XMLS.keys()) + ['inhand_transfer']}"
        )

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
    apply_scene_variant(env.model, scene)
    env._task = task

    if wrist_fov != 58.0:
        import math
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
