"""Object pose randomization for xdof-sim scenes.

Provides per-task ``SceneRandomizer`` subclasses that perturb free-jointed
objects around their nominal XML positions.  Collision avoidance uses
two-stage rejection sampling:

1. Fast pairwise XY clearance check (no mj_forward needed).
2. Full MuJoCo contact check after applying the proposed placement.

``RandomizationState`` stores *absolute* positions and quaternions (not
deltas), so replay is a direct blind write into qpos — no dependency on
nominal positions or the sampler implementation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PerturbRange:
    """Translation and yaw perturbation range for one object.

    For free-jointed objects set ``joint_name`` to the joint name and leave
    ``fixed_body=False`` (default).  For fixed bodies that should be moved by
    writing directly to ``model.body_pos`` / ``model.body_quat``, set
    ``fixed_body=True`` and use the body name in ``joint_name``.

    Deltas are relative to the object's nominal position at the time
    ``randomize()`` is called (i.e. after ``mj_resetData``), so they remain
    valid even if the XML default positions change.
    """

    joint_name: str
    delta_x: tuple[float, float]       # (min, max) metres
    delta_y: tuple[float, float]       # (min, max) metres
    delta_z: tuple[float, float] = field(default=(0.0, 0.0))   # stay on table
    delta_yaw: tuple[float, float] = field(default=(-np.pi, np.pi))
    fixed_body: bool = False  # if True, treat joint_name as a body name


@dataclass
class RandomizationState:
    """Serialisable randomization outcome.

    Stores absolute positions and quaternions — fully self-contained for
    replay without re-running the sampler.  The ``seed`` field is retained
    for audit / debugging only; replay always uses ``object_states`` directly.
    """

    seed: int
    object_states: dict[str, dict[str, list[float]]]
    # e.g. {"bottle_1_joint": {"pos": [x,y,z], "quat": [w,x,y,z]}, ...}

    def to_dict(self) -> dict[str, Any]:
        return {"seed": self.seed, "object_states": self.object_states}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RandomizationState":
        return cls(seed=d["seed"], object_states=d["object_states"])


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------


def _quat_from_yaw(yaw: float) -> np.ndarray:
    """wxyz quaternion for a rotation of ``yaw`` radians about world Z."""
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two wxyz quaternions: result = q1 * q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


# ---------------------------------------------------------------------------
# Base randomizer
# ---------------------------------------------------------------------------


class SceneRandomizer:
    """Base class for scene object randomization with two-stage rejection sampling.

    Subclasses only need to define ``perturbations`` (and optionally override
    ``min_clearance_m``, ``max_tries``, or ``table_bounds``).  The base class
    handles reading nominal positions, sampling, collision checking, and state
    serialisation.

    ``table_bounds`` is checked on every sample as Stage 0 (before the cheap
    pairwise distance check).  It is expressed as absolute world XY coordinates:
    ``(x_min, x_max, y_min, y_max)``.  Objects whose absolute position falls
    outside these bounds are immediately rejected, so delta ranges can be set
    generously without risk of objects falling off the edge.

    The default matches the standard sim table:
      centre (0.6, 0), half-extents (0.2975, 0.65) minus a 0.06 m edge margin.
    """

    perturbations: list[PerturbRange] = []
    max_tries: int = 200
    min_clearance_m: float = 0.03  # minimum XY centre-to-centre distance
    # Absolute XY workspace limits — overrideable per task if the table differs.
    table_bounds: tuple[float, float, float, float] = (0.36, 0.82, -0.55, 0.55)

    def __init__(self) -> None:
        # Cache fixed-body nominal positions on first read so that subsequent
        # randomizations don't drift (mj_resetData restores data.qpos but not
        # model.body_pos, so we must remember the original XML values ourselves).
        self._fixed_body_nominals: dict[str, tuple[np.ndarray, np.ndarray]] | None = None

    def randomize(
        self,
        model: Any,
        data: Any,
        seed: int | None = None,
    ) -> RandomizationState:
        """Sample a collision-free placement, apply it to data, and return state."""
        if not self.perturbations:
            return RandomizationState(seed=seed or 0, object_states={})

        rng = np.random.default_rng(seed)

        # Nominal positions come from the current qpos (caller should have
        # called mj_resetData + mj_forward before invoking randomize).
        nominals = self._read_nominals(model, data)

        last_states: dict[str, dict] = {}
        for attempt in range(self.max_tries):
            states = self._sample_once(nominals, rng)
            last_states = states

            # Stage 0: table bounds — free, no geometry queries needed.
            if not self._bounds_ok(states):
                continue

            # Stage 1: fast pairwise XY distance (no mj_forward cost).
            if not self._pairwise_ok(states):
                continue

            # Stage 2: full MuJoCo contact check.
            self._apply_states(model, data, states)
            mujoco.mj_forward(model, data)
            if not self._contacts_ok(model, data):
                continue

            return RandomizationState(seed=seed or 0, object_states=states)

        logger.warning(
            "%s: no collision-free placement found after %d tries — using last sample",
            type(self).__name__,
            self.max_tries,
        )
        self._apply_states(model, data, last_states)
        mujoco.mj_forward(model, data)
        return RandomizationState(seed=seed or 0, object_states=last_states)

    def apply(self, model: Any, data: Any, state: RandomizationState) -> None:
        """Restore a previously saved state without sampling (for replay)."""
        self._apply_states(model, data, state.object_states)
        mujoco.mj_forward(model, data)

    # -- internals -----------------------------------------------------------

    def _read_nominals(
        self, model: Any, data: Any
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Read current pos and quat for each perturbation.

        Free-jointed objects read from ``data.qpos``; fixed bodies read from
        a cache populated on the first call (the XML default values).  We must
        cache because ``_apply_states`` writes to ``model.body_pos`` and
        ``mj_resetData`` does not restore it, which would cause the nominal to
        drift on every reset if we re-read from the model each time.
        """
        # Populate fixed-body cache on first call (before any writes).
        if self._fixed_body_nominals is None:
            self._fixed_body_nominals = {}
            for p in self.perturbations:
                if p.fixed_body:
                    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, p.joint_name)
                    if body_id >= 0:
                        self._fixed_body_nominals[p.joint_name] = (
                            model.body_pos[body_id].copy(),
                            model.body_quat[body_id].copy(),
                        )

        nominals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for p in self.perturbations:
            if p.fixed_body:
                if p.joint_name not in self._fixed_body_nominals:
                    logger.warning("Body '%s' not found in model — skipping", p.joint_name)
                    continue
                pos, quat = self._fixed_body_nominals[p.joint_name]
                nominals[p.joint_name] = (pos.copy(), quat.copy())
            else:
                jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, p.joint_name)
                if jnt_id < 0:
                    logger.warning("Joint '%s' not found in model — skipping", p.joint_name)
                    continue
                adr = int(model.jnt_qposadr[jnt_id])
                nominals[p.joint_name] = (
                    data.qpos[adr: adr + 3].copy(),
                    data.qpos[adr + 3: adr + 7].copy(),
                )
        return nominals

    def _sample_once(
        self,
        nominals: dict[str, tuple[np.ndarray, np.ndarray]],
        rng: np.random.Generator,
    ) -> dict[str, dict[str, list[float]]]:
        """Draw one set of absolute object placements from nominal + delta.

        The delta ranges are intersected with ``table_bounds`` before sampling
        so every draw is guaranteed to land on the table.  This avoids the
        exponential failure rate of rejection sampling when large delta ranges
        are combined with a bounded workspace.
        """
        x_min, x_max, y_min, y_max = self.table_bounds
        states: dict[str, dict[str, list[float]]] = {}
        for p in self.perturbations:
            if p.joint_name not in nominals:
                continue
            nom_pos, nom_quat = nominals[p.joint_name]

            # Clamp the effective delta range so the absolute position stays
            # within table_bounds regardless of how large the delta is set.
            eff_dx = (
                max(p.delta_x[0], x_min - nom_pos[0]),
                min(p.delta_x[1], x_max - nom_pos[0]),
            )
            eff_dy = (
                max(p.delta_y[0], y_min - nom_pos[1]),
                min(p.delta_y[1], y_max - nom_pos[1]),
            )
            # If nominal is outside bounds (shouldn't happen), sample at 0.
            if eff_dx[0] > eff_dx[1]:
                eff_dx = (0.0, 0.0)
            if eff_dy[0] > eff_dy[1]:
                eff_dy = (0.0, 0.0)

            new_pos = nom_pos + np.array([
                rng.uniform(*eff_dx),
                rng.uniform(*eff_dy),
                rng.uniform(*p.delta_z),
            ])
            # Yaw perturbation applied on top of the nominal orientation.
            q_yaw = _quat_from_yaw(rng.uniform(*p.delta_yaw))
            new_quat = _quat_mul(q_yaw, nom_quat)

            states[p.joint_name] = {
                "pos": new_pos.tolist(),
                "quat": new_quat.tolist(),
            }
        return states

    def _apply_states(
        self,
        model: Any,
        data: Any,
        states: dict[str, dict[str, list[float]]],
    ) -> None:
        """Write absolute pos/quat for each object.

        Tries joint lookup first (free-jointed objects → ``data.qpos``).
        Falls back to body lookup (fixed bodies → ``model.body_pos/quat``).
        """
        for name, s in states.items():
            jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jnt_id >= 0:
                adr = int(model.jnt_qposadr[jnt_id])
                data.qpos[adr: adr + 3] = s["pos"]
                data.qpos[adr + 3: adr + 7] = s["quat"]
            else:
                # Fixed body — write directly into model geometry.
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                if body_id >= 0:
                    model.body_pos[body_id] = s["pos"]
                    model.body_quat[body_id] = s["quat"]

    def _bounds_ok(self, states: dict[str, dict[str, list[float]]]) -> bool:
        """Return False if any object's absolute XY position is outside table_bounds."""
        x_min, x_max, y_min, y_max = self.table_bounds
        for s in states.values():
            x, y = s["pos"][0], s["pos"][1]
            if not (x_min <= x <= x_max and y_min <= y <= y_max):
                return False
        return True

    def _pairwise_ok(self, states: dict[str, dict[str, list[float]]]) -> bool:
        """Return False if any two objects are closer than min_clearance_m in XY."""
        if self.min_clearance_m <= 0.0 or len(states) < 2:
            return True
        positions = [np.array(s["pos"][:2]) for s in states.values()]
        n = len(positions)
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(positions[i] - positions[j]) < self.min_clearance_m:
                    return False
        return True

    def _contacts_ok(self, model: Any, data: Any) -> bool:
        """Return False if any two randomized object bodies are in contact."""
        obj_body_ids: set[int] = set()
        for p in self.perturbations:
            if p.fixed_body:
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, p.joint_name)
                if body_id >= 0:
                    obj_body_ids.add(body_id)
            else:
                jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, p.joint_name)
                if jnt_id >= 0:
                    obj_body_ids.add(int(model.jnt_bodyid[jnt_id]))

        for c in range(data.ncon):
            contact = data.contact[c]
            b1 = int(model.geom_bodyid[contact.geom1])
            b2 = int(model.geom_bodyid[contact.geom2])
            if b1 in obj_body_ids and b2 in obj_body_ids:
                return False
        return True


# ---------------------------------------------------------------------------
# Color randomization helpers
# ---------------------------------------------------------------------------

# Preset mug color palette (RGBA, alpha=1.0).  Includes the original green so
# the default appearance is part of the distribution.
_MUG_COLOR_PALETTE: list[tuple[float, float, float, float]] = [
    (0.172, 0.780, 0.435, 1.0),  # original green
    (0.850, 0.325, 0.098, 1.0),  # red-orange
    (0.929, 0.694, 0.125, 1.0),  # amber
    (0.494, 0.184, 0.557, 1.0),  # purple
    (0.301, 0.745, 0.933, 1.0),  # sky blue
    (0.635, 0.078, 0.184, 1.0),  # dark red
    (0.047, 0.482, 0.863, 1.0),  # blue
    (0.960, 0.960, 0.960, 1.0),  # near-white
    (0.173, 0.173, 0.173, 1.0),  # near-black
]

# Probability of applying a random color (vs. keeping the XML default) each episode.
_COLOR_RANDOMIZE_PROB = 1.0


def _apply_mat_color(
    model: Any, mat_name: str, rgba: tuple[float, float, float, float]
) -> None:
    """Set a MuJoCo material's RGBA at runtime."""
    mat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, mat_name)
    if mat_id >= 0:
        model.mat_rgba[mat_id] = rgba
    else:
        logger.warning("Material '%s' not found in model — skipping color randomization", mat_name)


# ---------------------------------------------------------------------------
# Per-task randomizers
# ---------------------------------------------------------------------------


class BottlesRandomizer(SceneRandomizer):
    min_clearance_m = 0.08
    perturbations = [
        # Bottles placed near edges so range covers a large area of the table.
        # table_bounds check ensures nothing falls off regardless of delta size.
        PerturbRange("bottle_1_joint", delta_x=(-1.0, 1.0), delta_y=(-1.0, 1.0)),
        PerturbRange("bottle_2_joint", delta_x=(-1.0, 1.0), delta_y=(-1.0, 1.0)),
        PerturbRange("bottle_3_joint", delta_x=(-1.0, 1.0), delta_y=(-1.0, 1.0)),
        PerturbRange("bottle_4_joint", delta_x=(-1.0, 1.0), delta_y=(-1.0, 1.0)),
        # Bin: moderate range to keep it reachable; small yaw (asymmetric shape).
        PerturbRange("bin_joint", delta_x=(-0.05, 0.03), delta_y=(-0.2, 0.2),
                     delta_yaw=(-0.5, 0.5)),
    ]


class MarkerRandomizer(SceneRandomizer):
    min_clearance_m = 0.03
    perturbations = [
        PerturbRange("marker_joint", delta_x=(-0.10, 0.10), delta_y=(-0.14, 0.14)),
        PerturbRange("cap_joint",    delta_x=(-0.08, 0.08), delta_y=(-0.08, 0.08)),
    ]


_PLATE_TINT_PALETTE: list[tuple[float, float, float, float]] = [
    (1.000, 1.000, 1.000, 1.0),  # white (original)
    (1.000, 0.960, 0.860, 1.0),  # warm cream
    (0.950, 0.930, 0.900, 1.0),  # off-white / linen
    (0.850, 0.900, 1.000, 1.0),  # powder blue
    (0.880, 0.950, 0.880, 1.0),  # sage green
    (1.000, 0.880, 0.870, 1.0),  # blush pink
    (0.900, 0.870, 0.950, 1.0),  # lavender
    (0.870, 0.920, 0.940, 1.0),  # slate grey-blue
]


class DishRackRandomizer(SceneRandomizer):
    min_clearance_m = 0.1
    perturbations = [
        PerturbRange("dishrack",    delta_x=(-0.10, 0.09), delta_y=(-0.15, 0.15), delta_yaw=(-0.25, 0.25), fixed_body=True),
        PerturbRange("plate_joint", delta_x=(-0.35, 0.25), delta_y=(-0.35, 0.35)),
    ]

    def randomize(self, model: Any, data: Any, seed: int | None = None) -> RandomizationState:
        state = super().randomize(model, data, seed)
        rng = np.random.default_rng(seed)
        _apply_mat_color(model, "plate_mat", _PLATE_TINT_PALETTE[rng.integers(len(_PLATE_TINT_PALETTE))])
        return state

    def _sample_once(
        self,
        nominals: dict[str, tuple[np.ndarray, np.ndarray]],
        rng: np.random.Generator,
    ) -> dict[str, dict[str, list[float]]]:
        states = super()._sample_once(nominals, rng)
        # 50% chance: reflect the scene about the XZ plane (negate all Y coords)
        # and rotate the rack 180 about Z so it faces the opposite direction.
        if rng.integers(2) == 1:
            q_180 = _quat_from_yaw(np.pi)
            for key, s in states.items():
                s["pos"][1] = -s["pos"][1]
                if key == "dishrack":
                    s["quat"] = _quat_mul(q_180, np.array(s["quat"])).tolist()
        return states


class BlocksRandomizer(SceneRandomizer):
    """Per-block perturbation for the 26-letter blocks scene.

    Blocks are arranged in a 4-row grid with 55 mm spacing; perturbations
    scatter blocks across the top half of the table (±80 mm x/y, ±0.25 rad yaw).
    """
    min_clearance_m = 0.01
    perturbations = [
        PerturbRange(f"block_{letter}_jnt",
                     delta_x=(-0.08, 0.08),
                     delta_y=(-0.45, 0.45),
                     delta_yaw=(-0.25, 0.25))
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ]


class MugTreeRandomizer(SceneRandomizer):
    min_clearance_m = 0.12
    perturbations = [
        PerturbRange("mug_tree", delta_x=(-0.10, 0.10), delta_y=(-0.20, 0.20),
                     delta_yaw=(-0.25, 0.25), fixed_body=True),
        PerturbRange("mug_1_jnt", delta_x=(-0.10, 0.10), delta_y=(-0.3, 0.3)),
        PerturbRange("mug_2_jnt", delta_x=(-0.10, 0.10), delta_y=(-0.3, 0.3)),
    ]

    def randomize(self, model: Any, data: Any, seed: int | None = None) -> RandomizationState:
        state = super().randomize(model, data, seed)
        rng = np.random.default_rng(seed)
        if rng.random() < _COLOR_RANDOMIZE_PROB:
            for mat_name in ("mug_1_color", "mug_2_color"):
                _apply_mat_color(model, mat_name, _MUG_COLOR_PALETTE[rng.integers(len(_MUG_COLOR_PALETTE))])
        return state


class MugFlipRandomizer(SceneRandomizer):
    # Mugs start upside-down (quat=[0,1,0,0]); yaw rotation is still world-Z.
    # Tray is a fixed body; mugs are sampled relative to the tray's new position
    # so they stay on the tray after randomization.
    min_clearance_m = 0.03
    perturbations = [
        PerturbRange("tray", delta_x=(-0.10, 0.05), delta_y=(-0.3, 0.3),
                     delta_yaw=(-0.25, 0.25), fixed_body=True),
        # Mug deltas here are small offsets *relative to the tray* (not absolute).
        # The actual sampling is handled by the overridden _sample_once below.
        PerturbRange("mug_1_jnt", delta_x=(-0.035, 0.000), delta_y=(-0.005, 0.025)),
        PerturbRange("mug_2_jnt", delta_x=(-0.000, 0.035), delta_y=(-0.025, 0.005)),
    ]

    def _sample_once(
        self,
        nominals: dict[str, tuple[np.ndarray, np.ndarray]],
        rng: np.random.Generator,
    ) -> dict[str, dict[str, list[float]]]:
        x_min, x_max, y_min, y_max = self.table_bounds
        states: dict[str, dict[str, list[float]]] = {}

        # 1. Sample the tray (fixed body) using the standard clamped-delta logic.
        tray_p = next(p for p in self.perturbations if p.joint_name == "tray")
        tray_nom_pos, tray_nom_quat = nominals["tray"]
        eff_dx = (max(tray_p.delta_x[0], x_min - tray_nom_pos[0]),
                  min(tray_p.delta_x[1], x_max - tray_nom_pos[0]))
        eff_dy = (max(tray_p.delta_y[0], y_min - tray_nom_pos[1]),
                  min(tray_p.delta_y[1], y_max - tray_nom_pos[1]))
        tray_new_pos = tray_nom_pos + np.array([
            rng.uniform(*eff_dx),
            rng.uniform(*eff_dy),
            rng.uniform(*tray_p.delta_z),
        ])
        q_yaw = _quat_from_yaw(rng.uniform(*tray_p.delta_yaw))
        tray_new_quat = _quat_mul(q_yaw, tray_nom_quat)
        states["tray"] = {"pos": tray_new_pos.tolist(), "quat": tray_new_quat.tolist()}

        # 2. Sample each mug relative to the tray's new position.
        # Base position = tray_new_pos + the mug's original offset from the tray nominal.
        # A small additional per-mug delta adds variety within the tray footprint.
        tray_nom_xy = tray_nom_pos[:2]
        for p in self.perturbations:
            if p.joint_name == "tray":
                continue
            mug_nom_pos, mug_nom_quat = nominals[p.joint_name]
            offset = mug_nom_pos - tray_nom_pos   # fixed offset in tray-local frame
            base_pos = tray_new_pos + offset
            mug_new_pos = base_pos + np.array([
                rng.uniform(*p.delta_x),
                rng.uniform(*p.delta_y),
                rng.uniform(*p.delta_z),
            ])
            q_yaw_mug = _quat_from_yaw(rng.uniform(*p.delta_yaw))
            states[p.joint_name] = {
                "pos": mug_new_pos.tolist(),
                "quat": _quat_mul(q_yaw_mug, mug_nom_quat).tolist(),
            }

        return states

    def randomize(self, model: Any, data: Any, seed: int | None = None) -> RandomizationState:
        state = super().randomize(model, data, seed)
        rng = np.random.default_rng(seed)
        if rng.random() < _COLOR_RANDOMIZE_PROB:
            for mat_name in ("mug_1_color", "mug_2_color"):
                _apply_mat_color(model, mat_name, _MUG_COLOR_PALETTE[rng.integers(len(_MUG_COLOR_PALETTE))])
        return state


class PourRandomizer(SceneRandomizer):
    """Randomize mug, cup, and beads-inside-mug for the pour/screw scene.

    Beads translate with the mug (same dx/dy) so they stay inside it after reset.
    Mug and cup are placed independently; contacts checked between them only.
    """

    min_clearance_m = 0.08
    # Only mug and cup in perturbations — used for pairwise/contact checks.
    # Beads are injected by _sample_once below.
    perturbations = [
        PerturbRange("mug_1_jnt",  delta_x=(-0.12, 0.12), delta_y=(-0.15, 0.15)),
        PerturbRange("cup_1_jnt", delta_x=(-0.10, 0.10), delta_y=(-0.20, 0.20)),
    ]
    _bead_joints = [f"bead_{i}_jnt" for i in range(1, 11)]

    def _read_nominals(self, model: Any, data: Any) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        nominals = super()._read_nominals(model, data)
        for bead_name in self._bead_joints:
            jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, bead_name)
            if jnt_id >= 0:
                adr = int(model.jnt_qposadr[jnt_id])
                nominals[bead_name] = (
                    data.qpos[adr: adr + 3].copy(),
                    data.qpos[adr + 3: adr + 7].copy(),
                )
        return nominals

    def _sample_once(
        self,
        nominals: dict[str, tuple[np.ndarray, np.ndarray]],
        rng: np.random.Generator,
    ) -> dict[str, dict[str, list[float]]]:
        x_min, x_max, y_min, y_max = self.table_bounds
        states: dict[str, dict[str, list[float]]] = {}

        # Sample mug with standard clamped-delta logic.
        mug_p = next(p for p in self.perturbations if p.joint_name == "mug_1_jnt")
        mug_nom_pos, mug_nom_quat = nominals["mug_1_jnt"]
        eff_dx = (max(mug_p.delta_x[0], x_min - mug_nom_pos[0]),
                  min(mug_p.delta_x[1], x_max - mug_nom_pos[0]))
        eff_dy = (max(mug_p.delta_y[0], y_min - mug_nom_pos[1]),
                  min(mug_p.delta_y[1], y_max - mug_nom_pos[1]))
        mug_dx = rng.uniform(*eff_dx)
        mug_dy = rng.uniform(*eff_dy)
        mug_new_pos = mug_nom_pos + np.array([mug_dx, mug_dy, 0.0])
        q_yaw = _quat_from_yaw(rng.uniform(*mug_p.delta_yaw))
        states["mug_1_jnt"] = {
            "pos": mug_new_pos.tolist(),
            "quat": _quat_mul(q_yaw, mug_nom_quat).tolist(),
        }

        # Move beads by the same (dx, dy) as the mug so they stay inside it.
        for bead_name in self._bead_joints:
            if bead_name not in nominals:
                continue
            bead_nom_pos, bead_nom_quat = nominals[bead_name]
            states[bead_name] = {
                "pos": (bead_nom_pos + np.array([mug_dx, mug_dy, 0.0])).tolist(),
                "quat": bead_nom_quat.tolist(),
            }

        # Sample cup independently.
        cup_p = next(p for p in self.perturbations if p.joint_name == "cup_1_jnt")
        cup_nom_pos, cup_nom_quat = nominals["cup_1_jnt"]
        eff_dx = (max(cup_p.delta_x[0], x_min - cup_nom_pos[0]),
                  min(cup_p.delta_x[1], x_max - cup_nom_pos[0]))
        eff_dy = (max(cup_p.delta_y[0], y_min - cup_nom_pos[1]),
                  min(cup_p.delta_y[1], y_max - cup_nom_pos[1]))
        cup_new_pos = cup_nom_pos + np.array([
            rng.uniform(*eff_dx), rng.uniform(*eff_dy), 0.0,
        ])
        q_yaw_cup = _quat_from_yaw(rng.uniform(*cup_p.delta_yaw))
        states["cup_1_jnt"] = {
            "pos": cup_new_pos.tolist(),
            "quat": _quat_mul(q_yaw_cup, cup_nom_quat).tolist(),
        }

        return states

    def randomize(self, model: Any, data: Any, seed: int | None = None) -> RandomizationState:
        state = super().randomize(model, data, seed)
        rng = np.random.default_rng(seed)
        if rng.random() < _COLOR_RANDOMIZE_PROB:
            for mat_name in ("mug_1_color", "cup_body_color"):
                _apply_mat_color(model, mat_name, _MUG_COLOR_PALETTE[rng.integers(len(_MUG_COLOR_PALETTE))])
        return state


class SpellingRandomizer(SceneRandomizer):
    """Small per-block perturbation for the 26-letter block spelling scene."""

    min_clearance_m = 0.025
    perturbations = [
        PerturbRange(f"block_{letter}_jnt", delta_x=(-0.08, 0.08), delta_y=(-0.08, 0.08))
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ]


class DrawerRandomizer(SceneRandomizer):
    """Randomize the drawer frame position/yaw and marker positions.

    The drawer frame (drawer_body) is a fixed body moved via model.body_pos.
    Markers are free-jointed and sampled independently around the table, then
    shifted by the same (dx, dy) as the drawer so they remain nearby after reset.
    """

    min_clearance_m = 0.04
    perturbations = [
        PerturbRange("drawer_body", delta_x=(-0.08, 0.0), delta_y=(-0.10, 0.10),
                     delta_yaw=(-0.3, 0.3), fixed_body=True),
        *[
            PerturbRange(f"marker_{i}_joint", delta_x=(-0.08, 0.08), delta_y=(-0.10, 0.10))
            for i in range(1, 6)
        ],
    ]

    def _sample_once(
        self,
        nominals: dict[str, tuple[np.ndarray, np.ndarray]],
        rng: np.random.Generator,
    ) -> dict[str, dict[str, list[float]]]:
        states = super()._sample_once(nominals, rng)

        # Shift markers by the same (dx, dy) as the drawer so they stay near it.
        if "drawer_body" in states:
            drawer_nom_pos = nominals["drawer_body"][0]
            drawer_new_pos = np.array(states["drawer_body"]["pos"])
            dx = drawer_new_pos[0] - drawer_nom_pos[0]
            dy = drawer_new_pos[1] - drawer_nom_pos[1]
            for i in range(1, 6):
                key = f"marker_{i}_joint"
                if key in states:
                    states[key]["pos"][0] += dx
                    states[key]["pos"][1] += dy

        return states


class BallSortingRandomizer(SceneRandomizer):
    min_clearance_m = 0.06
    perturbations = [
        # Toy box: fixed body, randomized via model.body_pos (no physics).
        # Cylinders are rejection-sampled against it via the MuJoCo contact check.
        PerturbRange("ball-sorting-toy", delta_x=(-0.18, 0.08), delta_y=(-0.15, 0.15),
                     delta_yaw=(-0.3, 0.3), fixed_body=True),
        PerturbRange("cylinder-1", delta_x=(-0.5, 0.5), delta_y=(-0.5, 0.5)),
        PerturbRange("cylinder-2", delta_x=(-0.5, 0.5), delta_y=(-0.5, 0.5)),
        PerturbRange("cylinder-3", delta_x=(-0.5, 0.5), delta_y=(-0.5, 0.5)),
    ]


class ChessRandomizer(SceneRandomizer):
    """Small per-piece perturbation — chess pieces are tightly packed (~52mm grid)."""
    min_clearance_m = 0.02
    perturbations = [
        PerturbRange(jnt, delta_x=(-0.008, 0.008), delta_y=(-0.008, 0.008),
                     delta_yaw=(-0.25, 0.25))
        for jnt in [
            "black_bishop_1_jnt", "black_bishop_2_jnt", "black_king_1_jnt",
            "black_knight_1_jnt", "black_knight_2_jnt",
            "black_pawn_1_jnt", "black_pawn_2_jnt", "black_pawn_3_jnt",
            "black_pawn_4_jnt", "black_pawn_5_jnt", "black_pawn_6_jnt",
            "black_pawn_7_jnt", "black_pawn_8_jnt",
            "black_queen_1_jnt", "black_rook_1_jnt", "black_rook_2_jnt",
            "white_bishop_1_jnt", "white_bishop_2_jnt", "white_king_1_jnt",
            "white_knight_1_jnt", "white_knight_2_jnt",
            "white_pawn_1_jnt", "white_pawn_2_jnt", "white_pawn_3_jnt",
            "white_pawn_4_jnt", "white_pawn_5_jnt", "white_pawn_6_jnt",
            "white_pawn_7_jnt", "white_pawn_8_jnt",
            "white_queen_1_jnt", "white_rook_1_jnt", "white_rook_2_jnt",
        ]
    ]


# chess2 uses the same piece names as chess.
Chess2Randomizer = ChessRandomizer


# ---------------------------------------------------------------------------
# InHand Transfer randomizer — reloads the MuJoCo model on every reset
# ---------------------------------------------------------------------------

import xml.etree.ElementTree as _ET
from pathlib import Path as _Path

_MODELS_DIR = _Path(__file__).parent / "models"
_BASE_SCENE_XML = _MODELS_DIR / "yam_inhand_transfer_base.xml"
_LIGHTWHEEL_BASE = _MODELS_DIR / "assets_robocasa" / "objects_lightwheel" / "lightwheel"
_OBJAVERSE_BASE = _MODELS_DIR / "assets_robocasa" / "objaverse" / "objaverse"

_OBJAVERSE_CATEGORIES = {"rolling_pin", "water_bottle", "can", "ladle"}
# Approved object categories for the inhand_transfer task.
# Edit this list to add/remove objects from the randomization pool.
INHAND_OBJECT_CATEGORIES: list[str] = [
    # lightwheel pack
    "dish_brush", "whisk", "salt_and_pepper_shaker", "cream_cheese_stick",
    "cheese_grater", "pizza_cutter",
    # objaverse pack
    "rolling_pin", "water_bottle", "can", "ladle",
]
_INHAND_CATEGORIES = INHAND_OBJECT_CATEGORIES  # internal alias
_OBJ_DENSITY = 30.0
_CATEGORY_MESH_SCALE: dict[str, str] = {"wooden_spoon": "1 1 2.5"}
_X_MIN, _X_MAX = 0.42, 0.78
_Y_LEFT_MIN, _Y_LEFT_MAX = 0.10, 0.40
_Y_RIGHT_MIN, _Y_RIGHT_MAX = -0.40, -0.10
_OBJ_Z = 0.82


def _inhand_asset_base(category: str) -> _Path:
    return _OBJAVERSE_BASE if category in _OBJAVERSE_CATEGORIES else _LIGHTWHEEL_BASE


def _inhand_get_variants(category: str) -> list[_Path]:
    cat_dir = _inhand_asset_base(category) / category
    variants = []
    for d in sorted(cat_dir.iterdir()):
        if not d.is_dir():
            continue
        try:
            parsed = _inhand_parse_model_xml(d)
            if parsed["col_geoms"]:
                variants.append(d)
        except Exception:
            pass
    return variants


def _inhand_parse_model_xml(variant_dir: _Path) -> dict:
    root = _ET.parse(str(variant_dir / "model.xml")).getroot()
    asset_elem = root.find("asset")
    meshes, textures, materials = [], [], []
    if asset_elem is not None:
        for mesh in asset_elem.findall("mesh"):
            extra = {k: v for k, v in mesh.attrib.items() if k not in ("name", "file")}
            meshes.append({"name": mesh.get("name", ""), "file": mesh.get("file", ""), "extra": extra})
        for tex in asset_elem.findall("texture"):
            rel = tex.get("file", "")
            if rel:
                textures.append({"name": tex.get("name", ""), "file": rel, "type": tex.get("type", "2d")})
        for mat in asset_elem.findall("material"):
            materials.append({
                "name": mat.get("name", ""), "texture": mat.get("texture", ""),
                "rgba": mat.get("rgba", ""), "shininess": mat.get("shininess", ""),
                "specular": mat.get("specular", ""),
            })
    vis_geoms, col_geoms = [], []
    worldbody = root.find("worldbody")
    if worldbody is not None:
        for geom in worldbody.iter("geom"):
            cls = geom.get("class", "")
            if geom.get("name") == "reg_bbox":
                continue
            is_visual = cls == "visual" or (geom.get("contype") == "0" and cls != "region")
            if is_visual:
                vis_geoms.append({"mesh": geom.get("mesh", ""), "material": geom.get("material", "")})
            elif cls == "collision":
                col_geoms.append({"mesh": geom.get("mesh", "")})
    return {"meshes": meshes, "textures": textures, "materials": materials,
            "vis_geoms": vis_geoms, "col_geoms": col_geoms}


def _inhand_build_xml(category: str, variant_dir: _Path, x: float, y: float, z: float, yaw: float) -> str:
    base_text = _BASE_SCENE_XML.read_text()
    parsed = _inhand_parse_model_xml(variant_dir)
    prefix = "obj"
    cat_scale = _CATEGORY_MESH_SCALE.get(category)

    lines_asset = [f"    <!-- Object: {category}/{variant_dir.name} -->"]
    for m in parsed["meshes"]:
        extra = {**m["extra"]}
        if cat_scale and "scale" not in extra:
            extra["scale"] = cat_scale
        extra_str = "".join(f' {k}="{v}"' for k, v in extra.items())
        lines_asset.append(f'    <mesh file="{variant_dir / m["file"]}" name="{prefix}_{m["name"]}"{extra_str}/>')
    for t in parsed["textures"]:
        lines_asset.append(f'    <texture file="{variant_dir / t["file"]}" name="{prefix}_{t["name"]}" type="{t["type"]}"/>')
    for mat in parsed["materials"]:
        attrs = f'name="{prefix}_{mat["name"]}"'
        if mat["texture"]: attrs += f' texture="{prefix}_{mat["texture"]}"'
        if mat["rgba"]:    attrs += f' rgba="{mat["rgba"]}"'
        if mat["shininess"]: attrs += f' shininess="{mat["shininess"]}"'
        if mat["specular"]:  attrs += f' specular="{mat["specular"]}"'
        lines_asset.append(f"    <material {attrs}/>")

    w, s = np.cos(yaw / 2), np.sin(yaw / 2)
    lines_body = [
        f"    <!-- Task object: {category}/{variant_dir.name} -->",
        f'    <body name="task_object" pos="{x:.4f} {y:.4f} {z:.4f}" quat="{w:.6f} 0 0 {s:.6f}">',
        f'      <freejoint name="task_object_joint"/>',
    ]
    for vg in parsed["vis_geoms"]:
        mesh_attr = f'mesh="{prefix}_{vg["mesh"]}"' if vg["mesh"] else ""
        mat_attr = f' material="{prefix}_{vg["material"]}"' if vg["material"] else ""
        lines_body.append(
            f'      <geom type="mesh" {mesh_attr}{mat_attr} contype="0" conaffinity="0" group="2"'
            f' density="0" solimp="0.998 0.998 0.001" solref="0.001 1"/>')
    for cg in parsed["col_geoms"]:
        mesh_attr = f'mesh="{prefix}_{cg["mesh"]}"' if cg["mesh"] else ""
        lines_body.append(
            f'      <geom type="mesh" {mesh_attr} group="3" density="{_OBJ_DENSITY}"'
            f' friction="3.0 0.3 0.05" solimp="0.95 0.99 0.001" solref="0.004 1"/>')
    lines_body.append("    </body>")

    result = base_text.replace("<!-- TASK_ASSETS_PLACEHOLDER -->", "\n".join(lines_asset))
    result = result.replace("<!-- TASK_BODY_PLACEHOLDER -->", "\n".join(lines_body))
    result = result.replace('file="i2rt_yam/', f'file="{_MODELS_DIR}/assets/i2rt_yam/')
    return result


class InHandTransferRandomizer(SceneRandomizer):
    """Randomizer for inhand_transfer: swaps the entire MuJoCo model on each reset.

    Unlike other randomizers that perturb existing object positions, this one
    picks a new kitchen tool category/variant and reloads the model from XML so
    the mesh assets change.  Call ``bind_env(env)`` after construction.
    """

    perturbations: list = []  # no PerturbRange — we handle placement ourselves

    def __init__(self) -> None:
        super().__init__()
        self._env_ref = None
        self._rng = np.random.default_rng()
        # Pre-cache variant lists so we don't re-scan on every reset.
        self._variants: dict[str, list[_Path]] = {}

    def bind_env(self, env: Any) -> None:
        self._env_ref = env

    def _get_variants(self, category: str) -> list[_Path]:
        if category not in self._variants:
            self._variants[category] = _inhand_get_variants(category)
        return self._variants[category]

    def randomize(self, model: Any, data: Any, seed: int | None = None) -> RandomizationState:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        env = self._env_ref
        categories = _INHAND_CATEGORIES
        category = categories[int(self._rng.integers(0, len(categories)))]
        variants = self._get_variants(category)
        variant_dir = variants[int(self._rng.integers(0, len(variants)))]

        side = "left" if self._rng.random() < 0.5 else "right"
        x = float(self._rng.uniform(_X_MIN, _X_MAX))
        y = float(self._rng.uniform(_Y_LEFT_MIN, _Y_LEFT_MAX) if side == "left"
                  else self._rng.uniform(_Y_RIGHT_MIN, _Y_RIGHT_MAX))
        yaw = float(self._rng.uniform(-np.pi, np.pi))

        xml = _inhand_build_xml(category, variant_dir, x, y, _OBJ_Z, yaw)

        if env is not None:
            env.reload_from_xml(xml)
            mujoco.mj_resetData(env.model, env.data)
            env._set_qpos_from_state(env.get_init_q())
            jnt_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "task_object_joint")
            qadr = env.model.jnt_qposadr[jnt_id]
            w, s = np.cos(yaw / 2), np.sin(yaw / 2)
            env.data.qpos[qadr:qadr + 3] = [x, y, _OBJ_Z]
            env.data.qpos[qadr + 3:qadr + 7] = [w, 0, 0, s]
            mujoco.mj_forward(env.model, env.data)
            # Store metadata for callers
            env._inhand_category = category
            env._inhand_variant = variant_dir.name
            env._inhand_side = side

        return RandomizationState(
            seed=seed or 0,
            object_states={"task_object_joint": {"pos": [x, y, _OBJ_Z], "quat": [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]}},
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TASK_RANDOMIZERS: dict[str, SceneRandomizer] = {
    "bottles":      BottlesRandomizer(),
    "marker":       MarkerRandomizer(),
    "pour":         PourRandomizer(),
    "spelling":     SpellingRandomizer(),
    "drawer":       DrawerRandomizer(),
    "dishrack":     DishRackRandomizer(),
    "blocks":       BlocksRandomizer(),
    "mug_tree":     MugTreeRandomizer(),
    "mug_flip":     MugFlipRandomizer(),
    "ball_sorting": BallSortingRandomizer(),
    "chess":        ChessRandomizer(),
    "chess2":       Chess2Randomizer(),
    # "empty": no free-jointed objects to randomize
}
