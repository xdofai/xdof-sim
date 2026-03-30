"""Inverse kinematics module for xdof-sim.

Provides J-PARSE singularity-aware velocity IK using MuJoCo's native
Jacobian computation as the MJCF backend.
"""

from xdof_sim.ik.mujoco_ik import MuJoCoIKSolver

__all__ = ["MuJoCoIKSolver"]
