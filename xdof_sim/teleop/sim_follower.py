"""MuJoCo sim follower node that accepts ZMQ commands from GELLO leaders.

Subscribes to left_actions and right_actions ZMQ topics, applies them as
joint position targets in the MuJoCo sim, and renders camera views.

Usage:
    python -m xdof_sim.teleop.sim_follower --scene hybrid
"""

from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np

from xdof_sim.teleop.node import Node


class SimFollowerNode(Node):
    """Follower node that drives a MuJoCo sim from ZMQ leader commands."""

    def __init__(
        self,
        name: str = "sim_follower",
        control_rate: float = 30.0,
        left_leader: str = "left",
        right_leader: str = "right",
        scene: str = "hybrid",
        render: bool = True,
    ):
        super().__init__(name, control_rate)
        self.left_leader = left_leader
        self.right_leader = right_leader
        self.scene = scene
        self.render = render

        # Subscribe to leader topics
        self.create_subscriber(f"{left_leader}_actions", conflate=1)
        self.create_subscriber(f"{right_leader}_actions", conflate=1)

        # Publish joint state
        self.create_publisher(f"{name}_state")

    def initial_bootup(self, *args, **kwargs) -> None:
        import xdof_sim

        self.env = xdof_sim.make_env(
            scene=self.scene, render_cameras=self.render
        )
        obs, _ = self.env.reset()
        self._state = obs["state"].copy()
        print(f"[{self._name}] Sim follower ready. Waiting for leader commands...")

        # Publish initial state
        self.publish(f"{self._name}_state", self._state)

        # Wait for first leader message
        print(f"[{self._name}] Waiting for leader connections...")
        left_msg = self.subscribe(f"{self.left_leader}_actions", block=True)
        right_msg = self.subscribe(f"{self.right_leader}_actions", block=True)

        # Apply initial positions via interpolation
        left_pos = left_msg[0]
        right_pos = right_msg[0]
        combined = np.concatenate([left_pos, right_pos])

        current = self.env.get_init_q()
        steps = 50
        for i in range(steps + 1):
            alpha = i / steps
            target = (1 - alpha) * current + alpha * combined
            self.env._step_single(target)
        self._state = self.env.get_obs()["state"]
        self.publish(f"{self._name}_state", self._state)
        print(f"[{self._name}] Initial position set. Starting control loop.")

    def tick(self) -> None:
        # Get latest commands from both leaders (non-blocking)
        left_pos, left_extras = self.subscribe(
            f"{self.left_leader}_actions", block=False
        )
        right_pos, right_extras = self.subscribe(
            f"{self.right_leader}_actions", block=False
        )

        if left_pos is None and right_pos is None:
            return

        # Build 14D action from latest state + any new commands
        action = self._state.copy()
        if left_pos is not None:
            action[:7] = left_pos
        if right_pos is not None:
            action[7:] = right_pos

        # Step physics
        self.env._step_single(action)
        obs = self.env.get_obs()
        self._state = obs["state"].copy()

        # Publish updated state
        self.publish(f"{self._name}_state", self._state)

    def on_shutdown(self) -> None:
        if hasattr(self, "env"):
            self.env.close()
        print(f"[{self._name}] Sim follower shutdown.")


def main():
    parser = argparse.ArgumentParser(description="MuJoCo sim follower node")
    parser.add_argument(
        "--scene",
        type=str,
        default="hybrid",
        choices=["eval", "training", "hybrid"],
    )
    parser.add_argument("--left-leader", type=str, default="left")
    parser.add_argument("--right-leader", type=str, default="right")
    parser.add_argument("--control-rate", type=float, default=30.0)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    node = SimFollowerNode(
        scene=args.scene,
        left_leader=args.left_leader,
        right_leader=args.right_leader,
        control_rate=args.control_rate,
        render=not args.no_render,
    )
    node.run()


if __name__ == "__main__":
    main()
