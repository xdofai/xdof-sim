from __future__ import annotations

import unittest

import numpy as np

from xdof_sim.debug import TaskEvalDashboardState
from xdof_sim.task_eval import EvalDebugSpec, PlotSpec, ThresholdSpec
from xdof_sim.task_eval import TaskEvalResult


class TaskEvalDashboardStateTests(unittest.TestCase):
    def test_snapshot_tracks_current_and_history(self) -> None:
        state = TaskEvalDashboardState(
            task_name="bottles",
            prompt="throw plastic bottles in bin",
            evaluator_name="bottles_in_bin",
            debug_spec=EvalDebugSpec(
                plots=[
                    PlotSpec(
                        key="num_bottles_in_bin",
                        title="Bottles In Bin",
                        thresholds=[ThresholdSpec(value=2.0, label="success", direction="gt")],
                    )
                ]
            ).to_dict(),
            history_limit=4,
        )
        state.update(
            step=3,
            sim_time=0.1,
            result=TaskEvalResult(
                reward=np.asarray([0.5], dtype=np.float32),
                success=np.asarray([False]),
                metrics={
                    "num_bottles_in_bin": np.asarray([1], dtype=np.int32),
                    "bottles_in_bin": [["bottle_1"]],
                },
            ),
        )

        snapshot = state.snapshot(history_tail=16)
        self.assertTrue(snapshot["available"])
        self.assertEqual(snapshot["task_name"], "bottles")
        self.assertEqual(snapshot["prompt"], "throw plastic bottles in bin")
        self.assertEqual(snapshot["current"]["step"], 3)
        self.assertEqual(snapshot["current"]["metrics"]["num_bottles_in_bin"], 1)
        self.assertEqual(snapshot["history"][0]["reward"], 0.5)
        self.assertIn("num_bottles_in_bin", snapshot["numeric_metric_keys"])
        self.assertEqual(snapshot["debug_spec"]["plots"][0]["key"], "num_bottles_in_bin")

    def test_snapshot_handles_missing_evaluator(self) -> None:
        state = TaskEvalDashboardState(
            task_name="unknown",
            prompt="do something",
            evaluator_name=None,
        )
        state.update(step=0, sim_time=0.0, result=None)
        snapshot = state.snapshot()
        self.assertFalse(snapshot["available"])
        self.assertIsNone(snapshot["current"]["reward"])
        self.assertEqual(snapshot["history"], [])


if __name__ == "__main__":
    unittest.main()
