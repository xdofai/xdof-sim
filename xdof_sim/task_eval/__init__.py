"""Task evaluation helpers for automatic reward/success scoring."""

from xdof_sim.task_eval.base import TaskEvalResult, TaskEvaluator
from xdof_sim.task_eval.debug_spec import EvalDebugSpec, PlotSpec, ThresholdSpec
from xdof_sim.task_eval.bottles import BottlesInBinEvaluator
from xdof_sim.task_eval.registry import make_task_evaluator

__all__ = [
    "TaskEvalResult",
    "TaskEvaluator",
    "ThresholdSpec",
    "PlotSpec",
    "EvalDebugSpec",
    "BottlesInBinEvaluator",
    "make_task_evaluator",
]
