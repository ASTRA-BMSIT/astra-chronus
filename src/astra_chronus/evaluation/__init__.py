"""Evaluation harness package for Astra Chronus.

Provides trajectory splitting, metric computation (ADE/FDE in degrees),
and baseline predictors (constant velocity, Kalman).
"""

from astra_chronus.evaluation.baselines import (
    ConstantVelocityPredictor,
    KalmanPredictor,
)
from astra_chronus.evaluation.metrics import (
    CONTRACT_HORIZONS,
    EvaluationMetrics,
    compute_aggregate_metrics,
    compute_angular_displacement,
    compute_prediction_metrics,
)
from astra_chronus.evaluation.splits import (
    TrajectorySplit,
    split_trajectories,
)

__all__ = [
    "CONTRACT_HORIZONS",
    "ConstantVelocityPredictor",
    "EvaluationMetrics",
    "KalmanPredictor",
    "TrajectorySplit",
    "compute_aggregate_metrics",
    "compute_angular_displacement",
    "compute_prediction_metrics",
    "split_trajectories",
]
