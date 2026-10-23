"""Tests for the evaluation harness: splits, metrics, and baselines.

Each module gets the three cases CONTRIBUTING Step 4 asks for:
- the success path
- the validation failure
- the boundary case (including the critical whole-trajectory separation test)
"""

import pytest

from astra_chronus.evaluation.baselines import (
    ConstantVelocityPredictor,
    KalmanPredictor,
)
from astra_chronus.evaluation.metrics import (
    EvaluationMetrics,
    compute_aggregate_metrics,
    compute_angular_displacement,
    compute_prediction_metrics,
)
from astra_chronus.evaluation.splits import (
    TrajectorySplit,
    split_trajectories,
)
from astra_chronus.interfaces import Prediction, TrackState

T_NS = 1_730_000_000_033_000_000


def make_test_track(
    track_id: int = 1,
    az_deg: float = 90.0,
    el_deg: float = 10.0,
    az_rate_dps: float = -2.0,
    el_rate_dps: float = 0.5,
    confidence: float = 0.9,
) -> TrackState:
    """Helper creating a valid TrackState for tests."""
    return TrackState(
        track_id=track_id,
        t_ns=T_NS,
        az_deg=az_deg,
        el_deg=el_deg,
        az_rate_dps=az_rate_dps,
        el_rate_dps=el_rate_dps,
        bbox_scale=12.0,
        confidence=confidence,
    )


# -----------------------------------------------------------------------------
# 1. Trajectory Splits Tests
# -----------------------------------------------------------------------------


def test_split_trajectories_success_path() -> None:
    """Success path: whole trajectories partitioned into train, val, and test."""
    ids = [f"traj_{i:03d}" for i in range(100)]
    split = split_trajectories(
        ids,
        seed=42,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    assert isinstance(split, TrajectorySplit)
    assert len(split.train_ids) == 70
    assert len(split.val_ids) == 15
    assert len(split.test_ids) == 15
    # Total trajectories preserved
    all_assigned = set(split.train_ids) | set(split.val_ids) | set(split.test_ids)
    assert all_assigned == set(ids)


def test_no_trajectory_is_split_across_train_and_test() -> None:
    """Boundary case: a trajectory must NEVER appear in more than one partition.

    This directly addresses V0 Corrigendum Finding 5 (preventing data leakage
    across frame-based splits).
    """
    ids = [f"flight_{i}" for i in range(250)]
    split = split_trajectories(
        ids,
        seed=20261101,
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20,
    )

    train_set = set(split.train_ids)
    val_set = set(split.val_ids)
    test_set = set(split.test_ids)

    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)


def test_split_trajectories_reproducibility() -> None:
    """Boundary case: identical seeds yield identical splits regardless of input order."""
    ids_a = [f"traj_{i}" for i in range(50)]
    ids_b = list(reversed(ids_a))

    split_a = split_trajectories(ids_a, seed=123)
    split_b = split_trajectories(ids_b, seed=123)
    split_different_seed = split_trajectories(ids_a, seed=999)

    assert split_a.train_ids == split_b.train_ids
    assert split_a.test_ids == split_b.test_ids
    assert split_a.train_ids != split_different_seed.train_ids


def test_split_trajectories_validation_failures() -> None:
    """Failure path: invalid inputs raise descriptive ValueErrors."""
    # Empty input
    with pytest.raises(ValueError, match="must not be empty"):
        split_trajectories([], seed=1)

    # Duplicate trajectory IDs
    with pytest.raises(ValueError, match="duplicate entries"):
        split_trajectories(["traj_1", "traj_1"], seed=1)

    # Ratios not summing to 1.0
    with pytest.raises(ValueError, match="must sum to 1.0"):
        split_trajectories(
            ["traj_1", "traj_2"], seed=1, train_ratio=0.5, val_ratio=0.2, test_ratio=0.2
        )

    # Ratio outside [0.0, 1.0]
    with pytest.raises(ValueError, match="must be in \\[0.0, 1.0\\]"):
        split_trajectories(
            ["t1"], seed=1, train_ratio=-0.1, val_ratio=0.6, test_ratio=0.5
        )


# -----------------------------------------------------------------------------
# 2. Metrics Tests
# -----------------------------------------------------------------------------


def test_metrics_success_path() -> None:
    """Success path: ADE and FDE accurately calculated in degrees."""
    horizons = (0.25, 0.5, 1.0, 2.0)
    pred = Prediction(
        track_id=1,
        t_ns=T_NS,
        horizons_s=horizons,
        delta_az_deg=(-0.5, -1.0, -2.0, -4.0),
        delta_el_deg=(0.2, 0.4, 0.8, 1.6),
        sigma_deg=(0.1, 0.2, 0.4, 0.8),
    )
    # Ground truth with known fixed offsets:
    # At h=0.25: pred=(-0.5, 0.2), true=(-0.2, 0.6) -> diff=(-0.3, -0.4) -> dist = 0.5 deg
    # At h=0.50: pred=(-1.0, 0.4), true=(-1.0, 0.4) -> dist = 0.0 deg
    # At h=1.00: pred=(-2.0, 0.8), true=(-2.0, 1.8) -> diff=(0.0, -1.0) -> dist = 1.0 deg
    # At h=2.00: pred=(-4.0, 1.6), true=(-7.0, 5.6) -> diff=(3.0, -4.0) -> dist = 5.0 deg
    true_az = (-0.2, -1.0, -2.0, -7.0)
    true_el = (0.6, 0.4, 1.8, 5.6)

    metrics = compute_prediction_metrics(pred, true_az, true_el)

    assert isinstance(metrics, EvaluationMetrics)
    assert metrics.errors_deg == pytest.approx((0.5, 0.0, 1.0, 5.0))
    # ADE = (0.5 + 0.0 + 1.0 + 5.0) / 4 = 1.625 deg
    assert metrics.ade_deg == pytest.approx(1.625)
    # FDE = 5.0 deg (error at 2.0s)
    assert metrics.fde_deg == pytest.approx(5.0)


def test_metrics_custom_eval_horizons_subset() -> None:
    """Success path: evaluation over paper reporting subset (0.5, 1.0, 2.0s)."""
    horizons = (0.25, 0.5, 1.0, 2.0)
    pred = Prediction(
        track_id=1,
        t_ns=T_NS,
        horizons_s=horizons,
        delta_az_deg=(0.0, 1.0, 2.0, 3.0),
        delta_el_deg=(0.0, 0.0, 0.0, 0.0),
        sigma_deg=(0.1, 0.2, 0.3, 0.4),
    )
    true_az = (0.0, 2.0, 4.0, 6.0)
    true_el = (0.0, 0.0, 0.0, 0.0)

    # Evaluate only on (0.5, 1.0, 2.0)
    metrics = compute_prediction_metrics(
        pred, true_az, true_el, eval_horizons_s=(0.5, 1.0, 2.0)
    )
    assert metrics.horizons_s == (0.5, 1.0, 2.0)
    # Errors at 0.5: 1.0, at 1.0: 2.0, at 2.0: 3.0
    assert metrics.errors_deg == pytest.approx((1.0, 2.0, 3.0))
    assert metrics.ade_deg == pytest.approx(2.0)
    assert metrics.fde_deg == pytest.approx(3.0)


def test_metrics_zero_displacement_gives_zero_error() -> None:
    """Boundary case: exact prediction gives exactly 0.0 ADE and FDE."""
    horizons = (0.25, 0.5, 1.0, 2.0)
    pred = Prediction(
        track_id=1,
        t_ns=T_NS,
        horizons_s=horizons,
        delta_az_deg=(1.2, 2.4, 4.8, 9.6),
        delta_el_deg=(-0.3, -0.6, -1.2, -2.4),
        sigma_deg=(0.1, 0.2, 0.3, 0.4),
    )
    metrics = compute_prediction_metrics(
        pred,
        true_delta_az_deg=pred.delta_az_deg,
        true_delta_el_deg=pred.delta_el_deg,
    )
    assert metrics.ade_deg == 0.0
    assert metrics.fde_deg == 0.0
    assert all(e == 0.0 for e in metrics.errors_deg)


def test_metrics_azimuth_wrap_around_boundary() -> None:
    """Boundary case: angular difference across 360/0 degree modular boundary."""
    # Predicted +179 deg, Ground truth -179 deg -> difference is 2 deg, not 358 deg
    dist = compute_angular_displacement(179.0, 0.0, -179.0, 0.0)
    assert dist == pytest.approx(2.0)


def test_metrics_validation_failures() -> None:
    """Failure path: mismatched dimensions or invalid horizons raise."""
    pred = Prediction(
        track_id=1,
        t_ns=T_NS,
        horizons_s=(0.5, 1.0),
        delta_az_deg=(0.1, 0.2),
        delta_el_deg=(0.0, 0.0),
        sigma_deg=(0.1, 0.2),
    )
    # Mismatched length of ground truth
    with pytest.raises(ValueError, match="Ground truth length mismatch"):
        compute_prediction_metrics(pred, [0.1], [0.0])

    # Requesting horizon not present in prediction
    with pytest.raises(
        ValueError, match="Requested evaluation horizon 2.0s not present"
    ):
        compute_prediction_metrics(pred, [0.1, 0.2], [0.0, 0.0], eval_horizons_s=(2.0,))


def test_aggregate_metrics() -> None:
    """Success path: aggregation across multiple trajectories."""
    m1 = EvaluationMetrics(
        horizons_s=(1.0,), errors_deg=(2.0,), ade_deg=2.0, fde_deg=2.0
    )
    m2 = EvaluationMetrics(
        horizons_s=(1.0,), errors_deg=(4.0,), ade_deg=4.0, fde_deg=4.0
    )

    agg = compute_aggregate_metrics([m1, m2])
    assert agg["mean_ade_deg"] == pytest.approx(3.0)
    assert agg["mean_fde_deg"] == pytest.approx(3.0)
    assert agg["err_1.00s_deg"] == pytest.approx(3.0)

    with pytest.raises(ValueError, match="must not be empty"):
        compute_aggregate_metrics([])


# -----------------------------------------------------------------------------
# 3. Baselines Tests
# -----------------------------------------------------------------------------


def test_constant_velocity_baseline_success() -> None:
    """Success path: ConstantVelocityPredictor produces contract Prediction."""
    predictor = ConstantVelocityPredictor(
        horizons_s=(0.25, 0.5, 1.0, 2.0),
        sigma_rate_dps=0.4,
    )
    track = make_test_track(az_rate_dps=-2.0, el_rate_dps=1.5)
    pred = predictor.predict(track)

    assert isinstance(pred, Prediction)
    assert pred.track_id == track.track_id
    assert pred.t_ns == track.t_ns
    assert pred.horizons_s == (0.25, 0.5, 1.0, 2.0)
    assert pred.delta_az_deg == pytest.approx((-0.5, -1.0, -2.0, -4.0))
    assert pred.delta_el_deg == pytest.approx((0.375, 0.75, 1.5, 3.0))


def test_constant_velocity_uncertainty_widens_with_horizon() -> None:
    """Boundary case: sigma is monotonically increasing with horizon."""
    predictor = ConstantVelocityPredictor()
    pred = predictor.predict(make_test_track())
    assert all(
        later > earlier
        for earlier, later in zip(pred.sigma_deg, pred.sigma_deg[1:], strict=False)
    )


def test_constant_velocity_stationary_target_gives_zero_delta() -> None:
    """Boundary case: zero rate gives zero delta offsets."""
    predictor = ConstantVelocityPredictor()
    track = make_test_track(az_rate_dps=0.0, el_rate_dps=0.0)
    pred = predictor.predict(track)
    assert pred.delta_az_deg == (0.0, 0.0, 0.0, 0.0)
    assert pred.delta_el_deg == (0.0, 0.0, 0.0, 0.0)


def test_constant_velocity_validation_failures() -> None:
    """Failure path: invalid confidence or empty horizons raise ValueError."""
    with pytest.raises(ValueError, match="confidence must be in"):
        predictor = ConstantVelocityPredictor()
        predictor.predict(make_test_track(confidence=1.5))

    with pytest.raises(ValueError, match="positive"):
        ConstantVelocityPredictor(horizons_s=(-1.0, 0.5))


def test_kalman_baseline_success() -> None:
    """Success path: KalmanPredictor filters history and yields Prediction."""
    predictor = KalmanPredictor()
    history = [
        make_test_track(az_deg=90.0 + i * 0.1, az_rate_dps=3.0, el_rate_dps=-1.0)
        for i in range(5)
    ]
    pred = predictor.predict(history)

    assert isinstance(pred, Prediction)
    assert pred.horizons_s == (0.25, 0.5, 1.0, 2.0)
    assert len(pred.delta_az_deg) == 4
    assert len(pred.sigma_deg) == 4

    with pytest.raises(ValueError, match="history sequence must not be empty"):
        predictor.predict([])
