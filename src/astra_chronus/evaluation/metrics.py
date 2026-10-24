"""Evaluation metrics for counter-UAS trajectory prediction.

Reports Average Displacement Error (ADE) and Final Displacement Error (FDE)
in degrees, matching README section 7 and the frozen interface contract in
`astra_chronus.interfaces.Prediction`.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from astra_chronus.interfaces import Prediction

CONTRACT_HORIZONS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)


@dataclass(frozen=True)
class EvaluationMetrics:
    """Displacement error metrics across evaluation horizons.

    Attributes
    ----------
    horizons_s : tuple[float, ...]
        The evaluation horizon grid in seconds.
    errors_deg : tuple[float, ...]
        Angular displacement error in degrees at each horizon.
    ade_deg : float
        Average Displacement Error in degrees across all evaluated horizons.
    fde_deg : float
        Final Displacement Error in degrees at the furthest horizon.
    """

    horizons_s: tuple[float, ...]
    errors_deg: tuple[float, ...]
    ade_deg: float
    fde_deg: float


def compute_angular_displacement(
    pred_delta_az: float,
    pred_delta_el: float,
    true_delta_az: float,
    true_delta_el: float,
) -> float:
    """Calculate Euclidean angular displacement in degrees with azimuth wrap.

    Parameters
    ----------
    pred_delta_az : float
        Predicted azimuth offset in degrees.
    pred_delta_el : float
        Predicted elevation offset in degrees.
    true_delta_az : float
        Ground-truth azimuth offset in degrees.
    true_delta_el : float
        Ground-truth elevation offset in degrees.

    Returns
    -------
    float
        Displacement magnitude in degrees.
    """
    # Wrap azimuth difference across modular circular boundary [-180, 180)
    diff_az = ((pred_delta_az - true_delta_az + 180.0) % 360.0) - 180.0
    diff_el = pred_delta_el - true_delta_el
    return math.hypot(diff_az, diff_el)


def compute_prediction_metrics(
    prediction: Prediction,
    true_delta_az_deg: Sequence[float],
    true_delta_el_deg: Sequence[float],
    *,
    eval_horizons_s: Sequence[float] | None = None,
) -> EvaluationMetrics:
    """Compute ADE and FDE in degrees between a prediction and ground truth.

    Parameters
    ----------
    prediction : Prediction
        Predicted trajectory containing horizons, delta azimuths, and delta elevations.
    true_delta_az_deg : Sequence[float]
        Ground-truth azimuth offsets corresponding to `prediction.horizons_s`.
    true_delta_el_deg : Sequence[float]
        Ground-truth elevation offsets corresponding to `prediction.horizons_s`.
    eval_horizons_s : Sequence[float] | None, default None
        Subset of horizons (in seconds) to evaluate. If None, evaluates across
        all horizons declared in `prediction.horizons_s`.

    Returns
    -------
    EvaluationMetrics
        Container holding per-horizon errors, ADE in degrees, and FDE in degrees.

    Raises
    ------
    ValueError
        If offset lengths do not match prediction horizons, or if requested
        evaluation horizons are not present in the prediction.
    """
    n_horizons = len(prediction.horizons_s)
    if len(true_delta_az_deg) != n_horizons or len(true_delta_el_deg) != n_horizons:
        raise ValueError(
            f"Ground truth length mismatch: prediction has {n_horizons} horizons, "
            f"got {len(true_delta_az_deg)} az offsets and {len(true_delta_el_deg)} el offsets."
        )

    if eval_horizons_s is None:
        target_horizons = prediction.horizons_s
    else:
        target_horizons = tuple(eval_horizons_s)

    if not target_horizons:
        raise ValueError("eval_horizons_s must contain at least one horizon.")

    # Build index mapping from horizon to position in prediction
    horizon_idx_map = {h: idx for idx, h in enumerate(prediction.horizons_s)}

    errors: list[float] = []
    for h in target_horizons:
        if h not in horizon_idx_map:
            raise ValueError(
                f"Requested evaluation horizon {h}s not present in prediction "
                f"horizons {prediction.horizons_s}."
            )
        idx = horizon_idx_map[h]
        err = compute_angular_displacement(
            pred_delta_az=prediction.delta_az_deg[idx],
            pred_delta_el=prediction.delta_el_deg[idx],
            true_delta_az=true_delta_az_deg[idx],
            true_delta_el=true_delta_el_deg[idx],
        )
        errors.append(err)

    errors_tuple = tuple(errors)
    ade = sum(errors_tuple) / len(errors_tuple)
    fde = errors_tuple[-1]

    return EvaluationMetrics(
        horizons_s=target_horizons,
        errors_deg=errors_tuple,
        ade_deg=ade,
        fde_deg=fde,
    )


def compute_aggregate_metrics(
    metrics_list: Sequence[EvaluationMetrics],
) -> dict[str, float]:
    """Aggregate a collection of evaluation metrics across tracks.

    Parameters
    ----------
    metrics_list : Sequence[EvaluationMetrics]
        Collection of metrics computed for individual trajectory tracks.

    Returns
    -------
    dict[str, float]
        Dictionary with 'mean_ade_deg', 'mean_fde_deg', and mean error per horizon.

    Raises
    ------
    ValueError
        If `metrics_list` is empty or horizons across entries do not match.
    """
    if not metrics_list:
        raise ValueError("metrics_list must not be empty.")

    base_horizons = metrics_list[0].horizons_s
    for item in metrics_list:
        if item.horizons_s != base_horizons:
            raise ValueError("All evaluation metrics must share identical horizons.")

    n_samples = len(metrics_list)
    mean_ade = sum(m.ade_deg for m in metrics_list) / n_samples
    mean_fde = sum(m.fde_deg for m in metrics_list) / n_samples

    result: dict[str, float] = {
        "mean_ade_deg": mean_ade,
        "mean_fde_deg": mean_fde,
    }

    for i, h in enumerate(base_horizons):
        h_err = sum(m.errors_deg[i] for m in metrics_list) / n_samples
        result[f"err_{h:.2f}s_deg"] = h_err

    return result
