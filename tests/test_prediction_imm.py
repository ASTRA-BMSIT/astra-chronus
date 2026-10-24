"""Tests for the IMM-EKF prediction baseline.

These tests cover the public prediction contract, configuration validation,
motion transitions, numerical robustness, and deterministic history handling.
The coordinated-turn tests validate the documented angular-plane approximation,
not a full 3-D vehicle dynamics model.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from astra_chronus.interfaces import Prediction, TrackState
from astra_chronus.prediction.imm_ekf import (
    _SMALL_ANGLE_RAD,
    IMMEKF,
    _jacobian,
    _transition,
    _turn_coefficients,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "prediction" / "imm_ekf_v1.yaml"
EXPECTED_HORIZONS = (0.25, 0.5, 1.0, 2.0)
EXPECTED_MODELS = [
    "constant_velocity",
    "constant_acceleration",
    "coordinated_turn",
]


@pytest.fixture
def config():
    """Load the committed IMM-EKF configuration."""
    return deepcopy(load_config(CONFIG_PATH))


@pytest.fixture
def history():
    """Create three plausible, time-ordered observations for one track."""
    return (
        TrackState(
            track_id=1,
            t_ns=1_000_000_000,
            az_deg=10.0,
            el_deg=5.0,
            az_rate_dps=2.0,
            el_rate_dps=0.5,
            bbox_scale=0.02,
            confidence=0.95,
        ),
        TrackState(
            track_id=1,
            t_ns=1_100_000_000,
            az_deg=10.2,
            el_deg=5.05,
            az_rate_dps=2.0,
            el_rate_dps=0.5,
            bbox_scale=0.02,
            confidence=0.95,
        ),
        TrackState(
            track_id=1,
            t_ns=1_200_000_000,
            az_deg=10.4,
            el_deg=5.10,
            az_rate_dps=2.0,
            el_rate_dps=0.5,
            bbox_scale=0.02,
            confidence=0.95,
        ),
    )


def make_observation(
    *,
    track_id=1,
    t_ns=1_300_000_000,
    az_deg=10.6,
    el_deg=5.15,
    az_rate_dps=2.0,
    el_rate_dps=0.5,
):
    """Build an observation for focused validation and numerical tests."""
    return TrackState(
        track_id=track_id,
        t_ns=t_ns,
        az_deg=az_deg,
        el_deg=el_deg,
        az_rate_dps=az_rate_dps,
        el_rate_dps=el_rate_dps,
        bbox_scale=0.02,
        confidence=0.95,
    )


# Configuration contract and validation


def test_load_committed_config(config):
    """The committed configuration has the required model and horizons."""
    assert config["model"] == "imm_ekf"
    assert isinstance(config["seed"], int)
    assert isinstance(config["split"], str)
    assert config["split"].strip()
    assert config["horizons_s"] == list(EXPECTED_HORIZONS)
    assert config["motion_models"] == EXPECTED_MODELS


def test_load_config_missing_file_raises(tmp_path):
    """A missing configuration path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Configuration not found"):
        load_config(tmp_path / "missing.yaml")


@pytest.mark.parametrize("yaml_text", ["", "null\n", "- item\n", "42\n"])
def test_load_config_non_mapping_raises(tmp_path, yaml_text):
    """The configuration root must be a YAML mapping."""
    path = tmp_path / "config.yaml"
    path.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(TypeError, match="YAML mapping"):
        load_config(path)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("model", "other"),
        ("seed", True),
        ("seed", 1.5),
        ("split", ""),
        ("split", "   "),
        ("split", 7),
        ("split", None),
        ("horizons_s", [0.25, 0.5, 1.0]),
        ("motion_models", ["constant_velocity"]),
    ],
)
def test_load_config_rejects_invalid_top_level_values(tmp_path, config, key, value):
    """Invalid top-level contract values are rejected."""
    config[key] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises((TypeError, ValueError)):
        load_config(path)


def test_load_config_requires_a_split_reference(tmp_path, config):
    """A config without a split reference cannot be loaded."""
    del config["split"]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="Missing configuration keys"):
        load_config(path)


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("initialization", "min_observations"),
        ("initialization", "initial_position_std_deg"),
        ("noise", "measurement_std_deg"),
        ("noise", "acceleration_std_dps2"),
        ("imm", "initial_model_probabilities"),
        ("imm", "transition_matrix"),
    ],
)
def test_load_config_rejects_missing_nested_fields(tmp_path, config, section, field):
    """Required nested configuration fields cannot be omitted."""
    del config[section][field]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="Missing keys"):
        load_config(path)


@pytest.mark.parametrize("section", ["initialization", "noise", "imm"])
def test_load_config_rejects_non_mapping_sections(tmp_path, config, section):
    """Required nested configuration sections must be mappings."""
    config[section] = []
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(TypeError, match=f"section '{section}'"):
        load_config(path)


# Prediction interface and behavior


def test_predict_returns_prediction_interface(config, history):
    """Valid history produces the frozen Prediction interface."""
    result = IMMEKF(config).predict(history)

    assert isinstance(result, Prediction)
    assert result.track_id == history[-1].track_id
    assert result.t_ns == history[-1].t_ns


def test_prediction_matches_project_horizons(config, history):
    """Prediction output contains all required horizons and matching arrays."""
    result = IMMEKF(config).predict(history)

    assert result.horizons_s == EXPECTED_HORIZONS
    assert len(result.delta_az_deg) == len(EXPECTED_HORIZONS)
    assert len(result.delta_el_deg) == len(EXPECTED_HORIZONS)
    assert len(result.sigma_deg) == len(EXPECTED_HORIZONS)


def test_prediction_values_are_finite_and_uncertainty_nonnegative(config, history):
    """All predicted offsets and uncertainty values are finite."""
    result = IMMEKF(config).predict(history)
    values = result.delta_az_deg + result.delta_el_deg + result.sigma_deg

    assert np.all(np.isfinite(values))
    assert all(value >= 0.0 for value in result.sigma_deg)


def test_prediction_uncertainty_is_finite_for_each_horizon(config, history):
    """Each horizon has one finite uncertainty value."""
    result = IMMEKF(config).predict(history)

    assert len(result.sigma_deg) == len(result.horizons_s)
    assert all(np.isfinite(value) and value >= 0 for value in result.sigma_deg)


def test_insufficient_history_is_rejected(config, history):
    """History shorter than the configured minimum is rejected."""
    with pytest.raises(ValueError, match="observations are required"):
        IMMEKF(config).predict(history[:2])


def test_non_increasing_timestamps_are_rejected(config, history):
    """Repeated timestamps are invalid."""
    invalid_history = (
        history[0],
        history[1],
        make_observation(t_ns=history[1].t_ns, az_deg=10.3, el_deg=5.1),
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        IMMEKF(config).predict(invalid_history)


def test_decreasing_timestamps_are_rejected(config, history):
    """Out-of-order observations are invalid."""
    invalid_history = (history[0], history[2], history[1])

    with pytest.raises(ValueError, match="strictly increasing"):
        IMMEKF(config).predict(invalid_history)


def test_mismatched_track_ids_are_rejected(config, history):
    """Observations from different tracks cannot be mixed."""
    invalid_history = (
        history[0],
        make_observation(track_id=2, t_ns=history[1].t_ns),
        history[2],
    )

    with pytest.raises(ValueError, match="share one track_id"):
        IMMEKF(config).predict(invalid_history)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("az_deg", float("nan")),
        ("az_deg", float("inf")),
        ("el_deg", float("-inf")),
        ("az_rate_dps", float("nan")),
        ("el_rate_dps", float("inf")),
    ],
)
def test_non_finite_angular_history_is_rejected(config, history, field, bad_value):
    """Non-finite angles and angular rates are rejected."""
    values = {
        "track_id": 1,
        "t_ns": history[-1].t_ns,
        "az_deg": history[-1].az_deg,
        "el_deg": history[-1].el_deg,
        "az_rate_dps": history[-1].az_rate_dps,
        "el_rate_dps": history[-1].el_rate_dps,
    }
    values[field] = bad_value
    invalid_history = history[:2] + (make_observation(**values),)

    with pytest.raises(ValueError, match="finite angular values"):
        IMMEKF(config).predict(invalid_history)


def test_repeated_history_produces_identical_prediction(config, history):
    """A repeated call with the same history does not reuse stale filter state."""
    predictor = IMMEKF(config)

    assert predictor.predict(history) == predictor.predict(history)


def test_extended_history_matches_fresh_predictor(config, history):
    """A predictor reset on each call handles extended history deterministically."""
    predictor = IMMEKF(config)
    predictor.predict(history)
    extended_history = history + (make_observation(),)

    reused_predictor_result = predictor.predict(extended_history)
    fresh_predictor_result = IMMEKF(config).predict(extended_history)

    assert reused_predictor_result == fresh_predictor_result


def test_model_probabilities_remain_normalized_after_prediction(config, history):
    """Model probabilities remain finite and normalized after processing history."""
    predictor = IMMEKF(config)
    predictor.predict(history)

    assert np.all(np.isfinite(predictor.probabilities))
    assert np.all(predictor.probabilities >= 0.0)
    assert predictor.probabilities.sum() == pytest.approx(1.0)


def test_filter_covariances_are_symmetric_and_psd(config, history):
    """Posterior covariance matrices are symmetric and PSD within tolerance."""
    predictor = IMMEKF(config)
    predictor.predict(history)

    for covariance in predictor.covariances:
        assert np.allclose(covariance, covariance.T, atol=1e-9)
        assert np.linalg.eigvalsh(covariance).min() >= -1e-8


# Motion transition and Jacobian tests


@pytest.mark.parametrize(
    "model",
    ["constant_velocity", "constant_acceleration", "coordinated_turn"],
)
def test_transition_at_zero_dt_preserves_position_and_rates(model):
    """A zero time step does not move the angular position or rates."""
    state = np.array([10.0, 5.0, 2.0, 1.0, 4.0, 2.0, 30.0])

    result = _transition(state, 0.0, model)

    np.testing.assert_allclose(result[:4], state[:4], atol=1e-10)


def test_constant_velocity_transition():
    """Constant velocity advances angular position at a constant rate."""
    state = np.array([10.0, 5.0, 2.0, 1.0, 0.0, 0.0, 0.0])

    result = _transition(state, 0.5, "constant_velocity")

    assert result[0] == pytest.approx(11.0)
    assert result[1] == pytest.approx(5.5)
    assert result[2] == pytest.approx(2.0)
    assert result[3] == pytest.approx(1.0)
    np.testing.assert_allclose(result[4:], 0.0)


def test_constant_acceleration_transition():
    """Constant acceleration advances positions and angular rates."""
    state = np.array([10.0, 5.0, 2.0, 1.0, 4.0, 2.0, 0.0])

    result = _transition(state, 0.5, "constant_acceleration")

    assert result[0] == pytest.approx(11.5)
    assert result[1] == pytest.approx(5.75)
    assert result[2] == pytest.approx(4.0)
    assert result[3] == pytest.approx(2.0)
    assert result[6] == pytest.approx(0.0)


def test_coordinated_turn_zero_rate_is_straight_line_motion():
    """Zero turn rate reduces the angular-plane model to straight motion."""
    state = np.array([10.0, 5.0, 2.0, 1.0, 0.0, 0.0, 0.0])

    result = _transition(state, 0.5, "coordinated_turn")

    assert result[0] == pytest.approx(11.0)
    assert result[1] == pytest.approx(5.5)
    assert result[2] == pytest.approx(2.0)
    assert result[3] == pytest.approx(1.0)


def test_coordinated_turn_preserves_angular_rate_magnitude():
    """The angular-plane turn rotates the rate vector without scaling it."""
    state = np.array([10.0, 5.0, 2.0, 1.0, 0.0, 0.0, 30.0])

    result = _transition(state, 0.5, "coordinated_turn")

    assert np.hypot(result[2], result[3]) == pytest.approx(np.hypot(state[2], state[3]))


@pytest.mark.parametrize("dt", [-1.0, float("nan"), float("inf")])
def test_transition_rejects_invalid_dt(dt):
    """Negative or non-finite time intervals are rejected."""
    state = np.zeros(7)

    with pytest.raises(ValueError, match="dt must be finite"):
        _transition(state, dt, "constant_velocity")


@pytest.mark.parametrize(
    "state",
    [
        np.zeros(6),
        np.zeros(8),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float("nan")]),
    ],
)
def test_transition_rejects_invalid_state(state):
    """The state vector must contain seven finite values."""
    with pytest.raises(ValueError, match="seven finite values"):
        _transition(state, 0.1, "constant_velocity")


def test_transition_rejects_unknown_model():
    """Unknown motion model names are rejected."""
    with pytest.raises(ValueError, match="Unknown motion model"):
        _transition(np.zeros(7), 0.1, "unknown")


def test_jacobian_at_zero_dt_is_identity():
    """At zero elapsed time, the transition Jacobian is identity."""
    state = np.array([10.0, 5.0, 2.0, 1.0, 4.0, 2.0, 30.0])

    for model in EXPECTED_MODELS:
        np.testing.assert_allclose(
            _jacobian(state, 0.0, model),
            np.eye(7),
            atol=1e-8,
        )


# IMM configuration and numerical robustness


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("noise", "measurement_std_deg", float("nan")),
        ("noise", "acceleration_std_dps2", float("inf")),
        ("noise", "turn_rate_std_dps", 0.0),
        ("initialization", "initial_position_std_deg", float("nan")),
        ("initialization", "initial_rate_std_dps", -1.0),
    ],
)
def test_invalid_noise_and_initialization_values_are_rejected(
    config, section, key, value
):
    """Standard deviations must be finite and strictly positive."""
    config[section][key] = value

    with pytest.raises((TypeError, ValueError)):
        IMMEKF(config)


@pytest.mark.parametrize("value", [0, -1, 3.5, True])
def test_invalid_min_observations_is_rejected(config, value):
    """The minimum history length must be a positive integer."""
    config["initialization"]["min_observations"] = value

    with pytest.raises((TypeError, ValueError)):
        IMMEKF(config)


@pytest.mark.parametrize(
    "probabilities",
    [
        [0.0, 0.0, 0.0],
        [float("nan"), 0.5, 0.5],
        [float("inf"), 0.0, 0.0],
        [-0.1, 0.6, 0.5],
        [1.1, 0.0, -0.1],
        [0.2, 0.2, 0.2],
    ],
)
def test_invalid_initial_model_probabilities_are_rejected(config, probabilities):
    """Initial probabilities must be finite, bounded, and sum to one."""
    config["imm"]["initial_model_probabilities"] = probabilities

    with pytest.raises(ValueError):
        IMMEKF(config)


@pytest.mark.parametrize(
    "matrix",
    [
        [[0.0, 0.0, 0.0], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]],
        [[float("nan"), 0.0, 0.0], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]],
        [[1.1, -0.1, 0.0], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]],
        [[0.9, 0.1, 0.0], [0.9, 0.1, 0.0], [0.9, 0.1, 0.0]],
    ],
)
def test_invalid_transition_matrices_are_rejected(config, matrix):
    """Transition probabilities must be valid and allow every model to receive mass."""
    config["imm"]["transition_matrix"] = matrix

    with pytest.raises(ValueError):
        IMMEKF(config)


def test_log_likelihood_update_stays_normalized_for_large_residual(config):
    """Log-space likelihood normalization handles very unlikely observations."""
    predictor = IMMEKF(config)
    predictor.states = [np.zeros(7, dtype=float) for _ in predictor.models]
    predictor.covariances = [
        predictor.initial_covariance.copy() for _ in predictor.models
    ]

    predictor._update(make_observation(az_deg=10_000.0, el_deg=-10_000.0))

    assert np.all(np.isfinite(predictor.probabilities))
    assert np.all(predictor.probabilities >= 0.0)
    assert predictor.probabilities.sum() == pytest.approx(1.0)


def test_process_noise_is_symmetric_positive_semidefinite(config):
    """The configured diagonal process-noise approximation is PSD."""
    predictor = IMMEKF(config)

    for model in EXPECTED_MODELS:
        noise = predictor._process_noise(0.5, model)

        assert noise.shape == (7, 7)
        assert np.allclose(noise, noise.T)
        assert np.linalg.eigvalsh(noise).min() >= -1e-12


@pytest.mark.parametrize("dt", [-0.1, float("nan"), float("inf")])
def test_process_noise_rejects_invalid_dt(config, dt):
    """Process noise rejects invalid time intervals."""
    predictor = IMMEKF(config)

    with pytest.raises(ValueError, match="dt must be finite"):
        predictor._process_noise(dt, "constant_velocity")


# Prediction accuracy and azimuth wrap-around


def make_linear_track(
    *,
    az0,
    el0=2.0,
    az_rate=3.0,
    el_rate=1.0,
    count=30,
    dt_s=0.1,
    wrap=None,
):
    """Build a noiseless constant-velocity track, optionally wrapping azimuth."""
    track = []
    for index in range(count):
        t = index * dt_s
        az = az0 + az_rate * t
        if wrap is not None:
            az = wrap(az)
        track.append(
            make_observation(
                t_ns=1_000_000_000 + round(t * 1e9),
                az_deg=az,
                el_deg=el0 + el_rate * t,
                az_rate_dps=az_rate,
                el_rate_dps=el_rate,
            )
        )
    return tuple(track)


def wrap_signed(angle):
    """Wrap an angle into [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


def wrap_unsigned(angle):
    """Wrap an angle into [0, 360)."""
    return angle % 360.0


def test_noiseless_constant_velocity_track_matches_analytic_offsets(config):
    """Success path: a straight, steady track is extrapolated exactly."""
    prediction = IMMEKF(config).predict(make_linear_track(az0=5.0))

    for horizon, d_az, d_el in zip(
        prediction.horizons_s,
        prediction.delta_az_deg,
        prediction.delta_el_deg,
        strict=True,
    ):
        assert d_az == pytest.approx(3.0 * horizon, abs=1e-6)
        assert d_el == pytest.approx(1.0 * horizon, abs=1e-6)


@pytest.mark.parametrize(
    ("az0", "az_rate", "wrap"),
    [
        (178.0, 3.0, wrap_signed),  # crosses +180 -> -180
        (-178.0, -3.0, wrap_signed),  # crosses -180 -> +180
        (358.0, 3.0, wrap_unsigned),  # crosses 360 -> 0
        (-5.0, 3.0, wrap_unsigned),  # starts at 355 and crosses 360 -> 0
    ],
)
def test_azimuth_wrap_crossing_does_not_change_the_prediction(
    config, az0, az_rate, wrap
):
    """Boundary: wrapping the azimuth convention must not alter the offsets."""
    wrapped_track = make_linear_track(az0=az0, az_rate=az_rate, wrap=wrap)
    # Guard the premise: the wrapped track really jumps by about a full turn.
    jumps = np.abs(np.diff([item.az_deg for item in wrapped_track]))
    assert jumps.max() > 180.0

    wrapped = IMMEKF(config).predict(wrapped_track)
    unwrapped = IMMEKF(config).predict(make_linear_track(az0=az0, az_rate=az_rate))

    np.testing.assert_allclose(wrapped.delta_az_deg, unwrapped.delta_az_deg, atol=1e-9)
    np.testing.assert_allclose(wrapped.delta_el_deg, unwrapped.delta_el_deg, atol=1e-9)
    np.testing.assert_allclose(wrapped.sigma_deg, unwrapped.sigma_deg, atol=1e-9)
    for horizon, d_az in zip(wrapped.horizons_s, wrapped.delta_az_deg, strict=True):
        assert d_az == pytest.approx(az_rate * horizon, abs=1e-6)


def test_wrap_crossing_in_the_input_history_is_not_mutated(config):
    """The caller's history keeps its wrapped values after prediction."""
    history = make_linear_track(az0=178.0, wrap=wrap_signed)
    before = tuple(item.az_deg for item in history)

    IMMEKF(config).predict(history)

    assert tuple(item.az_deg for item in history) == before


# Analytic Jacobians


def numerical_jacobian(state, dt, model, epsilon=1e-6):
    """Central finite differences of ``_transition``, used as a reference."""
    result = np.zeros((7, 7))
    for column in range(7):
        offset = np.zeros(7)
        offset[column] = epsilon
        upper = _transition(state + offset, dt, model)
        lower = _transition(state - offset, dt, model)
        result[:, column] = (upper - lower) / (2.0 * epsilon)
    return result


@pytest.mark.parametrize("model", EXPECTED_MODELS)
@pytest.mark.parametrize("turn_rate", [0.0, 1e-3, -0.5, 3.0, -20.0, 50.0])
@pytest.mark.parametrize("dt", [0.01, 0.1, 2.0])
def test_analytic_jacobian_matches_finite_differences(model, turn_rate, dt):
    """Success path: the analytic Jacobian agrees with the transition."""
    state = np.array([10.0, 5.0, 3.0, -2.0, 0.5, -0.25, turn_rate])

    analytic = _jacobian(state, dt, model)
    reference = numerical_jacobian(state, dt, model)

    scale = max(1.0, float(np.max(np.abs(analytic))))
    np.testing.assert_allclose(analytic / scale, reference / scale, atol=1e-6)


def test_coordinated_turn_jacobian_keeps_turn_rate_coupling_at_zero():
    """Boundary: at zero turn rate the position still depends on turn rate.

    The first-order coupling is rate * dt**2 / 2, converted from degrees per
    second to radians per second. A numerical Jacobian loses it to rounding.
    """
    dt = 0.1
    state = np.array([0.0, 0.0, 3.0, 2.0, 0.0, 0.0, 0.0])
    to_rad = np.pi / 180.0

    jacobian = _jacobian(state, dt, "coordinated_turn")

    assert jacobian[1, 6] == pytest.approx(3.0 * dt**2 / 2.0 * to_rad, rel=1e-9)
    assert jacobian[0, 6] == pytest.approx(-2.0 * dt**2 / 2.0 * to_rad, rel=1e-9)


@pytest.mark.parametrize(
    ("state", "dt", "model", "message"),
    [
        (np.zeros(7), 0.1, "unknown_model", "Unknown motion model"),
        (np.zeros(7), -0.1, "constant_velocity", "dt must be finite"),
        (np.zeros(7), float("nan"), "constant_velocity", "dt must be finite"),
        (np.zeros(6), 0.1, "constant_velocity", "seven finite values"),
        (np.full(7, np.nan), 0.1, "constant_velocity", "seven finite values"),
    ],
)
def test_jacobian_rejects_invalid_inputs(state, dt, model, message):
    """Failure path: the Jacobian validates like the transition does."""
    with pytest.raises(ValueError, match=message):
        _jacobian(state, dt, model)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_turn_coefficients_are_continuous_at_the_series_boundary(sign):
    """Boundary: switching from series to closed form leaves no jump."""
    inside = _turn_coefficients(sign * _SMALL_ANGLE_RAD * (1.0 - 1e-9))
    outside = _turn_coefficients(sign * _SMALL_ANGLE_RAD * (1.0 + 1e-9))

    np.testing.assert_allclose(inside, outside, rtol=1e-6, atol=1e-9)


def test_coordinated_turn_transition_is_continuous_through_zero_turn_rate():
    """Boundary: tiny turn rates behave like straight-line motion."""
    state = np.array([0.0, 0.0, 3.0, 2.0, 0.0, 0.0, 0.0])
    straight = _transition(state, 0.5, "coordinated_turn")

    for turn_rate in (1e-9, -1e-9, 1e-4, -1e-4):
        perturbed = state + np.eye(7)[6] * turn_rate
        turning = _transition(perturbed, 0.5, "coordinated_turn")
        np.testing.assert_allclose(turning[:4], straight[:4], atol=1e-3)
