"""IMM-EKF prediction baseline.

The filter tracks world-frame azimuth and elevation (degrees) with three
motion models (constant velocity, constant acceleration, simplified
coordinated turn) and predicts angular offsets at the contract horizons.

Azimuth is unwrapped across the history window before filtering, so the
result does not depend on whether the upstream convention is [0, 360) or
(-180, 180]. Offsets are relative to the filtered current estimate.
"""

import math
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from astra_chronus.interfaces import Prediction, TrackState

# README section 4 fixes the horizon grid; the committed config must match it.
CONTRACT_HORIZONS_S = [0.25, 0.5, 1.0, 2.0]

# Motion models with a transition and Jacobian defined here.
_MODELS = ("constant_velocity", "constant_acceleration", "coordinated_turn")

# Below this turn angle (omega * dt, radians) the coordinated-turn terms use
# series expansions, because sin/cos forms cancel catastrophically near zero.
_SMALL_ANGLE_RAD = 1e-2


def load_config(config_path: Path) -> dict[str, Any]:
    """Load and validate the basic IMM-EKF configuration.

    Parameters
    ----------
    config_path : Path
        Path to the committed YAML configuration.

    Returns
    -------
    dict[str, Any]
        Parsed configuration values.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    TypeError
        If the file or a required section is not a mapping, or the seed is
        not an integer.
    ValueError
        If required configuration values are missing or invalid.
    """
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration not found: {config_path}")

    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise TypeError("Configuration must be a YAML mapping.")

    required_keys = {"model", "seed", "horizons_s", "motion_models"}
    missing_keys = required_keys - config.keys()
    if missing_keys:
        raise ValueError(f"Missing configuration keys: {sorted(missing_keys)}")

    if config["model"] != "imm_ekf":
        raise ValueError("model must be 'imm_ekf'.")

    if not isinstance(config["seed"], int) or isinstance(config["seed"], bool):
        raise TypeError("seed must be an integer.")

    if config["horizons_s"] != CONTRACT_HORIZONS_S:
        raise ValueError("horizons_s must match the project contract.")

    expected_models = [
        "constant_velocity",
        "constant_acceleration",
        "coordinated_turn",
    ]
    if config["motion_models"] != expected_models:
        raise ValueError("motion_models must contain the three required models.")

    required_sections = {
        "initialization": {
            "min_observations",
            "initial_position_std_deg",
            "initial_rate_std_dps",
            "initial_acceleration_std_dps2",
            "initial_turn_rate_std_dps",
        },
        "noise": {
            "measurement_std_deg",
            "acceleration_std_dps2",
            "turn_rate_std_dps",
        },
        "imm": {"initial_model_probabilities", "transition_matrix"},
    }
    for section, required_fields in required_sections.items():
        values = config.get(section)
        if not isinstance(values, dict):
            raise TypeError(f"Configuration section '{section}' must be a mapping.")
        missing_fields = required_fields - values.keys()
        if missing_fields:
            raise ValueError(f"Missing keys in '{section}': {sorted(missing_fields)}")

    return config


def _validate_state_and_dt(state: np.ndarray, dt: float) -> None:
    """Reject malformed state vectors and invalid time steps."""
    if state.ndim != 1 or state.size != 7 or not np.all(np.isfinite(state)):
        raise ValueError("state must contain seven finite values.")
    if not math.isfinite(dt) or dt < 0:
        raise ValueError("dt must be finite and non-negative.")


def _turn_coefficients(theta: float) -> tuple[float, float, float, float]:
    """Return sinc(t), g(t), sinc'(t) and g'(t) for g(t) = (1 - cos t) / t.

    These are the smooth functions behind the coordinated-turn position
    update and its derivative with respect to turn rate. Series are used for
    small ``theta`` so the limit at zero turn rate is exact.
    """
    if abs(theta) < _SMALL_ANGLE_RAD:
        t2 = theta * theta
        sinc = 1.0 - t2 / 6.0 + t2 * t2 / 120.0
        g = theta * (0.5 - t2 / 24.0 + t2 * t2 / 720.0)
        d_sinc = theta * (-1.0 / 3.0 + t2 / 30.0 - t2 * t2 / 840.0)
        d_g = 0.5 - t2 / 8.0 + t2 * t2 / 144.0
        return sinc, g, d_sinc, d_g

    sin_t = math.sin(theta)
    cos_t = math.cos(theta)
    sinc = sin_t / theta
    g = (1.0 - cos_t) / theta
    d_sinc = (theta * cos_t - sin_t) / theta**2
    d_g = (theta * sin_t - (1.0 - cos_t)) / theta**2
    return sinc, g, d_sinc, d_g


def _transition(
    state: np.ndarray,
    dt: float,
    model: str,
) -> np.ndarray:
    """Propagate the angular state using one motion model.

    The coordinated-turn case is a simplified rotation in the angular
    azimuth/elevation plane, not a full three-dimensional vehicle model.
    """
    _validate_state_and_dt(state, dt)

    # At zero elapsed time, every state component must remain unchanged.
    # This also makes the numerical Jacobian the identity at dt == 0.
    if dt == 0.0:
        return state.copy()

    az, el, az_rate, el_rate, az_acc, el_acc, turn_rate = state

    result = state.copy()

    if model == "constant_velocity":
        result[0] = az + az_rate * dt
        result[1] = el + el_rate * dt
        result[4:] = 0.0

    elif model == "constant_acceleration":
        result[0] = az + az_rate * dt + 0.5 * az_acc * dt**2
        result[1] = el + el_rate * dt + 0.5 * el_acc * dt**2
        result[2] = az_rate + az_acc * dt
        result[3] = el_rate + el_acc * dt
        result[6] = 0.0

    elif model == "coordinated_turn":
        # This is a simplified coordinated turn in the azimuth/elevation
        # angular plane, not a full 3-D physical drone-dynamics model.
        turn_rate_rad_s = math.radians(turn_rate)
        angle = turn_rate_rad_s * dt
        sinc, g, _, _ = _turn_coefficients(angle)
        sin_angle = math.sin(angle)
        cos_angle = math.cos(angle)
        result[0] = az + dt * (az_rate * sinc - el_rate * g)
        result[1] = el + dt * (az_rate * g + el_rate * sinc)
        result[2] = az_rate * cos_angle - el_rate * sin_angle
        result[3] = az_rate * sin_angle + el_rate * cos_angle

        result[4] = -turn_rate_rad_s * result[3]
        result[5] = turn_rate_rad_s * result[2]

    else:
        raise ValueError(f"Unknown motion model: {model}")

    return result


def _jacobian(
    state: np.ndarray,
    dt: float,
    model: str,
) -> np.ndarray:
    """Return the analytic Jacobian of ``_transition`` for one motion model.

    State order is azimuth, elevation, azimuth rate, elevation rate, azimuth
    acceleration, elevation acceleration, turn rate. At ``dt == 0`` the
    transition is the identity, so the Jacobian is too.
    """
    _validate_state_and_dt(state, dt)
    if model not in _MODELS:
        raise ValueError(f"Unknown motion model: {model}")

    jacobian = np.eye(7, dtype=float)
    if dt == 0.0:
        return jacobian

    if model == "constant_velocity":
        jacobian[0, 2] = dt
        jacobian[1, 3] = dt
        jacobian[4:, :] = 0.0  # accelerations and turn rate are reset to zero.

    elif model == "constant_acceleration":
        jacobian[0, 2] = dt
        jacobian[1, 3] = dt
        jacobian[0, 4] = 0.5 * dt**2
        jacobian[1, 5] = 0.5 * dt**2
        jacobian[2, 4] = dt
        jacobian[3, 5] = dt
        jacobian[6, :] = 0.0  # turn rate is reset to zero.

    else:
        az_rate, el_rate = state[2], state[3]
        to_rad = math.pi / 180.0
        omega = math.radians(state[6])
        theta = omega * dt
        sinc, g, d_sinc, d_g = _turn_coefficients(theta)
        sin_t = math.sin(theta)
        cos_t = math.cos(theta)

        # Position rows against the two rates.
        jacobian[0, 2] = dt * sinc
        jacobian[0, 3] = -dt * g
        jacobian[1, 2] = dt * g
        jacobian[1, 3] = dt * sinc
        # Rate rows against the two rates (a rotation).
        jacobian[2, 2] = cos_t
        jacobian[2, 3] = -sin_t
        jacobian[3, 2] = sin_t
        jacobian[3, 3] = cos_t
        # Acceleration rows: a = (-omega * vy', omega * vx'). The previous
        # accelerations are overwritten, so their diagonal terms vanish.
        jacobian[4, 4] = 0.0
        jacobian[5, 5] = 0.0
        jacobian[4, 2] = -omega * sin_t
        jacobian[4, 3] = -omega * cos_t
        jacobian[5, 2] = omega * cos_t
        jacobian[5, 3] = -omega * sin_t
        # Column for the turn rate, per degree per second.
        new_az_rate = az_rate * cos_t - el_rate * sin_t
        new_el_rate = az_rate * sin_t + el_rate * cos_t
        d_az_rate = -dt * new_el_rate
        d_el_rate = dt * new_az_rate
        jacobian[0, 6] = to_rad * dt**2 * (az_rate * d_sinc - el_rate * d_g)
        jacobian[1, 6] = to_rad * dt**2 * (az_rate * d_g + el_rate * d_sinc)
        jacobian[2, 6] = to_rad * d_az_rate
        jacobian[3, 6] = to_rad * d_el_rate
        jacobian[4, 6] = to_rad * (-new_el_rate - omega * d_el_rate)
        jacobian[5, 6] = to_rad * (new_az_rate + omega * d_az_rate)

    return jacobian


class IMMEKF:
    """Interacting multiple-model extended Kalman filter."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the filter from the committed configuration."""
        self.config = config
        self.models = tuple(config["motion_models"])
        self.transition_matrix = np.asarray(
            config["imm"]["transition_matrix"],
            dtype=float,
        )
        self.probabilities = np.asarray(
            config["imm"]["initial_model_probabilities"],
            dtype=float,
        )

        if self.transition_matrix.shape != (3, 3):
            raise ValueError("IMM transition_matrix must be 3 by 3.")

        if self.probabilities.shape != (3,):
            raise ValueError("IMM requires three initial model probabilities.")

        if not np.all(np.isfinite(self.transition_matrix)):
            raise ValueError("Transition-matrix probabilities must be finite.")

        if not np.all(np.isfinite(self.probabilities)):
            raise ValueError("Initial model probabilities must be finite.")

        if (
            np.any(self.transition_matrix < 0)
            or np.any(self.transition_matrix > 1)
            or np.any(self.probabilities < 0)
            or np.any(self.probabilities > 1)
        ):
            raise ValueError("IMM probabilities must be between 0 and 1.")

        if not np.allclose(self.transition_matrix.sum(axis=1), 1.0):
            raise ValueError("Each transition-matrix row must sum to one.")

        if np.any(self.transition_matrix.sum(axis=0) <= 0):
            raise ValueError(
                "Every IMM model must have a possible incoming transition."
            )

        if not np.isclose(self.probabilities.sum(), 1.0):
            raise ValueError("Initial model probabilities must sum to one.")

        if np.any(self.transition_matrix.T @ self.probabilities <= 0):
            raise ValueError(
                "Every model must have a positive predicted probability initially."
            )

        self.probabilities = self.probabilities / self.probabilities.sum()
        self.initial_probabilities = self.probabilities.copy()

        self.states: list[np.ndarray] = []
        self.covariances: list[np.ndarray] = []

        initialization = config["initialization"]
        self.min_observations = initialization["min_observations"]

        stds = np.array(
            [
                initialization["initial_position_std_deg"],
                initialization["initial_position_std_deg"],
                initialization["initial_rate_std_dps"],
                initialization["initial_rate_std_dps"],
                initialization["initial_acceleration_std_dps2"],
                initialization["initial_acceleration_std_dps2"],
                initialization["initial_turn_rate_std_dps"],
            ],
            dtype=float,
        )
        self.initial_covariance = np.diag(stds**2)

        noise = config["noise"]

        if (
            isinstance(self.min_observations, bool)
            or not isinstance(self.min_observations, int)
            or self.min_observations < 1
        ):
            raise ValueError("min_observations must be a positive integer.")

        noise_values = np.array(
            [
                noise["measurement_std_deg"],
                noise["acceleration_std_dps2"],
                noise["turn_rate_std_dps"],
            ],
            dtype=float,
        )

        if (
            not np.all(np.isfinite(stds))
            or not np.all(np.isfinite(noise_values))
            or np.any(stds <= 0)
            or np.any(noise_values <= 0)
        ):
            raise ValueError(
                "Initialization and noise standard deviations must be finite and positive."
            )

        self.measurement_variance = noise_values[0] ** 2
        self.acceleration_variance = noise_values[1] ** 2
        self.turn_rate_variance = noise_values[2] ** 2

        self.measurement_matrix = np.zeros((2, 7), dtype=float)
        self.measurement_matrix[0, 0] = 1.0
        self.measurement_matrix[1, 1] = 1.0
        self.measurement_noise = np.eye(2, dtype=float) * self.measurement_variance

    def _process_noise(self, dt: float, model: str) -> np.ndarray:
        """Construct the configured diagonal process-noise approximation.

        The diagonal terms are provisional heuristics, not calibrated process
        covariances. They should be validated against representative trajectories
        before interpreting ``sigma_deg`` as calibrated uncertainty.
        """
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and non-negative.")
        if model not in self.models:
            raise ValueError(f"Unknown motion model: {model}")

        noise = np.zeros((7, 7), dtype=float)
        noise[2, 2] = self.acceleration_variance * dt**2
        noise[3, 3] = self.acceleration_variance * dt**2
        noise[4, 4] = self.acceleration_variance * dt
        noise[5, 5] = self.acceleration_variance * dt

        if model == "coordinated_turn":
            noise[6, 6] = self.turn_rate_variance * dt

        return noise

    def _mix(self) -> None:
        """Mix model states and covariances using IMM probabilities."""
        previous_probabilities = self.probabilities.copy()
        predicted_probabilities = self.transition_matrix.T @ previous_probabilities

        if np.any(predicted_probabilities <= 0):
            raise ValueError("IMM model probabilities must remain positive.")

        mixed_states = []
        mixed_covariances = []

        for destination in range(len(self.models)):
            weights = (
                self.transition_matrix[:, destination]
                * previous_probabilities
                / predicted_probabilities[destination]
            )
            mixed_state = sum(
                weight * state
                for weight, state in zip(weights, self.states, strict=True)
            )
            mixed_covariance = np.zeros((7, 7), dtype=float)

            for weight, state, covariance in zip(
                weights, self.states, self.covariances, strict=True
            ):
                difference = state - mixed_state
                mixed_covariance += weight * (
                    covariance + np.outer(difference, difference)
                )

            mixed_states.append(mixed_state)
            mixed_covariances.append(mixed_covariance)

        self.states = mixed_states
        self.covariances = mixed_covariances
        self.probabilities = predicted_probabilities

    def _update(self, observation: TrackState) -> None:
        """Update all models with an angular-position measurement."""
        measurement = np.array(
            [observation.az_deg, observation.el_deg],
            dtype=float,
        )
        identity = np.eye(7, dtype=float)
        log_likelihoods = []

        for index, model in enumerate(self.models):
            state = self.states[index]
            covariance = self.covariances[index]

            residual = measurement - self.measurement_matrix @ state
            innovation_covariance = (
                self.measurement_matrix @ covariance @ self.measurement_matrix.T
                + self.measurement_noise
            )

            gain = np.linalg.solve(
                innovation_covariance,
                self.measurement_matrix @ covariance,
            ).T

            state = state + gain @ residual
            correction = identity - gain @ self.measurement_matrix
            covariance = (
                correction @ covariance @ correction.T
                + gain @ self.measurement_noise @ gain.T
            )

            sign, log_determinant = np.linalg.slogdet(innovation_covariance)
            if sign <= 0 or not math.isfinite(log_determinant):
                raise ValueError("Innovation covariance must be positive definite.")

            mahalanobis = residual @ np.linalg.solve(
                innovation_covariance,
                residual,
            )
            log_likelihood = -0.5 * (
                mahalanobis + log_determinant + 2.0 * math.log(2.0 * math.pi)
            )
            if not math.isfinite(float(log_likelihood)):
                raise ValueError("IMM model log-likelihood must be finite.")

            log_likelihoods.append(float(log_likelihood))
            self.states[index] = state
            self.covariances[index] = covariance

        with np.errstate(divide="ignore"):
            log_weights = np.log(self.probabilities) + np.asarray(log_likelihoods)

        maximum = float(np.max(log_weights))
        if not math.isfinite(maximum):
            raise ValueError("Unable to normalize IMM model probabilities.")

        weights = np.exp(log_weights - maximum)
        # Keep every model numerically reachable; this floor is negligible
        # relative to any representable meaningful posterior probability.
        weights = np.maximum(weights, np.finfo(float).tiny)
        total = float(weights.sum())
        if not math.isfinite(total) or total <= 0:
            raise ValueError("Invalid normalized IMM model probabilities.")

        self.probabilities = weights / total

    def _step(self, dt: float, observation: TrackState) -> None:
        """Propagate and update all motion models for one observation."""
        self._mix()

        for index, model in enumerate(self.models):
            state = self.states[index]
            covariance = self.covariances[index]
            jacobian = _jacobian(state, dt, model)

            self.states[index] = _transition(state, dt, model)
            self.covariances[index] = (
                jacobian @ covariance @ jacobian.T + self._process_noise(dt, model)
            )

        self._update(observation)

    def _combined_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the probability-weighted state and covariance."""
        state = sum(
            probability * model_state
            for probability, model_state in zip(
                self.probabilities, self.states, strict=True
            )
        )
        covariance = np.zeros((7, 7), dtype=float)

        for probability, model_state, model_covariance in zip(
            self.probabilities,
            self.states,
            self.covariances,
            strict=True,
        ):
            difference = model_state - state
            covariance += probability * (
                model_covariance + np.outer(difference, difference)
            )

        return state, covariance

    def predict(self, history: tuple[TrackState, ...]) -> Prediction:
        """Estimate a track and predict future angular offsets.

        Each call starts from the supplied history, so repeated calls are
        deterministic and overlapping history does not reuse stale filter
        state. Azimuth is unwrapped across the window before filtering.
        Offsets are measured from the filtered current estimate.

        Parameters
        ----------
        history : tuple[TrackState, ...]
            Time-ordered observations for one track.

        Returns
        -------
        Prediction
            Future angular offsets and uncertainty at each configured horizon.

        Raises
        ------
        ValueError
            If the history is too short, inconsistent, or contains
            non-finite angular measurements.
        """
        if len(history) < self.min_observations:
            raise ValueError(
                f"At least {self.min_observations} observations are required."
            )

        track_id = history[-1].track_id

        if any(item.track_id != track_id for item in history):
            raise ValueError("All history observations must share one track_id.")

        if any(
            current.t_ns <= previous.t_ns for previous, current in pairwise(history)
        ):
            raise ValueError("Observation timestamps must be strictly increasing.")

        if any(
            not all(
                math.isfinite(value)
                for value in (
                    item.az_deg,
                    item.el_deg,
                    item.az_rate_dps,
                    item.el_rate_dps,
                )
            )
            for item in history
        ):
            raise ValueError("Track history must contain finite angular values.")

        # Unwrap azimuth so a track crossing +/-180 (or 0/360) is continuous.
        # Only differences matter, so the angle convention upstream is free.
        unwrapped_az = np.unwrap(
            np.array([item.az_deg for item in history], dtype=float),
            period=360.0,
        )
        history = tuple(
            replace(item, az_deg=float(az))
            for item, az in zip(history, unwrapped_az, strict=True)
        )

        # Reinitialize on every call: the output depends only on this history.
        self.probabilities = self.initial_probabilities.copy()

        first = history[0]
        initial_state = np.array(
            [
                first.az_deg,
                first.el_deg,
                first.az_rate_dps,
                first.el_rate_dps,
                0.0,
                0.0,
                0.0,
            ],
            dtype=float,
        )
        self.states = [initial_state.copy() for _ in self.models]
        self.covariances = [self.initial_covariance.copy() for _ in self.models]

        last_t_ns = first.t_ns
        for observation in history[1:]:
            dt = (observation.t_ns - last_t_ns) / 1e9
            self._step(dt, observation)
            last_t_ns = observation.t_ns

        current_state, _ = self._combined_state()
        horizons = tuple(self.config["horizons_s"])
        delta_az = []
        delta_el = []
        sigma = []

        for horizon in horizons:
            future_states = []
            future_covariances = []

            for index, model in enumerate(self.models):
                state = self.states[index].copy()
                covariance = self.covariances[index].copy()
                jacobian = _jacobian(state, horizon, model)

                future_states.append(_transition(state, horizon, model))
                future_covariances.append(
                    jacobian @ covariance @ jacobian.T
                    + self._process_noise(horizon, model)
                )

            future_state = sum(
                probability * state
                for probability, state in zip(
                    self.probabilities, future_states, strict=True
                )
            )
            future_covariance = np.zeros((7, 7), dtype=float)

            for probability, state, covariance in zip(
                self.probabilities,
                future_states,
                future_covariances,
                strict=True,
            ):
                difference = state - future_state
                future_covariance += probability * (
                    covariance + np.outer(difference, difference)
                )

            delta_az.append(float(future_state[0] - current_state[0]))
            delta_el.append(float(future_state[1] - current_state[1]))
            sigma.append(
                float(
                    math.sqrt(
                        max(
                            0.0,
                            max(
                                future_covariance[0, 0],
                                future_covariance[1, 1],
                            ),
                        )
                    )
                )
            )

        return Prediction(
            track_id=track_id,
            t_ns=history[-1].t_ns,
            horizons_s=horizons,
            delta_az_deg=tuple(delta_az),
            delta_el_deg=tuple(delta_el),
            sigma_deg=tuple(sigma),
        )
