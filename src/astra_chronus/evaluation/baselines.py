"""Baseline trajectory predictors for counter-UAS evaluation.

Provides the reference baselines every learned predictor must beat:
1. Constant-velocity baseline (the fundamental floor V0 failed to clear).
2. Linear constant-velocity Kalman filter predictor.

Outputs strictly adhere to `astra_chronus.interfaces.Prediction`.
"""

import math
from collections.abc import Sequence

from astra_chronus.interfaces import Prediction, TrackState

DEFAULT_HORIZONS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)


class ConstantVelocityPredictor:
    """Constant-velocity baseline predictor.

    Extrapolates angular position linearly using current angular rates.
    This trivial baseline forms the performance floor in V0 corrigendum
    Finding 1: any learned model that cannot beat this baseline is worse
    than doing nothing clever.

    Parameters
    ----------
    horizons_s : tuple[float, ...], default (0.25, 0.5, 1.0, 2.0)
        The horizon grid in seconds, matching the contract in README section 4.
    sigma_rate_dps : float, default 0.5
        Estimated rate uncertainty standard deviation in degrees per second,
        used to model uncertainty growth with time.

    Attributes
    ----------
    horizons_s : tuple[float, ...]
        Configured prediction horizons.
    sigma_rate_dps : float
        Uncertainty rate scale.

    Raises
    ------
    ValueError
        If `horizons_s` is empty, contains non-positive numbers, or if
        `sigma_rate_dps` is negative.
    """

    def __init__(
        self,
        horizons_s: tuple[float, ...] = DEFAULT_HORIZONS,
        sigma_rate_dps: float = 0.5,
    ) -> None:
        if not horizons_s:
            raise ValueError("horizons_s must contain at least one horizon.")
        if any(h <= 0.0 for h in horizons_s):
            raise ValueError("All horizons must be strictly positive numbers.")
        if sigma_rate_dps < 0.0:
            raise ValueError("sigma_rate_dps must be non-negative.")

        self.horizons_s = horizons_s
        self.sigma_rate_dps = sigma_rate_dps

    def predict(
        self,
        track: TrackState,
        horizons_s: Sequence[float] | None = None,
    ) -> Prediction:
        """Predict angular offsets assuming constant angular velocity.

        Parameters
        ----------
        track : TrackState
            Current track state in world-frame azimuth and elevation.
        horizons_s : Sequence[float] | None, default None
            Evaluation horizons to forecast. If None, uses `self.horizons_s`.

        Returns
        -------
        Prediction
            A frozen prediction holding delta azimuth, delta elevation, and
            uncertainty values for each horizon.

        Raises
        ------
        ValueError
            If custom horizons are empty or non-positive, or if track confidence
            is outside [0.0, 1.0].
        """
        if track.confidence < 0.0 or track.confidence > 1.0:
            raise ValueError(
                f"Track confidence must be in [0.0, 1.0], got {track.confidence}"
            )

        target_horizons = self.horizons_s if horizons_s is None else tuple(horizons_s)
        if not target_horizons:
            raise ValueError("horizons_s must contain at least one horizon.")
        if any(h <= 0.0 for h in target_horizons):
            raise ValueError("All horizons must be strictly positive.")

        delta_az: list[float] = []
        delta_el: list[float] = []
        sigma: list[float] = []

        for h in target_horizons:
            delta_az.append(track.az_rate_dps * h)
            delta_el.append(track.el_rate_dps * h)
            # Uncertainty grows monotonically with forecast horizon
            sigma.append(self.sigma_rate_dps * h)

        return Prediction(
            track_id=track.track_id,
            t_ns=track.t_ns,
            horizons_s=target_horizons,
            delta_az_deg=tuple(delta_az),
            delta_el_deg=tuple(delta_el),
            sigma_deg=tuple(sigma),
        )


class KalmanPredictor:
    """Linear constant-velocity Kalman filter predictor for angle tracking.

    Estimates 2D decoupled [position, velocity] states for azimuth and
    elevation across the trajectory history. Propagates state and error
    covariances through F, H, Q, and R to yield calibrated posterior forecasts.

    Parameters
    ----------
    horizons_s : tuple[float, ...], default (0.25, 0.5, 1.0, 2.0)
        The horizon grid in seconds.
    process_noise : float, default 1.0
        Continuous process noise intensity (acceleration power spectral density).
    measurement_noise : float, default 0.1
        Measurement variance in degrees squared.

    Raises
    ------
    ValueError
        If noise parameters are non-positive or horizons are empty.
    """

    def __init__(
        self,
        horizons_s: tuple[float, ...] = DEFAULT_HORIZONS,
        process_noise: float = 1.0,
        measurement_noise: float = 0.1,
    ) -> None:
        if not horizons_s or any(h <= 0.0 for h in horizons_s):
            raise ValueError("horizons_s must contain positive horizons.")
        if process_noise <= 0.0 or measurement_noise <= 0.0:
            raise ValueError("Noise parameters must be strictly positive.")

        self.horizons_s = horizons_s
        self.q = process_noise
        self.r = measurement_noise

    def _filter_1d(
        self,
        positions: list[float],
        velocities: list[float],
        timestamps_s: list[float],
    ) -> tuple[float, float, tuple[float, float, float]]:
        """Filter a 1D coordinate series [pos, vel] using a Kalman filter.

        Returns
        -------
        tuple[float, float, tuple[float, float, float]]
            Final filtered position, velocity, and covariance elements (P00, P01, P11).
        """
        # Initial state estimate
        p = positions[0]
        v = velocities[0]
        # Initial covariance P
        p00 = self.r
        p01 = 0.0
        p11 = self.q

        r_vel = self.r * 2.0  # Velocity measurement variance

        for k in range(1, len(positions)):
            dt = timestamps_s[k] - timestamps_s[k - 1]
            if dt <= 0.0:
                dt = 1.0 / 30.0

            # 1. State prediction: x_pred = F * x
            p_pred = p + v * dt
            v_pred = v

            # 2. Covariance prediction: P_pred = F * P * F^T + Q
            dt2 = dt * dt
            dt3 = dt2 * dt
            q00 = (dt3 / 3.0) * self.q
            q01 = (dt2 / 2.0) * self.q
            q11 = dt * self.q

            pred_p00 = p00 + 2.0 * dt * p01 + dt2 * p11 + q00
            pred_p01 = p01 + dt * p11 + q01
            pred_p11 = p11 + q11

            # 3. Measurement update: S = P_pred + R
            s00 = pred_p00 + self.r
            s01 = pred_p01
            s10 = pred_p01
            s11 = pred_p11 + r_vel

            det_s = s00 * s11 - s01 * s10
            det_s = max(1e-12, det_s)

            inv_s00 = s11 / det_s
            inv_s01 = -s01 / det_s
            inv_s10 = -s10 / det_s
            inv_s11 = s00 / det_s

            # Kalman gain K = P_pred * S^-1
            k00 = pred_p00 * inv_s00 + pred_p01 * inv_s10
            k01 = pred_p00 * inv_s01 + pred_p01 * inv_s11
            k10 = pred_p01 * inv_s00 + pred_p11 * inv_s10
            k11 = pred_p01 * inv_s01 + pred_p11 * inv_s11

            # Innovation y = z - x_pred
            y_pos = positions[k] - p_pred
            y_vel = velocities[k] - v_pred

            # State update
            p = p_pred + k00 * y_pos + k01 * y_vel
            v = v_pred + k10 * y_pos + k11 * y_vel

            # Covariance update P = (I - K) * P_pred
            p00 = (1.0 - k00) * pred_p00 - k01 * pred_p01
            p01 = (1.0 - k00) * pred_p01 - k01 * pred_p11
            p11 = -k10 * pred_p01 + (1.0 - k11) * pred_p11

        return p, v, (p00, p01, p11)

    def predict(
        self,
        history: Sequence[TrackState],
        horizons_s: Sequence[float] | None = None,
    ) -> Prediction:
        """Filter the trajectory history and predict future angular offsets.

        Parameters
        ----------
        history : Sequence[TrackState]
            Ordered sequence of track states for one target.
        horizons_s : Sequence[float] | None, default None
            Forecast horizons in seconds. If None, uses `self.horizons_s`.

        Returns
        -------
        Prediction
            Predicted angular offsets and uncertainties derived from the
            posterior filter state.

        Raises
        ------
        ValueError
            If `history` is empty.
        """
        if not history:
            raise ValueError("history sequence must not be empty.")

        target_horizons = self.horizons_s if horizons_s is None else tuple(horizons_s)
        last_track = history[-1]

        timestamps = [s.t_ns * 1e-9 for s in history]
        az_pos = [s.az_deg for s in history]
        az_vel = [s.az_rate_dps for s in history]
        el_pos = [s.el_deg for s in history]
        el_vel = [s.el_rate_dps for s in history]

        _, v_az, cov_az = self._filter_1d(az_pos, az_vel, timestamps)
        _, v_el, cov_el = self._filter_1d(el_pos, el_vel, timestamps)

        delta_az: list[float] = []
        delta_el: list[float] = []
        sigma: list[float] = []

        for h in target_horizons:
            delta_az.append(v_az * h)
            delta_el.append(v_el * h)

            # Propagate posterior variance h seconds into the future
            var_az = (
                cov_az[0]
                + 2.0 * h * cov_az[1]
                + (h**2) * cov_az[2]
                + ((h**3) / 3.0) * self.q
            )
            var_el = (
                cov_el[0]
                + 2.0 * h * cov_el[1]
                + (h**2) * cov_el[2]
                + ((h**3) / 3.0) * self.q
            )
            total_sigma = math.sqrt(max(0.0, var_az) + max(0.0, var_el))
            sigma.append(total_sigma)

        return Prediction(
            track_id=last_track.track_id,
            t_ns=last_track.t_ns,
            horizons_s=target_horizons,
            delta_az_deg=tuple(delta_az),
            delta_el_deg=tuple(delta_el),
            sigma_deg=tuple(sigma),
        )
