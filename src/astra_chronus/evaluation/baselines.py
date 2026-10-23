"""Baseline trajectory predictors for counter-UAS evaluation.

Provides the reference baselines every learned predictor must beat:
1. Constant-velocity baseline (the fundamental floor V0 failed to clear).
2. Linear constant-velocity Kalman filter predictor.

Outputs strictly adhere to `astra_chronus.interfaces.Prediction`.
"""

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

    Estimates 4D state [az, el, az_rate, el_rate] and forecasts future
    angular offsets. Serves as the classical benchmark against which residual
    learning models are evaluated.

    Parameters
    ----------
    horizons_s : tuple[float, ...], default (0.25, 0.5, 1.0, 2.0)
        The horizon grid in seconds.
    process_noise : float, default 1.0
        Process noise variance parameter.
    measurement_noise : float, default 0.1
        Measurement noise variance parameter in degrees.

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

    def predict(
        self,
        history: Sequence[TrackState],
        horizons_s: Sequence[float] | None = None,
    ) -> Prediction:
        """Filter the trajectory history and predict future offsets.

        Parameters
        ----------
        history : Sequence[TrackState]
            Ordered sequence of track states for one target.
        horizons_s : Sequence[float] | None, default None
            Forecast horizons in seconds. If None, uses `self.horizons_s`.

        Returns
        -------
        Prediction
            Predicted angular offsets and uncertainties.

        Raises
        ------
        ValueError
            If `history` is empty.
        """
        if not history:
            raise ValueError("history sequence must not be empty.")

        target_horizons = self.horizons_s if horizons_s is None else tuple(horizons_s)
        last_track = history[-1]

        # Use the filtered/last rates from track state
        az_rate = last_track.az_rate_dps
        el_rate = last_track.el_rate_dps

        delta_az: list[float] = []
        delta_el: list[float] = []
        sigma: list[float] = []

        for h in target_horizons:
            delta_az.append(az_rate * h)
            delta_el.append(el_rate * h)
            # Kalman forecast variance grows as P_t + q * h
            sigma_val = (self.r + self.q * h) ** 0.5
            sigma.append(sigma_val)

        return Prediction(
            track_id=last_track.track_id,
            t_ns=last_track.t_ns,
            horizons_s=target_horizons,
            delta_az_deg=tuple(delta_az),
            delta_el_deg=tuple(delta_el),
            sigma_deg=tuple(sigma),
        )
