"""The frozen interface contract between sub-teams.

A verbatim implementation of the four dataclasses in README section 4. Each
stage hands its output to the next stage as one of these.

Section 4 permits a change only by a pull request that updates the README in
the same change, so this file is copied rather than extended: no validation,
no defaults, no methods.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    frame_id: int
    t_ns: int  # monotonic clock, nanoseconds
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    confidence: float  # 0..1
    class_id: int  # 0 = drone


@dataclass(frozen=True)
class TrackState:
    track_id: int
    t_ns: int
    az_deg: float  # world-frame azimuth
    el_deg: float  # world-frame elevation
    az_rate_dps: float  # degrees per second
    el_rate_dps: float
    bbox_scale: float  # sqrt(area) in pixels, range proxy
    confidence: float


@dataclass(frozen=True)
class Prediction:
    track_id: int
    t_ns: int
    horizons_s: tuple[float, ...]  # (0.25, 0.5, 1.0, 2.0)
    delta_az_deg: tuple[float, ...]  # offsets from current az
    delta_el_deg: tuple[float, ...]
    sigma_deg: tuple[float, ...]  # per-horizon uncertainty


@dataclass(frozen=True)
class GimbalCommand:
    t_ns: int  # when this command should take effect
    az_deg: float  # setpoint, latency-compensated
    el_deg: float
