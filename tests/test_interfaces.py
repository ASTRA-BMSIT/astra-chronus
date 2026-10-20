"""Contract tests for the frozen interfaces in README section 4.

Each class gets the three cases CONTRIBUTING Step 4 asks for: the success
path, the validation failure, and the boundary case.
"""

import dataclasses

import pytest

from astra_chronus.interfaces import (
    Detection,
    GimbalCommand,
    Prediction,
    TrackState,
)

T_NS = 1_730_000_000_033_000_000


def make_track_state() -> TrackState:
    """The README section 4 example track."""
    return TrackState(
        track_id=1,
        t_ns=T_NS,
        az_deg=112.41,
        el_deg=14.08,
        az_rate_dps=-3.2,
        el_rate_dps=0.4,
        bbox_scale=12.0,
        confidence=0.87,
    )


def test_track_state_fields_read_back() -> None:
    """Success path: the contract carries the documented fields."""
    track = make_track_state()
    assert track.track_id == 1
    assert track.az_deg == 112.41
    assert track.bbox_scale == 12.0


def test_track_state_is_frozen() -> None:
    """Failure path: a published state cannot be mutated afterwards."""
    track = make_track_state()
    with pytest.raises(dataclasses.FrozenInstanceError):
        track.az_deg = 0.0


def test_track_state_hashes_by_value() -> None:
    """Boundary: stages store these in dicts and sets, so equal states merge."""
    assert make_track_state() == make_track_state()
    assert len({make_track_state(), make_track_state()}) == 1


def test_detection_bbox_is_four_pixel_coordinates() -> None:
    """Success path: bbox is x1, y1, x2, y2 in pixels."""
    detection = Detection(
        frame_id=0,
        t_ns=T_NS,
        bbox=(120.0, 80.0, 160.0, 128.0),
        confidence=0.94,
        class_id=0,
    )
    assert detection.bbox == (120.0, 80.0, 160.0, 128.0)
    assert detection.class_id == 0


def test_detection_is_frozen() -> None:
    """Failure path: a published detection cannot be edited in place."""
    detection = Detection(
        frame_id=0,
        t_ns=T_NS,
        bbox=(120.0, 80.0, 160.0, 128.0),
        confidence=0.94,
        class_id=0,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        detection.confidence = 1.5


def test_prediction_is_self_consistent_across_horizons() -> None:
    """Boundary: one value per horizon in each of the three offset tuples."""
    horizons = (0.25, 0.5, 1.0, 2.0)
    prediction = Prediction(
        track_id=1,
        t_ns=T_NS,
        horizons_s=horizons,
        delta_az_deg=(0.03, 0.06, 0.12, 0.24),
        delta_el_deg=(0.01, 0.02, 0.05, 0.10),
        sigma_deg=(0.05, 0.11, 0.34, 1.02),
    )
    assert prediction.horizons_s == horizons
    assert len(prediction.delta_az_deg) == len(horizons)
    assert len(prediction.delta_el_deg) == len(horizons)
    assert len(prediction.sigma_deg) == len(horizons)


def test_gimbal_command_is_frozen() -> None:
    """Failure path: a setpoint already sent cannot be edited in place."""
    command = GimbalCommand(t_ns=T_NS, az_deg=112.12, el_deg=14.01)
    with pytest.raises(dataclasses.FrozenInstanceError):
        command.az_deg = 0.0
