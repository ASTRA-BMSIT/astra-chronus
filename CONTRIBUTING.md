# Contributing to Astra Chronus

Thank you for contributing! This guide shows how the pipeline works, how to
add to it, and the standards every pull request must meet. Want to pick up
work? Jump to
[How Work Is Claimed and Credited](#how-work-is-claimed-and-credited).

Astra Chronus is a counter-UAS **research** project of the Astra club,
BMSIT: detect, track, predict, and lock a single drone from a camera feed.
Anyone can contribute - the core team raises most pull requests, but external
contributors are welcome for bug fixes, documentation, dataset-license
verification, baselines, and experiments.

All contributions are subject to the safety rules in the
[README](README.md#12-legal-and-safety-non-negotiable): simulation-only for
interception, nothing that engages a real aircraft, nothing that endangers
people or property. **We cannot accept contributions that violate these
rules, however good the code is.**

---

## How the Pipeline Works

### Architecture Overview

```
Camera Frame
    ↓
Detection (fine-tuned YOLO-family, small-object cascade)
    ↓
Detection (bbox, confidence, t_ns)
    ↓
Tracking (ByteTrack / OC-SORT + Kalman)
    ↓
State Estimation (pixels → azimuth/elevation + gimbal encoders → world frame)
    ↓
TrackState (az, el, rates, bbox_scale, confidence)
    ↓
IMM-EKF Baseline ──→ Residual Predictor (GRU/TCN) ──→ Prediction
    (constant velocity,                    (delta_az, delta_el,
     constant accel,                        sigma per horizon)
     coordinated turn)
    ↓
Control (PID/MPC + measured-latency compensation)
    ↓
GimbalCommand (latency-compensated setpoints)
    ↓
Pan-Tilt Gimbal Lock
    ↘
      Intercept path planning — SIMULATION ONLY (PX4/Gazebo or AirSim)
```

### Data Lifecycle

Every stage communicates through the **frozen dataclasses** defined in
[README section 4](README.md#4-interfaces-week-1-contract) and kept in
`src/astra_chronus/interfaces.py`. These are the contract between
sub-teams; a change to any of them is a PR that updates the README in the
same commit.

1. **Detection** (`src/astra_chronus/detection/`)
   Emits `Detection` per frame: bounding box in pixels, confidence,
   monotonic `t_ns`. Runs full-frame at a low rate to acquire, then a cropped
   region-of-interest at full camera rate.

2. **Tracking** (`src/astra_chronus/tracking/`)
   Associates detections into tracks (ByteTrack or OC-SORT). Single target
   in V1, so no re-identification network.

3. **State Estimation** (`src/astra_chronus/tracking/`)
   Converts pixels to angles using calibration, adds gimbal encoder angles,
   and produces `TrackState` in **world-frame azimuth/elevation**. Working in
   pixel space past this point is a bug: the model would learn the gimbal's
   motion instead of the drone's
   ([README section 3.2](README.md#32-tracking-and-state-estimation)).

4. **Prediction** (`src/astra_chronus/prediction/`)
   The IMM-EKF predicts forward; the learned residual model predicts the
   correction to that prediction. Output is a `Prediction`: angular offsets
   at 0.25 / 0.5 / 1 / 2 s with per-horizon uncertainty.

5. **Control** (`src/astra_chronus/control/`)
   Turns the prediction into a `GimbalCommand`, aimed where the target will
   be after the **measured** end-to-end latency, not where it is now.

6. **Simulation** (`src/astra_chronus/simulation/`)
   PX4/Gazebo or AirSim hooks and intercept logic. This package must never
   be imported by control, tracking, detection, or prediction: the
   simulation-only boundary lives in this dependency direction.

---

## Example Flow: One Capture, End to End

All values below are **illustrative** - they show the shape of each payload,
not measurements. Real latencies are measured on the actual hardware from
day one, never assumed
([README section 7](README.md#7-evaluation-harness-and-metrics)).

### Input Frame

```
Camera frame at t_ns = 1730000000000000000, drone at ~40 m, ~12 px wide
```

### Detection Output

```python
Detection(
    frame_id=1042,
    t_ns=1730000000000000000,
    bbox=(912.0, 341.0, 924.0, 353.0),
    confidence=0.87,
    class_id=0,
)
```

### Tracking + State Estimation Output

```python
TrackState(
    track_id=1,
    t_ns=1730000000033000000,   # +33 ms (frame + inference)
    az_deg=112.41,               # world frame: pixel offset + encoder angle
    el_deg=14.08,
    az_rate_dps=-3.2,
    el_rate_dps=0.4,
    bbox_scale=12.0,             # sqrt(area) in pixels, range proxy
    confidence=0.87,
)
```

### Prediction Output

```python
Prediction(
    track_id=1,
    t_ns=1730000000033000000,
    horizons_s=(0.25, 0.5, 1.0, 2.0),
    delta_az_deg=(-0.80, -1.60, -3.19, -6.31),
    delta_el_deg=(0.10, 0.20, 0.41, 0.83),
    sigma_deg=(0.05, 0.11, 0.34, 1.02),   # widens with horizon
)
```

### Gimbal Command Output

```python
GimbalCommand(
    t_ns=1730000000123000000,   # +123 ms: when the gimbal can act
    az_deg=112.12,              # aim point includes latency lead
    el_deg=14.12,
)
```

If a stage cannot produce its output (tracker lost the target, predictor
uncertainty above the configured gate), it fails loudly with a typed
exception - never with a silent default value.

---

## How to Add a New Module

### Step 1: Check the Contract

Read
[README section 4](README.md#4-interfaces-week-1-contract) (frozen
interfaces) and section 3 (your stage's design). If your change needs a new
field on a frozen dataclass, open an issue first and agree it with the
project lead. Do not widen a dataclass in the same PR that consumes it.

### Step 2: Implement in Your Sub-Team Directory

```python
# src/astra_chronus/prediction/residual_gru.py
"""Residual GRU: learns the correction the IMM-EKF misses."""

from astra_chronus.interfaces import Prediction, TrackState


class ResidualGRU:
    """Predict angular-offset corrections on top of the IMM-EKF.

    The IMM is always run first; this model only predicts what physics
    misses ([README section 3.3](README.md#33-prediction-the-research-core)).
    Smooth and evasive trajectories are reported separately, never pooled.

    Parameters
    ----------
    hidden_size : int
        GRU hidden units. Keep the model small - 100K to 2M parameters so
        it runs in under a millisecond on the Jetson (README section 6).
    horizon_weights : tuple of float
        Loss weights for the 0.25, 0.5, 1 and 2 s horizons. Short horizons
        drive lock quality; unweighted, the 2 s error dominates training.

    Attributes
    ----------
    MIN_HISTORY_FRAMES : int
        Minimum track length the model needs, at camera rate.

    Raises
    ------
    ValueError
        If `horizon_weights` does not have one weight per horizon.
    """

    MIN_HISTORY_FRAMES: int = 30

    def __init__(
        self,
        hidden_size: int = 128,
        horizon_weights: tuple[float, ...] = (1.0, 1.0, 0.7, 0.5),
    ) -> None:
        if len(horizon_weights) != 4:
            raise ValueError(
                "horizon_weights needs one weight per horizon: "
                "0.25, 0.5, 1, 2 s"
            )
        self.hidden_size = hidden_size
        self.horizon_weights = horizon_weights

    def predict(self, history: tuple[TrackState, ...]) -> Prediction:
        """Predict future angular offsets for one track.

        Parameters
        ----------
        history : tuple of TrackState
            Last 1 to 3 s of world-frame track, at camera rate. Never
            clean ground truth: pass tracks that went through the real
            detector and tracker, or through injected detector noise
            (jitter, missed frames, ID flips).

        Returns
        -------
        Prediction
            Offsets at 0.25, 0.5, 1 and 2 s with per-horizon uncertainty,
            on the same horizon grid as the IMM baseline.

        Raises
        ------
        ValueError
            If `history` is shorter than `MIN_HISTORY_FRAMES`. Fail loudly;
            do not fall back to a constant-velocity guess.
        """
        if len(history) < self.MIN_HISTORY_FRAMES:
            raise ValueError(
                f"history has {len(history)} frames, "
                f"need at least {self.MIN_HISTORY_FRAMES}"
            )
        ...  # run the IMM, run the GRU on its residual, return corrected offsets
```

### Step 3: Register It in a Committed Config

```yaml
# configs/prediction/residual_gru_v1.yaml
model: residual_gru
hidden_size: 128
horizon_weights: [1.0, 1.0, 0.7, 0.5]
seed: 20261101
split_ref: sim_v1_evasive   # produced by scripts/make_splits.py --seed 20261101
training_data:
  source: sim
  noise_injection: detector_passed   # never clean ground truth
```

Hyperparameters live in `configs/`, one YAML per experiment, committed. A
hardcoded threshold in a source file will be flagged in review. Splits are
**generated** from a committed script and seed - `data/` is gitignored and
never holds committed files
([README section 9](README.md#9-proposed-repository-structure)).

### Step 4: Add Tests

```python
# tests/test_residual_gru.py
import pytest

from astra_chronus.interfaces import TrackState
from astra_chronus.prediction.residual_gru import ResidualGRU


def make_track(n_frames: int) -> tuple[TrackState, ...]:
    """A short constant-speed track, n frames at camera rate."""
    return tuple(
        TrackState(
            track_id=1,
            t_ns=i * 33_000_000,
            az_deg=90.0 - 0.03 * i,
            el_deg=10.0,
            az_rate_dps=-0.9,
            el_rate_dps=0.0,
            bbox_scale=12.0,
            confidence=0.9,
        )
        for i in range(n_frames)
    )


def test_horizon_grid_is_preserved() -> None:
    """The correction keeps the contract's horizon grid."""
    model = ResidualGRU()
    pred = model.predict(make_track(model.MIN_HISTORY_FRAMES))
    assert pred.horizons_s == (0.25, 0.5, 1.0, 2.0)
    assert len(pred.delta_az_deg) == len(pred.horizons_s)


def test_short_history_raises() -> None:
    """Shorter than the model window fails loudly."""
    model = ResidualGRU()
    with pytest.raises(ValueError, match="history"):
        model.predict(make_track(2))


def test_wrong_horizon_weights_raise_at_init() -> None:
    """Constructor validation catches bad configs early."""
    with pytest.raises(ValueError, match="horizon_weights"):
        ResidualGRU(horizon_weights=(1.0, 1.0))


def test_uncertainty_widens_with_horizon() -> None:
    """Boundary: sigma is monotonically non-decreasing in horizon."""
    model = ResidualGRU()
    pred = model.predict(make_track(model.MIN_HISTORY_FRAMES))
    assert all(
        later >= earlier
        for earlier, later in zip(pred.sigma_deg, pred.sigma_deg[1:])
    )
```

Test the success path, the validation failure, and the boundary case. New
behavior without tests blocks the PR.

### Step 5: Run the Checks Locally

```bash
ruff check src tests scripts
ruff format --check src tests scripts
bandit -r src -ll -q
pytest tests/test_residual_gru.py -v
```

CI runs the same three gates
([`.github/workflows/ci.yml`](.github/workflows/ci.yml)). Run the tests
related to your change, not the full suite on every commit.

### Step 6: Run the Experiment and Log It

```bash
python scripts/train_prediction.py --config configs/prediction/residual_gru_v1.yaml
python scripts/eval_prediction.py  --config configs/prediction/residual_gru_v1.yaml
```

Then add a row to the experiment log
([`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md)):

```text
DATE: <YYYY-MM-DD>
CONFIG: configs/prediction/residual_gru_v1.yaml (seed <seed>)
SPLIT: <split_ref> (held out)
RESULT: ADE@1s <deg> | FDE@2s <deg> | IMM baseline ADE@1s <deg>
HARDWARE: <device> | predictor latency <ms> p95
NOTES: <failure cases, lost tracks, anything a reviewer will ask>
```

Numbers without a log row will be changed to `TBD` in review. This rule
exists because the work may become a paper: a number that cannot be traced
to a committed config and seed cannot be published.

### Step 7: Open the Pull Request

- Branch: `<github-username>/<short-description>`
  (e.g. `your-name/residual-gru-predictor`)
- Title: `<type>(<area>): <imperative summary>` -
  `feat(prediction): add residual GRU corrector on IMM baseline`
- Body closes the issue and includes:
  - **What** changed and **why**,
  - **How tested** (which pytest commands, which config run),
  - **Experiment log** row (for anything that moves a metric),
  - **Demo evidence** for anything visible
    (see below),
  - **AI disclosure**: state whether AI tools helped write the code and
    what for. We keep this in the record because the work may become a
    paper.

### Demo Requirement for Anything Visible

Contributors must include proof that their change works. Accepted formats:

- Screenshot of a plot, metrics table, or console output,
- Short demo clip of a simulation run.

The evidence must clearly show:

- The config and seed that ran,
- The metric values, and
- The IMM-EKF baseline next to your numbers.

Expected format:

```text
RUN: eval_prediction.py --config configs/prediction/residual_gru_v1.yaml
STATUS: success
RESULT:
  ADE@1s: <deg>          baseline IMM: <deg>
  FDE@2s: <deg>          baseline IMM: <deg>
  latency p95: <ms>      hardware: <device>
```

---

## Code Quality Standards

### Typing and Documentation

- **Python 3.10+**, type hints on **every** public function signature and
  class attribute.
- NumPy-style docstrings: `Parameters`, `Returns`, `Raises`; classes also
  document `Attributes`.
- Comments explain *why*, not *what*. An uncommented line is preferred over
  a narration comment.
- Lint and format with ruff before pushing: `ruff check .` and
  `ruff format .` must be clean.

### Data Structures and Units

- Frozen dataclasses for anything passed between stages. Mutability across
  a stage boundary is a review blocker. No `dict`-as-record where a
  dataclass expresses the contract.
- Degrees, degrees per second, nanoseconds, everywhere. Pixel space and
  angle space never mix inside one field. Conversions happen once, at the
  calibration boundary, and are tested against a known calibration.
- Calibration constants come from `configs/`, never hardcoded.

### Error Handling

- Typed exceptions (`TrackerLostError`, `CalibrationMissingError`), never
  bare `except Exception:` with a fallback value.
- A lost target is a normal event and is handled explicitly by Control
  (re-acquire or hold), not swallowed in Tracking.
- No mutable module-level state, no hidden globals, no blocking I/O inside
  async functions.

### Honesty (This Is a Research Repo)

- A metric appears only with a committed config, a seed, and an experiment
  log row.
- Baselines are reported honestly. The IMM-EKF must be properly tuned; a
  strawman baseline that our model beats proves nothing.
- Smooth and evasive trajectories are reported separately, never only
  pooled.
- Failure cases are documented, not hidden.

### Testing

- Success path, validation failure, boundary case - minimum for any new
  behavior, in the same PR.
- No wall-clock timing assertions in unit tests; latency is measured by the
  dedicated benchmark script.
- No network access in unit tests; mock external calls.
- Run `pytest tests/test_<module>.py -q` - the tests related to your change.

---

## Development Workflow

### 1. Sync with main

```bash
git fetch origin
git rebase origin/main
```

### 2. Create a Feature Branch

```bash
git checkout -b your-name/residual-gru-predictor
```

### 3. Implement, Test, Log

```bash
pytest tests/test_residual_gru.py -q
ruff check src tests scripts
```

### 4. Commit with a Ticket Reference

```bash
git add src/astra_chronus/prediction/residual_gru.py \
        tests/test_residual_gru.py \
        configs/prediction/residual_gru_v1.yaml \
        docs/EXPERIMENT_LOG.md
git commit -m "feat(prediction): add residual GRU corrector on IMM baseline

- ResidualGRU predicts corrections to the IMM-EKF prediction
- Config carries seed and evasive-split reference
- Tests cover success, short-history failure, horizon weighting

Closes #12"
```

- Short subject, 50 characters or fewer, imperative mood.
- Body wrapped at 72 characters, explaining why (not just what).
- Prefixes: `feat:` `fix:` `enh:` `doc:` `cos:` `test:` `chore:`.
- One logical change per commit; split unrelated work into separate PRs.

### 5. Push and Create the PR

```bash
git push origin your-name/residual-gru-predictor
```

### 6. Review Feedback

Two automated passes run before a human sees the PR:

- **CodeRabbit** (`.coderabbit.yaml`): bug hunt plus project-specific rules -
  simulation-only boundary, frozen-interface drift, unit mixing, clean
  ground-truth training, weak baselines, missing seeds, untraced numbers.
  Treat its comments as a first review: fix what is right, reply on what
  is not.
- **CI**: ruff, bandit, pytest. It stays green on docs-only changes.

Then respond to comments, push fixes, and re-request review. One core-team
approval merges. Reviews aim for the next push, not the same hour; a PR
idle for three days is escalated in the team channel.

---

## Interface Compatibility

The frozen dataclasses are consumed by four sub-teams at once.

1. **Do not remove or rename fields** - that breaks every downstream stage
   in the same commit.
2. **New fields must be optional** with a documented default, or land as a
   dedicated interface PR that updates
   [README section 4](README.md#4-interfaces-week-1-contract).
3. **Deprecations get a warning and one release cycle** before removal.
4. Every consumer of a changed interface is listed in the PR description, so
   the reviewer can check them.
5. Rebase or merge `main` before the final review pass; no force-pushes to
   `main`, ever.

---

## Performance Considerations

- **Latency is a first-class metric.** The per-stage budget is set in week 1
  and measured end to end on the actual Jetson; each stage records `t_ns`
  so the budget can be attributed. A lock that lags by about 150 ms misses a
  fast drone.
- **The predictor runs in under a millisecond** (100K-2M parameters). A
  change that pushes it over is caught in the latency benchmark.
- **The cascade exists for speed**: full-frame detector at low rate, ROI at
  full rate. Do not run full-frame inference per frame without a measured
  justification.
- **The tracker is not the bottleneck**; the detector is. Optimize in that
  order.

---

## How Work Is Claimed and Credited

1. Find or open an **issue** describing the work. Check the README's
   [open decisions](README.md#14-open-decisions-and-facts-to-verify) and the
   first-tasks column in [Core Team](#core-team) before starting something
   new.
2. **Comment on the issue to claim it.** Core team: you get assigned.
   External: you get a "claimed by community" note; first useful PR wins the
   task.
3. Open a **branch** named `<github-username>/<short-description>`, for
   example `your-name/detector-training-script`.
4. Open a **PR against `main`** that closes the issue, following the steps
   above.
5. **AI disclosure:** every PR states whether AI tools were used to write
   the code and what for.
6. On merge, **add your row to the Contributions table** in the same PR or a
   follow-up: your name, what the PR did, its number.

Credit is the table plus git history. There are no other rankings.

---

## Core Team

Club members raise the project's PRs; external contributors are welcome
under the same rules. Workstreams below are placeholders until the first
team meeting - starting assignments, not permanent rules. Members get
listed here with their name and workstream as they claim their first task.

| Member | Role | Workstream | First task |
|---|---|---|---|
| Reyyi Shreyas | President, project lead | Evaluation harness, interface contracts | Land the harness skeleton with fixed splits |
| _TBD - first meeting_ | Perception | Detection: fine-tuning, small-object tricks, cascade | Set up detector training on a public dataset |
| _TBD - first meeting_ | Tracking and prediction | ByteTrack/OC-SORT, IMM baseline, predictor models | Implement the IMM-EKF baseline |
| _TBD - first meeting_ | Simulation and data | PX4/Gazebo or AirSim, dataset ingestion, licenses | Fill `docs/DATA_LICENSES.md` with verified entries |
| _TBD - first meeting_ | Control and hardware | Gimbal, latency, Jetson deployment | Measure the end-to-end latency budget |
| _TBD - first meeting_ | Paper and documentation | Writing, experiment log, submissions | Record the first runs in `docs/EXPERIMENT_LOG.md` |

Faculty guide: **TBD** (sign-off and authorship role per the written
authorship rules, agreed before month 2).

### Contributions Table

Every merged PR adds one row. Written by the PR author, reviewed like code.
`Work done` states what the contribution did, not effort spent.

| Contributor | Work done | PR |
|---|---|---|
| Reyyi Shreyas | V1 architecture README, research plan and PDF, contributor guide, CI and review setup | direct pushes on `main` |
| _Your row here_ | _What your merged PR did_ | _#N_ |

External contributors: same table, same rules. A merged PR is a row; that is
the whole credit system.

---

## Good First Contributions

- Verify a dataset's license and add the entry to
  [`docs/DATA_LICENSES.md`](docs/DATA_LICENSES.md) (Anti-UAV, DUT Anti-UAV,
  Drone-vs-Bird, MAV-VID all need this).
- Log the first runs in [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md)
  once the harness lands.
- Write a test for an existing module that has none.
- Fix docs: broken links, unclear steps, missing rationale.
- Reproduce a baseline (Kalman/IMM) and file the numbers in the experiment
  log via a PR.

Larger contributions (new model, new sensor, new dataset pipeline): open a
design issue first, discuss, then implement. No surprise 2,000-line PRs.

---

## Safety and Compliance

- No code that engages real aircraft, jams, or spoofs. Interception stays in
  simulation, enforced by review and by the CodeRabbit rules.
- No real-flight contributions without documented campus approval and
  DGCA-compliant procedures linked in the PR.
- No capability claims in docs or PR text that have not been demonstrated
  and logged.
- Dataset licenses are recorded in `docs/DATA_LICENSES.md` before a dataset
  is used in training.

See
[README section 12](README.md#12-legal-and-safety-non-negotiable) for the
full rules.

---

## Questions?

- **Where does my change go?** The directory map above; ask in the Astra
  club team channel if still unclear.
- **Can I touch a frozen interface?** Open an issue and agree with the
  project lead first.
- **How do I reproduce a number?** The experiment log row carries the config
  path and seed; the run is `scripts/eval_*.py --config ...`.
- **Review stalled?** Tag the project lead in the team channel; three days
  idle is escalation by default.

Ask before you spend a day on something unclear. Questions cost minutes;
wrong assumptions cost weeks - the project lead's job is keeping interfaces
and evaluation honest, so use that.

---

**Thank you for making Astra Chronus better!**
