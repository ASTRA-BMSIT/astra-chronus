# Astra Chronus

**Counter-UAS detect, track, predict, and lock pipeline.** Research project of the
Astra defence-tech club, BMSIT: from a single camera feed, find one drone, track
it, predict where it will be up to 2 seconds ahead, and keep a pan-tilt camera
locked on it.

This repo is the **V1 plan and architecture contract** the team builds from. When
a decision changes, update this file in the same pull request.

**What you'll find here:**

- **The V1 plan and pipeline** - scope, the six pipeline stages, loss and reward
  design, data plan, metrics, timeline, and roles (sections 1-16 below).
- **Frozen interfaces** - the four dataclasses (`Detection`, `TrackState`,
  `Prediction`, `GimbalCommand`) every sub-team codes against (section 4).
- **Research plan** - literature, venues, authorship, and the iDEX/DRDO route
  ([docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md), plus its PDF).
- **Honesty records** - the shared [experiment log](docs/EXPERIMENT_LOG.md),
  [dataset licenses](docs/DATA_LICENSES.md), and the
  [V0 corrigendum](docs/V0_CORRIGENDUM.md) correcting the earlier research
  report's "verified" claim with re-run measurements.
- **Contributor guide and CI** - how to claim work, branch and commit style, and
  the automated review that gates every PR
  ([CONTRIBUTING.md](CONTRIBUTING.md), `.coderabbit.yaml`, CI).

**Status:** V1 planning. No code in this repo yet.
**Prior work (V0):** [Astra-chronus-ai](https://github.com/reyyishreyas/Astra-chronus-ai-)
is the earlier interception simulator (radar-telemetry simulation, 2-layer GRU
trajectory predictor, FastAPI + React console). It stays as a reference. This
repo is the V1 rebuild around a real camera feed.
**Related:** [astra-uav](https://github.com/reyyishreyas/astra-uav) is the club's
separate GPS-denied navigation track. Do not mix the two.

---

## Architecture Overview

The full chain, from camera to gimbal lock. Interception exists only in
simulation.

```text
Camera Frame
    ↓
Detection (fine-tuned YOLO-family, small-object cascade)
    ↓  Detection (bbox, confidence, t_ns)
Tracking (ByteTrack / OC-SORT + Kalman)
    ↓
State Estimation (pixels → azimuth/elevation + gimbal encoders → world frame)
    ↓  TrackState (az, el, rates, bbox_scale, confidence)
IMM-EKF Baseline ──→ Residual Predictor (GRU/TCN) ──→ Prediction
  (constant velocity,               (delta_az, delta_el,
   constant accel,                    sigma per horizon)
   coordinated turn)
    ↓
Control (PID/MPC + measured-latency compensation)
    ↓  GimbalCommand (latency-compensated setpoints)
Pan-Tilt Gimbal Lock
    ↘
      Intercept path planning — SIMULATION ONLY (PX4/Gazebo or AirSim)
```

The stage-by-stage rationale is in [section 2](#2-pipeline-architecture); the
frozen dataclasses each arrow carries are in
[section 4](#4-interfaces-week-1-contract). The rule that shapes everything:
**always build the classical baseline first** (Kalman/IMM) - a learned model only
counts as a contribution if it beats a properly tuned one.

---

## Getting Started

This repo is in **V1 planning**: the plan and interfaces are here, and code lands
starting with the evaluation harness (see
[section 15](#15-first-two-weeks)). Until then, getting started means getting
oriented and picking up a task.

**1. Clone and read the plan.**

```bash
git clone https://github.com/ASTRA-BMSIT/astra-chronus.git
cd astra-chronus
```

Then skim this README (the contract) and
[docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md) (the paper route).

**2. Pick up work.** Read [CONTRIBUTING.md](CONTRIBUTING.md), then claim an issue
or start from
[Good first contributions](CONTRIBUTING.md#good-first-contributions) - verifying
a dataset license or filling the experiment log needs no code.

**3. Before you push, run the gates.** Once you've added code, every PR must pass
ruff, bandit, and pytest; CI runs the same
([`.github/workflows/ci.yml`](.github/workflows/ci.yml)):

```bash
ruff check . && ruff format --check .
bandit -r src -ll -q
pytest tests/ -q
```

CI skips these on the docs-only tree today and activates them the moment `src/`
and `tests/` exist. Python 3.10+; the full environment spec lands with the first
code PR.

---

## 1. V1 goal

From a single camera feed: detect one drone, track it, predict its position
0.5 to 2 seconds ahead, and keep a pan-tilt camera locked on it.

The full chain:

```
detect -> track -> predict -> lock -> intercept (simulation only)
```

**In V1**

- One target at a time, camera only, edge hardware (Jetson-class).
- Interception logic runs in a simulator only. On real hardware the "lock" is
  the gimbal holding the target at frame centre. No payload, no engagement.

**Explicitly out of V1**

- Swarms, radar or RF sensing, jamming, spoofing, or anything that engages a
  real aircraft.

The simulation-only boundary keeps the project legally safe, makes success
measurable, and still produces a full research story.

---

## 2. Pipeline architecture

| Stage | What it does | V1 approach | Why |
|---|---|---|---|
| **Detection** | Find a small, fast drone, sometimes a few pixels wide | Fine-tune a pretrained YOLO-family detector (COCO weights); RT-DETR as comparison | Data-hungry problem, low-level features transfer. Nobody trains detectors from scratch |
| **Tracking** | Keep identity across frames through clutter | ByteTrack or OC-SORT + Kalman filter, no re-ID needed for single target | Proven, lightweight |
| **State estimation** | Smooth noisy observations into a state | IMM-EKF (constant velocity, constant acceleration, coordinated-turn models) | The baseline every ML model must beat |
| **Prediction** | Estimate position 0.25, 0.5, 1, 2 s ahead | Train small from scratch: residual GRU/TCN on top of IMM | Research contribution. No pretrained model exists for this, and the model is tiny |
| **Lock (control)** | Point the sensor where the target will be | PID or MPC with latency compensation | Predictable, debuggable |
| **Intercept** | Chase path in simulation | Proportional navigation; RL (PPO/SAC) as stretch goal | Classical baseline first |

**Rule:** always build the classical baseline first (Kalman/IMM). Any learned
model only counts as a contribution if it beats that baseline.

### Design at a glance: fine-tune, from scratch, or classical

| Stage | Approach | Reason |
|---|---|---|
| Detection | Fine-tune pretrained | Needs millions of labels otherwise |
| Tracking | Classical, no training | ByteTrack/OC-SORT + Kalman already works |
| State estimation | Classical IMM-EKF | Baseline and fallback |
| Maneuver prediction | From scratch, small (about 100K to 2M parameters, runs in under a millisecond) | The research contribution |
| Control | Classical PID/MPC in V1 | Predictable, debuggable |
| Intercept policy | Proportional navigation, optional RL later | RL is a stretch goal |

---

## 3. Stage details

### 3.1 Detection

- Candidates: YOLO-family small/nano as main, RT-DETR as comparison.
  **Check current versions and licenses before committing. Ultralytics models
  are AGPL-licensed, which matters if we ever publish or productize.**
- Training recipe: start from COCO weights, fine-tune on public anti-UAV
  datasets then on club footage, add hard negatives (birds, kites, planes,
  distant lights) to cut false positives.
- Small-object tricks (drones are often a few pixels wide):
  - higher input resolution (for example 1280),
  - an extra high-resolution detection head,
  - tiled inference for wide frames,
  - temporal input: stack 2 to 3 frames or add a frame-difference channel so
    motion separates a drone from clutter. Keep this as an ablation.
- **Cascade for latency:** full-frame detector at a low rate to acquire the
  target, then a cropped region-of-interest at full camera rate. Small crops
  are fast.
- Export to TensorRT FP16 (INT8 later) for Jetson.

### 3.2 Tracking and state estimation

- ByteTrack or OC-SORT for frame-to-frame association.
- Convert pixels to **angles** using camera calibration, then work in azimuth,
  elevation, and angular rates.
- The camera sits on a moving gimbal: **add gimbal encoder angles to get
  world-frame angles.** Skipping this is the most common mistake - the model
  would otherwise learn the gimbal's motion instead of the drone's.
- Run an **IMM-EKF** with constant-velocity, constant-acceleration, and
  coordinated-turn models. This is the benchmark.
- Range from bounding-box size and known drone dimensions at first; a second
  camera for stereo later.

### 3.3 Prediction (the research core)

- **Input:** last 1 to 3 s of world-frame angles, angular rates, bbox scale,
  detection confidence, at camera rate.
- **Output:** future angular offsets at 0.25, 0.5, 1, and 2 s, plus
  uncertainty per horizon.
- Architecture options, simplest first:
  1. **Residual GRU or TCN on top of IMM** - the network predicts only the
     correction to the Kalman prediction, so it learns what physics misses.
     This is the main model.
  2. **Small Transformer encoder** - an ablation; expect it to need more data.
  3. **Mixture density head** - for multimodal output. An evasive drone can go
     left or right; a single-point prediction averages both into a useless
     middle.
- Training data, in order of volume:
  1. procedural trajectories (waypoints, minimum-snap curves, random aggressive
     and evasive maneuvers) - unlimited with ground truth,
  2. physics sim (PX4 with Gazebo, AirSim, or Isaac Sim),
  3. real club flights with GPS and IMU ground truth, for fine-tuning and
     testing.
- **Critical rule:** train on tracks passed through the real detector and
  tracker, or on synthetically injected noise (jitter, missed frames, ID
  flips). Training on clean ground truth collapses on real detector output.

### 3.4 Control and intercept

- Gimbal control: PID or MPC, aimed at where the target will be **after the
  measured end-to-end latency**, not where it is now.
- Intercept in simulation starts with proportional navigation. RL (PPO or SAC)
  is a later extension, compared against that baseline.

---

## 4. Interfaces (week-1 contract)

These dataclasses are the contract between sub-teams. Change them only by PR
that updates this file.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Detection:
    frame_id: int
    t_ns: int                    # monotonic clock, nanoseconds
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    confidence: float            # 0..1
    class_id: int                # 0 = drone

@dataclass(frozen=True)
class TrackState:
    track_id: int
    t_ns: int
    az_deg: float                # world-frame azimuth
    el_deg: float                # world-frame elevation
    az_rate_dps: float           # degrees per second
    el_rate_dps: float
    bbox_scale: float            # sqrt(area) in pixels, range proxy
    confidence: float

@dataclass(frozen=True)
class Prediction:
    track_id: int
    t_ns: int
    horizons_s: tuple[float, ...]       # (0.25, 0.5, 1.0, 2.0)
    delta_az_deg: tuple[float, ...]     # offsets from current az
    delta_el_deg: tuple[float, ...]
    sigma_deg: tuple[float, ...]        # per-horizon uncertainty

@dataclass(frozen=True)
class GimbalCommand:
    t_ns: int                    # when this command should take effect
    az_deg: float                # setpoint, latency-compensated
    el_deg: float
```

Conventions:

- Units: degrees, degrees per second, nanoseconds. Never mix pixel space and
  angle space inside one field.
- World-frame angles = pixel-derived offset + gimbal encoder angle. Every
  consumer reads world-frame only.
- Latency: each stage records `t_ns`; the control stage consumes
  `t_ns + measured_latency`. The latency budget is measured from day one
  (Section 10), not assumed.

---

## 5. Loss and reward design

**Predictor (supervised - this is where reward-shaped penalties live):**

- Ground truth future position exists for every training track, so supervised
  learning beats reinforcement learning here: simpler, faster, more stable.
- **Distance-proportional penalty.** The further the predicted point is from
  the true one, the larger the penalty, measured as angular error (what the
  gimbal cares about), not raw coordinates.
- No plain hit/miss reward: it gives no gradient about *how close* the
  prediction was.
- **Huber loss** (quadratic for small errors, linear for large) so one wild
  miss does not dominate training.
- **Weight by horizon:** 0.25 s and 2 s are not equally hard or equally
  important. Short horizons drive lock quality, long ones drive intercept
  planning. Without weighting, the 2 s error dominates and ruins short-term
  accuracy.
- **Uncertainty-aware NLL (second step):** the model also predicts how sure it
  is. Confident and wrong is punished harder than uncertain and wrong, and the
  controller can widen its search when the predictor is unsure.
- **Baseline-relative weighting:** predicting "the drone stays still" scores
  well on smooth flight. Weight loss higher on maneuver segments where the IMM
  baseline fails - that is where the contribution lives.

**Controller (RL, simulation only, stretch goal):**

- Positive for each frame the target stays near image centre, negative
  proportional to distance from centre, small penalties for jerky motion and
  control effort, large bonus for a successful simulated intercept. This is
  PPO/SAC training.
- **Reward hacking is the risk** (for example zooming out until everything
  looks "near centre"). Guard by testing on held-out scenarios and checking
  real metrics, never the reward alone.

**No online learning in V1.** Field updates can drift or collapse the model
silently on noisy detections. Instead log predictions and outcomes during
field tests and retrain offline. The logs become the real-flight fine-tune set.

---

## 6. Data plan

| Source | Use | Notes |
|---|---|---|
| Public datasets: Anti-UAV, DUT Anti-UAV, Drone-vs-Bird, MAV-VID | Train and study | **Verify availability and licenses before use** |
| Simulation: PX4 + Gazebo, AirSim, or Isaac Sim | Unlimited target trajectories with ground truth (smooth, aggressive, evasive) | |
| Own footage: club drones in a permitted open area | Fine-tune and test | Log GPS and IMU next to the video as ground truth |

Dataset licenses go in `docs/DATA_LICENSES.md` as they are confirmed. No
dataset enters training until its license entry exists.

---

## 7. Evaluation harness and metrics

Decide these now, measure them the same way every run. The harness is the
first thing to build: it makes the paper credible and the demo honest.

| Stage | Metric |
|---|---|
| Detection | Precision and recall versus distance and target size |
| Tracking | Track continuity, ID switches |
| Prediction | ADE and FDE (average / final displacement error, in degrees) at 0.5, 1, and 2 s, against the Kalman/IMM baseline |
| Lock | Percentage of time the target stays within X pixels of centre (X fixed before testing), plus end-to-end latency |
| Sim intercept | Success rate across fixed scenarios |

**Experiment matrix (ablations):**

- Detector: with and without temporal input; with and without tiling.
- Predictor: IMM only vs GRU residual vs Transformer vs mixture density, on
  smooth and evasive trajectories reported separately.
- Training data: sim-only vs sim plus real fine-tuning (measures the
  sim-to-real gap).
- Lock quality vs end-to-end latency, with and without prediction lead.

Fixed splits, honest baselines, everything reproducible from a logged config.
A shared experiment log with every run recorded is mandatory.

**Latency budget:** set the per-stage budget in week 1 and report measured
values from the first integration on. Proposed starting split (to confirm
after measurement): camera frame, detector, tracker, predictor, control, each
measured end to end on the actual Jetson. A lock that lags by even about
150 ms misses a fast drone, so latency is a first-class metric, not a footnote.

---

## 8. Research claim and contributions

The work targets one paper and one defence-style submission from the same
core. The paper claim (one sentence, everything else cut if it does not
support it):

> *A learned residual predictor on top of an IMM-EKF reduces angular
> prediction error on evasive drone maneuvers at 1 to 2 s horizons under
> realistic detector noise, and improves gimbal lock stability on edge
> hardware.*

Planned contributions:

1. A noise-aware training and evaluation protocol (train on tracks passed
   through a real detector, not clean ground truth).
2. A benchmark split separating smooth and evasive trajectories, released
   with code.
3. A residual GRU or TCN predictor with uncertainty, compared against IMM,
   plain GRU, and Transformer.
4. An end-to-end latency study showing how much prediction lead improves lock
   quality on edge hardware.

The full research plan (literature, venues, authorship, iDEX route) is in
[docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md) and its PDF.

---

## 9. Proposed repository structure

```
astra-chronus/
├── README.md                  # this file
├── docs/
│   ├── RESEARCH_PLAN.md       # paper and submission plan
│   ├── RESEARCH_PLAN.pdf
│   ├── EXPERIMENT_LOG.md      # every run: config, seed, results, hardware
│   └── DATA_LICENSES.md       # one entry per dataset, confirmed licenses
├── configs/                   # one YAML per experiment, committed
├── src/astra_chronus/
│   ├── detection/             # detector training + inference + cascade
│   ├── tracking/              # ByteTrack/OC-SORT + calibration + encoders
│   ├── prediction/            # IMM baseline, GRU/TCN, Transformer, MDN
│   ├── control/               # PID/MPC, latency compensation
│   ├── simulation/            # PX4/Gazebo or AirSim hooks, intercept logic
│   └── evaluation/            # harness, metrics, plots
├── scripts/                   # train_*.py, eval_*.py, one entry per task
├── tests/                     # per-module gates, run in CI
├── notebooks/                 # analysis only, no training code
└── data/                      # gitignored; download scripts only
```

Rules: training never lives in notebooks; every run is launched from a
committed config; `data/` is never committed.

---

## 10. Timeline

About six months, currently sketched as **November 2026 to April 2027 -
shift to match the semester** and re-date this table when confirmed.

| Month | Focus |
|---|---|
| 1 | Literature review, dataset and sim setup, define metrics, form the team |
| 2 | Detection and tracking baseline on public data; start writing related work |
| 3 | Kalman/IMM prediction baseline; build the evaluation harness |
| 4 | Learned prediction models, compared against baseline |
| 5 | Hardware integration, gimbal control, latency tuning |
| 6 | Simulated intercept, field tests, paper draft, demo |

Start writing the paper's related work and method sections in month 2, not
month 6. Late writing is how student papers end up thin.

---

## 11. Team structure

Split into sub-teams. Club members raise the project's PRs; assignments are
**provisional until the first team meeting**. External contributors are
welcome - see [CONTRIBUTING.md](CONTRIBUTING.md).

| Sub-team | Scope | Owner |
|---|---|---|
| Project lead | Interfaces between stages, evaluation honesty | Reyyi Shreyas |
| Perception | Detection, small-object work, cascade | open - first meeting |
| Tracking and prediction | ByteTrack, IMM, GRU/Transformer | open - first meeting |
| Simulation and data | PX4/Gazebo, dataset ingestion, licenses | open - first meeting |
| Control and hardware | Gimbal, latency, Jetson deployment | open - first meeting |
| Paper and documentation | Writing, experiment log, submissions | open - first meeting (separate from project lead) |

- **President / project lead:** Reyyi Shreyas - keeps interfaces between
  stages clean and the evaluation honest.
- **Faculty guide:** TBD (formal sign-off pending).
- Work claimed, credited, and reviewed per
  [CONTRIBUTING.md](CONTRIBUTING.md): branch naming, commit style, the
  contributions table, and the AI-disclosure rule for PRs.
- **Authorship rules agreed in writing before month 2:** contribution
  thresholds, author order logic, and the faculty guide's role. This is the
  most common source of club conflict - settle it early.
- Keep a shared experiment log; reviewers will ask.

---

## 12. Legal and safety (non-negotiable)

- Counter-drone systems are tightly regulated in India. Jamming, spoofing, and
  any physical interception of aircraft are not something a student club can
  do without authorization. **Simulation-only for intercept.**
- Fly only the club's own drones, in permitted areas, within DGCA rules, with
  campus administration's written approval.
- Nothing is built or tested that endangers people or property. Brief BMSIT
  management early. Framed as detection, tracking, and research, the project
  is much easier to support, and the college may help reach DRDO or
  defence-tech mentors.
- No dual-use overreach: nothing published that works as an attack guide, and
  no capability claimed that has not been demonstrated.

---

## 13. Risks

| Risk | Why it hurts | Mitigation |
|---|---|---|
| Small-object detection at range | Depends on camera optics as much as the model | Decide lens and resolution early; test with real footage |
| Sim-to-real gap in trajectories | Sim-trained predictor fails on real flights | Noise-aware training; real fine-tune data in month 5 |
| Latency | Undermines good prediction | Budget and measure from day one; cascade detector |
| Dataset licenses | Blocks publication or submission | Verify before use; record in `docs/DATA_LICENSES.md` |
| Scope creep (swarms, RF, second sensor) | V1 never finishes | Camera-only, single-target, written here |
| Weak baselines | Paper gets rejected | Beat a properly tuned IMM, not a strawman |

---

## 14. Open decisions and facts to verify

- [ ] Which GPU and Jetson are actually available (decides detector size).
- [ ] Own flight data available, or V1 is public data plus simulation only.
- [ ] Mixture density head in V1, or after the residual baseline works.
- [ ] Camera-only for V1, or plan a second sensor for later.
- [ ] Goal balance: paper first, demo first, or both.
- [ ] Verify dataset licenses, Ultralytics AGPL terms, iDEX/DRDO eligibility
      rules, and current competition deadlines (they change every year).
- [ ] Confirm timeline dates and faculty sign-off.

---

## 15. First two weeks

1. Faculty guide sign-off and BMSIT's written approval for future flight tests.
2. Authorship rules and sub-team owners, in writing.
3. Interface contracts (Section 4) merged as the first PR.
4. Latency budget numbers fixed.
5. Then build the **evaluation harness** first - baselines and fixed splits.
   Everything else is measured against it.

---

## 16. Contributing

Core team and external contributors: see
[CONTRIBUTING.md](CONTRIBUTING.md) for how work is claimed, the branch and
commit conventions, code standards, and the contributions table that records
who did what.

---

## Contact and community

- **Astra defence-tech club, BMSIT** - this is a club research project. The
  project lead is Reyyi Shreyas (president); the faculty guide is TBD pending
  formal sign-off.
- **Where to ask:** the Astra club team channel for members; open a GitHub
  issue for anything that should be public or that external contributors can
  help with.
- **Before you start something unclear:** ask first. Questions cost minutes;
  wrong assumptions cost weeks.
- **Safety:** this project is simulation-only for interception and follows the
  rules in [section 12](#12-legal-and-safety-non-negotiable). Contributions that
  cross that boundary cannot be accepted, however good the code.
