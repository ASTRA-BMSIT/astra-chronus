# Astra Chronus - Research Plan

**Paper and academic submission plan.** Companion to the team
[README](../README.md). Audience: the Astra club research team, the BMSIT
department, and the faculty guide (TBD).

This document is planning material. It contains no results - every number in
it is a target or a definition, never a measurement. Results enter the
experiment log first, then this file.

---

## 1. Research problem

Small low-altitude drones near a perimeter are hard to counter with camera
systems for three compounding reasons: the target is a few pixels wide at
range, its flight can turn evasive without warning, and the whole pipeline
(detection, inference, gimbal motion) adds latency that a naive
point-at-where-it-is controller cannot absorb.

**Research gap:** classical trackers (Kalman / IMM) predict well on smooth
flight and degrade sharply on sudden maneuvers, because they assume fixed
motion models. Learned sequence models can capture maneuvers, but published
evaluations typically train them on clean ground-truth trajectories, which
does not match what a real detector produces.

**Research questions:**

1. Does a small learned residual predictor on top of an IMM-EKF reduce
   angular prediction error on evasive maneuvers at 0.5-2 s horizons, under
   realistic detector noise, versus IMM alone?
2. Does that improvement translate into measurable gimbal lock stability on
   edge hardware?
3. How large is the sim-to-real gap when a predictor trained in simulation
   is fine-tuned on a small set of real flights?
4. How does prediction lead trade off against end-to-end latency?

---

## 2. The claim (one sentence)

> *A learned residual predictor on top of an IMM-EKF reduces angular
> prediction error on evasive drone maneuvers at 1 to 2 s horizons under
> realistic detector noise, and improves gimbal lock stability on edge
> hardware.*

Everything in the paper must serve this sentence. Contributions we can list:

1. A noise-aware training and evaluation protocol (train on tracks passed
   through a real detector, not clean ground truth).
2. A benchmark split separating smooth and evasive trajectories, released
   with code.
3. A residual GRU or TCN predictor with uncertainty, compared against IMM,
   plain GRU, and Transformer.
4. An end-to-end latency study showing how prediction lead improves lock
   quality on edge hardware.

---

## 3. Method summary

| Component | Approach | Role in the paper |
|---|---|---|
| Detection | Fine-tuned YOLO-family detector, small-object tricks, cascade | Source of realistic noise |
| Tracking | ByteTrack / OC-SORT + camera calibration + gimbal encoders | Produces world-frame angle tracks |
| State estimation | IMM-EKF (constant velocity, constant acceleration, coordinated turn) | **Baseline to beat** |
| Prediction | Residual GRU/TCN on IMM output; Transformer ablation; mixture density head for multimodal futures | Main contribution |
| Control | PID/MPC with latency compensation | Deployment metric (lock stability) |
| Interception | Proportional navigation, simulation only | Secondary demo, not a paper claim |

Loss: Huber, horizon-weighted (0.25 / 0.5 / 1 / 2 s), plus NLL uncertainty
term as the second training step. Model size target: 100K-2M parameters,
sub-millisecond inference. Full design in [README Section 5](../README.md#5-loss-and-reward-design).

---

## 4. Experimental protocol

### Datasets

- Public: Anti-UAV, DUT Anti-UAV, Drone-vs-Bird, MAV-VID.
  **Verify availability, terms, and licenses before any training run; record
  each confirmed license in `DATA_LICENSES.md`.**
- Simulation: PX4 + Gazebo or AirSim - unlimited labeled trajectories
  (smooth, aggressive, evasive).
- Own footage: club drones in permitted areas with GPS/IMU ground truth,
  for fine-tuning and the sim-to-real experiment.

### Splits and hygiene

- Split by trajectory or flight session, never by frame (frames from one
  flight leak across a random split).
- Separate smooth and evasive subsets; report them separately, never only a
  pooled average.
- The evaluation harness consumes fixed, committed splits. Changing a split
  is a versioned change.

### Baselines

1. Constant velocity Kalman filter.
2. IMM-EKF (the primary baseline).
3. Persistence predictor ("stays where it is") - the lazy baseline that
   exposes easy smooth-flight scores.
4. Plain GRU on track history (no residual connection).

### Metrics

- **Prediction:** ADE and FDE in degrees at 0.5, 1, and 2 s horizons,
  smooth and evasive reported separately, with the delta over IMM.
- **Detection:** precision / recall versus distance and target size.
- **Tracking:** track continuity, ID switches.
- **Lock:** percent of time target within a fixed pixel radius of centre;
  end-to-end latency.
- **Calibration (if MDN/uncertainty lands):** coverage of predicted
  uncertainty intervals.
- Report spread (bootstrap confidence intervals) on headline numbers, not
  single runs. Multiple seeds for learned models.

### Ablation matrix

| Axis | Variants |
|---|---|
| Predictor | IMM only / GRU residual / Transformer / mixture density |
| Trajectory type | smooth / evasive (always split) |
| Training data | sim only / sim + real fine-tune |
| Input quality | clean ground truth vs detector-passed tracks |
| Detector | temporal input on/off, tiling on/off |
| Control | prediction lead on/off vs latency |

---

## 5. Timeline aligned with the project

Months match the README timeline (Nov 2026 - Apr 2027, re-date when
confirmed):

| Month | Research milestone |
|---|---|
| 1 | Literature review complete, datasets cleared, metrics frozen |
| 2 | Related work + method sections started (writing begins here) |
| 3 | Baselines run, harness public in the repo |
| 4 | Learned models + ablations run, tables drafted |
| 5 | Field data collected, sim-to-real experiment, latency study |
| 6 | Paper draft, technical report, submission targets chosen |

---

## 6. Publication route

- **arXiv first**, to establish priority, then:
- **Workshops** (robotics / vision / tracking / UAV perception / edge AI)
  are the realistic first target for a first paper. A journal extension can
  follow with the field results.
- Anti-UAV style challenges running alongside major conferences are another
  venue: the harness built here is directly reusable.
- **Check current deadlines yourself** - they shift every year. Working
  backwards from a spring 2027 submission fits the six-month plan.

Reviewers will want: a reusable evaluation setup, honest failure cases, and
baselines that were properly tuned (a strawman IMM gets a paper rejected).

---

## 7. Indian defence and innovation route (parallel track)

Programs change; **confirm current eligibility and rules on official sites
before promising anything to the team.**

- **iDEX / Defence Innovation Organisation challenges:** typically need an
  incorporated entity (startup or MSME), not a college club. Plan: watch for
  counter-UAS and drone-detection problem statements and match the work to
  them; options are the BMSIT incubation cell, a startup registered by
  members after graduation, or a faculty-led collaboration.
- **Accessible to a club now:** Smart India Hackathon defence problem
  statements, DRDO and service-affiliated student challenges, industry
  drone/AI hackathons. Use these as early validation.
- **Mentors:** local DRDO labs in Bengaluru (for example the Centre for
  Artificial Intelligence and Robotics), faculty with defence
  collaborations, alumni in the sector. The college connection matters here.
- **IP:** settle with BMSIT early who owns the code and future IP, and what
  will be open-sourced. Surprises here later can block both the paper and a
  startup route.

What these evaluators judge, same as journals: a real problem, rigorous
evidence, a clear path to deployment, and a safe, legal framing. Evidence
means public-benchmark numbers others can reproduce, the fixed-split
harness, field logs including failures, latency and power measurements on
the real edge hardware, and a stated TRL with what it takes to reach the
next one.

---

## 8. Academic integration (college and guide)

- **Faculty guide:** TBD - sign-off on the project, likely
  co-author; agree the guide's identity and authorship role explicitly in
  the written authorship rules.
- **Authorship rules in writing before month 2:** contribution thresholds,
  author order logic, project lead (Reyyi Shreyas) versus paper lead
  (assigned separately). The most common source of club conflict.
- **Department angle:** the work can support a B.E. capstone / academic
  credit; confirm with the department what documentation is required.
- **Shared experiment log:** every run recorded and reproducible. Reviewers
  and the department both ask.
- **Open parts:** publish the evaluation code, simulation setup, and a
  technical report. Keep field footage of sensitive locations out of the
  repo.

---

## 9. Literature review checklist

Search and read across these threads (seed terms, not an exhaustive list):

- [ ] Counter-UAS survey literature (camera, RF, radar approaches and their
      limits)
- [ ] Small-object detection in aerial imagery (tiling, temporal heads)
- [ ] Multi-object tracking: ByteTrack, OC-SORT, and their evaluation
      practice
- [ ] Maneuver prediction: IMM variants, interacting multiple model
      estimation, coordinated-turn models
- [ ] Learned trajectory forecasting (residual / physics-informed
      architectures, uncertainty heads, multimodal prediction)
- [ ] Drone-vs-bird discrimination (false-positive control)
- [ ] Latency-aware visual servoing and gimbal control
- [ ] Existing anti-UAV benchmarks and their license terms

Keep the reading list in this file's repo (a `docs/LITERATURE.md`), one line
per paper: citation, what it claims, what it does not cover, where we cite
it.

---

## 10. What would kill this research (and the fix)

- **Simulation-only results** - criticized for the sim-to-real gap. Fix:
  even a small set of real flights.
- **Weak baseline** - beat a properly tuned IMM, not a strawman.
- **Overclaiming** - state what works at what range and in what conditions;
  report failure cases. This builds credibility with reviewers and defence
  evaluators alike.
- **Scope creep** - camera-only, single-target V1, written in the README.
- **Late writing** - start month 2.
- **License or IP surprises** - verify before use, settle IP early.
