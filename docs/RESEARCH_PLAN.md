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

---

## 11. Related work

Status: first draft for review (issue #5). Rules for this section:

- Every entry names a real paper with a link, says what it did, and says what
  it does not cover for us.
- Numbers are copied from the cited source with its units and horizon. They
  are not comparable with each other unless the units and horizons match, and
  none is a measurement of ours.
- If a claim could not be sourced, it says `TBD` instead of an assertion.
- Entries were checked against the paper page or the DOI registry. Entries
  marked "abstract only" were not read in full.

### 11.1 Counter-UAS detection and tracking

**Benchmarks.**

- Jiang et al., [Anti-UAV: A Large Multi-Modal Benchmark for UAV
  Tracking](https://arxiv.org/abs/2101.08466) (arXiv 2021). Introduces the
  Anti-UAV dataset with "more than 300 video pairs" and "over 580k manually
  annotated bounding boxes", plus a tracker called DFSC. Abstract only. It is
  a tracking benchmark, so it reports no trajectory prediction.
- Huang et al., [Anti-UAV410: A Thermal Infrared Benchmark and Customized
  Scheme for Tracking Drones in the
  Wild](https://doi.org/10.1109/TPAMI.2023.3335338) (IEEE TPAMI 46(5),
  2024). Thermal infrared only, with 410 videos and over 438K annotated
  boxes, and a tracker called SiamDT. Abstract only. Prediction is not
  covered.
- Xie et al., [CST Anti-UAV: A Thermal Infrared Benchmark for Tiny UAV
  Tracking in Complex Scenes](https://arxiv.org/abs/2507.23473) (ICCV 2025
  Workshops). 220 sequences, over 240k boxes, 20 trackers benchmarked. The
  best reaches "35.92% state accuracy", which the authors cite as evidence
  that tiny UAVs in complex scenes remain hard. Thermal infrared only.
  Prediction is not covered.

**Drone versus bird.** Coluccia et al., [Drone vs. Bird Detection: Deep
Learning Algorithms and Results from a Grand
Challenge](https://doi.org/10.3390/s21082824) (Sensors 2021). Compares three
approaches from the 2020 challenge. Per the abstract, sequences from a
moving camera and very distant drones are the most challenging, and the
approaches are complementary on correct detection rate, false alarm rate and
average precision. Abstract only. A later review covers the 2023 edition:
Coluccia et al., [The Drone-vs-Bird Detection Grand Challenge at ICASSP 2023:
A Review of Methods and Results](https://doi.org/10.1109/ojsp.2024.3379073)
(IEEE OJSP 2024), not yet read. This line of work scores detection, not what
the target does next.

**Small objects.** Akyon et al., [Slicing Aided Hyper Inference and
Fine-tuning for Small Object Detection](https://arxiv.org/abs/2202.06934)
(ICIP 2022). Tiled inference (SAHI) raised AP by 6.8% (FCOS), 5.1% (VFNet)
and 5.3% (TOOD) without fine-tuning, and by a cumulative 12.7%, 13.4% and
14.5% with slicing-aided fine-tuning, on VisDrone and xView. These are aerial
datasets, not counter-UAS footage, so the gain for our targets is `TBD` until
we measure it. Drone-specific detectors beyond this are `TBD`, not yet
surveyed.

**Multi-object tracking.**

- Zhang et al., [ByteTrack: Multi-Object Tracking by Associating Every
  Detection Box](https://arxiv.org/abs/2110.06864) (arXiv 2021). Associates
  low-score detections as well, and reports "80.3 MOTA, 77.3 IDF1 and 63.1
  HOTA" on the MOT17 test set at 30 FPS on a single V100. MOT17 is
  pedestrians, not drones.
- Cao et al., [Observation-Centric SORT: Rethinking SORT for Robust
  Multi-Object Tracking](https://arxiv.org/abs/2203.14360) (CVPR 2023).
  Argues that Kalman-filter trackers assume linear motion and accumulate
  error while a target is unobserved, and reports "700+ FPS on a single CPU".
  Not evaluated on drones.

Neither tracker predicts 0.5 to 2 s ahead. Both only associate detections,
so the prediction horizon in section 1 is outside their scope.

### 11.2 Trajectory prediction for aerial targets

**Classical baselines.**

- Blom and Bar-Shalom, [The interacting multiple model algorithm for systems
  with Markovian switching coefficients](https://doi.org/10.1109/9.1299)
  (IEEE Trans. Automatic Control 33(8), 780-783, 1988). The original IMM
  reference for the primary baseline in section 4. Not read in full.
- Li and Jilkov, [Survey of maneuvering target tracking. Part I: dynamic
  models](https://doi.org/10.1109/TAES.2003.1261132) (IEEE Trans. Aerospace
  and Electronic Systems 39(4), 1333-1364, 2003). Source for the choice of
  motion models in the IMM-EKF. Not read in full.

**Learned prediction against classical baselines.** Reported results differ
by dataset, and most are in pixels.

- Becker et al., [Generating Synthetic Training Data for Deep Learning-Based
  UAV Trajectory Prediction](https://arxiv.org/abs/2107.00422) (ROBOVIS
  2021). An RNN-MDN trained only on synthetic trajectories, evaluated on real
  Anti-UAV sequences against a constant-velocity Kalman filter and linear
  interpolation. The metric is final displacement error (FDE) in pixels at 8,
  10 and 12 frames, with no seconds given because the Anti-UAV frame rate is
  not stated. The paper notes that no camera motion compensation is applied
  and that errors at longer horizons are high "due to the maneuvering
  behavior of the UAVs". Its Table 1 has a second column per horizon that the
  text does not define, so it is not quoted. The caption says 16 frames but
  the headings say 12.

  | FDE (px) | EO 8 frames | EO 10 | EO 12 | IR 8 | IR 10 | IR 12 |
  |---|---|---|---|---|---|---|
  | RNN-MDN | 60.304 | 82.780 | 121.453 | 20.849 | 41.378 | 61.459 |
  | Kalman filter (CV) | 81.061 | 110.041 | 179.459 | 22.340 | 43.172 | 68.012 |
  | Linear interpolation | 86.998 | 119.106 | 183.558 | 24.235 | 45.434 | 69.665 |

  The baseline is constant-velocity, not an IMM.

- Hari Prasanth S. M. et al., [Event-Only Drone Trajectory Forecasting with
  RPM-Modulated Kalman Filtering](https://arxiv.org/abs/2603.01997) (arXiv
  2026, preprint). On the FRED dataset, ADE and FDE in pixels at 0.4 s and
  0.8 s. The authors report that all classical methods beat the deep models,
  and that the LSTM and Transformer use only past ground-truth boxes as
  input. Table I:

  | Method | ADE 0.4 s | ADE 0.8 s | FDE 0.4 s | FDE 0.8 s |
  |---|---|---|---|---|
  | LSTM | 171.9 | 353.3 | 53.04 | 96.44 |
  | Transformer | 124.8 | 292.2 | 46.74 | 87.68 |
  | CNN + Transformer (RGB + event) | 121.3 | 281.5 | 45.23 | 85.78 |
  | Linear extrapolation | 16.45 | 37.93 | 32.90 | 83.32 |
  | Kalman filter | 16.40 | 37.08 | 32.80 | 81.54 |
  | Kalman + RPM (event) | 15.35 | 34.85 | 31.16 | 77.76 |

  Rows for the other two CNN + Transformer variants are omitted here. This
  is a preprint and has not been peer reviewed.

**Hybrids of IMM and learning.**

- Zhu et al., [UAV Trajectory Tracking via RNN-enhanced IMM-KF with ADS-B
  Data](https://arxiv.org/abs/2312.15721) (arXiv 2023). An RNN adapts the
  noise matrices of an IMM-KF. Reports total RMSE "decreases by 28.56%"
  against the traditional IMM-KF, "via comprehensive simulations". Abstract
  only. Simulation, and it adjusts the filter instead of correcting its
  output.
- Zhang et al., [Deep Uncertainty-aware Tracking for Maneuvering
  Targets](https://arxiv.org/abs/2410.14422) (IEEE Trans. Aerospace and
  Electronic Systems, 2026). A data-driven network for maneuvering targets
  that models measurement uncertainty. Abstract only, and it is a radar
  setting with no numbers in the abstract.

### 11.3 Camera-based tracking with latency compensation

Li, Wang and Ramanan, [Towards Streaming
Perception](https://arxiv.org/abs/2005.10420) (ECCV 2020). Argues that by the
time an algorithm finishes a frame the world has changed, and proposes a
"streaming accuracy" metric combining latency and accuracy. Finds that
tracking and forecasting emerge as useful internal representations. It is
evaluated on urban driving video, with no gimbal and no drone. Abstract only.

Latency-compensated gimbal and visual-servoing control for pan-tilt cameras:
`TBD`. No source has been checked yet.

### 11.4 What our contribution would have to be

The four planned contributions are from README section 8.

**1. Noise-aware training and evaluation protocol.** The sources read here
feed learned models clean inputs. Becker et al. add 1.5-px white Gaussian
noise (per the paper's Kalman observation-noise assumption) to the synthetic
generated trajectory positions used to train the RNN; the ANTI-UAV annotated
centres are their evaluation ground truth, not the noisy training inputs. The
FRED paper states that its LSTM and Transformer use past ground-truth boxes.
To be
publishable we would have to show that predictors trained on tracks produced
by a real detector and tracker behave differently from the same predictors
trained on ground truth, and report both.

**2. Smooth and evasive split.** The tables above report one pooled number
per method. Becker et al. note that errors grow with maneuvering, but no
table separates smooth from evasive flight. To be publishable the split must
be fixed and committed, with a stated rule for what counts as evasive, and
every headline number must be reported on both subsets.

**3. Residual predictor with uncertainty.** The FRED paper reports classical
filters beating deep models on FRED, and Becker et al. compare only against a
constant-velocity filter. Zhu et al. use an IMM-KF but adapt the filter
instead of learning a residual, in simulation. To be publishable the residual
model must beat a properly tuned IMM-EKF, not a constant-velocity filter, on
the evasive subset, with the IMM tuned on the same data, and its uncertainty
must be calibrated.

**4. Latency study on edge hardware.** Li et al. supply the metric idea but
not a gimbal result. To be publishable we would have to measure lock
stability with and without prediction lead against end-to-end latency on the
real edge hardware. Any prior study of this exact setup is `TBD`.

### 11.5 The published numbers we have to beat

The best published ADE and FDE at 1 s and 2 s for a camera-based drone
trajectory predictor, in degrees: **`TBD`**. None of the sources checked
reports that. Closest, with different units and horizons, from the FRED
paper: the best entry is Kalman + RPM, with ADE 15.35 px and 34.85 px
and FDE 31.16 px and 77.76 px at 0.4 s and 0.8 s. Those pixel values depend
on the camera, so they cannot be converted to degrees or compared at 1 s and
2 s without the camera intrinsics. Closing this gap is the main follow-up:
the table needs a source that reports ADE and FDE in degrees, or a stated
conversion from pixels with the dataset's intrinsics.
