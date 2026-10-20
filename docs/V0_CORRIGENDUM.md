# V0 corrigendum: what is measured and what is only described

**Date:** 20 October 2026
**Corrects:** the status line and the accuracy implications in
`ASTRA_CHRONOS_Research_Report.pdf`, a document circulated to the team that is
not stored in this repository.
**Does not dispute:** `ASTRA_CHRONUS_.pdf`, the architecture review prepared by
Shaun Thomas. I re-ran its two headline findings independently and both hold.
**Harnesses:** [`tools/verify_finding1.py`](tools/verify_finding1.py) and
[`tools/verify_finding5.py`](tools/verify_finding5.py).

## Why this file exists

The research report opens with `STATUS: VERIFIED RESEARCH` and then describes a
working kill-chain: the GRU forecasts the drone 10 seconds ahead, the threat
score is computed, a collision point is solved, an interceptor flies, the drone
is neutralised.

The description of *what the code contains* is accurate. The word **verified**
is not. Nothing in that report was measured, and when the pipeline is run
headlessly the predictor turns out to be worse than a trivial guess.

Both harnesses are committed here so anyone can re-run the numbers rather than
take them on trust, in line with the rule in
[`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md): a number without a row does not
exist.

---

## 1. What was run

Both harnesses take the path to a V0 clone, because they import `src/` and need
the checkpoint from that repository:

```bash
git clone https://github.com/reyyishreyas/Astra-chronus-ai- /tmp/astra-v0
python docs/tools/verify_finding1.py /tmp/astra-v0
python docs/tools/verify_finding5.py /tmp/astra-v0
```

Common method for both:

- the V0 project's own live generator
  (`src/simulation/drone_motion.py`, `RealisticDroneMotionGenerator`),
- a 50-step window at `dt = 0.1` (5.0 s of history),
- the 13 features in the order `src/simulation/environment.py:46` defines them,
- checkpoint `models/best_model.pt`,
- seeds 0-7 for each of the five behaviours,
- device: Apple MPS.

No code in the V0 repository was modified to run these.

For Finding 1 the model is rolled out 20 steps (2.0 s) and compared with where
the drone actually went. The baseline is constant velocity extrapolated from
the last two rows of the same window, which is the least clever guess
available.

---

## 2. Finding 1: the predictor is worse than doing nothing clever, confirmed

Error in metres at 2 seconds ahead, mean over 8 seeds per behaviour.

| Behaviour | GRU ADE / FDE | Constant-velocity ADE / FDE |
|---|---|---|
| normal | 89.3 / 138.6 | 0.9 / 2.5 |
| loitering | 84.4 / 125.5 | 1.7 / 4.7 |
| evasive | 111.6 / 166.9 | 0.9 / 2.3 |
| fast approach | 50.6 / 83.3 | 19.7 / 43.2 |
| hovering | 57.6 / 87.7 | 3.7 / 6.8 |
| **overall** | **78.7 / 120.4** | **5.4 / 11.9** |

**The trained model is 14.6x worse than constant velocity on average**, and
worse on every single behaviour.

My magnitudes are lower than the review's table, which reports 417.3 for normal
and 539.9 for evasive. The direction and the conclusion are identical.
Hovering reproduces almost exactly (57.6 / 87.7 against 57.6 / 87.3), which
suggests the two harnesses agree and the gap elsewhere comes from window and
seed choices. **Quote these numbers as this run's numbers, not as the
review's.**

ADE (average displacement error) is the average distance between predicted and
actual position across the horizon. FDE (final displacement error) is the same
distance at the last step.

---

## 3. Finding 5: the classifier cannot see the two most dangerous behaviours, confirmed

Share of ticks, 8 seeds per behaviour, sliding 50-step window.

| Simulated behaviour | Classifier output | Peak live speed |
|---|---|---|
| normal | loitering 50%, normal 50% | 27.6 m/s |
| loitering | loitering 97%, normal 3% | 20.5 m/s |
| evasive | **normal 94%**, loitering 6% | 33.5 m/s |
| fast approach | **evasive 63%**, normal 35% | 47.9 m/s |
| hovering | hovering 100% | 7.3 m/s |

Loitering (97%) and hovering (100%) reproduce the review exactly. Evasive and
fast approach are the two that matter, and they are the two it gets wrong.

The mechanism is arithmetic, not a mystery. `classifier.py:76` requires
`mean_speed > 80` to call something a fast approach. The fastest the live
generator ever produces is **47.9 m/s** (`drone_motion.py:115` clips at 48.0).
That rule can never fire. The evasive rule at `classifier.py:78` needs
`speed_var > 500`, which does not occur either.

---

## 4. Confirmed by reading the code

**Finding 2, trained on a different world than it runs in.** Three generators,
not one:

- training data: `src/data_engine/trajectory_generator.py:11`,
  `vel = np.random.rand(3) * 5` (each component 0 to 5, so roughly 0-8 m/s)
- a second one: `data_gen.py`, imported by `src/main.py:22`
- live simulation: `src/simulation/drone_motion.py`, 14-24 m/s cruise,
  18-30 m/s evasive, 28-42 m/s dive

Training drones are several times slower than live drones. This is the most
likely root cause of Finding 1.

**Finding 3, inputs are never scaled.** `src/data_engine/preprocessing.py:4`
imports `StandardScaler` and never uses it anywhere in `src/`. The predictor
emits absolute `x, y, z` from its final layer rather than a displacement, so
anything outside the training area is extrapolation.

**Finding 4, the displayed uncertainty is not computed.**
`src/simulation/environment.py:93` assigns `np.linspace(2, 20, 20)` directly as
`pred_uncertainty`. It grows in a straight line regardless of input. The Monte
Carlo dropout routine does exist (`src/trajectory_ai/predictor.py:86`) but has
no caller on the live path; only `path_predictor.py:32` references it, from
inside its own unused wrapper.

**Finding 6, engagement ignores the threat score.**
`src/backend/server.py`, in the predicting phase, moves to `PHASE_FIRED` as
soon as an aim point exists. The threat score is appended to `threat_history`
for display a few lines above, but the launch decision never reads it, and
there is no operator approval step.

---

## 5. What was not verified

Stated plainly so nobody mistakes relay for measurement:

- **Finding 7** (the GRU neutralises 8-10 of 10 runs while the fallback gets
  10 of 10 on the first shot) and **Finding 8** (browser-driven clock, global
  state, no seed): not re-run. Those remain the review's.
- The exact magnitudes in the review's supporting table. Only hovering
  reproduced closely.
- The "~25 GB" dataset size. `data/` in the V0 repo holds one README and the
  dataset is downloaded separately, so the size cannot be checked. The
  trajectory count and noise do match config
  (`src/data_engine/config.py:1`: 30,000 trajectories, noise 1.5).
- The 10-second horizon. The rollout is real: `server.py:18` sets
  `PRED_STEPS = 100` at `dt = 0.1`. This run measured 2.0 s, not 10 s. Do not
  quote a 10-second accuracy number from anyone yet.

---

## 6. Claim by claim

| Research report says | Status | Evidence |
|---|---|---|
| `STATUS: VERIFIED RESEARCH` | **Not supported** | No experiment log rows exist; two headless runs contradict the accuracy implication |
| GRU projects the drone up to 10 s ahead | Rollout exists, accuracy **refuted at 2 s** | `server.py:18`; section 2 above, 14.6x worse than constant velocity |
| Solves the collision point, launches, neutralises | **Accurate as description** | aim-point and proximity-fuse code exist and run |
| 0-100 threat score drives the response | Score exists, **does not gate anything** | `server.py`, section 4 |
| Monte Carlo dropout gives commanders a trustworthy error bubble | Routine exists, **never called on the live path**; stored band is a fixed line | `predictor.py:86`, `environment.py:93` |
| 30,000 trajectories, Gaussian noise 1.5 | **Matches config** | `data_engine/config.py:1` |
| 5 realistic UAV behaviours | **Accurate** | generator supports all five |

---

## 7. One decision still open

The team is currently spread across two architectures:

- **V0, `Astra-chronus-ai-`**: radar-telemetry simulation, interception
  simulator, 2-layer GRU. Built. This is what the review tested.
- **V1, this repository**: real camera feed, detect-track-predict-lock,
  interception only inside simulation. Planned, and the README says no code
  yet.
- One teammate's research notes the problem statement asks for
  **simulation-first** while the V1 README describes a **camera-based**
  approach.

That conflict cannot be settled from inside this repository. The problem
statement text is needed before anyone rebuilds anything a third time.
