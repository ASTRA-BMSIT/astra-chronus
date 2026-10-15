# Contributing to Astra Chronus

Anyone can contribute: this is an open-source counter-UAS research project of
the Astra club, BMSIT. The core team raises most pull requests, but external
contributors are welcome for bug fixes, documentation, datasets licences
verification, baselines, and experiments.

All contributions are subject to the safety rules in the
[README](README.md#12-legal-and-safety-non-negotiable): simulation-only for
interception, nothing that engages a real aircraft, nothing that endangers
people or property. We cannot accept contributions that violate these rules,
however good the code is.

---

## 1. Core team

Club members raise the project's PRs; external contributors are welcome
under the same rules. Workstreams below are placeholders until the first
team meeting - starting assignments, not permanent rules. Members get
listed here with their name and workstream as they claim their first task.

| Member | Role | Workstream | First task |
|---|---|---|---|
| Reyyi Shreyas | President, project lead | Evaluation harness, interface contracts | Land the harness skeleton with fixed splits |
| _TBD - first meeting_ | Perception | Detection: fine-tuning, small-object tricks, cascade | Set up detector training on a public dataset |
| _TBD - first meeting_ | Tracking and prediction | ByteTrack/OC-SORT, IMM baseline, predictor models | Implement the IMM-EKF baseline |
| _TBD - first meeting_ | Simulation and data | PX4/Gazebo or AirSim, dataset ingestion, licences | Fill `docs/DATA_LICENSES.md` with verified entries |
| _TBD - first meeting_ | Control and hardware | Gimbal, latency, Jetson deployment | Measure the end-to-end latency budget |
| _TBD - first meeting_ | Paper and documentation | Writing, experiment log, submissions | Set up the shared experiment log |

Faculty guide: **TBD** (sign-off and authorship role per the written
authorship rules).

---

## 2. How work is claimed and credited

1. Find or open an **issue** describing the work. Check the README's
   [open decisions](README.md#14-open-decisions-and-facts-to-verify) and the
   first-tasks column above before starting something new.
2. **Comment on the issue to claim it.** Core team: you get assigned.
   External: you get a "claimed by community" note; first useful PR wins the
   task.
3. Open a **branch** named `<github-username>/<short-description>`, for
   example `your-name/detector-training-script`.
4. Open a **PR against `main`** that closes the issue. The PR description
   states what changed, why, and how it was tested.
5. **AI disclosure:** every PR states whether AI tools were used to write
   the code and what for. We keep this in the record because the work may
   become a paper.
6. On merge, **add your row to the Contributions table** (Section 5) in the
   same PR or a follow-up: your name, what the PR did, its number.

Credit is the table plus git history. There are no other rankings.

---

## 3. Code standards

- **Python 3.10+**, matching the package style:
  - type hints on all public functions,
  - NumPy-style docstrings (`Parameters`, `Returns`, `Raises`),
  - comments on non-obvious lines only.
- **Lint and format with ruff** before pushing: `ruff check .` and
  `ruff format .` must be clean.
- **Tests are part of the change.** New behaviour needs tests under `tests/`
  in the same PR. Run the tests related to your change, not the full suite
  on every commit:
  `pytest tests/test_<module>.py -q`.
- **Configs live in `configs/`**, one YAML per experiment, committed. A
  result is not valid if it cannot be reproduced from a committed config and
  seed.
- **No training code in notebooks.** Notebooks are for analysis and plots
  only. Training goes in `scripts/` launched from a config.
- **Never commit data.** `data/` is gitignored; commit download or generation
  scripts instead.
- **Interfaces are frozen by the README** (Section 4): changing a
  dataclass is a PR that updates the README in the same commit. Coordinate
  with the project lead first.
- **Metrics honesty:** no number goes in a doc, issue, or PR title unless it
  was measured by a committed config run and is in the experiment log.
  Unmeasured values are written as `TBD`, never estimated.

---

## 4. Commit style

- Short subject, 50 characters or fewer, imperative mood.
- Body wrapped at 72 characters, explaining why (not just what).
- Prefixes: `fix:` `enh:` `doc:` `cos:` `test:` `feat:` `chore:`.
- One logical change per commit; split unrelated work into separate PRs.

---

## 5. Contributions table

Every merged PR adds one row. Written by the PR author, reviewed like
code. `Work done` states what the contribution did, not effort spent.

| Contributor | Work done | PR |
|---|---|---|
| Reyyi Shreyas | V1 architecture README, research plan and PDF, contributor guide | direct pushes on `main` |
| _Your row here_ | _What your merged PR did_ | _#N_ |

External contributors: same table, same rules. A merged PR is a row; that is
the whole credit system.

---

## 6. Good first contributions

- Verify a dataset's licence and add the entry to `docs/DATA_LICENSES.md`
  (Anti-UAV, DUT Anti-UAV, Drone-vs-Bird, MAV-VID all need this).
- Run the latency-measurement script on available hardware and log results.
- Write a test for an existing module that has none.
- Fix docs: broken links, unclear steps, missing rationale.
- Reproduce a baseline (Kalman/IMM) and file the numbers in the experiment
  log via a PR.

Larger contributions (new model, new sensor, new dataset pipeline): open a
design issue first, discuss, then implement. No surprise 2,000-line PRs.

---

## 7. Review and merge

- One approval from a core-team member merges. The project lead reviews
  interface changes; sub-team owners review their own area.
- CI must be green (lint, format, tests).
- Reviews aim for the next push, not the same hour. Core team: review
  others' PRs promptly; a PR stalled three days gets escalated in the team
  channel.
- Rebase or merge `main` before the final review pass; no force-pushes to
  `main`, ever.

---

## 8. Questions

Ask in the Astra club team channel before you spend a day on something
unclear. Questions cost minutes; wrong assumptions cost weeks - the project
lead's job is keeping interfaces and evaluation honest, so use that.
