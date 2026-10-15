# Experiment Log

Every run is recorded here, in the same PR that reports its numbers. This
log is why a claimed metric is trustworthy: a number without a row does not
exist. Reviewers ask for the row; if it is missing, the number is changed to
`TBD`.

Rules (README section 7):

- A row is added by the PR author, reviewed like code.
- Each row names a **committed config** and a **seed**, so anyone can
  re-run it.
- **Smooth and evasive trajectories are logged separately** - never only a
  pooled number.
- The **IMM-EKF baseline is logged next to every learned model**, so the
  comparison is honest.
- Failed runs are logged too, with a note on what failed. Failures are
  evidence, not embarrassment.

## Format

One block per run:

```text
DATE: <YYYY-MM-DD>
CONFIG: <path under configs/> (seed <seed>)
SPLIT: <split_ref> (<train or held out>)
RESULT: <stage metric> <value> | <baseline metric> <value> | ...
HARDWARE: <device> | <latency metric> <value>
NOTES: <failure cases, lost tracks, anything a reviewer will ask>
```

## Runs

_No runs logged yet._

The harness lands first (see the first-tasks table in
[CONTRIBUTING.md](../CONTRIBUTING.md)); the first rows here are its smoke
runs. Until then, every performance number in the repository is a target or
a definition, not a measurement.
