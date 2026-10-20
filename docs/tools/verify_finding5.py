"""Independent check of the ASTRA-CHRONOS architecture review, Finding 5.

Finding 5 says the rule-based behaviour classifier's thresholds do not fit the
drones it watches: the "fast approach" rule needs a mean speed above 80 m/s
but live drones never reach it, and the two most dangerous behaviours are the
two it mislabels most often.

This harness slides the V0 project's own rule-based classifier over windows of
live-simulation trajectories and reports the share of ticks each behaviour is
labelled with, plus the peak speed actually produced.

It must run against a clone of the V0 repository, because it imports
``src/simulation`` and ``src/threat_engine``.

Usage
-----
    python docs/tools/verify_finding5.py /path/to/Astra-chronus-ai-

Requires: numpy (torch is imported by the classifier module but unused here).
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

DT = 0.1
SEQ = 50
MODES = ["normal", "loitering", "evasive", "fast_approach", "hovering"]
SEEDS = list(range(8))


def resolve_root() -> Path:
    """Locate the V0 clone from argv or the ASTRA_V0 environment variable."""
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).expanduser().resolve()
    env = os.environ.get("ASTRA_V0")
    if env:
        return Path(env).expanduser().resolve()
    raise SystemExit(
        "usage: python verify_finding5.py /path/to/Astra-chronus-ai-\n"
        "(or set ASTRA_V0 to that path)"
    )


def features(pos: np.ndarray) -> np.ndarray:
    """13 features: xyz | v | a | speed | heading | turn_rate | curvature."""
    vel = np.gradient(pos, DT, axis=0)
    acc = np.gradient(vel, DT, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    heading = np.arctan2(vel[:, 1], vel[:, 0])
    turn = np.gradient(heading)
    curv = np.abs(turn) / (speed + 1e-6)
    return np.column_stack([pos, vel, acc, speed, heading, turn, curv])


def run(root: Path) -> None:
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "src"))
    from simulation.drone_motion import RealisticDroneMotionGenerator
    from threat_engine.classifier import BehaviourClassifier

    gen = RealisticDroneMotionGenerator(dt=DT)
    clf = BehaviourClassifier(use_rules=True)
    print(f"v0 repo: {root}")
    print(f"{'simulated behaviour':<20}{'classifier output (share of ticks)':<52}{'peak speed'}")
    for mode in MODES:
        counts: Counter[str] = Counter()
        peak_speed = 0.0
        for seed in SEEDS:
            saved = np.random.get_state()
            np.random.seed(seed)
            pos, _v, _a = gen.generate(mode, length=180)
            np.random.set_state(saved)

            feats = features(pos)
            peak_speed = max(
                peak_speed,
                float(np.linalg.norm(np.gradient(pos, DT, axis=0), axis=1).max()),
            )
            for start in range(0, len(feats) - SEQ + 1, 5):
                counts[clf.classify(feats[start:start + SEQ])["label"]] += 1

        total = sum(counts.values())
        parts = "  ".join(f"{k} {100 * v / total:.0f}%" for k, v in counts.most_common())
        print(f"{mode:<20}{parts:<52}{peak_speed:6.1f} m/s")


if __name__ == "__main__":
    run(resolve_root())
