"""Independent check of the ASTRA-CHRONOS architecture review, Finding 1.

Finding 1 says the trained GRU trajectory predictor is on average about 100x
less accurate than simply assuming the drone keeps flying in a straight line.

This harness reuses the V0 project's own live-simulation generator and
checkpoint loader: it generates a live trajectory, builds the 50-step /
13-feature window the way ``src/simulation/environment.py`` does, rolls the
trained model out 20 steps (2.0 s at dt = 0.1), and compares against the
drone's actual future positions. The constant-velocity baseline is computed
from the same window.

It must run against a clone of the V0 repository, because it imports
``src/simulation`` and needs ``models/best_model.pt``.

Usage
-----
    python docs/tools/verify_finding1.py /path/to/Astra-chronus-ai-

Requires: numpy, torch.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

DT = 0.1
SEQ = 50
HORIZON = 20  # 2.0 s
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
        "usage: python verify_finding1.py /path/to/Astra-chronus-ai-\n"
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
    from trajectory_ai.predictor import TrajectoryPredictor

    gen = RealisticDroneMotionGenerator(dt=DT)
    pred = TrajectoryPredictor.from_checkpoint(root / "models" / "best_model.pt")
    print(f"v0 repo: {root}")
    print(f"model input_size={pred.input_size} device={pred.device}")
    print(f"{'behaviour':<16}{'GRU ADE/FDE':>22}{'CV ADE/FDE':>20}")
    rows = []
    for mode in MODES:
        g_err, b_err = [], []
        for seed in SEEDS:
            # save/restore global numpy RNG so each seed starts identically
            saved = np.random.get_state()
            np.random.seed(seed)
            pos, _vel, _acc = gen.generate(mode, length=SEQ + HORIZON + 10)
            np.random.set_state(saved)

            window = features(pos)[:SEQ]
            future = pos[SEQ : SEQ + HORIZON]

            gru = pred.predict(window, steps=HORIZON)
            p0, p1 = window[-2, :3], window[-1, :3]
            cv = np.array([p1 + (p1 - p0) * (k + 1) for k in range(HORIZON)])

            g_err.append(
                (
                    float(np.linalg.norm(gru - future, axis=1).mean()),
                    float(np.linalg.norm(gru - future, axis=1)[-1]),
                )
            )
            b_err.append(
                (
                    float(np.linalg.norm(cv - future, axis=1).mean()),
                    float(np.linalg.norm(cv - future, axis=1)[-1]),
                )
            )
        g, b = np.array(g_err), np.array(b_err)
        rows.append((g, b))
        print(
            f"{mode:<16}{g[:, 0].mean():8.1f} /{g[:, 1].mean():8.1f}"
            f"{b[:, 0].mean():11.1f} /{b[:, 1].mean():8.1f}"
        )

    allg = np.vstack([r[0] for r in rows])
    allb = np.vstack([r[1] for r in rows])
    print(
        f"\nOVERALL  GRU ADE {allg[:, 0].mean():.1f}  FDE {allg[:, 1].mean():.1f}"
        f" | CV ADE {allb[:, 0].mean():.1f}  FDE {allb[:, 1].mean():.1f}"
    )
    print(f"ratio GRU/CV ADE: {allg[:, 0].mean() / allb[:, 0].mean():.1f}x")


if __name__ == "__main__":
    run(resolve_root())
