"""Generate reproducible whole-trajectory splits from a committed config.

Enforces whole-trajectory separation: a trajectory must never appear in both
train and test (README section 7, V0 Corrigendum Finding 5).

Usage
-----
python scripts/make_splits.py --config configs/evaluation/split_config.yaml
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

# Ensure src is on sys.path when running script standalone from repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from astra_chronus.evaluation.splits import TrajectorySplit, split_trajectories


def load_config(config_path: Path) -> dict[str, Any]:
    """Load configuration from a YAML file.

    Parameters
    ----------
    config_path : Path
        Path to the YAML configuration file.

    Returns
    -------
    dict[str, Any]
        Parsed configuration mapping.

    Raises
    ------
    FileNotFoundError
        If `config_path` does not exist.
    TypeError
        If `config_path` cannot be parsed as a YAML mapping.
    """
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    content = config_path.read_text(encoding="utf-8")
    loaded = yaml.safe_load(content)
    if not isinstance(loaded, dict):
        raise TypeError(f"Configuration must be a valid YAML mapping: {config_path}")

    return loaded


def main() -> int:
    """CLI entrypoint for generating whole-trajectory splits."""
    parser = argparse.ArgumentParser(
        description="Generate whole-trajectory train/val/test splits from a config."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to committed split YAML configuration.",
    )
    parser.add_argument(
        "--trajectories",
        nargs="*",
        default=None,
        help="Optional list of trajectory IDs to split.",
    )
    parser.add_argument(
        "--num-synthetic",
        type=int,
        default=50,
        help="Number of synthetic IDs to generate if none provided.",
    )

    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except (FileNotFoundError, TypeError) as err:
        print(f"Error loading configuration: {err}", file=sys.stderr)
        return 1

    try:
        seed = int(config["seed"])
        train_ratio = float(config["train_ratio"])
        val_ratio = float(config["val_ratio"])
        test_ratio = float(config["test_ratio"])
    except (KeyError, TypeError, ValueError) as err:
        print(f"Invalid configuration parameters: {err!r}", file=sys.stderr)
        return 1

    if args.trajectories:
        trajectory_ids = args.trajectories
    else:
        # Default scenario IDs if no raw IDs are passed
        num_ids = args.num_synthetic
        trajectory_ids = [f"traj_{i:04d}" for i in range(num_ids)]

    try:
        split: TrajectorySplit = split_trajectories(
            trajectory_ids=trajectory_ids,
            seed=seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
    except ValueError as err:
        print(f"Split generation failed: {err}", file=sys.stderr)
        return 1

    summary = {
        "seed": split.seed,
        "train_count": len(split.train_ids),
        "val_count": len(split.val_ids),
        "test_count": len(split.test_ids),
        "total_count": len(trajectory_ids),
        "train_ids": split.train_ids,
        "val_ids": split.val_ids,
        "test_ids": split.test_ids,
    }

    output_path_str = config.get("output_file")
    if output_path_str:
        out_path = Path(output_path_str)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote split reference to {out_path}")

    print(
        f"Generated whole-trajectory split (seed={split.seed}): "
        f"train={len(split.train_ids)}, val={len(split.val_ids)}, test={len(split.test_ids)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
