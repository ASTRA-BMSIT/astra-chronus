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

# Ensure src is on sys.path when running script standalone from repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from astra_chronus.evaluation.splits import TrajectorySplit, split_trajectories


def parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse basic key-value YAML without requiring external dependencies."""
    result: dict[str, Any] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip()
        val = val.strip()

        # Parse inline list e.g. [0.25, 0.5, 1.0, 2.0]
        if val.startswith("[") and val.endswith("]"):
            items = [item.strip() for item in val[1:-1].split(",") if item.strip()]
            parsed_items: list[Any] = []
            for item in items:
                try:
                    parsed_items.append(float(item))
                except ValueError:
                    parsed_items.append(item.strip("\"'"))
            result[key] = parsed_items
        elif (
            val.startswith('"')
            and val.endswith('"')
            or val.startswith("'")
            and val.endswith("'")
        ):
            result[key] = val[1:-1]
        else:
            try:
                if "." in val:
                    result[key] = float(val)
                else:
                    result[key] = int(val)
            except ValueError:
                result[key] = val
    return result


def load_config(config_path: Path) -> dict[str, Any]:
    """Load configuration from YAML file.

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
    """
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    content = config_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import-untyped]

        loaded = yaml.safe_load(content)
        if isinstance(loaded, dict):
            return loaded
    except ImportError:
        pass

    return parse_simple_yaml(content)


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
    except FileNotFoundError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    seed = int(config.get("seed", 20261101))
    train_ratio = float(config.get("train_ratio", 0.70))
    val_ratio = float(config.get("val_ratio", 0.15))
    test_ratio = float(config.get("test_ratio", 0.15))

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
