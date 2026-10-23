"""Whole-trajectory splits with reproducible seeds.

A trajectory must never appear in both train and test. In counter-UAS tracking,
consecutive frames from the same flight share dynamics, lighting and camera
aspect. Splitting by frame leaks motion models between train and test and masks
generalization failures (README section 7, V0 Corrigendum Finding 5).
"""

import random
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class TrajectorySplit:
    """A reproducible train/val/test split of trajectory identifiers.

    Attributes
    ----------
    train_ids : tuple[str, ...]
        Trajectory identifiers assigned to the training split.
    val_ids : tuple[str, ...]
        Trajectory identifiers assigned to the validation split.
    test_ids : tuple[str, ...]
        Trajectory identifiers assigned to the test split.
    seed : int
        Random seed used to generate the split.
    train_ratio : float
        Fraction of trajectories allocated to training.
    val_ratio : float
        Fraction of trajectories allocated to validation.
    test_ratio : float
        Fraction of trajectories allocated to testing.
    """

    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    seed: int
    train_ratio: float
    val_ratio: float
    test_ratio: float


def split_trajectories(
    trajectory_ids: Sequence[str],
    *,
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> TrajectorySplit:
    """Split trajectory identifiers into mutually exclusive partitions.

    Splits whole trajectories deterministically based on a committed seed.
    Guarantees that train, validation, and test subsets are completely
    disjoint.

    Parameters
    ----------
    trajectory_ids : Sequence[str]
        Unique identifiers for whole flight trajectories.
    seed : int
        Random seed ensuring split reproducibility.
    train_ratio : float, default 0.70
        Proportion of trajectories for training.
    val_ratio : float, default 0.15
        Proportion of trajectories for validation.
    test_ratio : float, default 0.15
        Proportion of trajectories for testing.

    Returns
    -------
    TrajectorySplit
        A frozen container holding the disjoint train, val, and test ID tuples.

    Raises
    ------
    ValueError
        If `trajectory_ids` is empty, contains duplicate IDs, or if the
        proportions do not sum to 1.0 within numerical tolerance.
    """
    if not trajectory_ids:
        raise ValueError("trajectory_ids must not be empty.")

    # Guard against duplicate trajectory IDs causing data leakage across splits
    unique_ids = sorted(set(trajectory_ids))
    if len(unique_ids) != len(trajectory_ids):
        raise ValueError("trajectory_ids contains duplicate entries.")

    for ratio_name, ratio_val in (
        ("train_ratio", train_ratio),
        ("val_ratio", val_ratio),
        ("test_ratio", test_ratio),
    ):
        if ratio_val < 0.0 or ratio_val > 1.0:
            raise ValueError(f"{ratio_name} must be in [0.0, 1.0], got {ratio_val}")

    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-5:
        raise ValueError(
            f"Split ratios must sum to 1.0 (train={train_ratio}, val={val_ratio}, "
            f"test={test_ratio}, sum={total_ratio})"
        )

    # Deterministic shuffle from sorted base ensures invariance to input order
    rng = random.Random(seed)
    shuffled = list(unique_ids)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = round(n_total * train_ratio)
    n_val = round(n_total * val_ratio)
    # Give remainder to test to ensure exact preservation of all trajectories
    n_test = n_total - n_train - n_val

    # Adjust rounding corner cases if any slice became negative
    if n_test < 0:
        n_test = 0
        n_val = n_total - n_train

    train_ids = tuple(shuffled[:n_train])
    val_ids = tuple(shuffled[n_train : n_train + n_val])
    test_ids = tuple(shuffled[n_train + n_val :])

    return TrajectorySplit(
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
