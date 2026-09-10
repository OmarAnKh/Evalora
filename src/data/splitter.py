from typing import Dict

from datasets import Dataset


def split_dataset(
    dataset: Dataset,
    label_key: str = "score",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, Dataset]:
    """Split a dataset into train/validation/test subsets.

    Raises ValueError when ratios are invalid or the dataset is too small.
    """
    if label_key not in dataset.column_names:
        raise ValueError(f"Missing '{label_key}' in dataset columns.")

    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError("Split ratios must sum to a positive value.")
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0.")
    if train_ratio <= 0:
        raise ValueError("Train ratio must be greater than 0.")

    if len(dataset) < 2:
        raise ValueError("Need at least 2 records to split the dataset.")

    test_total = val_ratio + test_ratio
    if test_total == 0:
        return {
            "train": dataset,
            "validation": dataset.select([]),
            "test": dataset.select([]),
        }

    try:
        train_test = dataset.train_test_split(
            test_size=test_total,
            seed=seed,
        )
    except ValueError as exc:
        raise ValueError(f"Split failed: {exc}") from exc

    if val_ratio == 0:
        return {
            "train": train_test["train"],
            "validation": train_test["train"].select([]),
            "test": train_test["test"],
        }

    test_fraction = test_ratio / test_total
    try:
        test_valid = train_test["test"].train_test_split(
            test_size=test_fraction,
            seed=seed,
        )
    except ValueError as exc:
        raise ValueError(f"Validation/test split failed: {exc}") from exc

    return {
        "train": train_test["train"],
        "validation": test_valid["train"],
        "test": test_valid["test"],
    }
