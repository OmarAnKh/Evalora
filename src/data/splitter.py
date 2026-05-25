from typing import Dict

from datasets import ClassLabel, Dataset, Features


def split_dataset(
    dataset: Dataset,
    label_key: str = "score",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, Dataset]:
    """Split a dataset into stratified train/val/test subsets.

    Raises ValueError when ratios are invalid or stratification is not possible.
    """
    if label_key not in dataset.column_names:
        raise ValueError(f"Missing '{label_key}' in dataset columns.")

    label_column = f"__{label_key}_label"
    dataset_for_split = dataset
    score_feature = dataset.features.get(label_key)
    if not isinstance(score_feature, ClassLabel):
        unique_scores = sorted(set(dataset[label_key]))
        class_label = ClassLabel(num_classes=len(unique_scores))
        features = dataset.features.copy()
        features[label_column] = class_label
        dataset_for_split = (
            dataset.map(
                lambda row: {label_column: unique_scores.index(row[label_key])}
            )
            .cast(Features(features))
        )
    else:
        features = dataset.features.copy()
        features[label_column] = score_feature
        dataset_for_split = dataset.map(lambda row: {label_column: row[label_key]}).cast(
            Features(features)
        )

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
        train_test = dataset_for_split.train_test_split(
            test_size=test_total,
            seed=seed,
            stratify_by_column=label_column,
        )
    except ValueError as exc:
        raise ValueError(f"Stratified split failed: {exc}") from exc

    if val_ratio == 0:
        train_dataset = train_test["train"].remove_columns(label_column)
        test_dataset = train_test["test"].remove_columns(label_column)
        return {
            "train": train_dataset,
            "validation": train_dataset.select([]),
            "test": test_dataset,
        }

    test_fraction = test_ratio / test_total
    try:
        test_valid = train_test["test"].train_test_split(
            test_size=test_fraction,
            seed=seed,
            stratify_by_column=label_column,
        )
    except ValueError as exc:
        raise ValueError(f"Stratified validation/test split failed: {exc}") from exc

    train_dataset = train_test["train"].remove_columns(label_column)
    validation_dataset = test_valid["train"].remove_columns(label_column)
    test_dataset = test_valid["test"].remove_columns(label_column)
    return {
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset,
    }
