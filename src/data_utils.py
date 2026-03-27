"""
data_utils.py — Data loading & validation helpers.
"""

import numpy as np
from sklearn.datasets import load_iris


def get_feature_names():
    return ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def validate_features(features: list) -> bool:
    """
    Validate that input features are correct:
    - Must have exactly 4 values
    - All values must be positive numbers
    - Values must be within realistic Iris measurement ranges
    """
    if len(features) != 4:
        return False
    ranges = [(4.0, 8.0), (2.0, 5.0), (1.0, 7.0), (0.1, 3.0)]
    for val, (lo, hi) in zip(features, ranges):
        if not isinstance(val, (int, float)):
            return False
        if val < lo or val > hi:
            return False
    return True


def split_dataset(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split dataset into train / val / test sets.
    Returns: X_train, X_val, X_test, y_train, y_val, y_test
    """
    from sklearn.model_selection import train_test_split

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def data_summary(X, y):
    """Return a quick summary dict for the dataset."""
    return {
        "num_samples": len(X),
        "num_features": X.shape[1],
        "classes": np.unique(y).tolist(),
        "class_counts": {int(c): int(np.sum(y == c)) for c in np.unique(y)},
    }
