"""
test_data_utils.py — Unit tests for src/data_utils.py

Run with:  pytest tests/test_data_utils.py -v
"""

import pytest
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_utils import validate_features, split_dataset, data_summary, get_feature_names
from model import load_data


# ──────────────────────────────────────────────
# Feature Validation Tests
# ──────────────────────────────────────────────

class TestValidateFeatures:
    def test_valid_setosa_features(self):
        assert validate_features([5.1, 3.5, 1.4, 0.2]) is True

    def test_valid_virginica_features(self):
        assert validate_features([6.3, 3.3, 6.0, 2.5]) is True

    def test_too_few_features(self):
        assert validate_features([5.1, 3.5, 1.4]) is False

    def test_too_many_features(self):
        assert validate_features([5.1, 3.5, 1.4, 0.2, 9.9]) is False

    def test_empty_features(self):
        assert validate_features([]) is False

    def test_out_of_range_sepal_length(self):
        # sepal_length must be 4.0–8.0; 0.5 is out of range
        assert validate_features([0.5, 3.5, 1.4, 0.2]) is False

    def test_out_of_range_petal_width(self):
        # petal_width must be 0.1–3.0; 10.0 is out of range
        assert validate_features([5.1, 3.5, 1.4, 10.0]) is False

    def test_string_value_is_invalid(self):
        assert validate_features([5.1, "wide", 1.4, 0.2]) is False

    def test_none_value_is_invalid(self):
        assert validate_features([5.1, None, 1.4, 0.2]) is False


# ──────────────────────────────────────────────
# Dataset Splitting Tests
# ──────────────────────────────────────────────

class TestSplitDataset:
    @pytest.fixture(scope="class")
    def splits(self):
        X, y = load_data()
        return split_dataset(X, y, test_size=0.2, val_size=0.1)

    def test_total_samples_preserved(self, splits):
        X_train, X_val, X_test, y_train, y_val, y_test = splits
        total = len(X_train) + len(X_val) + len(X_test)
        assert total == 150

    def test_test_set_is_20_percent(self, splits):
        _, _, X_test, _, _, _ = splits
        assert len(X_test) == 30  # 20% of 150

    def test_no_overlap_between_splits(self, splits):
        """Ensure train/val/test indices are disjoint."""
        X_train, X_val, X_test, _, _, _ = splits
        # Convert rows to tuples for set comparison
        train_set = set(map(tuple, X_train))
        val_set = set(map(tuple, X_val))
        test_set = set(map(tuple, X_test))
        assert len(train_set & test_set) == 0
        assert len(train_set & val_set) == 0

    def test_labels_match_features(self, splits):
        X_train, X_val, X_test, y_train, y_val, y_test = splits
        assert len(X_train) == len(y_train)
        assert len(X_val) == len(y_val)
        assert len(X_test) == len(y_test)


# ──────────────────────────────────────────────
# Data Summary Tests
# ──────────────────────────────────────────────

class TestDataSummary:
    @pytest.fixture(scope="class")
    def summary(self):
        X, y = load_data()
        return data_summary(X, y)

    def test_num_samples(self, summary):
        assert summary["num_samples"] == 150

    def test_num_features(self, summary):
        assert summary["num_features"] == 4

    def test_classes(self, summary):
        assert summary["classes"] == [0, 1, 2]

    def test_class_counts_are_balanced(self, summary):
        """Iris is perfectly balanced: 50 samples per class."""
        for cls, count in summary["class_counts"].items():
            assert count == 50, f"Class {cls} has {count} samples, expected 50"


# ──────────────────────────────────────────────
# Feature Names Tests
# ──────────────────────────────────────────────

class TestFeatureNames:
    def test_returns_four_names(self):
        names = get_feature_names()
        assert len(names) == 4

    def test_names_are_strings(self):
        for name in get_feature_names():
            assert isinstance(name, str)
