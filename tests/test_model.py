"""
test_model.py — Unit tests for src/model.py

Run with:  pytest tests/test_model.py -v
"""

import pytest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import load_data, preprocess, train, evaluate, predict, CLASS_NAMES


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture(scope="module")
def iris_splits():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="module")
def trained_model(iris_splits):
    X_train, X_test, y_train, y_test = iris_splits
    X_train_s, X_test_s, scaler = preprocess(X_train, X_test)
    model = train(X_train_s, y_train)
    return model, scaler, X_test_s, y_test


# ──────────────────────────────────────────────
# Data Tests
# ──────────────────────────────────────────────

class TestDataLoading:
    def test_load_data_shape(self):
        X, y = load_data()
        assert X.shape == (150, 4), "Iris should have 150 samples and 4 features"
        assert y.shape == (150,)

    def test_labels_are_valid(self):
        _, y = load_data()
        assert set(y) == {0, 1, 2}, "Labels should be 0, 1, 2"

    def test_no_nan_values(self):
        X, y = load_data()
        assert not np.isnan(X).any(), "Feature matrix must have no NaNs"


# ──────────────────────────────────────────────
# Preprocessing Tests
# ──────────────────────────────────────────────

class TestPreprocessing:
    def test_scaler_output_shape(self, iris_splits):
        X_train, X_test, y_train, y_test = iris_splits
        X_train_s, X_test_s, scaler = preprocess(X_train, X_test)
        assert X_train_s.shape == X_train.shape
        assert X_test_s.shape == X_test.shape

    def test_train_mean_near_zero(self, iris_splits):
        X_train, X_test, _, _ = iris_splits
        X_train_s, _, _ = preprocess(X_train, X_test)
        means = X_train_s.mean(axis=0)
        assert np.allclose(means, 0, atol=1e-6), "Scaled train features should have ~0 mean"

    def test_train_std_near_one(self, iris_splits):
        X_train, X_test, _, _ = iris_splits
        X_train_s, _, _ = preprocess(X_train, X_test)
        stds = X_train_s.std(axis=0)
        assert np.allclose(stds, 1, atol=1e-6), "Scaled train features should have ~1 std"


# ──────────────────────────────────────────────
# Training Tests
# ──────────────────────────────────────────────

class TestTraining:
    def test_model_trains_without_error(self, iris_splits):
        X_train, X_test, y_train, _ = iris_splits
        X_train_s, _, _ = preprocess(X_train, X_test)
        model = train(X_train_s, y_train)
        assert model is not None

    def test_model_has_correct_classes(self, trained_model):
        model, scaler, _, _ = trained_model
        assert list(model.classes_) == [0, 1, 2]

    def test_model_has_coef(self, trained_model):
        model, _, _, _ = trained_model
        assert hasattr(model, "coef_")
        assert model.coef_.shape[1] == 4  # 4 features


# ──────────────────────────────────────────────
# Evaluation Tests
# ──────────────────────────────────────────────

class TestEvaluation:
    def test_accuracy_above_dev_threshold(self, trained_model):
        """DEV threshold: accuracy >= 0.80"""
        model, _, X_test_s, y_test = trained_model
        acc, _ = evaluate(model, X_test_s, y_test)
        assert acc >= 0.80, f"Accuracy {acc:.2f} is below DEV threshold of 0.80"

    def test_accuracy_above_stg_threshold(self, trained_model):
        """STG threshold: accuracy >= 0.90"""
        model, _, X_test_s, y_test = trained_model
        acc, _ = evaluate(model, X_test_s, y_test)
        assert acc >= 0.90, f"Accuracy {acc:.2f} is below STG threshold of 0.90"

    def test_accuracy_above_prod_threshold(self, trained_model):
        """PROD threshold: accuracy >= 0.93"""
        model, _, X_test_s, y_test = trained_model
        acc, _ = evaluate(model, X_test_s, y_test)
        assert acc >= 0.93, f"Accuracy {acc:.2f} is below PROD threshold of 0.93"

    def test_report_contains_all_classes(self, trained_model):
        model, _, X_test_s, y_test = trained_model
        _, report = evaluate(model, X_test_s, y_test)
        for cls in CLASS_NAMES:
            assert cls in report, f"Class '{cls}' missing from report"


# ──────────────────────────────────────────────
# Prediction Tests
# ──────────────────────────────────────────────

class TestPrediction:
    def test_predict_returns_valid_class(self, trained_model):
        model, scaler, _, _ = trained_model
        result = predict(model, scaler, [5.1, 3.5, 1.4, 0.2])
        assert result in CLASS_NAMES

    def test_predict_setosa(self, trained_model):
        """Classic setosa sample — should predict setosa."""
        model, scaler, _, _ = trained_model
        result = predict(model, scaler, [5.1, 3.5, 1.4, 0.2])
        assert result == "setosa"

    def test_predict_virginica(self, trained_model):
        """Classic virginica sample — should predict virginica."""
        model, scaler, _, _ = trained_model
        result = predict(model, scaler, [6.3, 3.3, 6.0, 2.5])
        assert result == "virginica"

    def test_predict_output_is_string(self, trained_model):
        model, scaler, _, _ = trained_model
        result = predict(model, scaler, [5.8, 2.7, 5.1, 1.9])
        assert isinstance(result, str)
