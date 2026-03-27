"""
test_integration.py — Integration tests that mirror CI/CD quality gates.

These tests simulate what runs in each environment:
  - dev  → quick smoke tests, permissive thresholds
  - stg  → full pipeline integration, tighter thresholds
  - prod → strict accuracy gate + model artifact check

Run all:        pytest tests/test_integration.py -v
Run dev only:   pytest tests/test_integration.py -v -m dev
Run stg only:   pytest tests/test_integration.py -v -m stg
Run prod only:  pytest tests/test_integration.py -v -m prod
"""

import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import load_data, preprocess, train, evaluate, save_model, load_model, predict
from data_utils import validate_features, split_dataset
from config import get_config
from sklearn.model_selection import train_test_split


# ──────────────────────────────────────────────
# Pytest Markers
# ──────────────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line("markers", "dev: tests that run on the dev branch")
    config.addinivalue_line("markers", "stg: tests that run on the stg branch")
    config.addinivalue_line("markers", "prod: tests that run on the prod branch")


# ──────────────────────────────────────────────
# Shared Fixture
# ──────────────────────────────────────────────

@pytest.fixture(scope="module")
def full_pipeline():
    """Run the complete ML pipeline and return all artefacts."""
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_s, X_test_s, scaler = preprocess(X_train, X_test)
    model = train(X_train_s, y_train)
    acc, report = evaluate(model, X_test_s, y_test)
    return {
        "model": model,
        "scaler": scaler,
        "accuracy": acc,
        "report": report,
        "X_test": X_test_s,
        "y_test": y_test,
    }


# ──────────────────────────────────────────────
# DEV — Smoke Tests
# ──────────────────────────────────────────────

@pytest.mark.dev
class TestDevPipeline:
    """
    DEV branch checks: 'does it run at all?'
    Fast, lenient. Catch obvious breakage early.
    """

    def test_pipeline_runs_end_to_end(self, full_pipeline):
        """The whole pipeline should complete without exceptions."""
        assert full_pipeline["model"] is not None
        assert full_pipeline["scaler"] is not None

    def test_accuracy_above_dev_threshold(self, full_pipeline):
        """Dev threshold is 0.80 — very permissive for experiments."""
        acc = full_pipeline["accuracy"]
        assert acc >= 0.80, f"[DEV] Accuracy {acc:.2f} too low (< 0.80)"

    def test_single_prediction_works(self, full_pipeline):
        """A single prediction should return a valid class name."""
        result = predict(
            full_pipeline["model"],
            full_pipeline["scaler"],
            [5.1, 3.5, 1.4, 0.2],
        )
        assert result in ["setosa", "versicolor", "virginica"]

    def test_feature_validator_works(self):
        assert validate_features([5.1, 3.5, 1.4, 0.2]) is True
        assert validate_features([99, 99, 99, 99]) is False


# ──────────────────────────────────────────────
# STG — Integration / Pre-release Tests
# ──────────────────────────────────────────────

@pytest.mark.stg
class TestStgPipeline:
    """
    STG branch checks: 'is this ready for release?'
    Mirrors production behaviour. Tighter thresholds.
    """

    def test_accuracy_above_stg_threshold(self, full_pipeline):
        """Staging threshold is 0.90."""
        acc = full_pipeline["accuracy"]
        assert acc >= 0.90, f"[STG] Accuracy {acc:.2f} below staging gate (< 0.90)"

    def test_model_save_and_reload(self, full_pipeline, tmp_path):
        """Model must survive a save/load round-trip."""
        save_model(full_pipeline["model"], full_pipeline["scaler"], path=str(tmp_path))
        loaded_model, loaded_scaler = load_model(path=str(tmp_path))

        # Reloaded model must give identical predictions
        original_preds = full_pipeline["model"].predict(full_pipeline["X_test"])
        reloaded_preds = loaded_model.predict(full_pipeline["X_test"])
        assert list(original_preds) == list(reloaded_preds), "Predictions differ after reload"

    def test_report_contains_precision_recall(self, full_pipeline):
        report = full_pipeline["report"]
        assert "precision" in report
        assert "recall" in report
        assert "f1-score" in report

    def test_all_classes_predicted(self, full_pipeline):
        """Model must not degenerate to predicting only one class."""
        import numpy as np
        preds = full_pipeline["model"].predict(full_pipeline["X_test"])
        unique_preds = set(np.unique(preds).tolist())
        assert len(unique_preds) >= 2, "Model is only predicting one class — degenerate model!"

    def test_dataset_split_integrity(self):
        """Validate the 3-way split produces non-overlapping sets."""
        X, y = load_data()
        splits = split_dataset(X, y)
        X_train, X_val, X_test = splits[0], splits[1], splits[2]
        train_set = set(map(tuple, X_train))
        test_set = set(map(tuple, X_test))
        assert len(train_set & test_set) == 0, "Train and test sets overlap!"


# ──────────────────────────────────────────────
# PROD — Strict Quality Gates
# ──────────────────────────────────────────────

@pytest.mark.prod
class TestProdPipeline:
    """
    PROD branch checks: 'is this safe to serve real users?'
    Strictest thresholds. Model artifact must exist.
    """

    def test_accuracy_above_prod_threshold(self, full_pipeline):
        """Production threshold is 0.93."""
        acc = full_pipeline["accuracy"]
        assert acc >= 0.93, f"[PROD] Accuracy {acc:.2f} below production gate (< 0.93)"

    def test_model_artifacts_exist_after_save(self, full_pipeline, tmp_path):
        """Both model.pkl and scaler.pkl must be written to disk."""
        save_model(full_pipeline["model"], full_pipeline["scaler"], path=str(tmp_path))
        assert os.path.exists(os.path.join(str(tmp_path), "model.pkl"))
        assert os.path.exists(os.path.join(str(tmp_path), "scaler.pkl"))

    def test_config_prod_threshold_matches_test(self):
        """Config threshold and test threshold must be in sync."""
        os.environ["ENV"] = "prod"
        cfg = get_config()
        assert cfg["min_accuracy_threshold"] == 0.93

    def test_known_samples_predict_correctly(self, full_pipeline):
        """
        Regression test: hard-coded canonical samples must always
        predict the right class in production.
        """
        canonical = [
            ([5.1, 3.5, 1.4, 0.2], "setosa"),
            ([7.0, 3.2, 4.7, 1.4], "versicolor"),
            ([6.3, 3.3, 6.0, 2.5], "virginica"),
        ]
        model, scaler = full_pipeline["model"], full_pipeline["scaler"]
        for features, expected in canonical:
            result = predict(model, scaler, features)
            assert result == expected, (
                f"[PROD] Canonical sample {features} predicted '{result}', expected '{expected}'"
            )

    def test_invalid_features_are_rejected(self):
        """Production must never accept out-of-range inputs."""
        bad_inputs = [
            [],
            [1, 2, 3],
            [99, 99, 99, 99],
            [5.1, "x", 1.4, 0.2],
        ]
        for bad in bad_inputs:
            assert validate_features(bad) is False, f"Bad input {bad} was not rejected"
