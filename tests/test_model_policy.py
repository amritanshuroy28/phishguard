import pickle

import numpy as np
import pytest

from backend.services.ensemble_service import (
    PredictionError,
    UNIFIED_FEATURES,
    UltimateMLService,
)
from ml_pipeline.train_calibrated import canonicalize_url, choose_threshold, metric_report


class ConstantProbabilityModel:
    def __init__(self, probability):
        self.probability = probability

    def predict_proba(self, features):
        return np.tile([1 - self.probability, self.probability], (len(features), 1))


def write_bundle(path, probability=0.75, threshold=0.8, certified=True):
    with path.open("wb") as handle:
        pickle.dump(
            {
                "model": ConstantProbabilityModel(probability),
                "features": UNIFIED_FEATURES,
                "feature_names": UNIFIED_FEATURES,
                "model_version": "test-certified",
                "decision_policy": {"phishing_probability_threshold": threshold},
                "certification": {"certified": certified},
            },
            handle,
        )


def test_canonicalize_url_removes_non_identifying_variation():
    assert canonicalize_url(" HTTP://Example.COM:80/#ignored ") == "http://example.com"
    assert canonicalize_url("example.com/path?a=1") == "https://example.com/path?a=1"
    assert canonicalize_url("mailto:security@example.com") is None


def test_threshold_selection_enforces_legitimate_false_positive_budget():
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    probabilities = np.array([0.01, 0.05, 0.10, 0.15, 0.90, 0.80, 0.85, 0.90, 0.95, 0.99])

    threshold, report = choose_threshold(labels, probabilities, max_legitimate_fpr=0.01)

    assert threshold > 0.90
    assert report["legitimate_false_positive_rate"] == 0.0
    assert report["balanced_accuracy"] >= 0.5


def test_metric_report_counts_legitimate_false_positives():
    report = metric_report(
        np.array([0, 0, 1, 1]), np.array([0.3, 0.9, 0.2, 0.8]), threshold=0.5
    )

    assert report["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert report["legitimate_false_positive_rate"] == 0.5
    assert report["balanced_accuracy"] == 0.5


def test_runtime_uses_persisted_threshold_and_raw_probability(tmp_path):
    model_path = tmp_path / "model.pkl"
    write_bundle(model_path, probability=0.75, threshold=0.8)

    service = UltimateMLService(model_path)
    prediction, probability = service.predict(np.zeros(len(UNIFIED_FEATURES)))

    assert service.loaded
    assert prediction is False
    assert probability == 0.75
    assert service.get_info()["decision_policy"]["phishing_probability_threshold"] == 0.8


def test_runtime_rejects_uncertified_bundle_and_feature_mismatch(tmp_path):
    model_path = tmp_path / "uncertified.pkl"
    write_bundle(model_path, certified=False)
    service = UltimateMLService(model_path)

    assert not service.loaded
    with pytest.raises(PredictionError, match="No certified"):
        service.predict(np.zeros(len(UNIFIED_FEATURES)))
