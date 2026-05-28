"""
tests/test_score.py
===================
Unit tests for src/score.py.

All tests mock the global _model so no real model file is needed.

Tests cover:
  - test_run_returns_predictions_and_probabilities
  - test_run_rejects_missing_features
  - test_run_rejects_oversized_batch
  - test_run_returns_latency_ms
"""

import importlib
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def score_module():
    """
    Import (or re-import) score.py with a fresh module state so each test
    gets a clean _model global.
    """
    import score
    importlib.reload(score)
    return score


@pytest.fixture()
def mock_model():
    """A sklearn-like model mock that returns deterministic predictions."""
    model = MagicMock()
    # predict returns label 1 for every sample
    model.predict.side_effect = lambda X: np.ones(len(X), dtype=int)
    # predict_proba returns [0.2, 0.8] for every sample (binary)
    model.predict_proba.side_effect = lambda X: np.tile([0.2, 0.8], (len(X), 1))
    model.classes_ = np.array([0, 1])
    return model


@pytest.fixture()
def patched_score(score_module, mock_model):
    """
    Inject the mock model into score._model so run() uses it without
    touching disk or Azure.
    """
    score_module._model = mock_model
    return score_module


def _make_payload(n: int = 1) -> str:
    """Return a valid JSON scoring payload with n records."""
    record = {
        "alcohol": 13.2,
        "malic_acid": 1.78,
        "ash": 2.14,
        "alcalinity_of_ash": 11.2,
        "magnesium": 100.0,
        "total_phenols": 2.65,
        "flavanoids": 2.76,
        "nonflavanoid_phenols": 0.26,
        "proanthocyanins": 1.28,
        "color_intensity": 4.38,
        "hue": 1.05,
        "od280/od315_of_diluted_wines": 3.40,
        "proline": 1050.0,
    }
    return json.dumps({"data": [record] * n})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRunReturnsPredictionsAndProbabilities:
    def test_predictions_key_present(self, patched_score):
        result = json.loads(patched_score.run(_make_payload(1)))
        assert "predictions" in result

    def test_probabilities_key_present(self, patched_score):
        result = json.loads(patched_score.run(_make_payload(1)))
        assert "probabilities" in result

    def test_prediction_count_matches_input(self, patched_score):
        n = 3
        result = json.loads(patched_score.run(_make_payload(n)))
        assert len(result["predictions"]) == n

    def test_probability_rows_match_input(self, patched_score):
        n = 5
        result = json.loads(patched_score.run(_make_payload(n)))
        assert len(result["probabilities"]) == n

    def test_predictions_are_integers(self, patched_score):
        result = json.loads(patched_score.run(_make_payload(2)))
        for pred in result["predictions"]:
            assert isinstance(pred, (int, float))

    def test_probabilities_sum_to_one(self, patched_score):
        result = json.loads(patched_score.run(_make_payload(2)))
        for row in result["probabilities"]:
            assert abs(sum(row) - 1.0) < 1e-6


class TestRunRejectsMissingFeatures:
    def test_missing_one_feature_returns_error(self, patched_score):
        record = {
            "alcohol": 13.2,
            # malic_acid deliberately omitted
            "ash": 2.14,
            "alcalinity_of_ash": 11.2,
            "magnesium": 100.0,
            "total_phenols": 2.65,
            "flavanoids": 2.76,
            "nonflavanoid_phenols": 0.26,
            "proanthocyanins": 1.28,
            "color_intensity": 4.38,
            "hue": 1.05,
            "od280/od315_of_diluted_wines": 3.40,
            "proline": 1050.0,
        }
        payload = json.dumps({"data": [record]})
        result = json.loads(patched_score.run(payload))
        assert "error" in result

    def test_empty_data_list_returns_error(self, patched_score):
        payload = json.dumps({"data": []})
        result = json.loads(patched_score.run(payload))
        assert "error" in result

    def test_missing_data_key_returns_error(self, patched_score):
        payload = json.dumps({"records": []})
        result = json.loads(patched_score.run(payload))
        assert "error" in result


class TestRunRejectsOversizedBatch:
    def test_1001_records_rejected(self, patched_score):
        result = json.loads(patched_score.run(_make_payload(1001)))
        assert "error" in result

    def test_1000_records_accepted(self, patched_score):
        result = json.loads(patched_score.run(_make_payload(1000)))
        assert "predictions" in result

    def test_error_message_mentions_limit(self, patched_score):
        result = json.loads(patched_score.run(_make_payload(1500)))
        assert "error" in result
        # The error message should mention the batch limit
        assert "1000" in str(result["error"]) or "limit" in str(result["error"]).lower()


class TestRunReturnsLatencyMs:
    def test_latency_ms_key_present(self, patched_score):
        result = json.loads(patched_score.run(_make_payload(1)))
        assert "latency_ms" in result

    def test_latency_ms_is_non_negative(self, patched_score):
        result = json.loads(patched_score.run(_make_payload(1)))
        assert result["latency_ms"] >= 0

    def test_latency_ms_is_numeric(self, patched_score):
        result = json.loads(patched_score.run(_make_payload(1)))
        assert isinstance(result["latency_ms"], (int, float))
