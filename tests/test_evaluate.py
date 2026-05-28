"""
tests/test_evaluate.py
======================
Unit tests for src/evaluate.py.

The Azure ML client is mocked via the mock_azure_client fixture from conftest.py.

Tests cover:
  - test_passes_when_no_champion_exists    : first deploy always passes
  - test_passes_when_improvement_exceeds_threshold
  - test_fails_when_no_improvement
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import evaluate  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_metrics(tmp_path: Path, metrics: dict) -> str:
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(metrics))
    return str(path)


GOOD_METRICS = {
    "accuracy": 0.95,
    "f1_weighted": 0.95,
    "precision_weighted": 0.94,
    "recall_weighted": 0.95,
    "auc_roc_weighted": 0.98,
}

CHAMPION_METRICS_LOW = {
    "f1_weighted": "0.930",   # challenger is better by > 0.005
    "accuracy": "0.928",
}

CHAMPION_METRICS_HIGH = {
    "f1_weighted": "0.948",   # challenger 0.95 - 0.948 = 0.002 < 0.005 threshold
    "accuracy": "0.947",
}


# ---------------------------------------------------------------------------
# Test: no existing champion
# ---------------------------------------------------------------------------

class TestPassesWhenNoChampionExists:
    """When no model is registered in Azure ML the evaluation gate must pass."""

    def test_fetch_returns_none_when_no_models(self, mock_azure_client):
        """fetch_champion_metrics returns None when model list is empty."""
        mock_azure_client.models.list.return_value = []
        with patch("azure.ai.ml.MLClient", return_value=mock_azure_client), \
             patch("azure.identity.DefaultAzureCredential"):
            result = evaluate.fetch_champion_metrics(
                model_name="wine-quality-classifier",
                workspace="ws",
                resource_group="rg",
                subscription="sub",
            )
        assert result is None

    def test_evaluate_passes_with_none_champion(self, tmp_path, mock_azure_client):
        """evaluate() exits 0 when champion is None (first deployment)."""
        mock_azure_client.models.list.return_value = []
        metrics_path = write_metrics(tmp_path, GOOD_METRICS)

        with patch("azure.ai.ml.MLClient", return_value=mock_azure_client), \
             patch("azure.identity.DefaultAzureCredential"):
            new_metrics = evaluate.load_new_metrics(metrics_path)
            champion = evaluate.fetch_champion_metrics(
                model_name="wine-quality-classifier",
                workspace="ws",
                resource_group="rg",
                subscription="sub",
            )
            # When champion is None the gate should pass
            assert evaluate.should_deploy(new_metrics, champion, primary_metric="f1_weighted")


# ---------------------------------------------------------------------------
# Test: challenger clearly beats champion
# ---------------------------------------------------------------------------

class TestPassesWhenImprovementExceedsThreshold:
    def test_improvement_above_threshold_returns_true(self, tmp_path, mock_azure_client):
        """
        Challenger f1=0.95, champion f1=0.930 => delta=0.02 > 0.005 threshold.
        should_deploy must return True.
        """
        mock_model = MagicMock()
        mock_model.version = "3"
        mock_model.tags = CHAMPION_METRICS_LOW
        mock_azure_client.models.list.return_value = [mock_model]

        metrics_path = write_metrics(tmp_path, GOOD_METRICS)
        new_metrics = evaluate.load_new_metrics(metrics_path)

        with patch("azure.ai.ml.MLClient", return_value=mock_azure_client), \
             patch("azure.identity.DefaultAzureCredential"):
            champion = evaluate.fetch_champion_metrics(
                model_name="wine-quality-classifier",
                workspace="ws",
                resource_group="rg",
                subscription="sub",
            )

        result = evaluate.should_deploy(
            new_metrics, champion, primary_metric="f1_weighted", min_improvement=0.005
        )
        assert result is True

    def test_large_improvement_passes(self, tmp_path, mock_azure_client):
        poor_champion_tags = {"f1_weighted": "0.700", "accuracy": "0.680"}
        mock_model = MagicMock()
        mock_model.version = "1"
        mock_model.tags = poor_champion_tags
        mock_azure_client.models.list.return_value = [mock_model]

        metrics_path = write_metrics(tmp_path, GOOD_METRICS)
        new_metrics = evaluate.load_new_metrics(metrics_path)

        with patch("azure.ai.ml.MLClient", return_value=mock_azure_client), \
             patch("azure.identity.DefaultAzureCredential"):
            champion = evaluate.fetch_champion_metrics(
                model_name="wine-quality-classifier",
                workspace="ws",
                resource_group="rg",
                subscription="sub",
            )

        assert evaluate.should_deploy(new_metrics, champion, primary_metric="f1_weighted")


# ---------------------------------------------------------------------------
# Test: challenger does not beat champion by enough
# ---------------------------------------------------------------------------

class TestFailsWhenNoImprovement:
    def test_below_threshold_returns_false(self, tmp_path, mock_azure_client):
        """
        Challenger f1=0.95, champion f1=0.948 => delta=0.002 < 0.005 threshold.
        should_deploy must return False.
        """
        mock_model = MagicMock()
        mock_model.version = "5"
        mock_model.tags = CHAMPION_METRICS_HIGH
        mock_azure_client.models.list.return_value = [mock_model]

        metrics_path = write_metrics(tmp_path, GOOD_METRICS)
        new_metrics = evaluate.load_new_metrics(metrics_path)

        with patch("azure.ai.ml.MLClient", return_value=mock_azure_client), \
             patch("azure.identity.DefaultAzureCredential"):
            champion = evaluate.fetch_champion_metrics(
                model_name="wine-quality-classifier",
                workspace="ws",
                resource_group="rg",
                subscription="sub",
            )

        result = evaluate.should_deploy(
            new_metrics, champion, primary_metric="f1_weighted", min_improvement=0.005
        )
        assert result is False

    def test_equal_score_returns_false(self, tmp_path, mock_azure_client):
        """Exactly equal to champion is not sufficient — must strictly exceed threshold."""
        same_score_tags = {"f1_weighted": str(GOOD_METRICS["f1_weighted"])}
        mock_model = MagicMock()
        mock_model.version = "4"
        mock_model.tags = same_score_tags
        mock_azure_client.models.list.return_value = [mock_model]

        metrics_path = write_metrics(tmp_path, GOOD_METRICS)
        new_metrics = evaluate.load_new_metrics(metrics_path)

        with patch("azure.ai.ml.MLClient", return_value=mock_azure_client), \
             patch("azure.identity.DefaultAzureCredential"):
            champion = evaluate.fetch_champion_metrics(
                model_name="wine-quality-classifier",
                workspace="ws",
                resource_group="rg",
                subscription="sub",
            )

        result = evaluate.should_deploy(
            new_metrics, champion, primary_metric="f1_weighted", min_improvement=0.005
        )
        assert result is False

    def test_regression_returns_false(self, tmp_path, mock_azure_client):
        """If the new model is actually worse, should_deploy must return False."""
        superior_champion = {"f1_weighted": "0.990", "accuracy": "0.989"}
        mock_model = MagicMock()
        mock_model.version = "10"
        mock_model.tags = superior_champion
        mock_azure_client.models.list.return_value = [mock_model]

        worse_metrics = {**GOOD_METRICS, "f1_weighted": 0.920}
        metrics_path = write_metrics(tmp_path, worse_metrics)
        new_metrics = evaluate.load_new_metrics(metrics_path)

        with patch("azure.ai.ml.MLClient", return_value=mock_azure_client), \
             patch("azure.identity.DefaultAzureCredential"):
            champion = evaluate.fetch_champion_metrics(
                model_name="wine-quality-classifier",
                workspace="ws",
                resource_group="rg",
                subscription="sub",
            )

        result = evaluate.should_deploy(
            new_metrics, champion, primary_metric="f1_weighted", min_improvement=0.005
        )
        assert result is False
