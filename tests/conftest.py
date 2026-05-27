"""
Pytest fixtures shared across the test suite.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Make src importable without installing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import FeatureEngineer  # noqa: E402


@pytest.fixture(scope="session")
def wine_dataset():
    """Full wine dataset as a (X, y) tuple."""
    wine = load_wine(as_frame=True)
    X = wine.frame.drop(columns=["target"])
    y = wine.frame["target"]
    return X, y


@pytest.fixture(scope="session")
def sample_data(wine_dataset):
    """Small balanced sample — 60 rows — for fast unit tests."""
    X, y = wine_dataset
    # Stratified sample: 20 rows per class
    parts = []
    for cls in y.unique():
        mask = y == cls
        parts.append(X[mask].head(20))
    X_small = pd.concat(parts).reset_index(drop=True)
    y_small = pd.concat([y[y == cls].head(20) for cls in y.unique()]).reset_index(drop=True)
    return X_small, y_small


@pytest.fixture(scope="session")
def trained_model(sample_data):
    """
    A quickly trained RandomForest pipeline for tests that need a live model.

    Scope=session keeps training to one run regardless of how many tests use it.
    """
    X, y = sample_data
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    fe = FeatureEngineer(apply_polynomial=True, poly_degree=2, clip_outliers=True)
    rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=1)
    pipeline = Pipeline([("feature_engineer", fe), ("classifier", rf)])
    pipeline.fit(X_train, y_train)
    return pipeline


@pytest.fixture()
def mock_azure_client():
    """
    Mocked Azure ML MLClient so tests never hit Azure.
    """
    with patch("azure.ai.ml.MLClient") as MockClient:
        instance = MockClient.return_value

        # Mock model listing
        mock_model = MagicMock()
        mock_model.version = "5"
        mock_model.tags = {
            "f1_weighted": "0.940",
            "accuracy": "0.938",
        }
        instance.models.list.return_value = [mock_model]
        instance.models.create_or_update.return_value = mock_model

        yield instance


@pytest.fixture()
def sample_score_payload():
    """A valid JSON scoring payload for the score.py endpoint."""
    return {
        "data": [
            {
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
        ]
    }
