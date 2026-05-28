"""Shared utilities for the wine quality ML pipeline."""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a configured logger.

    In Azure-hosted jobs the APPLICATIONINSIGHTS_CONNECTION_STRING env var is
    expected to be set.  When present an OpenCensus Azure Monitor handler is
    attached so logs flow into Application Insights automatically.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    conn_str = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
    if conn_str:
        try:
            from opencensus.ext.azure.log_exporter import AzureLogHandler

            azure_handler = AzureLogHandler(connection_string=conn_str)
            azure_handler.setFormatter(
                logging.Formatter("%(message)s")
            )
            logger.addHandler(azure_handler)
            logger.debug("Azure Monitor log handler attached.")
        except ImportError:
            logger.warning(
                "opencensus-ext-azure not installed; skipping Azure Monitor logging."
            )

    return logger


def load_config(path: str) -> dict:
    """Load a YAML config file and return it as a dict."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh) or {}


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible feature engineering transformer.

    Steps applied:
    1. IQR-based outlier clipping on all numeric columns.
    2. Optional polynomial features for a selected subset of columns.
    3. Log1p transformation for heavily right-skewed columns.
    """

    # Columns that benefit most from polynomial expansion in wine quality data
    POLY_COLS = ["alcohol", "volatile acidity", "sulphates", "citric acid"]
    # Columns with strong right skew
    LOG_COLS = ["residual sugar", "free sulfur dioxide", "total sulfur dioxide"]

    def __init__(
        self,
        apply_polynomial: bool = True,
        poly_degree: int = 2,
        clip_outliers: bool = True,
        iqr_multiplier: float = 1.5,
    ):
        self.apply_polynomial = apply_polynomial
        self.poly_degree = poly_degree
        self.clip_outliers = clip_outliers
        self.iqr_multiplier = iqr_multiplier

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":
        """Compute IQR bounds and fit polynomial features transformer."""
        self._feature_names_in = list(X.columns)

        if self.clip_outliers:
            self._lower = {}
            self._upper = {}
            for col in X.select_dtypes(include=[np.number]).columns:
                q1 = X[col].quantile(0.25)
                q3 = X[col].quantile(0.75)
                iqr = q3 - q1
                self._lower[col] = q1 - self.iqr_multiplier * iqr
                self._upper[col] = q3 + self.iqr_multiplier * iqr

        if self.apply_polynomial:
            poly_input_cols = [c for c in self.POLY_COLS if c in X.columns]
            self._poly_cols = poly_input_cols
            if poly_input_cols:
                self._poly = PolynomialFeatures(
                    degree=self.poly_degree,
                    interaction_only=False,
                    include_bias=False,
                )
                self._poly.fit(X[poly_input_cols])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering and return an enriched DataFrame."""
        X = X.copy()

        # Step 1: Clip outliers
        if self.clip_outliers and hasattr(self, "_lower"):
            for col, low in self._lower.items():
                if col in X.columns:
                    X[col] = X[col].clip(lower=low, upper=self._upper[col])

        # Step 2: Log1p for skewed columns
        for col in self.LOG_COLS:
            if col in X.columns:
                X[f"{col}_log1p"] = np.log1p(X[col])

        # Step 3: Polynomial features
        if self.apply_polynomial and hasattr(self, "_poly") and self._poly_cols:
            poly_input_cols = [c for c in self._poly_cols if c in X.columns]
            if poly_input_cols:
                poly_array = self._poly.transform(X[poly_input_cols])
                poly_names = self._poly.get_feature_names_out(poly_input_cols)
                poly_df = pd.DataFrame(
                    poly_array, columns=poly_names, index=X.index
                )
                # Drop raw source columns that are already in poly output
                for col in poly_input_cols:
                    if col in poly_df.columns:
                        poly_df = poly_df.drop(columns=[col])
                X = pd.concat([X, poly_df], axis=1)

        return X
