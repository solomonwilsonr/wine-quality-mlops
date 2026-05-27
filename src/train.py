"""
Training script for the Wine Quality Classifier.

Usage (local):
    python src/train.py --experiment-name my-experiment --n-estimators 200

Usage (Azure ML job via azure/train_job.yml):
    Invoked automatically by the GitHub Actions pipeline.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from utils import FeatureEngineer, setup_logging

logger = setup_logging(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train wine quality classifier.")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to CSV data file. Defaults to sklearn built-in wine dataset.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="wine-quality-classifier",
        help="Registered model name in Azure ML.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="wine-quality-training",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Directory to write model artefacts and metrics.json.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=None,
        help="Override n_estimators (skips grid search when set).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Override max_depth (skips grid search when set).",
    )
    return parser.parse_args()


def load_data(data_path: str | None) -> tuple[pd.DataFrame, pd.Series]:
    """Load wine dataset either from a CSV or the sklearn built-in."""
    if data_path and Path(data_path).exists():
        logger.info("Loading data from %s", data_path)
        df = pd.read_csv(data_path)
        # Expect last column to be target
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
    else:
        logger.info("Loading built-in sklearn wine dataset.")
        wine = load_wine(as_frame=True)
        X = wine.frame.drop(columns=["target"])
        y = wine.frame["target"]
    return X, y


def build_pipeline(n_estimators: int | None, max_depth: int | None) -> Pipeline:
    """Construct the sklearn Pipeline with feature engineering + classifier."""
    fe = FeatureEngineer(apply_polynomial=True, poly_degree=2, clip_outliers=True)
    rf = RandomForestClassifier(
        n_estimators=n_estimators or 100,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    return Pipeline([("feature_engineer", fe), ("classifier", rf)])


def run_grid_search(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Run 5-fold GridSearchCV over RF hyperparameters."""
    param_grid = {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_split": [2, 5],
        "classifier__min_samples_leaf": [1, 2],
    }
    logger.info("Running GridSearchCV (cv=5)...")
    gs = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    gs.fit(X_train, y_train)
    logger.info("Best params: %s", gs.best_params_)
    logger.info("Best CV F1: %.4f", gs.best_score_)
    return gs.best_estimator_, gs.best_params_, gs.best_score_


def compute_metrics(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute and return a full set of evaluation metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    le = LabelEncoder()
    y_test_enc = le.fit_transform(y_test)

    # One-vs-rest AUC for multi-class
    auc = roc_auc_score(y_test_enc, y_proba, multi_class="ovr", average="weighted")

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
        "precision_weighted": float(precision_score(y_test, y_pred, average="weighted")),
        "recall_weighted": float(recall_score(y_test, y_pred, average="weighted")),
        "auc_roc_weighted": float(auc),
    }


def main():
    args = parse_args()

    # Configure MLflow — azure/azureml sets MLFLOW_TRACKING_URI automatically
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run() as run:
        logger.info("MLflow run ID: %s", run.info.run_id)
        mlflow.set_tags(
            {
                "git_sha": os.getenv("GIT_SHA", "local"),
                "pipeline_run_id": os.getenv("RUN_ID", "local"),
                "model_name": args.model_name,
            }
        )

        # ---- Data --------------------------------------------------------
        X, y = load_data(args.data_path)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
        )
        mlflow.log_params(
            {
                "test_size": args.test_size,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "n_features": X.shape[1],
            }
        )
        logger.info("Train: %d samples, Test: %d samples", len(X_train), len(X_test))

        # ---- Training ----------------------------------------------------
        pipeline = build_pipeline(args.n_estimators, args.max_depth)

        if args.n_estimators is None and args.max_depth is None:
            # Full grid search path (used in CI)
            best_pipeline, best_params, best_cv_score = run_grid_search(
                pipeline, X_train, y_train
            )
            mlflow.log_params({k.replace("classifier__", ""): v for k, v in best_params.items()})
            mlflow.log_metric("cv_f1_weighted", best_cv_score)
        else:
            # Fast path: caller specified hyperparameters (local dev / testing)
            logger.info("Fitting pipeline with user-supplied hyperparameters.")
            best_pipeline = pipeline.fit(X_train, y_train)

        # ---- Evaluation --------------------------------------------------
        metrics = compute_metrics(best_pipeline, X_test, y_test)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        logger.info("Metrics: %s", json.dumps(metrics, indent=2))

        # ---- Artefacts ---------------------------------------------------
        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            artifact_path="model",
            registered_model_name=None,  # Registration happens in register_model.py
            input_example=X_test.head(3),
            signature=mlflow.models.infer_signature(X_test, best_pipeline.predict(X_test)),
        )

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w") as fh:
            json.dump(
                {
                    **metrics,
                    "run_id": run.info.run_id,
                    "experiment_name": args.experiment_name,
                },
                fh,
                indent=2,
            )
        mlflow.log_artifact(str(metrics_path), artifact_path="model_metrics")
        logger.info("Metrics saved to %s", metrics_path)

    logger.info("Training complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
