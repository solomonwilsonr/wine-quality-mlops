"""
register_model.py
=================
Reads metrics from a training run, then registers the model artefact in
Azure ML with those metrics stored as tags for champion/challenger comparison.

Emits a GitHub Actions output variable:
    ::set-output name=model_version::<N>

Usage:
    python scripts/register_model.py \
        --subscription  $AZURE_SUBSCRIPTION_ID \
        --resource-group $AZURE_RESOURCE_GROUP \
        --workspace     $AZURE_ML_WORKSPACE \
        --model-path    ./outputs/model \
        --metrics-path  ./outputs/metrics.json \
        --model-name    wine-quality-classifier
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Register a trained model in Azure ML.")
    p.add_argument("--subscription",   required=True, help="Azure subscription ID.")
    p.add_argument("--resource-group", required=True, help="Azure resource group name.")
    p.add_argument("--workspace",      required=True, help="Azure ML workspace name.")
    p.add_argument("--model-path",     required=True, help="Local path to the model directory.")
    p.add_argument(
        "--metrics-path",
        required=True,
        help="Path to metrics.json produced by train.py.",
    )
    p.add_argument(
        "--model-name",
        default="wine-quality-classifier",
        help="Name to register the model under in Azure ML.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_metrics(metrics_path: str) -> dict:
    path = Path(metrics_path)
    if not path.exists():
        logger.error("Metrics file not found: %s", metrics_path)
        sys.exit(1)
    with open(path) as fh:
        metrics = json.load(fh)
    logger.info("Loaded metrics: %s", metrics)
    return metrics


def build_tags(metrics: dict) -> dict[str, str]:
    """
    Azure ML model tags must be string-valued.
    We store every numeric metric as a string so evaluate.py can read them back.
    """
    tags = {k: str(v) for k, v in metrics.items() if isinstance(v, (int, float))}
    tags["registered_by"] = "github-actions"
    return tags


def register_model(
    subscription: str,
    resource_group: str,
    workspace: str,
    model_path: str,
    model_name: str,
    tags: dict[str, str],
) -> str:
    """Register model in Azure ML; return the new version string."""
    try:
        from azure.ai.ml import MLClient
        from azure.ai.ml.entities import Model
        from azure.ai.ml.constants import AssetTypes
        from azure.identity import DefaultAzureCredential
    except ImportError as exc:
        logger.error("Missing Azure SDK dependency: %s", exc)
        sys.exit(1)

    logger.info(
        "Connecting to workspace '%s' in resource group '%s' (subscription: %s).",
        workspace, resource_group, subscription,
    )
    client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=subscription,
        resource_group_name=resource_group,
        workspace_name=workspace,
    )

    model_asset = Model(
        path=model_path,
        name=model_name,
        description="Wine Quality Classifier — RandomForest pipeline trained on UCI wine-quality-red dataset.",
        type=AssetTypes.MLFLOW_MODEL,
        tags=tags,
    )

    logger.info("Registering model '%s' from path '%s'...", model_name, model_path)
    registered = client.models.create_or_update(model_asset)

    logger.info(
        "Registered model '%s', version %s.",
        registered.name, registered.version,
    )
    return registered.version


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    metrics = load_metrics(args.metrics_path)
    tags = build_tags(metrics)

    version = register_model(
        subscription=args.subscription,
        resource_group=args.resource_group,
        workspace=args.workspace,
        model_path=args.model_path,
        model_name=args.model_name,
        tags=tags,
    )

    # Emit GitHub Actions output consumed by downstream steps.
    # GITHUB_OUTPUT env var is set in Actions runners; fall back to legacy syntax.
    import os
    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as fh:
            fh.write(f"model_version={version}\n")
            fh.write(f"model_name={args.model_name}\n")
    else:
        # Legacy set-output syntax (still supported in many runners)
        print(f"::set-output name=model_version::{version}")
        print(f"::set-output name=model_name::{args.model_name}")

    logger.info("Done. model_name=%s  model_version=%s", args.model_name, version)


if __name__ == "__main__":
    main()
