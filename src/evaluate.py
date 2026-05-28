"""
Model evaluation gate.

Compares a newly trained model's metrics against the current champion model
registered in Azure ML. Exits 0 if the new model is better, exits 1 otherwise.

This script is called by the CI pipeline before registering the new model.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)

# Primary metric used for champion/challenger comparison
PRIMARY_METRIC = "f1_weighted"
# Minimum improvement threshold required to justify a new deployment
MIN_IMPROVEMENT = 0.005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate new model vs champion.")
    parser.add_argument(
        "--metrics-path",
        required=True,
        help="Path to metrics.json produced by train.py.",
    )
    parser.add_argument("--model-name", required=True, help="Registered model name.")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--resource-group", required=True)
    parser.add_argument("--subscription", required=True)
    parser.add_argument(
        "--primary-metric",
        default=PRIMARY_METRIC,
        help="Metric to compare (must be present in metrics.json).",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=MIN_IMPROVEMENT,
        help="Fractional improvement required to deploy (e.g. 0.005 = 0.5%%).",
    )
    return parser.parse_args()


def load_new_metrics(metrics_path: str) -> dict:
    """Load metrics produced by the current training run."""
    path = Path(metrics_path)
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    with open(path) as fh:
        return json.load(fh)


def fetch_champion_metrics(model_name: str, workspace: str, resource_group: str, subscription: str) -> dict | None:
    """
    Fetch metrics from the latest champion model version in Azure ML.

    Returns None if no registered model exists yet (first deploy).
    """
    try:
        from azure.ai.ml import MLClient
        from azure.identity import DefaultAzureCredential

        client = MLClient(
            DefaultAzureCredential(),
            subscription_id=subscription,
            resource_group_name=resource_group,
            workspace_name=workspace,
        )

        # Get latest version
        versions = list(client.models.list(name=model_name))
        if not versions:
            logger.info("No existing model '%s' found — first deployment.", model_name)
            return None

        latest = sorted(versions, key=lambda m: int(m.version), reverse=True)[0]
        champion_metrics = latest.tags or {}
        logger.info(
            "Champion model: %s version %s, tags: %s",
            model_name, latest.version, champion_metrics,
        )

        # Tags are stored as strings; convert numeric ones
        numeric_metrics = {}
        for k, v in champion_metrics.items():
            try:
                numeric_metrics[k] = float(v)
            except (ValueError, TypeError):
                pass
        return numeric_metrics if numeric_metrics else None

    except ImportError:
        logger.warning("azure-ai-ml not installed; skipping champion fetch.")
        return None
    except Exception as exc:
        logger.error("Could not fetch champion metrics: %s", exc)
        return None


def evaluate(
    new_metrics: dict,
    champion_metrics: dict | None,
    primary_metric: str,
    min_improvement: float = MIN_IMPROVEMENT,
) -> bool:
    """
    Return True if the new model should be deployed.

    Rules:
    - If no champion exists, always deploy.
    - If primary_metric is missing from new metrics, refuse to deploy.
    - Deploy only if new_score > champion_score + min_improvement.
    """
    if primary_metric not in new_metrics:
        logger.error("Primary metric '%s' not found in new metrics.", primary_metric)
        return False

    new_score = new_metrics[primary_metric]
    logger.info("New model %s: %.6f", primary_metric, new_score)

    if champion_metrics is None:
        logger.info("No champion model — deploying unconditionally.")
        return True

    champion_score = champion_metrics.get(primary_metric)
    if champion_score is None:
        logger.warning(
            "Champion lacks '%s' metric — deploying conservatively.", primary_metric
        )
        return True

    logger.info("Champion model %s: %.6f", primary_metric, champion_score)
    improvement = new_score - champion_score
    logger.info(
        "Improvement: %.6f (threshold: %.6f)", improvement, min_improvement
    )

    if improvement >= min_improvement:
        logger.info("New model is better — will deploy.")
        return True
    else:
        logger.warning(
            "New model does NOT improve on champion by %.3f%% — skipping deploy.",
            min_improvement * 100,
        )
        return False


# Alias used by tests and scripts for a more descriptive name.
should_deploy = evaluate


def main() -> int:
    args = parse_args()

    new_metrics = load_new_metrics(args.metrics_path)
    logger.info("New model metrics: %s", json.dumps(new_metrics, indent=2))

    champion_metrics = fetch_champion_metrics(
        args.model_name, args.workspace, args.resource_group, args.subscription
    )

    deploy_decision = evaluate(
        new_metrics,
        champion_metrics,
        args.primary_metric,
        args.min_improvement,
    )

    # Write result for downstream pipeline steps
    result = {
        "should_deploy": deploy_decision,
        "new_metrics": new_metrics,
        "champion_metrics": champion_metrics,
        "primary_metric": args.primary_metric,
    }
    print(json.dumps(result, indent=2))

    return 0 if deploy_decision else 1


if __name__ == "__main__":
    sys.exit(main())
