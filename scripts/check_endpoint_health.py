"""
check_endpoint_health.py
========================
Polls an Azure ML managed online endpoint with a sample scoring request and
validates the response. Retries on failure with configurable back-off.

Exit codes:
    0 — endpoint is healthy
    1 — endpoint failed after all retries

Usage:
    python scripts/check_endpoint_health.py \
        --endpoint-name  wine-quality-prod \
        --subscription   $AZURE_SUBSCRIPTION_ID \
        --resource-group $AZURE_RESOURCE_GROUP \
        --workspace      $AZURE_ML_WORKSPACE \
        --max-retries    5 \
        --retry-delay    10
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

# Path to the sample request payload (relative to repo root)
DEFAULT_SAMPLE_REQUEST = Path(__file__).parent.parent / "data" / "sample_request.json"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Health-check an Azure ML online endpoint.")
    p.add_argument("--endpoint-name",  required=True, help="Azure ML endpoint name.")
    p.add_argument("--subscription",   required=True, help="Azure subscription ID.")
    p.add_argument("--resource-group", required=True, help="Azure resource group name.")
    p.add_argument("--workspace",      required=True, help="Azure ML workspace name.")
    p.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum number of polling attempts before declaring failure (default: 5).",
    )
    p.add_argument(
        "--retry-delay",
        type=int,
        default=10,
        help="Seconds to wait between retries (default: 10).",
    )
    p.add_argument(
        "--sample-request",
        default=str(DEFAULT_SAMPLE_REQUEST),
        help="Path to JSON file containing the sample scoring payload.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Azure helpers
# ---------------------------------------------------------------------------

def get_scoring_uri_and_key(
    endpoint_name: str,
    subscription: str,
    resource_group: str,
    workspace: str,
) -> tuple[str, str]:
    """Return (scoring_uri, primary_key) for the given endpoint."""
    try:
        from azure.ai.ml import MLClient
        from azure.identity import DefaultAzureCredential
    except ImportError as exc:
        logger.error("Missing Azure SDK dependency: %s", exc)
        sys.exit(1)

    client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=subscription,
        resource_group_name=resource_group,
        workspace_name=workspace,
    )

    endpoint = client.online_endpoints.get(name=endpoint_name)
    scoring_uri = endpoint.scoring_uri
    logger.info("Endpoint scoring URI: %s", scoring_uri)

    keys = client.online_endpoints.get_keys(name=endpoint_name)
    primary_key = keys.primary_key
    return scoring_uri, primary_key


# ---------------------------------------------------------------------------
# Health check logic
# ---------------------------------------------------------------------------

def load_sample_request(sample_path: str) -> bytes:
    path = Path(sample_path)
    if not path.exists():
        logger.error("Sample request file not found: %s", sample_path)
        sys.exit(1)
    with open(path) as fh:
        payload = json.load(fh)
    return json.dumps(payload).encode("utf-8")


def check_response(response_text: str) -> bool:
    """Return True if the response looks healthy (contains 'predictions')."""
    try:
        body = json.loads(response_text)
    except json.JSONDecodeError:
        logger.error("Response is not valid JSON: %.200s", response_text)
        return False

    if "predictions" not in body:
        logger.error("Response missing 'predictions' key. Got keys: %s", list(body.keys()))
        return False

    logger.info("Response contains 'predictions' with %d result(s).", len(body["predictions"]))
    return True


def poll_endpoint(
    scoring_uri: str,
    primary_key: str,
    payload: bytes,
    max_retries: int,
    retry_delay: int,
) -> bool:
    """
    POST to the scoring URI up to max_retries times.
    Returns True on the first healthy response, False if all attempts fail.
    """
    import urllib.request
    import urllib.error

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {primary_key}",
    }

    for attempt in range(1, max_retries + 1):
        logger.info("Health check attempt %d / %d ...", attempt, max_retries)
        try:
            req = urllib.request.Request(
                scoring_uri,
                data=payload,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                status = resp.status
                body = resp.read().decode("utf-8")

            logger.info("HTTP %d response received.", status)

            if status == 200 and check_response(body):
                logger.info("Endpoint is HEALTHY.")
                return True

            logger.warning("Unexpected HTTP %d or bad response body.", status)

        except urllib.error.HTTPError as exc:
            logger.warning("HTTP error on attempt %d: %s %s", attempt, exc.code, exc.reason)
        except urllib.error.URLError as exc:
            logger.warning("URL error on attempt %d: %s", attempt, exc.reason)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Unexpected error on attempt %d: %s", attempt, exc)

        if attempt < max_retries:
            logger.info("Waiting %d s before next attempt...", retry_delay)
            time.sleep(retry_delay)

    logger.error("Endpoint health check FAILED after %d attempt(s).", max_retries)
    return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    scoring_uri, primary_key = get_scoring_uri_and_key(
        endpoint_name=args.endpoint_name,
        subscription=args.subscription,
        resource_group=args.resource_group,
        workspace=args.workspace,
    )

    payload = load_sample_request(args.sample_request)

    healthy = poll_endpoint(
        scoring_uri=scoring_uri,
        primary_key=primary_key,
        payload=payload,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
    )

    sys.exit(0 if healthy else 1)


if __name__ == "__main__":
    main()
