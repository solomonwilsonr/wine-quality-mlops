# Wine Quality Classifier — ML CI/CD Pipeline

A production-grade machine-learning CI/CD system that trains, evaluates, registers, and deploys a
wine quality classifier on **Azure Machine Learning**, with full observability via **Application
Insights** and automated alerting to Slack and e-mail.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GitHub Repository                           │
│                                                                     │
│  src/          scripts/        azure/          tests/               │
│  ├─ train.py   ├─ register_   ├─ train_job    ├─ test_train.py     │
│  ├─ score.py   │   model.py   │   .yml        ├─ test_score.py     │
│  ├─ evaluate   └─ check_end   ├─ environment  └─ test_evaluate.py  │
│  │   .py           point_     │   .yml                             │
│  └─ utils.py       health.py  ├─ endpoint.yml                      │
│                               ├─ deployment   data/                 │
│                               │   .yml        ├─ winequality.csv   │
│                               └─ monitoring   └─ sample_request    │
│                                   .bicep          .json            │
└──────────────────────┬──────────────────────────────────────────────┘
                       │  push / PR
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    GitHub Actions                                    │
│                                                                     │
│  PR Validation          CI/CD Pipeline (main branch)                │
│  ┌────────────┐         ┌───────────┐  ┌──────────┐  ┌──────────┐ │
│  │ lint+test  │         │  quality  │  │  train   │  │ evaluate │ │
│  │ security   │         │  gate     │  │  (AML)   │  │ champion │ │
│  │ scan       │         └───────────┘  └──────────┘  └────┬─────┘ │
│  └────────────┘                                            │ pass  │
│                                                       ┌────▼─────┐ │
│                                                       │ register │ │
│                                                       │  model   │ │
│                                                       └────┬─────┘ │
│                                                       ┌────▼─────┐ │
│                                                       │  deploy  │ │
│                                                       │ endpoint │ │
│                                                       └────┬─────┘ │
│                                                       ┌────▼─────┐ │
│                                                       │  health  │ │
│                                                       │  check   │ │
│                                                       └──────────┘ │
└──────────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Azure                                        │
│                                                                     │
│  Azure ML Workspace                  Azure Monitor                  │
│  ┌──────────────────────────┐        ┌─────────────────────────┐   │
│  │  wine-quality-cluster    │        │  Log Analytics Workspace │   │
│  │  wine-quality-env        │        │  Application Insights    │   │
│  │  wine-quality-dataset    │        │  3× Alert Rules          │   │
│  │  wine-quality-classifier │        │  Action Group            │   │
│  │    (model registry)      │        │  (email + Slack)         │   │
│  │  wine-quality-prod       │        └─────────────────────────┘   │
│  │    (online endpoint)     │                                       │
│  └──────────────────────────┘                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

| Tool | Minimum Version | Notes |
|------|----------------|-------|
| Python | 3.9 | 3.10+ also supported |
| Azure CLI | 2.53 | `az login` before local runs |
| Azure ML CLI extension | 2.x | `az extension add -n ml` |
| Bicep CLI | 0.23 | for monitoring stack deployment |
| GitHub CLI | 2.x | for secrets management |

Azure resources that must exist **before** the first pipeline run:

- Azure ML Workspace (`AZURE_ML_WORKSPACE`)
- Compute cluster named `wine-quality-cluster` (Standard_DS3_v2, min 0 / max 4 nodes)
- Registered dataset `wine-quality-dataset` pointing at `data/winequality.csv`

---

## Quickstart

```bash
# 1. Clone and enter the repo
git clone https://github.com/<org>/wine-quality-classifier.git
cd wine-quality-classifier

# 2. Create a virtual environment and install dev dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# 3. Run the full test suite locally
pytest                          # uses pyproject.toml config

# 4. Train locally (uses sklearn built-in wine dataset as fallback)
python src/train.py --n-estimators 100 --max-depth 10

# 5. Trigger the full CI/CD pipeline
git push origin main            # GitHub Actions takes over from here
```

---

## GitHub Secrets Setup

All secrets are stored as **repository secrets** (`Settings > Secrets and variables > Actions`).

| Secret Name | Description | Where to Find |
|-------------|-------------|---------------|
| `AZURE_CLIENT_ID` | Service principal client ID (OIDC federation) | App registration in Azure AD |
| `AZURE_TENANT_ID` | Azure Active Directory tenant ID | Azure AD overview page |
| `AZURE_SUBSCRIPTION_ID` | Azure subscription ID | Subscriptions blade |
| `AZURE_RESOURCE_GROUP` | Resource group containing the ML workspace | Azure portal |
| `AZURE_ML_WORKSPACE` | Azure ML workspace name | ML Studio overview |
| `SLACK_WEBHOOK_URL` | Incoming webhook URL for Slack alerts | Slack App settings |
| `APPLICATIONINSIGHTS_CONNECTION_STRING` | App Insights connection string | App Insights > Properties |

### Creating the Federated Identity (OIDC — recommended)

```bash
# Create a service principal
az ad sp create-for-rbac --name "wine-quality-cicd" --role contributor \
  --scopes /subscriptions/<SUB_ID>/resourceGroups/<RG> \
  --sdk-auth

# Add federated credential for GitHub Actions
az ad app federated-credential create \
  --id <APP_ID> \
  --parameters '{
    "name": "github-actions",
    "issuer": "https://token.actions.githubusercontent.com",
    "subject": "repo:<org>/<repo>:ref:refs/heads/main",
    "audiences": ["api://AzureADTokenExchange"]
  }'
```

---

## Pipeline Stages

### PR Validation (`.github/workflows/pr-validation.yml`)

Runs on every pull request targeting `main`.

| Step | What it does |
|------|-------------|
| **Lint** | `flake8 src/ scripts/ tests/` — enforces PEP 8 (max line 100) |
| **Format check** | `black --check` + `isort --check` |
| **Unit tests** | `pytest tests/` with coverage report |
| **Security scan** | `bandit -r src/ scripts/` — no high-severity findings allowed |
| **Coverage gate** | Fails if coverage drops below 75% |

### CI/CD Pipeline (`.github/workflows/ci-cd-pipeline.yml`)

Runs on push to `main`.

| Stage | Job | Description |
|-------|-----|-------------|
| 1 | **quality-gate** | Lint, tests, security scan (same as PR) |
| 2 | **train** | Submits `azure/train_job.yml` to Azure ML; waits for completion |
| 3 | **evaluate** | Runs `src/evaluate.py` — compares new model to champion (must exceed F1 by ≥ 0.5%) |
| 4 | **register** | Runs `scripts/register_model.py` — registers model with metric tags |
| 5 | **deploy** | Applies `azure/endpoint.yml` + `azure/deployment.yml` via `az ml` |
| 6 | **health-check** | Runs `scripts/check_endpoint_health.py` — 5 retries, 10 s delay |

If any stage fails the pipeline halts and a `PipelineFailure` custom event is sent to Application Insights, triggering an alert.

---

## Monitoring & Alerts

The monitoring stack is deployed via `azure/monitoring.bicep`:

```bash
az deployment group create \
  --resource-group $AZURE_RESOURCE_GROUP \
  --template-file azure/monitoring.bicep \
  --parameters \
      environmentName=prod \
      alertEmail=oncall@example.com \
      slackWebhookUrl=$SLACK_WEBHOOK_URL
```

### Alert Rules

| Alert | Condition | Severity | Evaluation Window |
|-------|-----------|----------|-------------------|
| Endpoint P95 Latency | `RequestLatency_P95 > 2000 ms` | P2 | 5 min |
| Endpoint Error Rate | `RequestsFailedRate > 5%` | P1 | 5 min |
| Pipeline Failure | Custom event `PipelineFailure` in App Insights | P1 | 5 min |

All alerts fire to both e-mail and Slack via the Action Group.

### Querying Logs

```kusto
-- Recent endpoint latency (Log Analytics)
AmlOnlineEndpointConsoleLog
| where TimeGenerated > ago(1h)
| summarize p95_ms = percentile(DurationMs, 95) by bin(TimeGenerated, 5m)
| render timechart

-- Pipeline failure events
customEvents
| where name == "PipelineFailure"
| project timestamp, customDimensions
| order by timestamp desc
```

---

## Local Development Guide

### Running Tests

```bash
# All tests with coverage
pytest

# Single module
pytest tests/test_train.py -v

# Skip slow integration tests
pytest -m "not integration"
```

### Linting & Formatting

```bash
# Format code
black src/ scripts/ tests/
isort src/ scripts/ tests/

# Check only (CI mode)
black --check src/ scripts/ tests/
flake8 src/ scripts/ tests/

# Security scan
bandit -r src/ scripts/
```

### Training Locally

```bash
# Fast run with explicit hyperparameters (skips grid search)
python src/train.py --n-estimators 100 --max-depth 10

# With a local data file
python src/train.py \
  --data-path data/winequality.csv \
  --experiment-name local-experiment \
  --n-estimators 200
```

### Running the Scoring Script Locally

```bash
# Start a local mock endpoint (set model dir env var)
AZUREML_MODEL_DIR=./outputs python -c "
import score, json
score.init()
payload = open('data/sample_request.json').read()
print(score.run(payload))
"
```

### Deploying the Monitoring Stack

```bash
az deployment group create \
  --resource-group $AZURE_RESOURCE_GROUP \
  --template-file azure/monitoring.bicep \
  --parameters environmentName=prod alertEmail=you@example.com \
               slackWebhookUrl=https://hooks.slack.com/...
```

---

## Extending the Pipeline

### Adding a New Feature

1. Update `src/utils.py` — add transformation to `FeatureEngineer`.
2. Add unit test in `tests/test_train.py`.
3. Open a PR — the PR validation workflow runs automatically.

### Changing the Model Algorithm

1. Edit `src/train.py` — replace `RandomForestClassifier` with your estimator.
2. Update `azure/environment.yml` if new dependencies are needed.
3. Bump the environment version so Azure ML rebuilds the image.

### Adding a New Alert

Add a new `resource` block to `azure/monitoring.bicep` following the patterns for
`latencyAlert` or `pipelineFailureAlert`, then redeploy the Bicep template.

### Blue/Green Traffic Shifting

```bash
# Canary: send 10% of traffic to new green deployment
az ml online-endpoint update \
  --name wine-quality-prod \
  --resource-group $AZURE_RESOURCE_GROUP \
  --workspace-name $AZURE_ML_WORKSPACE \
  --traffic "blue=90 green=10"

# Full cutover after validation
az ml online-endpoint update \
  --name wine-quality-prod \
  --resource-group $AZURE_RESOURCE_GROUP \
  --workspace-name $AZURE_ML_WORKSPACE \
  --traffic "blue=0 green=100"
```

---

## Project Structure

```
.
├── .github/
│   └── workflows/
│       ├── ci-cd-pipeline.yml      # Main CI/CD pipeline
│       └── pr-validation.yml       # PR quality gate
├── azure/
│   ├── train_job.yml               # Azure ML CommandJob definition
│   ├── environment.yml             # Azure ML Environment (conda + Docker)
│   ├── endpoint.yml                # Managed Online Endpoint
│   ├── deployment.yml              # Blue deployment slot
│   └── monitoring.bicep            # Log Analytics + App Insights + Alerts
├── data/
│   ├── winequality.csv             # UCI Wine Quality (red) dataset
│   └── sample_request.json         # Sample scoring payload for health checks
├── scripts/
│   ├── register_model.py           # Registers trained model in Azure ML
│   └── check_endpoint_health.py    # Polls endpoint after deployment
├── src/
│   ├── train.py                    # Training pipeline
│   ├── score.py                    # Azure ML scoring script
│   ├── evaluate.py                 # Champion/challenger evaluation gate
│   └── utils.py                    # Shared utilities (FeatureEngineer, logging)
├── tests/
│   ├── conftest.py                 # Shared pytest fixtures
│   ├── test_train.py               # Unit tests for train.py
│   ├── test_score.py               # Unit tests for score.py
│   └── test_evaluate.py            # Unit tests for evaluate.py
├── pyproject.toml                  # black, isort, pytest configuration
├── setup.cfg                       # flake8 configuration
├── requirements.txt                # Runtime dependencies
└── requirements-dev.txt            # Dev + CI dependencies
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
