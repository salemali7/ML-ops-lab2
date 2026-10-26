# AI381 Customer Churn Prediction Service


[CI Pipeline](https://www.google.com/search?q=https://github.com/just-ai381/lab1-churn-pipeline/actions/workflows/ci.yml)

Productionized customer churn training pipeline (`src/train/train.py`) and FastAPI inference microservice (`src/app/main.py`) with Pydantic input validation, automated `pytest` verification, multi-stage non-root Docker packaging, and a supply-chain-hardened GitHub Actions CI/CD pipeline.

## Start the Service with One Command

Build the multi-stage container (which trains `model.joblib` during the build stage) and start the non-root FastAPI service on port `8000` with a single command:

```bash
docker build -t churn-api . && docker run --rm -p 8000:8000 churn-api
```

Or in a local Python environment:

```bash
pip install -r requirements.txt && python -m src.train.train --data churn_v2.csv --model-out model.joblib && uvicorn src.app.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints & Example Requests

### 1. Health & Model Provenance (`GET /health`)

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model_version": "churn_v2-caa1408b",
  "dataset": "churn_v2.csv",
  "metrics": {
    "f1": 0.587,
    "roc_auc": 0.792
  }
}
```

### 2. Churn Prediction (`POST /predict`)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

```json
{
  "churn_probability": 0.895,
  "churn_prediction": 1,
  "model_version": "churn_v2-caa1408b"
}
```

## Repository Structure

| Path | Description |
| :--- | :--- |
| `src/train/train.py` | CLI training script that validates input columns, normalizes `monthly_fee`/`monthly_charges`, fits the pipeline, and embeds SHA-256 dataset/model provenance metadata. |
| `src/app/main.py` | FastAPI service exposing `/health` and `/predict` with strict Pydantic input validation (`CustomerRequest`). |
| `tests/test_api.py` | `pytest` suite covering API contract, model loading, input schema validation (`422`), boundary edge cases, and deterministic predictions. |
| `Dockerfile` | Multi-stage, non-root (`UID 10001`) Dockerfile pinned to `python:3.11.9-slim-bookworm@sha256:35cec96...` with a built-in `HEALTHCHECK`. |
| `.github/workflows/ci.yml` | GitHub Actions workflow with all third-party actions pinned to full 40-character commit SHAs: runs `ruff` + `pytest` on every PR, and builds, scans with Trivy, and pushes to GHCR (`ghcr.io/<owner>/<repo>/churn-api:<commit-sha>`) on `main`. |

## Running Linter & Tests Locally

```bash
pip install -r requirements.txt ruff pytest httpx
ruff check src/ tests/
pytest -v
```

## Branch Protection & Pull Request Workflow

### 1. Protect `main` with Required Checks

Via the GitHub CLI (`gh`), run the following command (replacing `<OWNER>/<REPO>` with your repository slug) to enforce Pull Requests and require the `Lint and Test` status check to pass before merging into `main`:

```bash
gh api \
  --method PUT \
  -H "Accept: application/vnd.github+json" \
  /repos/<OWNER>/<REPO>/branches/main/protection \
  -f "required_status_checks[strict]=true" \
  -f "required_status_checks[contexts][]=Lint and Test" \
  -F "enforce_admins=true" \
  -f "required_pull_request_reviews[dismiss_stale_reviews]=true" \
  -F "required_pull_request_reviews[required_approving_review_count]=0" \
  -F "restrictions=null"
```

Or in the GitHub Web UI:
1. Navigate to **Settings → Branches → Add branch ruleset / Add branch protection rule**.
2. Set **Branch name pattern** to `main`.
3. Enable **Require a pull request before merging**.
4. Enable **Require status checks to pass before merging**, check **Require branches to be up to date before merging**, and add **`Lint and Test`** as a required check.

### 2. Merge Work Through a Pull Request

```bash
git checkout -b feat/mlops-hardening
git add src/ tests/ Dockerfile .github/workflows/ci.yml README.md
git commit -m "feat: refactor training/API, add pytest suite, non-root Dockerfile, and SHA-pinned CI"
git push -u origin feat/mlops-hardening

# Open PR and auto-merge once the required Lint and Test check turns green
gh pr create --base main --head feat/mlops-hardening --title "Productionize churn service and CI/CD pipeline" --fill
gh pr merge --squash --auto
```
