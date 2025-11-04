## Sales Forecasting on Vertex AI (BigQuery + KFP v2 + Endpoints)

This repository implements an end-to-end forecasting use case on Google Cloud using BigQuery, Vertex AI Pipelines (Kubeflow v2), Vertex AI Model Registry & Endpoints, Artifact Registry (containers), and Poetry for dependency management.

### What this repo demonstrates
- **Data ingestion to BigQuery**: Safe staging → final table pattern from a CSV in GCS.
- **Feature engineering**: Temporal, lag, and rolling features, plus one-hot categorical.
- **Model training**: LightGBM regressor with evaluation and artifact export.
- **Hyperparameter tuning (Vertex AI-ready)**: Training script exposes LightGBM params via CLI flags and reports an objective metric to Vertex AI Hyperparameter Tuning.
- **Pipelines (KFP v2)**: Orchestrate ingest → train → predict; optional register & deploy.
- **Custom serving**: FastAPI app packaged for Vertex AI Endpoints.
- **Predictions**: Online (Endpoint) and Batch Prediction jobs.
- **Monitoring**: Vertex AI Model Deployment Monitoring job (tabular drift).
- **Reproducibility**: Poetry + Docker multi-stage builds; Cloud Build CI optional.

### Architecture
- **BigQuery**: Final features table `daily_sales_features` consumed by training and inference.
- **Training**: `train.py` reads from BigQuery, engineers features, trains LightGBM, saves `model.joblib`.
- **Pipeline (KFP v2)**: Components run inside your image: `data_ingest_op` → `train_op` → `predict_op` → optional `register_and_deploy_op`.
- **Serving**: Custom container (`Dockerfile.serve`) runs FastAPI app that loads `model.joblib` from Vertex artifacts or GCS.
- **Prediction**: Online requests hit Vertex Endpoint; Batch jobs consume GCS/BigQuery and write outputs to GCS/BigQuery.


## Prerequisites
- Google Cloud project (owner or equivalent permissions for setup)
- Installed locally: `gcloud`, `docker`, `python 3.12`, `poetry`
- Enabled APIs in your project: `aiplatform.googleapis.com`, `bigquery.googleapis.com`, `artifactregistry.googleapis.com`, `cloudbuild.googleapis.com`, `run.googleapis.com`
- Artifact Registry Docker repo (the scripts can create it if missing)

### Authenticate and set default project
```bash
# Login and set project
gcloud auth login
gcloud config set project <PROJECT_ID>

# Application Default Credentials (used by SDKs)
gcloud auth application-default login
```


## Quickstart (most direct path)
1) Create `.env` at repo root (adjust values as needed):
```bash
PROJECT_ID="my-forecast-project-18870"
REGION="us-central1"
REPO="sales-forecast-repo"
IMAGE="sales-forecast"
SERVE_IMAGE="sales-forecast-serve"
DOCKER_IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}:latest"
SERVING_IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVE_IMAGE}:latest"
PIPELINE_ROOT="gs://${PROJECT_ID}-staging/pipeline_root"
SERVICE_ACCOUNT="ds-pipeline-sa@${PROJECT_ID}.iam.gserviceaccount.com"

# BigQuery and data
DATASET="gk2_takeaway_sales"
STAGING_TABLE="staging_daily_sales"
FINAL_TABLE="daily_sales_features"
LOCATION="US"
GCS_URI="gs://${PROJECT_ID}-staging/input/${FINAL_TABLE}.csv"

# Optional: central model artifact location for cross-step sharing
MODEL_GCS_URI="gs://${PROJECT_ID}-staging/models/model.joblib"
```

2) Put your CSV in GCS so ingestion can load it (the path must match `GCS_URI`):
```bash
# Example: copy your local CSV
gsutil cp /path/to/your/daily_sales_features.csv "gs://${PROJECT_ID}-staging/input/${FINAL_TABLE}.csv"
```

3) Build image and submit the KFP pipeline to Vertex AI:
```bash
# Builds/pushes image, compiles pipeline JSON, submits run (async)
./scripts/deploy_and_run.sh
```

Tip: To deploy an endpoint in the same run, set `ENABLE_DEPLOY=true` (and ensure `SERVING_IMAGE_URI` is set). The script will build/push the serving image if needed and the pipeline will register and deploy automatically.

4) (Optional, if you did not enable deployment in step 3) Build serving image, register model, deploy endpoint:
```bash
# Builds/pushes serving image, registers model from artifacts, deploys endpoint
./scripts/build_and_deploy_serving.sh
# Endpoint ID is saved in .vertex_endpoint_id
```

5) Call the endpoint for an online prediction:
```bash
ENDPOINT_ID=$(cat .vertex_endpoint_id)
python scripts/endpoint_predict.py \
  --project-id "${PROJECT_ID}" \
  --region "${REGION}" \
  --endpoint-id "${ENDPOINT_ID}" \
  --instances '[{"feature1": 1.2, "feature2": 3.4}]'
```


## Data ingestion
Source CSV is loaded to a staging table with explicit schema, then transformed into the final features table.
- Code: `src/sales_forecast/data_ingest.py`
- Env: `PROJECT_ID`, `DATASET`, `STAGING_TABLE`, `FINAL_TABLE`, `GCS_URI`, `LOCATION`

Run standalone:
```bash
python -m sales_forecast.data_ingest
```

Notes:
- The current script creates/overwrites the final table without partitioning. For large time-series, consider date partitioning and clustering in BigQuery.


## Feature engineering
- Code: `src/sales_forecast/features.py`
- Adds temporal features (year, month, day, dow, week_of_year, etc.), lag features `[1,2,3,7,14]`, rolling mean/std windows `[2..7]`, and one-hot encodes `holiday_name` if present.
- Drops initial rows with NaNs introduced by lag/rolling.


## Training
- Code: `src/sales_forecast/train.py`
- Reads features from `PROJECT_ID.DATASET.FINAL_TABLE`, trains LightGBM, evaluates on the most recent 20% by time, saves `model.joblib`.

Run standalone (local or in container):
```bash
# Save locally
python -m sales_forecast.train --model-path /tmp/model.joblib

# Save locally and upload to GCS
python -m sales_forecast.train \
  --model-path /tmp/model.joblib \
  --model-gcs-uri "gs://${PROJECT_ID}-staging/models/model.joblib"
```


## Hyperparameter Tuning (Vertex AI)
- Status: The training script is hyperparameter-tuning ready. It accepts LightGBM params as CLI flags and reports a single objective metric (MAE/RMSE/R²) using the `hypertune` library so Vertex AI can select the best trial.
- Default behavior: The provided pipeline and Cloud Run Jobs execute a single training run (no HPT) unless you explicitly create a Vertex Hyperparameter Tuning Job.

Supported hyperparameter flags (subset):
- `--learning-rate`, `--num-leaves`, `--max-depth`, `--min-child-samples`, `--feature-fraction`, `--bagging-fraction`, `--bagging-freq`, `--lambda-l1`, `--lambda-l2`, `--min-split-gain`, `--n-estimators`, `--objective`, `--metric`, `--boosting-type`, `--seed`, `--n-jobs`, `--early-stopping-rounds`.

Metric reporting:
- Pass `--hpt-metric-name mae` (or `rmse`, `r2`) to choose the objective. Alternatively set `HPT_METRIC_NAME` in the environment.

Run a Vertex AI Hyperparameter Tuning Job (example in Python):
```python
from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt

PROJECT_ID = "my-forecast-project-18870"
REGION = "us-central1"
IMAGE_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/sales-forecast-repo/sales-forecast:latest"
SERVICE_ACCOUNT = f"ds-pipeline-sa@{PROJECT_ID}.iam.gserviceaccount.com"

aiplatform.init(project=PROJECT_ID, location=REGION)

custom_job = aiplatform.CustomJob(
    display_name="sf-train",
    worker_pool_specs=[{
        "machine_spec": {"machine_type": "n1-standard-4"},
        "replica_count": 1,
        "container_spec": {
            "image_uri": IMAGE_URI,
            "command": ["python", "-m", "sales_forecast.train"],
            "args": [
                "--model-path", "/tmp/model.joblib",
                "--hpt-metric-name", "mae"  # or "rmse" / "r2"
            ],
        },
    }],
)

hpt_job = aiplatform.HyperparameterTuningJob(
    display_name="sf-lgbm-hpt",
    custom_job=custom_job,
    metric_spec={"mae": "minimize"},  # must match --hpt-metric-name
    parameter_spec={
        "learning_rate": hpt.DoubleParameterSpec(min=0.01, max=0.2, scale="log"),
        "num_leaves": hpt.IntegerParameterSpec(min=31, max=255, scale="linear"),
        "max_depth": hpt.IntegerParameterSpec(min=-1, max=15, scale="linear"),
        "feature_fraction": hpt.DoubleParameterSpec(min=0.6, max=1.0, scale="linear"),
        "bagging_fraction": hpt.DoubleParameterSpec(min=0.6, max=1.0, scale="linear"),
        "n_estimators": hpt.IntegerParameterSpec(min=200, max=2000, scale="linear"),
    },
    max_trial_count=20,
    parallel_trial_count=4,
)

hpt_job.run(service_account=SERVICE_ACCOUNT, sync=True)

best = hpt_job.trials[0]
print("Best metric:", best.final_measurement.metrics)
print("Best params:", best.parameters)
```

Notes:
- During tuning, avoid passing a shared `--model-gcs-uri` to prevent trials from overwriting the same path.
- After HPT finishes, retrain once with the best parameters to produce a single promoted artifact:
```bash
python -m sales_forecast.train \
  --model-path /tmp/model.joblib \
  --model-gcs-uri "gs://${PROJECT_ID}-staging/models/model.joblib" \
  --learning-rate <best> --num-leaves <best> --max-depth <best> \
  --feature-fraction <best> --bagging-fraction <best> --n-estimators <best> \
  --hpt-metric-name mae
```


## Inference (offline script)
- Code: `src/sales_forecast/predict.py`
- Loads model from local path or GCS, queries recent days from BigQuery to rebuild lag/rolling features, appends next day, and outputs a single-day forecast.

Run standalone:
```bash
# From local artifact
python -m sales_forecast.predict --model-path /tmp/model.joblib

# From GCS artifact
python -m sales_forecast.predict --model-gcs-uri "gs://${PROJECT_ID}-staging/models/model.joblib"
```


## Vertex AI Pipelines (Kubeflow v2)
- Code: `pipeline/forecast_pipeline.py`
- Components (all run in your container image):
  - `data_ingest_op()` → creates/refreshes tables
  - `train_op()` → trains LightGBM and emits `dsl.Model` artifact
  - `predict_op(model_artifact)` → runs inference using the trained artifact
  - `register_and_deploy_op(model_artifact, ...)` → optional registration + deployment

Note: The deploy step is gated. It runs only when `ENABLE_DEPLOY=true` and `SERVING_IMAGE_URI` is set and differs from `DOCKER_IMAGE_URI`. Otherwise it is skipped.

HPT note: The current pipeline runs a single training task and does not launch a Hyperparameter Tuning Job. To tune, run a separate Vertex HPT job (see section above), then retrain once with best params.

Compile and submit directly:
```bash
# Compile pipeline JSON
python pipeline/forecast_pipeline.py  # compiles to forecast_pipeline.json and submits a run
```

Or use the helper script (builds/pushes image, compiles, submits):
```bash
./scripts/deploy_and_run.sh
```


### Pipeline artifacts vs model artifacts (PIPELINE_ROOT vs models directory)

- **PIPELINE_ROOT** is a GCS prefix where Vertex/KFP stores run-time execution data: component outputs (artifacts), logs, and metadata for each pipeline run.
  - Defined in code and in the compiled JSON as `defaultPipelineRoot`.
  - Example: `gs://<project>-staging/pipeline_root/<run-id>/artifacts/...`.
  - This is managed per-run and can be cleaned up independently of serving.
- **Compiled pipeline JSON (`forecast_pipeline.json`)** is a local file generated when compiling the pipeline. It is not stored under `PIPELINE_ROOT`.
- **Model artifacts for serving** are promoted to a separate, stable GCS location (e.g., `gs://<project>-staging/models/run-<run-id>/model.joblib`) before registration.
  - Reason: separation of concerns, clearer versioning, stable URIs for serving, and independence from pipeline-run lifecycle.
  - You technically could point the Model Registry to a path inside `PIPELINE_ROOT`, but this couples serving to run retention/cleanup and is not recommended.

### How KFP artifacts are passed (dsl.OutputPath / dsl.Input)

- Declaring a parameter as `dsl.OutputPath(dsl.Model)` makes KFP inject a writable file path at runtime. Your code should save the model to that path.
- Downstream steps receive `dsl.Input[dsl.Model]` and should read via `.path`.
- The training step in this repo saves to the injected output path; Vertex mirrors that artifact under `PIPELINE_ROOT` for the run.

### Registration uses a pointer, not a file copy

- The pipeline copies the trained model from the KFP artifact to a stable models folder in GCS and builds an `artifact_uri` (directory) for registration.
- `aiplatform.Model.upload(artifact_uri=...)` creates a Model Registry entry that references your GCS folder; it does not re-upload the file.
- During deployment/serving, Vertex sets `AIP_STORAGE_URI` in your container so the app can download the model from that GCS location.


## Alternative: Cloud Run Jobs orchestration
- Code: `scripts/run_cloudrun_pipeline.sh`
- Creates three Cloud Run Jobs (ingest → train → predict) and executes them sequentially.

Run:
```bash
./scripts/run_cloudrun_pipeline.sh
```


## Serving (Vertex AI Endpoints)
- App: `src/sales_forecast/serve_app.py` (FastAPI)
- Container: `Dockerfile.serve` (exposes port 8080 for Vertex probes)
- Model load order:
  1. `MODEL_PATH` if provided and exists
  2. `MODEL_GCS_URI` if provided (downloads `model.joblib`)
  3. `AIP_STORAGE_URI` (Vertex-injected artifact dir)

Build and deploy end-to-end:
```bash
./scripts/build_and_deploy_serving.sh
# Saves endpoint resource name in .vertex_endpoint_id
```

If you enabled deployment via `ENABLE_DEPLOY=true` in the pipeline, you do not need to run this script.

Manual call example:
```bash
ENDPOINT_ID=$(cat .vertex_endpoint_id)
python scripts/endpoint_predict.py \
  --project-id "${PROJECT_ID}" \
  --region "${REGION}" \
  --endpoint-id "${ENDPOINT_ID}" \
  --instances '[{"feature1": 1.2, "feature2": 3.4}]'
```


## Batch Prediction
- Code: `scripts/batch_predict.py`

Example (GCS → GCS):
```bash
python scripts/batch_predict.py \
  --project-id "${PROJECT_ID}" \
  --region "${REGION}" \
  --model "projects/${PROJECT_ID}/locations/${REGION}/models/<MODEL_ID>" \
  --input-uri "gs://${PROJECT_ID}-staging/predict/input.jsonl" \
  --output-uri-prefix "gs://${PROJECT_ID}-staging/predict/output/" \
  --instances-format jsonl \
  --predictions-format jsonl
```


## Monitoring (Model Deployment Monitoring)
- Code: `scripts/setup_model_monitoring.py`

Create a monitoring job (hourly schedule, optional email alert):
```bash
python scripts/setup_model_monitoring.py \
  --project-id "${PROJECT_ID}" \
  --region "${REGION}" \
  --endpoint-id "${ENDPOINT_ID}" \
  --display-name "sales-forecast-monitor" \
  --sampling-rate 0.5 \
  --alert-email you@example.com
```

Notes:
- For production, define per-feature drift thresholds.


## Notebooks
- `notebooks/EDA.ipynb` — exploratory analysis.
- `notebooks/Demo_End_to_End.ipynb` — queries BigQuery and calls your deployed endpoint.

Run locally with Poetry’s environment:
```bash
poetry install --no-root --only main
poetry run jupyter lab
```

Before running the demo notebook set:
```bash
export PROJECT_ID
export REGION
export DATASET
export FINAL_TABLE
export ENDPOINT_ID=$(cat .vertex_endpoint_id)
```


## CI/CD (Cloud Build)
- File: `cloudbuild.yaml`
- Steps:
  - Authenticate to Artifact Registry
  - Build and push image (tagged with `$COMMIT_SHA`)
  - Compile the pipeline using that image tag
  - Optionally submit a pipeline run

Trigger from your repo (e.g., via Cloud Build triggers) to build on push.


## Configuration (env vars)
Common variables (can be set in `.env` and read by scripts/components):
- **PROJECT_ID**: GCP project id
- **REGION**: e.g., `us-central1`
- **REPO**, **IMAGE**, **DOCKER_IMAGE_URI**: container image for pipeline steps
- **SERVE_IMAGE**, **SERVING_IMAGE_URI**: container image for serving
- **PIPELINE_ROOT**: `gs://<bucket>/pipeline_root`
- **SERVICE_ACCOUNT**: service account email running pipeline/jobs
- **DATASET**, **STAGING_TABLE**, **FINAL_TABLE**, **LOCATION**
- **GCS_URI**: CSV path in GCS for ingestion
- **MODEL_GCS_URI**: optional central model artifact URI
- **ENABLE_DEPLOY**: set to `true` to let the pipeline register+deploy using `SERVING_IMAGE_URI` (default `false`).
- **HPT_METRIC_NAME**: optional; when set, the training script reports this objective for Vertex HPT (`mae`, `rmse`, or `r2`).

Scripts auto-load `.env` if present.


## Quotas, roles, and troubleshooting
- **Quotas**:
  - Endpoint deploy uses Vertex Online Prediction quotas (vCPUs, memory, endpoints, deployed models).
  - Registration uses Model Registry API quotas (control-plane), not training vCPUs.
- **Permissions**:
  - Service account typically needs: Vertex AI Admin, BigQuery User, Storage Admin, Artifact Registry Writer, Cloud Build Service Account (if CI), Run Admin (for Cloud Run Jobs path).
- **Common issues**:
  - `QUOTA_EXCEEDED` when deploying endpoint → reduce machine size/replicas or request quota increase.
  - `PERMISSION_DENIED` on BigQuery/GCS → ensure SA roles and dataset/bucket IAM.
  - No `ENDPOINT_ID` → ensure deploy step completed; check `.vertex_endpoint_id`.


## Cost and cleanup
- Costs accrue for BigQuery storage/queries, container build storage, endpoint instances, monitoring jobs.
- Cleanup when done:
```bash
# Delete endpoint (will undeploy models)
gcloud ai endpoints delete ${ENDPOINT_ID} --region ${REGION}

# (Optional) Delete models by display name
gcloud ai models list --region ${REGION} --filter="displayName=sales-forecast-model"
# then delete by ID(s)

# Remove Artifact Registry images
gcloud artifacts docker images delete "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}:latest" --delete-tags --quiet || true

gcloud artifacts docker images delete "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVE_IMAGE}:latest" --delete-tags --quiet || true

# Remove pipeline runs and compiled JSON as needed
rm -f forecast_pipeline.json .vertex_endpoint_id || true
```


## Project structure
```text
/ (repo root)
  Dockerfile
  Dockerfile.serve
  cloudbuild.yaml
  forecast_pipeline.json (generated)
  APPLICATION_FLOW.md
  notebooks/
    EDA.ipynb
    Demo_End_to_End.ipynb
  pipeline/
    forecast_pipeline.py
  deploy/
    vertex.py
  scripts/
    deploy_and_run.sh
    run_cloudrun_pipeline.sh
    build_and_deploy_serving.sh
    endpoint_predict.py
    batch_predict.py
    setup_model_monitoring.py
  src/
    sales_forecast/
      __init__.py
      config.py
      data_ingest.py
      features.py
      train.py
      predict.py
      serve_app.py
  pyproject.toml
  poetry.lock
  README.md
```


## Key points (what to highlight)
- Why BigQuery staging → final pattern and (optional) partitioning/clustering.
- Feature choices (lags/rolling, holidays) and alignment at serving time.
- Reproducibility with Poetry + Docker; the same image powers KFP steps.
- KFP design (artifacts, dependencies, optional register/deploy step).
- Endpoint SLOs vs. machine type/replicas; monitoring strategy.
- Batch vs. online prediction trade-offs.
