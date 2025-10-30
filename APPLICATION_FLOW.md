### Overview
Below are the complete data and application flows for all supported ways to run this project end-to-end. This explains what to run, what happens next, which Google Cloud services are used, and why each exists. Assumption: you have the necessary quotas (for both pipelines and online prediction) and the required IAM roles.

### Key Google Cloud services used
- BigQuery: analytics database that stores your tables.
- Cloud Storage (GCS): object store for CSV input and model artifacts.
- Artifact Registry: stores your built Docker container images.
- Vertex AI Pipelines (Kubeflow v2): orchestrates your ML steps (ingest → train → predict → optional register/deploy).
- Vertex AI Model Registry: catalog of your registered model versions (points to artifacts in GCS and a serving image).
- Vertex AI Endpoints (Online Prediction): hosts your model for real-time predictions.
- Cloud Build (optional): CI that builds and compiles the pipeline on push.
- Cloud Run Jobs (alternative orchestration): serverless jobs for ingest/train/predict without pipelines.

### Workflows you can run
There are three main ways to run the full flow. Choose one:

- A) Vertex AI Pipelines end-to-end (recommended when quotas are ready; serving/deploy is optional and gated).
- B) Cloud Run Jobs orchestration (alternative to pipelines; does not deploy an endpoint by itself).
- C) Split path: pipeline (or jobs) for ingest/train/predict, then a separate script to register/deploy an endpoint.

I’ll walk through each option in detail.

---

## A) Vertex AI Pipelines end-to-end
Run this if you want an automated flow inside Vertex AI Pipelines, with optional automatic register/deploy.

- Files to run:
  - `scripts/deploy_and_run.sh` (one-stop: builds image, compiles pipeline, submits a run; optionally builds serving image if deployment is enabled)
  - or `python pipeline/forecast_pipeline.py` (compiles pipeline JSON and submits a run using your prebuilt image)

- Serving option (updated):
  - Deployment is controlled by `ENABLE_DEPLOY`. When `ENABLE_DEPLOY=true` and a `SERVING_IMAGE_URI` is provided, the pipeline’s optional deploy step runs.
  - `scripts/deploy_and_run.sh` will also ensure the serving image exists: if `ENABLE_DEPLOY=true` and the image is missing, it builds/pushes `Dockerfile.serve` to `SERVING_IMAGE_URI` automatically.
  - If `ENABLE_DEPLOY=false` (default), the pipeline runs ingest → train → offline predict only.

What happens when you run `./scripts/deploy_and_run.sh`:
1) Local build and publish
   - Tools: Docker and Artifact Registry
   - The script builds your container image from `Dockerfile`, tags it as `DOCKER_IMAGE_URI`, and pushes it to Artifact Registry.
   - Purpose: package code + dependencies so all pipeline steps run in a controlled, reproducible environment.

2) Optional serving image build/push (only if `ENABLE_DEPLOY=true`)
   - Tools: Docker and Artifact Registry
   - If the serving image referenced by `SERVING_IMAGE_URI` does not exist, the script builds it from `Dockerfile.serve` and pushes it.
   - Purpose: provide a container that runs the FastAPI server (`sales_forecast.serve_app:app`) for Vertex AI Endpoints.

3) Pipeline compilation and submission
   - Tools: kfp SDK, Vertex AI Pipelines
   - `pipeline/forecast_pipeline.py` compiles to `forecast_pipeline.json`, then a Vertex AI Pipeline Job is submitted.
   - Purpose: define a graph of steps to run on managed infrastructure.

4) Pipeline runs these components (inside Vertex AI Pipelines):
   - `data_ingest_op()`:
     - Code invoked: `src/sales_forecast/data_ingest.py`
     - Services used: BigQuery and GCS
     - What it does:
       - Ensures the BigQuery dataset exists.
       - Loads your CSV from `GCS_URI` into a staging table (explicit schema).
       - Creates or replaces the final table `PROJECT_ID.DATASET.daily_sales_features` via SQL.
       - After completion: BigQuery has a clean, final features table sorted by date.
   - `train_op()`:
     - Code invoked: `src/sales_forecast/train.py`
     - Services used: BigQuery (data read), LightGBM (training)
     - What it does:
       - Reads the final BigQuery table.
       - Builds in-memory features (temporal, lag, rolling, one-hot) and splits by time.
       - Trains a LightGBM model, evaluates, and saves `model.joblib` to the component’s output path (Vertex mirrors it under your Pipeline Root bucket in GCS).
       - After completion: a model artifact is available as a pipeline output.
   - `predict_op(model_artifact)`:
     - Code invoked: `src/sales_forecast/predict.py`
     - Services used: BigQuery
     - What it does:
       - Loads the just-trained `model.joblib` (from the artifact path).
       - Reads the last 30 days from the `daily_sales_features` table.
       - Rebuilds the lag/rolling features for the combined data and predicts the next day’s value.
       - After completion: logs a one-day forecast (visible in job logs).
   - Optional: `register_and_deploy_op(...)` (gated)
     - Trigger condition: only runs if `ENABLE_DEPLOY=true` and `SERVING_IMAGE_URI` is set and differs from `DOCKER_IMAGE_URI`.
     - Code invoked: inline logic in `pipeline/forecast_pipeline.py` (copy artifact to GCS, find latest model) + `deploy/vertex.py` (register and deploy)
     - Services used: Cloud Storage, Vertex AI Model Registry, Vertex AI Endpoints
     - What it does:
       - Copies the model artifact from the component’s local path into a well-structured GCS folder under `models_gcs_dir/run-<run_id>/` to serve as the `artifact_uri`.
       - Registers the model in Vertex AI Model Registry using `artifact_uri` and `SERVING_IMAGE_URI`.
       - Looks up the latest registered model by display name and deploys it to an endpoint (creating the endpoint if needed) with your machine type and replica settings.
       - After completion: an online Endpoint is live and ready for real-time predictions.

Notes and outcomes:
- After the pipeline run, your final BigQuery table exists; a trained model artifact exists in GCS (pipeline root and/or the models directory); optionally (if enabled), a Vertex AI Endpoint exists and serves predictions.
- To call the live endpoint later, use `scripts/endpoint_predict.py` or the demo notebook.

---

## B) Cloud Run Jobs orchestration (no-Vertex-Pipelines path)
Run this if you prefer serverless jobs to orchestrate ingestion → training → prediction, without using Vertex AI Pipelines.

- File to run:
  - `./scripts/run_cloudrun_pipeline.sh`

What happens when you run it:
1) Local build and publish
   - Tools: Docker and Artifact Registry
   - Builds your container from `Dockerfile` and pushes to `DOCKER_IMAGE_URI`.

2) Create/update three Cloud Run Jobs
   - Tools: Cloud Run Jobs
   - Jobs created (or updated):
     - `sf-data-ingest`: runs `python -m sales_forecast.data_ingest`
     - `sf-train`: runs `python -m sales_forecast.train --model-path /tmp/model.joblib --model-gcs-uri ${MODEL_GCS_URI}`
     - `sf-predict`: runs `python -m sales_forecast.predict --model-gcs-uri ${MODEL_GCS_URI}`

3) Execute jobs sequentially and wait for completion
   - `sf-data-ingest`:
     - Services used: GCS and BigQuery
     - Loads CSV to staging table, then creates final table `daily_sales_features`.
   - `sf-train`:
     - Services used: BigQuery (data read), Cloud Storage (artifact write), LightGBM (training)
     - Trains the model and uploads `model.joblib` to `MODEL_GCS_URI`.
   - `sf-predict`:
     - Services used: BigQuery (data read), Cloud Storage (artifact read)
     - Downloads the model from GCS, rebuilds features for the last 30 days plus one day ahead, and prints the forecast to logs.

Notes and outcomes:
- This path does not register or deploy an online endpoint by itself.
- If you want an endpoint after this, run `./scripts/build_and_deploy_serving.sh` (see option C below).

---

## C) Split path: pipeline (or jobs) for ingest/train/predict, then a dedicated script to register/deploy
Run this if you want to separate training and serving concerns or if you used Cloud Run Jobs.

- Files to run:
  - First, one of:
    - `./scripts/deploy_and_run.sh` (or `python pipeline/forecast_pipeline.py`) to ingest/train/predict; or
    - `./scripts/run_cloudrun_pipeline.sh` to do the same with Cloud Run Jobs.
  - Then:
    - `./scripts/build_and_deploy_serving.sh` to build the serving image and register/deploy your model to an endpoint.

What happens when you run `./scripts/build_and_deploy_serving.sh`:
1) Local build and publish (serving image)
   - Tools: Docker and Artifact Registry
   - Builds `Dockerfile.serve` into `SERVING_IMAGE_URI` and pushes it.
   - Purpose: a container that boots a FastAPI app (`src/sales_forecast/serve_app.py`) with Uvicorn on port 8080.

2) Register the model
   - Code invoked: `deploy/vertex.py register`
   - Services used: Cloud Storage (artifact URI), Vertex AI Model Registry
   - Uses your model artifacts in GCS (either from a prior pipeline run’s upload step or your Cloud Run training step).
   - Creates a new model version (resource) with `SERVING_IMAGE_URI` attached.

3) Deploy to an endpoint
   - Code invoked: `deploy/vertex.py deploy`
   - Services used: Vertex AI Endpoints (Online Prediction)
   - Creates or reuses an endpoint, deploys the latest registered model to it (machine type, min/max replicas).
   - Saves the endpoint id to `.vertex_endpoint_id`.

4) Make online predictions
   - File to run: `python scripts/endpoint_predict.py --project-id ... --region ... --endpoint-id ... --instances '[{...}]'`
   - Services used: Vertex AI Endpoints
   - Sends JSON instances to the live model, gets JSON predictions back.
   - Note: online predictions are returned in the response; they are not saved to BigQuery by default.

Notes and outcomes:
- This path puts the serving lifecycle in your control with an explicit, dedicated script.
- If you already registered/deployed from inside the pipeline, you don’t have to run this script again.

---

## What each project file contributes in these flows
- `src/sales_forecast/data_ingest.py`: creates dataset if needed, loads CSV from GCS to staging, builds final table `daily_sales_features` with proper types. Uses BigQuery.
- `src/sales_forecast/features.py`: builds in-memory features (temporal, lag, rolling, one-hot) used by training and inference code.
- `src/sales_forecast/train.py`: reads BigQuery, builds features, trains LightGBM, evaluates, and saves `model.joblib`. Can upload to GCS when a URI is provided.
- `src/sales_forecast/predict.py`: loads the model (local or from GCS), fetches recent data from BigQuery, rebuilds features, and predicts the next day. Prints result to logs.
- `src/sales_forecast/serve_app.py`: FastAPI app for online predictions on Vertex AI Endpoints; loads model from `AIP_STORAGE_URI`, `MODEL_GCS_URI`, or `MODEL_PATH`.
- `pipeline/forecast_pipeline.py`: defines KFP v2 components and the pipeline graph; compiles and submits to Vertex AI Pipelines; the deploy step is optional and gated.
- `deploy/vertex.py`: utilities to register a model to Vertex AI Model Registry and deploy it to an Endpoint.
- `scripts/deploy_and_run.sh`: builds the training image, optionally builds the serving image (when `ENABLE_DEPLOY=true`), compiles the pipeline, submits a run.
- `scripts/run_cloudrun_pipeline.sh`: sets up and runs three Cloud Run Jobs (ingest → train → predict).
- `scripts/build_and_deploy_serving.sh`: builds the serving image, registers the model, deploys to an endpoint, saves the endpoint id.
- `scripts/endpoint_predict.py`: simple client to call a Vertex AI Endpoint for online predictions.
- `scripts/batch_predict.py`: creates a Vertex AI Batch Prediction job (input/output can be GCS or BigQuery).
- `scripts/setup_model_monitoring.py`: creates a Vertex AI Model Deployment Monitoring job (feature drift monitoring).

---

## What happens to your data throughout
- Input CSV: stored in GCS at `GCS_URI`.
- Staging table: temporary BigQuery table with explicit schema (overwritten on ingest).
- Final table: BigQuery table `PROJECT_ID.DATASET.daily_sales_features` used for training and inference queries.
- Model artifact: `model.joblib` saved to Vertex AI pipeline artifacts (if using pipelines) and/or uploaded to GCS (if configured).
- Online prediction: requests go to Vertex AI Endpoint and responses are returned to the caller; not stored by default.
- Batch prediction (optional): if you run `scripts/batch_predict.py` with BigQuery output configured, predictions are written to BigQuery; if configured for GCS output, predictions are written to the specified GCS prefix.

---

## Which file do I run for each path?
- Vertex Pipelines end-to-end:
  - Primary: `./scripts/deploy_and_run.sh` (builds training image; optionally builds serving image and enables deploy if configured)
  - Alternate: `python pipeline/forecast_pipeline.py` (ensure `DOCKER_IMAGE_URI` points to an already-pushed image; if deploying, also pre-build/push `SERVING_IMAGE_URI` and set `ENABLE_DEPLOY=true`)

- Cloud Run Jobs path (no endpoint deployment):
  - `./scripts/run_cloudrun_pipeline.sh`
  - Then run `./scripts/build_and_deploy_serving.sh` if you want an endpoint.

- Dedicated serving deployment:
  - `./scripts/build_and_deploy_serving.sh` (assumes a trained model artifact exists in GCS and builds/pushes `Dockerfile.serve`)

- Online call:
  - `python scripts/endpoint_predict.py --project-id ... --region ... --endpoint-id ... --instances '[{...}]'`

- Batch prediction:
  - `python scripts/batch_predict.py --project-id ... --region ... --model ... --input-uri ... --output-uri-prefix ... --predictions-format bigquery|jsonl`

---

## Simple summary
- Your data starts as a CSV in GCS. The ingestion step loads it safely into BigQuery and creates a clean final table.
- Training reads from this final table, builds in-memory features, trains LightGBM, and outputs a `model.joblib` artifact.
- Inference reads the last days from the final table, reconstructs features, and makes a next-day forecast.
- If you register and deploy, Vertex AI hosts your model behind an endpoint so applications can send JSON and receive predictions instantly.
- You can orchestrate all of this in Vertex AI Pipelines (recommended), run it as Cloud Run Jobs (alternative), or split training and serving into separate runs as needed.
