#!/usr/bin/env bash
set -euo pipefail

# Load .env if present and export variables
if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

PROJECT_ID="${PROJECT_ID:-my-forecast-project-18870}"
REGION="${REGION:-us-central1}"
REPO="${REPO:-sales-forecast-repo}"
IMAGE="${IMAGE:-sales-forecast}"
DOCKER_IMAGE_URI="${DOCKER_IMAGE_URI:-${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}:latest}"
PIPELINE_ROOT="${PIPELINE_ROOT:-gs://${PROJECT_ID}-staging/pipeline_root}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-ds-pipeline-sa@${PROJECT_ID}.iam.gserviceaccount.com}"
LOCATION="${LOCATION:-US}"

# Model artifact location in GCS for cross-step sharing
MODEL_GCS_URI="${MODEL_GCS_URI:-gs://${PROJECT_ID}-staging/model/model.joblib}"

echo "Project:        ${PROJECT_ID}"
echo "Region:         ${REGION}"
echo "Image URI:      ${DOCKER_IMAGE_URI}"
echo "Service acct:   ${SERVICE_ACCOUNT}"
echo "Model GCS URI:  ${MODEL_GCS_URI}"

gcloud config set project "${PROJECT_ID}" >/dev/null

# Ensure services
gcloud services enable run.googleapis.com artifactregistry.googleapis.com --project "${PROJECT_ID}" >/dev/null

# Build & push image if not present
gcloud auth configure-docker "${REGION}-docker.pkg.dev" -q
docker build -t "${DOCKER_IMAGE_URI}" .
docker push "${DOCKER_IMAGE_URI}"

# Helper: create-or-update a Cloud Run Job
cr_job_upsert() {
  local name="$1"; shift
  if gcloud run jobs describe "$name" --region "${REGION}" >/dev/null 2>&1; then
    gcloud run jobs update "$name" --image "${DOCKER_IMAGE_URI}" --region "${REGION}" "$@"
  else
    gcloud run jobs create "$name" --image "${DOCKER_IMAGE_URI}" --region "${REGION}" "$@"
  fi
}

# Common env vars for the container
COMMON_ENVS="PROJECT_ID=${PROJECT_ID},REGION=${REGION},DATASET=${DATASET:-gk2_takeaway_sales},STAGING_TABLE=${STAGING_TABLE:-staging_daily_sales},FINAL_TABLE=${FINAL_TABLE:-daily_sales_features},LOCATION=${LOCATION}"

# 1) Data ingest job
cr_job_upsert "sf-data-ingest" \
  --service-account "${SERVICE_ACCOUNT}" \
  --set-env-vars "${COMMON_ENVS}" \
  --command python \
  --args -m,sales_forecast.data_ingest

# 2) Train job (saves locally then uploads to GCS)
cr_job_upsert "sf-train" \
  --service-account "${SERVICE_ACCOUNT}" \
  --set-env-vars "${COMMON_ENVS}" \
  --command python \
  --args -m,sales_forecast.train,--model-path,/tmp/model.joblib,--model-gcs-uri,${MODEL_GCS_URI}

# 3) Predict job (downloads from GCS)
cr_job_upsert "sf-predict" \
  --service-account "${SERVICE_ACCOUNT}" \
  --set-env-vars "${COMMON_ENVS}" \
  --command python \
  --args -m,sales_forecast.predict,--model-gcs-uri,${MODEL_GCS_URI}

# Execute sequentially and wait for completion
gcloud run jobs execute sf-data-ingest --region "${REGION}" --wait
gcloud run jobs execute sf-train --region "${REGION}" --wait
gcloud run jobs execute sf-predict --region "${REGION}" --wait

echo "Cloud Run pipeline completed."


