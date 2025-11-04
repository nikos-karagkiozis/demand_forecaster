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
ENABLE_HPT="${ENABLE_HPT:-false}"
HPT_MAX_TRIALS="${HPT_MAX_TRIALS:-20}"
HPT_PARALLEL_TRIALS="${HPT_PARALLEL_TRIALS:-4}"
HPT_METRIC_NAME="${HPT_METRIC_NAME:-mae}"
HPT_METRIC_GOAL="${HPT_METRIC_GOAL:-minimize}"
HPT_MACHINE_TYPE="${HPT_MACHINE_TYPE:-n1-standard-4}"
BEST_PARAMS_URI="${BEST_PARAMS_URI:-gs://${PROJECT_ID}-staging/hpt/best_params.json}"
HPT_PARAM_SPACE_GCS="${HPT_PARAM_SPACE_GCS:-}"
HPT_PARAM_SPACE_JSON="${HPT_PARAM_SPACE_JSON:-}"
HPT_TRIAL_SERVICE_ACCOUNT="${HPT_TRIAL_SERVICE_ACCOUNT:-}"

# Model artifact location in GCS for cross-step sharing
MODEL_GCS_URI="${MODEL_GCS_URI:-gs://${PROJECT_ID}-staging/models/model.joblib}"

echo "Project:        ${PROJECT_ID}"
echo "Region:         ${REGION}"
echo "Image URI:      ${DOCKER_IMAGE_URI}"
echo "Service acct:   ${SERVICE_ACCOUNT}"
echo "Model GCS URI:  ${MODEL_GCS_URI}"
echo "Enable HPT:     ${ENABLE_HPT}"

gcloud config set project "${PROJECT_ID}" >/dev/null

# Ensure services
gcloud services enable run.googleapis.com artifactregistry.googleapis.com aiplatform.googleapis.com --project "${PROJECT_ID}" >/dev/null

# Ensure Artifact Registry repo exists (idempotent)
if ! gcloud artifacts repositories describe "${REPO}" --location="${REGION}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud artifacts repositories create "${REPO}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="Sales forecast repo"\
    --project "${PROJECT_ID}"
fi

# Ensure the bucket for MODEL_GCS_URI exists (idempotent)
if [[ "${MODEL_GCS_URI}" == gs://*/* ]]; then
  BUCKET="$(echo "${MODEL_GCS_URI#gs://}" | cut -d/ -f1)"
  if ! gsutil ls -b "gs://${BUCKET}" >/dev/null 2>&1; then
    gsutil mb -l "${LOCATION}" "gs://${BUCKET}"
  fi
fi

# Ensure the staging bucket exists (used by Vertex CustomJobs/HPT)
STAGING_BUCKET="${STAGING_BUCKET:-${PROJECT_ID}-staging}"
if ! gsutil ls -b "gs://${STAGING_BUCKET}" >/dev/null 2>&1; then
  gsutil mb -l "${LOCATION}" "gs://${STAGING_BUCKET}"
fi

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
  --args="-m,sales_forecast.data_ingest"

# 2) Train job (saves locally then uploads to GCS)
cr_job_upsert "sf-train" \
  --service-account "${SERVICE_ACCOUNT}" \
  --set-env-vars "${COMMON_ENVS}" \
  --command python \
  --args="-m,sales_forecast.train,--model-path,/tmp/model.joblib,--model-gcs-uri,${MODEL_GCS_URI}"

# 3) Predict job (downloads from GCS)
cr_job_upsert "sf-predict" \
  --service-account "${SERVICE_ACCOUNT}" \
  --set-env-vars "${COMMON_ENVS}" \
  --command python \
  --args="-m,sales_forecast.predict,--model-gcs-uri,${MODEL_GCS_URI}"

# 4) Optional: HPT submitter job and train-with-best job
if [[ "${ENABLE_HPT,,}" == "true" || "${ENABLE_HPT}" == "1" || "${ENABLE_HPT,,}" == "yes" || "${ENABLE_HPT,,}" == "on" ]]; then
  # Ensure the bucket for BEST_PARAMS_URI exists
  if [[ "${BEST_PARAMS_URI}" == gs://*/* ]]; then
    BUCKET_BEST="$(echo "${BEST_PARAMS_URI#gs://}" | cut -d/ -f1)"
    if ! gsutil ls -b "gs://${BUCKET_BEST}" >/dev/null 2>&1; then
      gsutil mb -l "${LOCATION}" "gs://${BUCKET_BEST}"
    fi
  fi

  # Build base args for HPT submitter
  HPT_ARGS="scripts/submit_hpt.py,\
--project-id,${PROJECT_ID},\
--region,${REGION},\
--image-uri,${DOCKER_IMAGE_URI},\
--staging-bucket,gs://${STAGING_BUCKET},\
--display-name,sf-hpt,\
--metric-name,${HPT_METRIC_NAME},\
--metric-goal,${HPT_METRIC_GOAL},\
--max-trials,${HPT_MAX_TRIALS},\
--parallel-trials,${HPT_PARALLEL_TRIALS},\
--machine-type,${HPT_MACHINE_TYPE},\
--best-params-output-gcs,${BEST_PARAMS_URI}"
  if [[ -n "${HPT_TRIAL_SERVICE_ACCOUNT}" ]]; then
    HPT_ARGS="${HPT_ARGS},--service-account,${HPT_TRIAL_SERVICE_ACCOUNT}"
  fi

  cr_job_upsert "sf-hpt" \
    --service-account "${SERVICE_ACCOUNT}" \
    --set-env-vars "${COMMON_ENVS}" \
    --command python \
    --args="${HPT_ARGS}"

  # Append optional param space flags if provided
  if [[ -n "${HPT_PARAM_SPACE_GCS}" ]]; then
    gcloud run jobs update "sf-hpt" --region "${REGION}" \
      --args="${HPT_ARGS},--param-space-gcs,${HPT_PARAM_SPACE_GCS}" >/dev/null
  elif [[ -n "${HPT_PARAM_SPACE_JSON}" ]]; then
    gcloud run jobs update "sf-hpt" --region "${REGION}" \
      --args="${HPT_ARGS},--param-space-json,${HPT_PARAM_SPACE_JSON}" >/dev/null
  fi

  cr_job_upsert "sf-train-best" \
    --service-account "${SERVICE_ACCOUNT}" \
    --set-env-vars "${COMMON_ENVS}" \
    --command python \
    --args="scripts/train_with_best_params.py,\
--best-params-gcs-uri,${BEST_PARAMS_URI},\
--model-path,/tmp/model.joblib,\
--model-gcs-uri,${MODEL_GCS_URI}"
fi

# Execute sequentially and wait for completion
gcloud run jobs execute sf-data-ingest --region "${REGION}" --wait
if [[ "${ENABLE_HPT,,}" == "true" || "${ENABLE_HPT}" == "1" || "${ENABLE_HPT,,}" == "yes" || "${ENABLE_HPT,,}" == "on" ]]; then
  gcloud run jobs execute sf-hpt --region "${REGION}" --wait
  gcloud run jobs execute sf-train-best --region "${REGION}" --wait
else
  gcloud run jobs execute sf-train --region "${REGION}" --wait
fi
gcloud run jobs execute sf-predict --region "${REGION}" --wait

echo "Cloud Run pipeline completed."


