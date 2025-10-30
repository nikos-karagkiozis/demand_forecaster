#!/usr/bin/env bash
set -euo pipefail

# Load .env if present and export variables
if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

# Required config with sensible defaults
PROJECT_ID="${PROJECT_ID:-my-forecast-project-18870}"
REGION="${REGION:-us-central1}"
REPO="${REPO:-sales-forecast-repo}"
IMAGE="${IMAGE:-sales-forecast}"
DOCKER_IMAGE_URI="${DOCKER_IMAGE_URI:-${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}:latest}"
PIPELINE_ROOT="${PIPELINE_ROOT:-gs://${PROJECT_ID}-staging/pipeline_root}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-ds-pipeline-sa@${PROJECT_ID}.iam.gserviceaccount.com}"
LOCATION="${LOCATION:-US}"
ENABLE_DEPLOY="${ENABLE_DEPLOY:-false}"
SERVE_IMAGE="${SERVE_IMAGE:-sales-forecast-serve}"
SERVING_IMAGE_URI="${SERVING_IMAGE_URI:-${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVE_IMAGE}:latest}"

echo "Project:        ${PROJECT_ID}"
echo "Region:         ${REGION}"
echo "Repo:           ${REPO}"
echo "Image:          ${IMAGE}"
echo "Image URI:      ${DOCKER_IMAGE_URI}"
echo "Serve image:    ${SERVE_IMAGE}"
echo "Serve URI:      ${SERVING_IMAGE_URI}"
echo "Pipeline root:  ${PIPELINE_ROOT}"
echo "Service acct:   ${SERVICE_ACCOUNT}"
echo "Location:       ${LOCATION}"
echo "Enable deploy:  ${ENABLE_DEPLOY}"

# Ensure gcloud is pointing at the correct project
gcloud config set project "${PROJECT_ID}" >/dev/null

# Enable required services (idempotent)
gcloud services enable \
  aiplatform.googleapis.com \
  artifactregistry.googleapis.com \
  --project "${PROJECT_ID}" >/dev/null

# Create Artifact Registry repo if it doesn't exist (idempotent)
if ! gcloud artifacts repositories describe "${REPO}" --location="${REGION}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud artifacts repositories create "${REPO}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="Sales forecast repo" \
    --project "${PROJECT_ID}"
fi

# Configure Docker to push to the regional registry
gcloud auth configure-docker "${REGION}-docker.pkg.dev" -q

# Ensure the staging bucket exists
STAGING_BUCKET="${PROJECT_ID}-staging"
if ! gsutil ls -b "gs://${STAGING_BUCKET}" >/dev/null 2>&1; then
  gsutil mb -l "${LOCATION}" "gs://${STAGING_BUCKET}"
fi

# Build and push the container
docker build -t "${DOCKER_IMAGE_URI}" .
docker push "${DOCKER_IMAGE_URI}"

# Optionally build and push the serving image for endpoint deployment
if [[ "${ENABLE_DEPLOY,,}" == "true" || "${ENABLE_DEPLOY}" == "1" || "${ENABLE_DEPLOY,,}" == "yes" || "${ENABLE_DEPLOY,,}" == "on" ]]; then
  if [[ "${SERVING_IMAGE_URI}" == "${DOCKER_IMAGE_URI}" ]]; then
    echo "[WARN] SERVING_IMAGE_URI equals DOCKER_IMAGE_URI. Serving image must be a dedicated image (Dockerfile.serve). Skipping deploy enablement."
    ENABLE_DEPLOY=false
  else
    echo "ENABLE_DEPLOY is true → ensuring serving image is available..."
    # Try to detect if the image already exists in the registry; if not, build and push it
    if docker manifest inspect "${SERVING_IMAGE_URI}" >/dev/null 2>&1; then
      echo "Serving image already exists: ${SERVING_IMAGE_URI} (skipping build)"
    else
      echo "Building and pushing serving image: ${SERVING_IMAGE_URI}"
      docker build -f Dockerfile.serve -t "${SERVING_IMAGE_URI}" .
      docker push "${SERVING_IMAGE_URI}"
    fi
  fi
fi

# Export values so the launcher picks them up
export PROJECT_ID REGION DOCKER_IMAGE_URI PIPELINE_ROOT SERVICE_ACCOUNT ENABLE_DEPLOY SERVING_IMAGE_URI

# Submit the pipeline via Poetry if available, else plain Python
if command -v poetry >/dev/null 2>&1; then
  poetry run python pipeline/forecast_pipeline.py
else
  python pipeline/forecast_pipeline.py
fi

echo "Done. Check Vertex AI pipelines UI for run status."


