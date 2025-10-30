#!/usr/bin/env bash
set -euo pipefail

# Load .env
if [ -f ".env" ]; then
  set -a; . ./.env; set +a
fi

PROJECT_ID="${PROJECT_ID:-my-forecast-project-18870}"
REGION="${REGION:-us-central1}"
REPO="${REPO:-sales-forecast-repo}"
SERVE_IMAGE="${SERVE_IMAGE:-sales-forecast-serve}"
SERVING_IMAGE_URI="${SERVING_IMAGE_URI:-${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVE_IMAGE}:latest}"

# Derive artifact directory from MODEL_GCS_URI if it points to a file
if [[ "${MODEL_GCS_URI:-}" == gs://*/*.* ]]; then
  ARTIFACT_URI="gs://$(echo "${MODEL_GCS_URI#gs://}" | awk -F/ '{print $1}')/$(dirname "${MODEL_GCS_URI#gs://*/}")/"
else
  ARTIFACT_URI="${MODEL_GCS_URI:-gs://${PROJECT_ID}-staging/models/}"
fi

echo "Project: ${PROJECT_ID}"
echo "Region:  ${REGION}"
echo "Serving image: ${SERVING_IMAGE_URI}"
echo "Artifact URI: ${ARTIFACT_URI}"

gcloud config set project "${PROJECT_ID}" >/dev/null
gcloud services enable artifactregistry.googleapis.com aiplatform.googleapis.com --project "${PROJECT_ID}" >/dev/null
gcloud auth configure-docker "${REGION}-docker.pkg.dev" -q

# Ensure repo exists
if ! gcloud artifacts repositories describe "${REPO}" --location="${REGION}" >/dev/null 2>&1; then
  gcloud artifacts repositories create "${REPO}" --repository-format=docker --location="${REGION}"
fi

# Build and push serving image
docker build -f Dockerfile.serve -t "${SERVING_IMAGE_URI}" .
docker push "${SERVING_IMAGE_URI}"

# Register model and deploy endpoint
python deploy/vertex.py register \
  --project-id "${PROJECT_ID}" \
  --region "${REGION}" \
  --display-name "sales-forecast-model" \
  --artifact-uri "${ARTIFACT_URI}" \
  --container-uri "${SERVING_IMAGE_URI}"

# Get latest model by display name and deploy (create endpoint if not exists)
MODEL_NAME=$(python - <<'PY'
from google.cloud import aiplatform as aipl
import os
proj=os.environ.get('PROJECT_ID'); reg=os.environ.get('REGION')
aipl.init(project=proj, location=reg)
ms=aipl.Model.list(filter="display_name=sales-forecast-model", order_by="create_time desc")
print(ms[0].resource_name)
PY
)

python deploy/vertex.py deploy \
  --project-id "${PROJECT_ID}" \
  --region "${REGION}" \
  --model-resource-name "${MODEL_NAME}" \
  --endpoint-display-name "sales-forecast-endpoint" \
  --machine-type "n1-standard-2" \
  --min-replicas 1 --max-replicas 1 \
  --save-endpoint-id .vertex_endpoint_id

echo "Endpoint deployed. ID saved in .vertex_endpoint_id"


