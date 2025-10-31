from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage


app = FastAPI()

_model = None  # loaded model singleton
_feature_names: Optional[List[str]] = None


class PredictRequest(BaseModel):
    instances: List[Dict[str, Any]]


def _download_from_gcs(gcs_uri: str, local_path: str) -> None:
    if not gcs_uri.startswith("gs://"):
        raise ValueError("GCS URI must start with gs://")
    bucket_name, blob_path = gcs_uri[5:].split("/", 1)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(local_path)


def _load_model() -> None:
    global _model, _feature_names
    if _model is not None:
        return

    # Vertex sets AIP_STORAGE_URI to the artifact directory
    artifact_dir = os.environ.get("AIP_STORAGE_URI")
    explicit_gcs = os.environ.get("MODEL_GCS_URI")
    local_model_path = os.environ.get("MODEL_PATH")
    model_filename = os.environ.get("MODEL_FILENAME", "model.joblib")

    if local_model_path and os.path.exists(local_model_path):
        model_path = local_model_path
    elif explicit_gcs:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = os.path.join(tmpdir, "model.joblib")
            _download_from_gcs(explicit_gcs, tmp)
            _model = joblib.load(tmp)
            _feature_names = getattr(_model, "feature_name_", None)
            return
    elif artifact_dir:
        # Expect the artifact directory to contain model.joblib
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = os.path.join(tmpdir, "model.joblib")
            gcs_uri = artifact_dir.rstrip("/") + "/" + model_filename
            _download_from_gcs(gcs_uri, tmp)
            _model = joblib.load(tmp)
            _feature_names = getattr(_model, "feature_name_", None)
            return
    else:
        raise RuntimeError("No model location provided. Set AIP_STORAGE_URI or MODEL_GCS_URI or MODEL_PATH.")

    _model = joblib.load(model_path)
    _feature_names = getattr(_model, "feature_name_", None)


@app.get("/")
def root() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "healthy"}


# Some Vertex components may probe a versioned path. Return 200 to unknown v1 probe paths.
@app.get("/v1/endpoints/{endpoint_id}/deployedModels/{deployed_model_id}")
def vertex_probe(endpoint_id: str, deployed_model_id: str) -> Dict[str, str]:
    return {"status": "ok", "endpoint": endpoint_id, "deployed_model": deployed_model_id}


@app.post("/predict")
def predict(req: PredictRequest) -> Dict[str, Any]:
    try:
        _load_model()
        if not req.instances:
            raise HTTPException(status_code=400, detail="instances is empty")

        df = pd.DataFrame(req.instances)
        if _feature_names:
            # Ensure required columns exist; fill missing with 0
            for c in _feature_names:
                if c not in df.columns:
                    df[c] = 0
            df = df[_feature_names]

        preds = _model.predict(df)
        return {"predictions": [float(x) for x in preds]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


