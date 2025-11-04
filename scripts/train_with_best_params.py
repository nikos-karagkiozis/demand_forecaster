#!/usr/bin/env python
"""
Read best hyperparameters from GCS (JSON) and invoke training with those params.

Example:
  python scripts/train_with_best_params.py \
    --best-params-gcs-uri gs://my-bucket/hpt/best_params.json \
    --model-path /tmp/model.joblib \
    --model-gcs-uri gs://my-bucket/models/model.joblib
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from typing import Any, Dict

from google.cloud import storage


def _download_gcs_to_text(gcs_uri: str) -> str:
    if not gcs_uri.startswith("gs://"):
        raise ValueError("gcs_uri must start with gs://")
    bucket_name, blob_path = gcs_uri[5:].split("/", 1)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    return blob.download_as_text()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train with best params read from GCS JSON")
    parser.add_argument("--best-params-gcs-uri", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-gcs-uri", required=False)
    parser.add_argument("--hpt-metric-name", required=False, default=None)
    args = parser.parse_args()

    data = _download_gcs_to_text(args.best_params_gcs_uri)
    best_params: Dict[str, Any] = json.loads(data)

    cmd = [
        "python", "-m", "sales_forecast.train",
        "--model-path", args.model_path,
    ]
    if args.model_gcs_uri:
        cmd.extend(["--model-gcs-uri", args.model_gcs_uri])
    if args.hpt_metric_name:
        cmd.extend(["--hpt-metric-name", args.hpt_metric_name])

    # Append best params as CLI flags (kebab-case)
    for k, v in best_params.items():
        flag = f"--{k.replace('_', '-')}"
        cmd.extend([flag, str(v)])

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()


