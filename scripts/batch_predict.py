#!/usr/bin/env python
"""
Create a Vertex AI Batch Prediction Job.

It reads instances from GCS (JSONL/CSV/BigQuery), writes predictions to GCS or BigQuery.

Usage (JSONL input, GCS output):
  python scripts/batch_predict.py \
    --project-id my-forecast-project-18870 \
    --region us-central1 \
    --model projects/123/locations/us-central1/models/456 \
    --input-uri gs://my-bucket/predict/input.jsonl \
    --output-uri-prefix gs://my-bucket/predict/output/
"""

from __future__ import annotations

import argparse
from typing import Optional

from google.cloud import aiplatform


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Vertex Batch Prediction Job")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--model", required=True, help="Model resource name or ID")
    parser.add_argument("--input-uri", required=True, help="GCS or BigQuery source (e.g., gs://... or bq://...")")
    parser.add_argument("--output-uri-prefix", required=True, help="GCS prefix or BigQuery sink")
    parser.add_argument("--instances-format", default="jsonl", choices=["jsonl", "csv", "bigquery"], help="Input format")
    parser.add_argument("--predictions-format", default="jsonl", choices=["jsonl", "csv", "bigquery"], help="Output format")
    parser.add_argument("--display-name", default="sales-forecast-batch-predict")
    args = parser.parse_args()

    aiplatform.init(project=args.project_id, location=args.region)

    model_name = args.model
    if not model_name.startswith("projects/"):
        model_name = f"projects/{args.project_id}/locations/{args.region}/models/{args.model}"

    job = aiplatform.BatchPredictionJob.create(
        job_display_name=args.display_name,
        model_name=model_name,
        gcs_source=args.input_uri if args.instances_format != "bigquery" else None,
        bigquery_source=args.input_uri if args.instances_format == "bigquery" else None,
        gcs_destination_prefix=args.output_uri_prefix if args.predictions_format != "bigquery" else None,
        bigquery_destination_prefix=args.output_uri_prefix if args.predictions_format == "bigquery" else None,
        instances_format=args.instances_format,
        predictions_format=args.predictions_format,
        generate_explanation=False,
        sync=True,
    )

    print("Batch prediction job:", job.resource_name)


if __name__ == "__main__":
    main()


