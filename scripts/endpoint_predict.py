#!/usr/bin/env python
"""
Call a Vertex AI Endpoint for online prediction.

Usage:
  python scripts/endpoint_predict.py \
    --project-id my-forecast-project-18870 \
    --region us-central1 \
    --endpoint-id 1234567890123456789 \
    --instances '[{"feature1": 1.2, "feature2": 3.4}]'
"""

from __future__ import annotations

import argparse
import json
from typing import Any, List

from google.cloud import aiplatform


def main() -> None:
    parser = argparse.ArgumentParser(description="Online prediction via Vertex Endpoint")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--endpoint-id", required=True, help="Endpoint resource id or full name")
    parser.add_argument("--instances", required=True, help="JSON list of instances")
    args = parser.parse_args()

    aiplatform.init(project=args.project_id, location=args.region)

    endpoint_name = args.endpoint_id
    if not endpoint_name.startswith("projects/"):
        endpoint_name = f"projects/{args.project_id}/locations/{args.region}/endpoints/{args.endpoint_id}"

    endpoint = aiplatform.Endpoint(endpoint_name)
    instances: List[Any] = json.loads(args.instances)
    predictions = endpoint.predict(instances=instances)
    print(json.dumps(predictions.predictions))


if __name__ == "__main__":
    main()


