"""
Utilities to register a model in Vertex AI Model Registry and deploy it to an Endpoint.

Usage examples:

Register a model artifact stored in GCS (created by training):
  python deploy/vertex.py register \
    --project-id my-forecast-project-18870 \
    --region us-central1 \
    --display-name sales-forecast-model \
    --artifact-uri gs://my-forecast-project-18870-staging/models/run-123/ \
    --container-uri us-central1-docker.pkg.dev/my-forecast-project-18870/sales-forecast-repo/sales-forecast:latest \
    --labels env=dev,owner=nik

Create an endpoint and deploy the latest version:
  python deploy/vertex.py deploy \
    --project-id my-forecast-project-18870 \
    --region us-central1 \
    --model-resource-name projects/123/locations/us-central1/models/456789 \
    --endpoint-display-name sales-forecast-endpoint \
    --machine-type n1-standard-4 \
    --min-replicas 1 --max-replicas 2

Re-deploy to an existing endpoint by ID:
  python deploy/vertex.py deploy \
    --project-id my-forecast-project-18870 \
    --region us-central1 \
    --model-resource-name projects/123/locations/us-central1/models/456789 \
    --endpoint-id 1234567890123456789 \
    --machine-type n1-standard-4
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional, Dict

from google.cloud import aiplatform


def parse_labels(labels_str: Optional[str]) -> Dict[str, str]:
    if not labels_str:
        return {}
    labels: Dict[str, str] = {}
    for pair in labels_str.split(","):
        if not pair:
            continue
        if "=" not in pair:
            continue
        k, v = pair.split("=", 1)
        labels[k.strip()] = v.strip()
    return labels


def register_model(
    project_id: str,
    region: str,
    display_name: str,
    artifact_uri: str,
    container_uri: str,
    labels: Optional[str] = None,
) -> aiplatform.Model:
    aiplatform.init(project=project_id, location=region)

    model = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=artifact_uri,
        serving_container_image_uri=container_uri,
        labels=parse_labels(labels),
        sync=True,
    )

    print("Registered model:", model.resource_name)
    return model


def get_or_create_endpoint(
    project_id: str,
    region: str,
    endpoint_display_name: Optional[str] = None,
    endpoint_id: Optional[str] = None,
) -> aiplatform.Endpoint:
    aiplatform.init(project=project_id, location=region)

    if endpoint_id:
        return aiplatform.Endpoint(endpoint_name=endpoint_id)

    # Try to find an endpoint by display name
    if endpoint_display_name:
        endpoints = aiplatform.Endpoint.list(filter=f"display_name={endpoint_display_name}")
        if endpoints:
            return endpoints[0]

    # Create new endpoint if none found
    endpoint = aiplatform.Endpoint.create(display_name=endpoint_display_name or "sales-forecast-endpoint")
    print("Created endpoint:", endpoint.resource_name)
    return endpoint


def deploy_model(
    project_id: str,
    region: str,
    model_resource_name: str,
    machine_type: str = "n1-standard-4",
    min_replicas: int = 1,
    max_replicas: int = 2,
    endpoint_display_name: Optional[str] = None,
    endpoint_id: Optional[str] = None,
    traffic_split: Optional[Dict[str, int]] = None,
) -> aiplatform.Endpoint:
    aiplatform.init(project=project_id, location=region)

    endpoint = get_or_create_endpoint(project_id, region, endpoint_display_name, endpoint_id)
    model = aiplatform.Model(model_resource_name)

    # Note: Model.deploy returns the Endpoint object in the Python SDK
    endpoint = model.deploy(
        endpoint=endpoint,
        machine_type=machine_type,
        min_replica_count=min_replicas,
        max_replica_count=max_replicas,
        traffic_percentage=100 if not traffic_split else None,
        traffic_split=traffic_split,
        sync=True,
    )

    # The 'id' of the DeployedModel can be found via endpoint.gca_resource.deployed_models.
    # For simplicity we print the endpoint resource name which uniquely identifies where the model is deployed.
    print("Deployed to Endpoint:", endpoint.resource_name)
    return endpoint


def write_endpoint_id(endpoint: aiplatform.Endpoint, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(endpoint.resource_name)
    print("Saved endpoint id to:", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Vertex AI model registration and deployment utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_reg = subparsers.add_parser("register", help="Register model to Vertex AI Model Registry")
    p_reg.add_argument("--project-id", required=True)
    p_reg.add_argument("--region", required=True)
    p_reg.add_argument("--display-name", required=True)
    p_reg.add_argument("--artifact-uri", required=True, help="GCS URI directory containing model artifacts")
    p_reg.add_argument("--container-uri", required=True, help="Serving container image URI")
    p_reg.add_argument("--labels", required=False, help="Comma-separated key=value labels")

    p_dep = subparsers.add_parser("deploy", help="Deploy a registered model to a Vertex Endpoint")
    p_dep.add_argument("--project-id", required=True)
    p_dep.add_argument("--region", required=True)
    p_dep.add_argument("--model-resource-name", required=True)
    p_dep.add_argument("--machine-type", default="n1-standard-4")
    p_dep.add_argument("--min-replicas", type=int, default=1)
    p_dep.add_argument("--max-replicas", type=int, default=2)
    p_dep.add_argument("--endpoint-display-name", required=False)
    p_dep.add_argument("--endpoint-id", required=False)
    p_dep.add_argument("--save-endpoint-id", required=False, help="Path to file to store endpoint resource name")
    p_dep.add_argument("--traffic-split", required=False, help='JSON mapping of deployed_model_id to traffic %')

    args = parser.parse_args()

    if args.command == "register":
        model = register_model(
            project_id=args.project_id,
            region=args.region,
            display_name=args.display_name,
            artifact_uri=args.artifact_uri,
            container_uri=args.container_uri,
            labels=getattr(args, "labels", None),
        )
        print(json.dumps({"model_resource_name": model.resource_name}))
    elif args.command == "deploy":
        traffic = None
        if getattr(args, "traffic_split", None):
            traffic = json.loads(args.traffic_split)
        endpoint = deploy_model(
            project_id=args.project_id,
            region=args.region,
            model_resource_name=args.model_resource_name,
            machine_type=args.machine_type,
            min_replicas=args.min_replicas,
            max_replicas=args.max_replicas,
            endpoint_display_name=getattr(args, "endpoint_display_name", None),
            endpoint_id=getattr(args, "endpoint_id", None),
            traffic_split=traffic,
        )
        if getattr(args, "save_endpoint_id", None):
            write_endpoint_id(endpoint, args.save_endpoint_id)


if __name__ == "__main__":
    main()


