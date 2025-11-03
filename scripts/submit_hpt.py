#!/usr/bin/env python
"""
Programmatically submit a Vertex AI Hyperparameter Tuning Job for the training container
and write the best trial's parameters to a provided GCS URI or local path.

Example:
  python scripts/submit_hpt.py \
    --project-id my-forecast-project-18870 \
    --region us-central1 \
    --image-uri us-central1-docker.pkg.dev/my-forecast-project-18870/sales-forecast-repo/sales-forecast:latest \
    --service-account ds-pipeline-sa@my-forecast-project-18870.iam.gserviceaccount.com \
    --display-name sf-hpt \
    --metric-name mae --metric-goal minimize \
    --max-trials 20 --parallel-trials 4 \
    --machine-type n1-standard-4 \
    --best-params-output-gcs gs://my-forecast-project-18870-staging/hpt/best_params.json

Optionally pass a param space JSON (inline) or a GCS path to load it from.
Param space JSON format:
{
  "learning_rate": {"type": "double", "min": 0.001, "max": 0.3, "scale": "log"},
  "num_leaves": {"type": "integer", "min": 16, "max": 512},
  "feature_fraction": {"type": "double", "min": 0.5, "max": 1.0},
  "bagging_fraction": {"type": "double", "min": 0.5, "max": 1.0},
  "bagging_freq": {"type": "integer", "min": 0, "max": 10},
  "lambda_l1": {"type": "double", "min": 1e-8, "max": 10.0, "scale": "log"},
  "lambda_l2": {"type": "double", "min": 1e-8, "max": 10.0, "scale": "log"},
  "min_child_samples": {"type": "integer", "min": 5, "max": 200},
  "max_depth": {"type": "integer", "min": -1, "max": 16}
}
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt
from google.cloud import storage


def _load_param_space(inline_json: str | None, gcs_uri: str | None) -> Dict[str, Any]:
    if inline_json:
        return json.loads(inline_json)
    if gcs_uri and gcs_uri.startswith("gs://"):
        bucket_name, blob_path = gcs_uri[5:].split("/", 1)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        data = blob.download_as_text()
        return json.loads(data)
    # default reasonable space for LightGBM
    return {
        "learning_rate": {"type": "double", "min": 0.001, "max": 0.3, "scale": "log"},
        "num_leaves": {"type": "integer", "min": 16, "max": 512},
        "feature_fraction": {"type": "double", "min": 0.5, "max": 1.0},
        "bagging_fraction": {"type": "double", "min": 0.5, "max": 1.0},
        "bagging_freq": {"type": "integer", "min": 0, "max": 10},
        "lambda_l1": {"type": "double", "min": 1e-8, "max": 10.0, "scale": "log"},
        "lambda_l2": {"type": "double", "min": 1e-8, "max": 10.0, "scale": "log"},
        "min_child_samples": {"type": "integer", "min": 5, "max": 200},
        "max_depth": {"type": "integer", "min": -1, "max": 16},
    }


def _to_parameter_spec(space: Dict[str, Any]) -> Dict[str, hpt._ParameterSpec]:
    spec: Dict[str, hpt._ParameterSpec] = {}
    for name, cfg in space.items():
        t = (cfg.get("type") or cfg.get("param_type") or "double").lower()
        scale = cfg.get("scale", "linear").lower()
        if t == "double":
            spec[name] = hpt.DoubleParameterSpec(min=cfg["min"], max=cfg["max"], scale=scale)
        elif t == "integer":
            spec[name] = hpt.IntegerParameterSpec(min=cfg["min"], max=cfg["max"], scale=scale)
        elif t == "categorical":
            spec[name] = hpt.CategoricalParameterSpec(values=cfg["values"])
        elif t == "discrete":
            spec[name] = hpt.DiscreteParameterSpec(values=cfg["values"], scale=scale)
        else:
            raise ValueError(f"Unsupported param type for {name}: {t}")
    return spec


def _write_json_to_path(data: Dict[str, Any], path: str) -> None:
    text = json.dumps(data)
    if path.startswith("gs://"):
        bucket_name, blob_path = path[5:].split("/", 1)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.upload_from_string(text, content_type="application/json")
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit Vertex AI Hyperparameter Tuning Job and save best params")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--image-uri", required=True, help="Training image URI for trials")
    parser.add_argument("--service-account", required=False, help="Service account for the trials")
    parser.add_argument("--display-name", default="sales-forecast-hpt")
    parser.add_argument("--metric-name", default="mae", help="Objective metric tag the trainer reports")
    parser.add_argument("--metric-goal", default="minimize", choices=["minimize", "maximize"])
    parser.add_argument("--max-trials", type=int, default=20)
    parser.add_argument("--parallel-trials", type=int, default=4)
    parser.add_argument("--machine-type", default="n1-standard-4")
    parser.add_argument("--param-space-json", required=False, help="Inline JSON string for parameter space")
    parser.add_argument("--param-space-gcs", required=False, help="gs:// path to JSON with parameter space")
    parser.add_argument("--best-params-output-gcs", required=True, help="Where to write best params JSON (gs:// or local path)")
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    args = parser.parse_args()

    aiplatform.init(project=args.project_id, location=args.region)

    # Trial job spec (CustomJob)
    worker_pool_specs = [{
        "machine_spec": {"machine_type": args.machine_type},
        "replica_count": 1,
        "container_spec": {
            "image_uri": args.image_uri,
            "command": ["python"],
            "args": [
                "-m", "sales_forecast.train",
                "--model-path", "/tmp/model.joblib",
                "--hpt-metric-name", args.metric_name,
                "--early-stopping-rounds", str(args.early_stopping_rounds),
            ],
            # Env can be added here if needed
        },
    }]

    custom_job = aiplatform.CustomJob(
        display_name=f"{args.display_name}-trial",
        worker_pool_specs=worker_pool_specs,
        service_account=args.service_account,
    )

    space = _load_param_space(args.param_space_json, args.param_space_gcs)
    parameter_spec = _to_parameter_spec(space)
    metric_spec = {args.metric_name: args.metric_goal}

    hp_job = aiplatform.HyperparameterTuningJob(
        display_name=args.display_name,
        custom_job=custom_job,
        metric_spec=metric_spec,
        parameter_spec=parameter_spec,
        max_trial_count=args.max_trials,
        parallel_trial_count=args.parallel_trials,
    )

    hp_job.run(sync=True)

    # Find best trial
    goal_minimize = args.metric_goal == "minimize"
    best_params: Dict[str, Any] = {}
    best_val: float | None = None

    # Use the underlying GCA trials for stability of fields
    gca = hp_job._gca_resource  # pylint: disable=protected-access
    for t in gca.trials or []:
        if not t.final_measurement or not t.final_measurement.metrics:
            continue
        m_val = None
        for m in t.final_measurement.metrics:
            if m.metric_id == args.metric_name:
                m_val = m.value
                break
        if m_val is None:
            continue
        if best_val is None or (goal_minimize and m_val < best_val) or (not goal_minimize and m_val > best_val):
            best_val = m_val
            best_params = {p.parameter_id: p.value for p in (t.parameters or [])}

    if not best_params:
        raise RuntimeError("No completed trials with objective metric; cannot determine best params.")

    _write_json_to_path(best_params, args.best_params_output_gcs)
    print(json.dumps({"best_params": best_params, "metric": args.metric_name, "goal": args.metric_goal}))


if __name__ == "__main__":
    main()


