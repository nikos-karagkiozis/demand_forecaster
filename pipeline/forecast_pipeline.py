# pipeline/forecast_pipeline.py
"""
Defines the Kubeflow pipeline for the sales forecasting project using KFP v2.

This pipeline orchestrates the data ingestion, model training, and prediction steps,
with each step running in its own containerized environment.
"""

from kfp import dsl
from kfp.compiler import Compiler
from google.cloud import aiplatform
import os
from dotenv import load_dotenv
from kfp import dsl as kfp_dsl

# --- Pipeline Configuration ---
# Load local .env if present for convenient local configuration
load_dotenv()

# Replace with your specific details (can be overridden via environment variables)
PROJECT_ID = os.environ.get("PROJECT_ID", "my-forecast-project-18870")
REGION = os.environ.get("REGION", "us-central1")
PIPELINE_ROOT = os.environ.get("PIPELINE_ROOT", f"gs://{PROJECT_ID}-staging/pipeline_root")

# This is the URI of the Docker image you will build and push to Artifact Registry
DOCKER_IMAGE_URI = os.environ.get(
    "DOCKER_IMAGE_URI",
    f"{REGION}-docker.pkg.dev/{PROJECT_ID}/sales-forecast-repo/sales-forecast:latest",
)

# The service account for running the pipeline components
SERVICE_ACCOUNT = os.environ.get(
    "SERVICE_ACCOUNT", f"ds-pipeline-sa@{PROJECT_ID}.iam.gserviceaccount.com"
)

# Optional HPT-in-pipeline configuration (can be overridden via env)
HPT_METRIC_NAME = os.environ.get("HPT_METRIC_NAME", "mae")
HPT_METRIC_GOAL = os.environ.get("HPT_METRIC_GOAL", "minimize")
HPT_MAX_TRIALS = int(os.environ.get("HPT_MAX_TRIALS", "20"))
HPT_PARALLEL_TRIALS = int(os.environ.get("HPT_PARALLEL_TRIALS", "4"))
HPT_MACHINE_TYPE = os.environ.get("HPT_MACHINE_TYPE", "n1-standard-4")
HPT_PARAM_SPACE_JSON = os.environ.get("HPT_PARAM_SPACE_JSON", "")

# --- Component Definitions ---
# Each component is a step in the pipeline. It runs a command in the specified container image.

@dsl.component(
    base_image=DOCKER_IMAGE_URI,
    packages_to_install=[] # Dependencies are already in the base image
)
def data_ingest_op():
    """Pipeline component to run the data ingestion script."""
    import subprocess
    # The command to execute inside the container
    # '-m' runs the module as a script
    subprocess.run(["python", "-m", "sales_forecast.data_ingest"], check=True)


@dsl.component(
    base_image=DOCKER_IMAGE_URI,
    packages_to_install=[]
)
def train_op(model_artifact: dsl.OutputPath(dsl.Model), hparams_json: str = ""):
    """Pipeline component to run the model training script and output the trained model."""
    import subprocess
    import json
    # KFP automatically provides a path for output artifacts
    args = ["python", "-m", "sales_forecast.train", "--model-path", model_artifact]
    if hparams_json:
        try:
            hp = json.loads(hparams_json)
            for k, v in hp.items():
                flag = f"--{k.replace('_', '-')}"
                args.extend([flag, str(v)])
        except Exception:
            pass
    subprocess.run(args, check=True)


@dsl.component(
    base_image=DOCKER_IMAGE_URI,
    packages_to_install=[]
)
def predict_op(model_artifact: dsl.Input[dsl.Model]): # Added model_artifact as an input
    """Pipeline component to run the prediction script using the trained model."""
    import subprocess
    # Pass the path of the input artifact to the predict.py script
    subprocess.run(["python", "-m", "sales_forecast.predict", "--model-path", model_artifact.path], check=True)


@dsl.component(
    base_image=DOCKER_IMAGE_URI,
    packages_to_install=[]
)
def hpt_submit_op(
    project_id: str,
    region: str,
    image_uri: str,
    service_account: str,
    metric_name: str = "mae",
    metric_goal: str = "minimize",
    max_trials: int = 20,
    parallel_trials: int = 4,
    machine_type: str = "n1-standard-4",
    param_space_json: str = "",
    best_params_json: dsl.OutputPath(str) = None,
):
    """Submit a Vertex HPT job and return the best params as JSON string.

    Note: This component uses the Vertex AI SDK from inside the pipeline step.
    """
    import json
    from google.cloud import aiplatform
    from google.cloud.aiplatform import hyperparameter_tuning as hpt

    aiplatform.init(project=project_id, location=region)

    # Worker pool spec mirrors the training step
    worker_pool_specs = [{
        "machine_spec": {"machine_type": machine_type},
        "replica_count": 1,
        "container_spec": {
            "image_uri": image_uri,
            "command": ["python"],
            "args": [
                "-m", "sales_forecast.train",
                "--model-path", "/tmp/model.joblib",
                "--hpt-metric-name", metric_name,
                "--early-stopping-rounds", "100",
            ],
        },
    }]

    custom_job = aiplatform.CustomJob(
        display_name="sales-forecast-hpt-trial",
        worker_pool_specs=worker_pool_specs,
        service_account=service_account,
    )

    # Default param space if none provided
    space = {
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
    if param_space_json:
        try:
            space = json.loads(param_space_json)
        except Exception:
            pass

    param_spec = {}
    for name, cfg in space.items():
        t = (cfg.get("type") or cfg.get("param_type") or "double").lower()
        scale = cfg.get("scale", "linear").lower()
        if t == "double":
            param_spec[name] = hpt.DoubleParameterSpec(min=cfg["min"], max=cfg["max"], scale=scale)
        elif t == "integer":
            param_spec[name] = hpt.IntegerParameterSpec(min=cfg["min"], max=cfg["max"], scale=scale)
        elif t == "categorical":
            param_spec[name] = hpt.CategoricalParameterSpec(values=cfg["values"])
        elif t == "discrete":
            param_spec[name] = hpt.DiscreteParameterSpec(values=cfg["values"], scale=scale)
        else:
            raise ValueError(f"Unsupported param type for {name}: {t}")

    metric_spec = {metric_name: metric_goal}
    hp_job = aiplatform.HyperparameterTuningJob(
        display_name="sales-forecast-hpt",
        custom_job=custom_job,
        metric_spec=metric_spec,
        parameter_spec=param_spec,
        max_trial_count=int(max_trials),
        parallel_trial_count=int(parallel_trials),
    )
    hp_job.run(sync=True)

    # Extract best params
    goal_minimize = metric_goal == "minimize"
    gca = hp_job._gca_resource  # pylint: disable=protected-access
    best_params = {}
    best_val = None
    for t in gca.trials or []:
        if not t.final_measurement or not t.final_measurement.metrics:
            continue
        val = None
        for m in t.final_measurement.metrics:
            if m.metric_id == metric_name:
                val = m.value
                break
        if val is None:
            continue
        if best_val is None or (goal_minimize and val < best_val) or (not goal_minimize and val > best_val):
            best_val = val
            best_params = {p.parameter_id: p.value for p in (t.parameters or [])}

    with open(best_params_json, "w", encoding="utf-8") as f:
        f.write(json.dumps(best_params))
def register_and_deploy_op(
    model_artifact: dsl.Input[dsl.Model],
    project_id: str,
    region: str,
    container_uri: str,
    models_gcs_dir: str,
    endpoint_display_name: str,
):
    """Registers the trained model to Vertex and deploys to an endpoint.

    Note: Upload requires GCS artifacts. We upload the local artifact to GCS first.
    """
    import os
    import subprocess
    from google.cloud import storage

    os.makedirs("/tmp/model", exist_ok=True)
    local_model_path = "/tmp/model/model.joblib"

    # Copy the KFP-provided model artifact to a specific local path
    subprocess.run(["cp", model_artifact.path, local_model_path], check=True)

    # Upload to GCS under models_gcs_dir
    storage_client = storage.Client(project=project_id)
    if not models_gcs_dir.startswith("gs://"):
        raise ValueError("models_gcs_dir must be a gs:// URI")
    bucket_name, rel_path = models_gcs_dir[5:].split("/", 1)
    bucket = storage_client.bucket(bucket_name)

    # Use run id from env if available to keep versions separate
    run_id = os.environ.get("KFP_RUN_ID", "manual")
    gcs_prefix = f"{rel_path.rstrip('/')}/run-{run_id}/"
    blob = bucket.blob(gcs_prefix + "model.joblib")
    blob.upload_from_filename(local_model_path)

    artifact_uri = f"gs://{bucket_name}/{gcs_prefix}"

    # Register and deploy using utility
    subprocess.run([
        "python", "deploy/vertex.py", "register",
        "--project-id", project_id,
        "--region", region,
        "--display-name", "sales-forecast-model",
        "--artifact-uri", artifact_uri,
        "--container-uri", container_uri,
    ], check=True)

    # Last created model is listed first; deploy it
    from google.cloud import aiplatform as aipl
    aipl.init(project=project_id, location=region)
    models = aipl.Model.list(filter="display_name=sales-forecast-model", order_by="create_time desc")
    if not models:
        raise RuntimeError("Model registration did not produce a Model resource.")
    model = models[0]

    subprocess.run([
        "python", "deploy/vertex.py", "deploy",
        "--project-id", project_id,
        "--region", region,
        "--model-resource-name", model.resource_name,
        "--endpoint-display-name", endpoint_display_name,
        "--machine-type", "n1-standard-4",
        "--min-replicas", "1",
        "--max-replicas", "1",
    ], check=True)


# --- Pipeline Definition ---
# This function defines the workflow and dependencies between components, including artifact passing.

@dsl.pipeline(
    name="sales-forecast-pipeline",
    description="An end-to-end pipeline for ingesting, training, and predicting sales.",
    pipeline_root=PIPELINE_ROOT
)
def forecast_pipeline():
    """Define the pipeline execution graph."""
    ingest_task = data_ingest_op() # Data ingestion step
    # Optional HPT inside pipeline
    enable_hpt = os.environ.get("ENABLE_HPT_IN_PIPELINE", "false").lower() in ("1", "true", "yes", "on")
    if enable_hpt:
        best_params = hpt_submit_op(
            project_id=PROJECT_ID,
            region=REGION,
            image_uri=DOCKER_IMAGE_URI,
            service_account=SERVICE_ACCOUNT,
            metric_name=HPT_METRIC_NAME,
            metric_goal=HPT_METRIC_GOAL,
            max_trials=HPT_MAX_TRIALS,
            parallel_trials=HPT_PARALLEL_TRIALS,
            machine_type=HPT_MACHINE_TYPE,
            param_space_json=HPT_PARAM_SPACE_JSON,
        ).after(ingest_task)
        train_task = train_op(hparams_json=best_params.outputs["best_params_json"]) # model_artifact output is implicit
        predict_task = predict_op(model_artifact=train_task.outputs["model_artifact"]).after(train_task)
    else:
        train_task = train_op().after(ingest_task) # Training step, outputs a model artifact
        predict_task = predict_op(model_artifact=train_task.outputs["model_artifact"]).after(train_task) # Prediction step, consumes the model artifact

    # Optional: Register and deploy
    # Enabled only when explicitly requested and a dedicated serving image is provided.
    enable_deploy = os.environ.get("ENABLE_DEPLOY", "false").lower() in ("1", "true", "yes", "on")
    serving_image_uri = os.environ.get("SERVING_IMAGE_URI")
    if enable_deploy and serving_image_uri and serving_image_uri != DOCKER_IMAGE_URI:
        _ = register_and_deploy_op(
            model_artifact=train_task.outputs["model_artifact"],
            project_id=PROJECT_ID,
            region=REGION,
            container_uri=serving_image_uri,
            models_gcs_dir=f"gs://{PROJECT_ID}-staging/models",
            endpoint_display_name="sales-forecast-endpoint",
        ).after(train_task)
    else:
        print("Skipping register_and_deploy_op: set ENABLE_DEPLOY=true and provide SERVING_IMAGE_URI distinct from DOCKER_IMAGE_URI to enable.")


# --- Pipeline Compilation and Execution ---
# This block allows you to compile and run the pipeline from the command line.

if __name__ == "__main__":
    # Compile the pipeline into a JSON file
    Compiler().compile(pipeline_func=forecast_pipeline, package_path="forecast_pipeline.json")

    # Initialize the Vertex AI client
    aiplatform.init(project=PROJECT_ID, location=REGION)

    # Define the pipeline job
    job = aiplatform.PipelineJob(
        display_name="sales-forecast-pipeline-run",
        template_path="forecast_pipeline.json",
        pipeline_root=PIPELINE_ROOT,
        enable_caching=True # Use cached results for unchanged steps
    )

    print("Submitting pipeline job to Vertex AI...")
    # Set sync=False to run the pipeline asynchronously.
    job.run(service_account=SERVICE_ACCOUNT, sync=False)
    print("Pipeline job submitted. Check the Vertex AI console for progress.")