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
def train_op(model_artifact: dsl.OutputPath(dsl.Model)):
    """Pipeline component to run the model training script and output the trained model."""
    import subprocess
    # KFP automatically provides a path for output artifacts
    subprocess.run(["python", "-m", "sales_forecast.train", "--model-path", model_artifact], check=True)


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

    # Optionally submit the pipeline, gated by env var
    submit_pipeline = os.environ.get("SUBMIT_PIPELINE", "false").lower() in ("1", "true", "yes", "on")
    if submit_pipeline:
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
    else:
        print("Skipping pipeline submission (set SUBMIT_PIPELINE=true to submit).")