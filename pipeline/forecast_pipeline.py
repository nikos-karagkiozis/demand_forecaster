# pipeline/forecast_pipeline.py
"""
Defines the Kubeflow pipeline for the sales forecasting project using KFP v2.

This pipeline orchestrates the data ingestion, model training, and prediction steps,
with each step running in its own containerized environment.
"""

from kfp import dsl
from kfp.compiler import Compiler
from google.cloud import aiplatform

# --- Pipeline Configuration ---
# Replace with your specific details
PROJECT_ID = "my-forecast-project-18870"
REGION = "us-central1"
PIPELINE_ROOT = f"gs://{PROJECT_ID}-staging/pipeline_root"

# This is the URI of the Docker image you will build and push to Artifact Registry
DOCKER_IMAGE_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/sales-forecast-repo/sales-forecast:latest"

# The service account for running the pipeline components
SERVICE_ACCOUNT = f"ds-pipeline-sa@{PROJECT_ID}.iam.gserviceaccount.com"

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