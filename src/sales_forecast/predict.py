# predict.py
"""
This script handles making predictions with the trained sales forecasting model.

It performs the following steps:
1. Loads the pre-trained LightGBM model.
2. Fetches the most recent data from BigQuery required for feature engineering.
3. Creates a placeholder for the future date to be predicted.
4. Applies the same feature engineering logic used in training.
5. Makes a prediction for the future date.
6. Prints the prediction.
"""

import os
import tempfile
import pandas as pd
import joblib
import argparse # Added import for argparse
from google.cloud import bigquery
from google.cloud import storage
import logging
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import from our project modules
from sales_forecast.config import config
from sales_forecast.features import generate_features, prepare_dataset_for_modeling


def download_file_from_gcs(gcs_uri: str, local_path: str) -> None:
    """Downloads a file from GCS to a local path."""
    if not gcs_uri.startswith("gs://"):
        raise ValueError("model_gcs_uri must start with 'gs://'")
    parts = gcs_uri[5:].split("/", 1)
    bucket_name = parts[0]
    blob_name = parts[1]
    client = storage.Client(project=config.bq.PROJECT_ID)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)
    logging.info(f"Downloaded model from {gcs_uri} to {local_path}")


def upload_file_to_gcs(local_path: str, gcs_uri: str) -> None:
    if not gcs_uri.startswith("gs://"):
        raise ValueError("gcs_uri must start with 'gs://'")
    bucket_name, blob_name = gcs_uri[5:].split("/", 1)
    client = storage.Client(project=config.bq.PROJECT_ID)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    logging.info(f"Uploaded file to {gcs_uri}")


def _derive_artifact_dir_from_model_gcs_uri(model_gcs_uri: str) -> str:
    if not model_gcs_uri.startswith("gs://"):
        raise ValueError("model_gcs_uri must start with 'gs://'")
    if model_gcs_uri.endswith("/"):
        return model_gcs_uri
    bucket_and_path = model_gcs_uri[5:]
    if "/" not in bucket_and_path:
        return model_gcs_uri + "/"
    bucket = bucket_and_path.split("/", 1)[0]
    path = bucket_and_path.split("/", 1)[1]
    if "/" in path:
        dir_path = path.rsplit("/", 1)[0] + "/"
    else:
        dir_path = ""
    return f"gs://{bucket}/{dir_path}"


def load_model(path: str):
    """Loads a pre-trained model from a joblib file."""
    logging.info(f"Loading model from {path}...")
    model = joblib.load(path)
    logging.info("Model loaded successfully.")
    return model


def get_inference_data(client: bigquery.Client, table_ref: str, days: int = 30) -> pd.DataFrame:
    """
    Fetches the last N days of data from BigQuery to use for inference.
    This data is needed to calculate lag and rolling features.

    Args:
        client (bigquery.Client): An authenticated BigQuery client.
        table_ref (str): The full ID of the table to query.
        days (int): The number of recent days of data to fetch.

    Returns:
        pd.DataFrame: The recent data from the BigQuery table.
    """
    logging.info(f"Fetching last {days} days of data for inference from {table_ref}...")
    query = f"""
        SELECT * FROM `{table_ref}`
        ORDER BY Date DESC
        LIMIT {days}
    """
    try:
        df = client.query(query).to_dataframe()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        logging.info(f"Successfully loaded {df.shape[0]} rows for inference.")
        return df
    except Exception as e:
        logging.error(f"Failed to load inference data from BigQuery: {e}")
        raise


def main(model_input_path: str | None = None, model_gcs_uri: str | None = None): # Modified to support GCS
    """Main function to orchestrate the model prediction pipeline."""
    logging.info("Starting prediction process...")

    # 1. Resolve and load Model
    if model_gcs_uri:
        with tempfile.TemporaryDirectory() as tmpdir:
            local_model = os.path.join(tmpdir, "model.joblib")
            download_file_from_gcs(model_gcs_uri, local_model)
            model = load_model(local_model)
    elif model_input_path:
        model = load_model(model_input_path)
    else:
        raise ValueError("Provide either --model-path or --model-gcs-uri")

    # 2. Get recent data for feature calculation
    client = bigquery.Client(project=config.bq.PROJECT_ID)
    table_ref = f"{config.bq.PROJECT_ID}.{config.bq.DATASET}.{config.bq.FINAL_TABLE}"
    inference_df = get_inference_data(client, table_ref, days=30)

    # 3. Create a placeholder for the next day's prediction
    last_date = inference_df['Date'].max()
    prediction_date = last_date + timedelta(days=1)
    future_df = pd.DataFrame([{'Date': prediction_date}])
    combined_df = pd.concat([inference_df, future_df], ignore_index=True)

    # 4. Generate features for the combined data
    featured_df = generate_features(combined_df)

    # 5. Isolate the feature set for the prediction date
    X_pred, _ = prepare_dataset_for_modeling(featured_df.tail(1), config.features.TARGET_COL)

    # 6. Align features to the model's expected schema (handle missing one-hot cols, order)
    feature_names = getattr(model, "feature_name_", None)
    if feature_names:
        for c in feature_names:
            if c not in X_pred.columns:
                X_pred[c] = 0
        X_pred = X_pred[feature_names]

    # 7. Make Prediction
    prediction = model.predict(X_pred)
    logging.info(f"Predicted sales for {prediction_date.date()}: {prediction[0]:.2f}")

    # 8. Plot last window actuals and the forecast point, upload alongside the model if possible
    try:
        os.makedirs("/tmp/plots", exist_ok=True)
        # Exclude the future row (last one after feature gen)
        hist_df = featured_df.iloc[:-1].copy()
        forecast_plot = "/tmp/plots/forecast_overlay.png"
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(hist_df["Date"], hist_df[config.features.TARGET_COL], label="actual (last window)", linewidth=2)
        ax.scatter([prediction_date], [float(prediction[0])], color="red", label="forecast (next day)")
        ax.set_title("Forecast Overlay: last window actuals and next-day forecast")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(forecast_plot, dpi=150)
        plt.close(fig)

        target_gcs_dir = None
        if model_gcs_uri:
            target_gcs_dir = _derive_artifact_dir_from_model_gcs_uri(model_gcs_uri)
        elif os.environ.get("MODEL_GCS_URI"):
            target_gcs_dir = _derive_artifact_dir_from_model_gcs_uri(os.environ["MODEL_GCS_URI"])

        if target_gcs_dir:
            gcs_uri = target_gcs_dir.rstrip("/") + "/plots/" + os.path.basename(forecast_plot)
            upload_file_to_gcs(forecast_plot, gcs_uri)
            logging.info(f"Uploaded forecast plot to {gcs_uri}")
    except Exception as e:
        logging.warning(f"Plot generation/upload skipped due to error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make sales forecast prediction.")
    parser.add_argument("--model-path", type=str, required=False,
                        help="Path to load the trained model from.")
    parser.add_argument("--model-gcs-uri", type=str, required=False,
                        help="GCS URI (gs://bucket/path) to download the trained model from.")
    args = parser.parse_args()
    main(getattr(args, "model_path", None), getattr(args, "model_gcs_uri", None)) # Pass parsed arguments