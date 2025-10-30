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

    # 6. Make Prediction
    prediction = model.predict(X_pred)
    logging.info(f"Predicted sales for {prediction_date.date()}: {prediction[0]:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make sales forecast prediction.")
    parser.add_argument("--model-path", type=str, required=False,
                        help="Path to load the trained model from.")
    parser.add_argument("--model-gcs-uri", type=str, required=False,
                        help="GCS URI (gs://bucket/path) to download the trained model from.")
    args = parser.parse_args()
    main(getattr(args, "model_path", None), getattr(args, "model_gcs_uri", None)) # Pass parsed arguments