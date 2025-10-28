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

import pandas as pd
import joblib
import argparse # Added import for argparse
from google.cloud import bigquery
import logging
from datetime import timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import from our project modules
from sales_forecast.config import config
from sales_forecast.features import generate_features, prepare_dataset_for_modeling


def load_model(path: str):
    """Loads a pre-trained model from a joblib file."""
    logging.info(f"Loading model from {path}...")
    try:
        model = joblib.load(path)
        logging.info("Model loaded successfully.")
        return model
    except FileNotFoundError:
        logging.error(f"Model file not found at {path}. Please run the training script first.")
        raise


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


def main(model_input_path: str): # Modified main to accept model_input_path
    """Main function to orchestrate the model prediction pipeline."""
    logging.info("Starting prediction process...")

    # 1. Load Model
    model = load_model(model_input_path) # Use the provided path

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
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to load the trained model from.")
    args = parser.parse_args()
    main(args.model_path) # Pass the parsed argument to main