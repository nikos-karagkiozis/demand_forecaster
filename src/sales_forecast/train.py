# train.py
"""
This script handles the training of the sales forecasting model.

It performs the following steps:
1. Loads data from BigQuery using settings from the config file.
2. Applies feature engineering functions from the features module.
3. Splits the data into training and validation sets based on time.
4. Trains a LightGBM regressor model.
5. Evaluates the model's performance on the validation set.
6. Saves the trained model to a file for later use.
"""

import os
import pandas as pd
import lightgbm as lgb
import joblib
import argparse # Added import for argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from google.cloud import bigquery
from google.cloud import storage
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import from our project modules
from sales_forecast.config import config
from sales_forecast.features import generate_features, prepare_dataset_for_modeling


def load_data_from_bq(client: bigquery.Client, table_ref: str) -> pd.DataFrame:
    """
    Loads the feature data from a specified BigQuery table.

    Args:
        client (bigquery.Client): An authenticated BigQuery client.
        table_ref (str): The full ID of the table to query (e.g., "project.dataset.table").

    Returns:
        pd.DataFrame: The data from the BigQuery table.
    """
    logging.info(f"Loading data from BigQuery table: {table_ref}")
    query = f"SELECT * FROM `{table_ref}` ORDER BY Date"
    try:
        df = client.query(query).to_dataframe()
        df['Date'] = pd.to_datetime(df['Date'])
        logging.info(f"Successfully loaded {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except Exception as e:
        logging.error(f"Failed to load data from BigQuery: {e}")
        raise


def split_data_by_time(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a time-series DataFrame into training and validation sets.

    Args:
        df (pd.DataFrame): The input DataFrame, sorted by date.
        test_size (float): The proportion of the dataset to allocate to the validation set.

    Returns:
        A tuple containing the training DataFrame and the validation DataFrame.
    """
    split_index = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_index]
    val_df = df.iloc[split_index:]
    logging.info(f"Data split into training ({len(train_df)} rows) and validation ({len(val_df)} rows).")
    return train_df, val_df


def upload_file_to_gcs(local_path: str, gcs_uri: str) -> None:
    """Uploads a local file to the specified GCS URI (gs://bucket/path)."""
    if not gcs_uri.startswith("gs://"):
        raise ValueError("model_gcs_uri must start with 'gs://'")
    parts = gcs_uri[5:].split("/", 1)
    bucket_name = parts[0]
    blob_name = parts[1] if len(parts) > 1 else os.path.basename(local_path)
    client = storage.Client(project=config.bq.PROJECT_ID)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    logging.info(f"Uploaded model to {gcs_uri}")


def main(model_output_path: str, model_gcs_uri: str | None = None): # Modified to accept optional GCS uri
    """Main function to orchestrate the model training pipeline."""
    logging.info("Starting model training process...")

    # 1. Load Data
    client = bigquery.Client(project=config.bq.PROJECT_ID)
    table_ref = f"{config.bq.PROJECT_ID}.{config.bq.DATASET}.{config.bq.FINAL_TABLE}"
    raw_df = load_data_from_bq(client, table_ref)

    # 2. Feature Engineering
    logging.info("Generating features...")
    featured_df = generate_features(raw_df)

    # 3. Data Splitting
    train_df, val_df = split_data_by_time(featured_df, test_size=0.2)

    X_train, y_train = prepare_dataset_for_modeling(train_df, config.features.TARGET_COL)
    X_val, y_val = prepare_dataset_for_modeling(val_df, config.features.TARGET_COL)

    # 4. Model Training
    logging.info("Training LightGBM model...")
    lgbm_params = {
        'objective': 'regression_l1',
        'metric': 'mae',
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
        'boosting_type': 'gbdt',
    }

    model = lgb.LGBMRegressor(**lgbm_params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='mae',
              callbacks=[lgb.early_stopping(100, verbose=True)])

    # 5. Model Evaluation
    logging.info("Evaluating model performance...")
    preds = model.predict(X_val)
    rmse = root_mean_squared_error(y_val, preds)
    mae = mean_absolute_error(y_val, preds)
    r2 = r2_score(y_val, preds)

    logging.info(f"Validation RMSE: {rmse:.4f}")
    logging.info(f"Validation MAE: {mae:.4f}")
    logging.info(f"Validation R^2: {r2:.4f}")

    # 6. Save Model
    # KFP handles the creation of the directory for output artifacts
    joblib.dump(model, model_output_path) # Use the provided path
    logging.info(f"Model saved to {model_output_path}")
    if model_gcs_uri:
        upload_file_to_gcs(model_output_path, model_gcs_uri)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sales forecasting model.")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to save the trained model artifact.")
    parser.add_argument("--model-gcs-uri", type=str, required=False,
                        help="Optional GCS URI (gs://bucket/path) to upload the trained model.")
    args = parser.parse_args()
    main(args.model_path, getattr(args, "model_gcs_uri", None)) # Pass the parsed arguments to main