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
import matplotlib.pyplot as plt
import seaborn as sns

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


def _derive_artifact_dir_from_model_gcs_uri(model_gcs_uri: str) -> str:
    """Return the artifact directory (gs://bucket/prefix/) from a model GCS URI which may be a file.

    Examples:
      - gs://bucket/models/model.joblib -> gs://bucket/models/
      - gs://bucket/models/run-123/model.joblib -> gs://bucket/models/run-123/
      - gs://bucket/models/ -> gs://bucket/models/
    """
    if not model_gcs_uri.startswith("gs://"):
        raise ValueError("model_gcs_uri must start with 'gs://'")
    # If it ends with a slash, assume it's already a directory
    if model_gcs_uri.endswith("/"):
        return model_gcs_uri
    # Otherwise, strip the last path component (filename)
    bucket_and_path = model_gcs_uri[5:]
    if "/" not in bucket_and_path:
        # Only bucket provided; append trailing slash
        return model_gcs_uri + "/"
    bucket = bucket_and_path.split("/", 1)[0]
    path = bucket_and_path.split("/", 1)[1]
    # dirname of path
    if "/" in path:
        dir_path = path.rsplit("/", 1)[0] + "/"
    else:
        dir_path = ""
    return f"gs://{bucket}/{dir_path}"


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
    # Ensure validation columns align to training (order + any missing filled with 0)
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)

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
    # Use best iteration if available (after early stopping)
    best_iter = getattr(model, "best_iteration_", None)
    preds = model.predict(X_val, num_iteration=best_iter)
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

    # 7. Generate and persist plots (validation + training quick checks)
    try:
        os.makedirs("/tmp/plots", exist_ok=True)

        # Actual vs predicted (validation)
        val_plot_path = "/tmp/plots/actual_vs_pred_val.png"
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(val_df["Date"], y_val.values, label="actual", linewidth=2)
        ax.plot(val_df["Date"], preds, label="pred", linewidth=2)
        ax.set_title("Validation: Actual vs Predicted")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(val_plot_path, dpi=150)
        plt.close(fig)

        # Predicted vs actual scatter (validation)
        scatter_path = "/tmp/plots/pred_vs_actual_val.png"
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.scatterplot(x=y_val.values, y=preds, ax=ax)
        lims = [min(y_val.min(), preds.min()), max(y_val.max(), preds.max())]
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Validation: Predicted vs Actual")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(scatter_path, dpi=150)
        plt.close(fig)

        # Residuals histogram (validation)
        resid_path = "/tmp/plots/residuals_hist_val.png"
        residuals = preds - y_val.values
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(residuals, bins=30, kde=True, ax=ax)
        ax.set_title("Validation Residuals Histogram")
        ax.set_xlabel("Residual (pred - actual)")
        fig.tight_layout()
        fig.savefig(resid_path, dpi=150)
        plt.close(fig)

        # Feature importance
        fi_path = "/tmp/plots/feature_importance.png"
        try:
            importances = getattr(model, "feature_importances_", None)
            if importances is not None:
                fi = pd.DataFrame({
                    "feature": X_train.columns,
                    "importance": importances
                }).sort_values("importance", ascending=False).head(30)
                fig, ax = plt.subplots(figsize=(10, max(4, len(fi) * 0.3)))
                sns.barplot(data=fi, x="importance", y="feature", ax=ax, orient="h")
                ax.set_title("Feature Importance (top 30)")
                fig.tight_layout()
                fig.savefig(fi_path, dpi=150)
                plt.close(fig)
        except Exception as e:
            logging.warning(f"Skipping feature importance plot: {e}")

        # Optional: training overlay (quick check)
        try:
        y_train_pred = model.predict(X_train, num_iteration=best_iter)
            train_plot_path = "/tmp/plots/actual_vs_pred_train.png"
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(train_df["Date"], y_train.values, label="actual", linewidth=1)
            ax.plot(train_df["Date"], y_train_pred, label="pred", linewidth=1)
            ax.set_title("Train: Actual vs Predicted")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(train_plot_path, dpi=150)
            plt.close(fig)
        except Exception as e:
            logging.warning(f"Skipping training overlay plot: {e}")

        # Upload plots to GCS alongside the model, if a target directory is known
        target_gcs_dir = None
        if model_gcs_uri:
            target_gcs_dir = _derive_artifact_dir_from_model_gcs_uri(model_gcs_uri)
        elif os.environ.get("MODEL_GCS_URI"):
            target_gcs_dir = _derive_artifact_dir_from_model_gcs_uri(os.environ["MODEL_GCS_URI"])

        if target_gcs_dir:
            for local_path in [val_plot_path, scatter_path, resid_path, fi_path]:
                if os.path.exists(local_path):
                    gcs_uri = target_gcs_dir.rstrip("/") + "/plots/" + os.path.basename(local_path)
                    upload_file_to_gcs(local_path, gcs_uri)
                    logging.info(f"Uploaded plot to {gcs_uri}")
    except Exception as e:
        logging.warning(f"Plot generation/upload skipped due to error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sales forecasting model.")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to save the trained model artifact.")
    parser.add_argument("--model-gcs-uri", type=str, required=False,
                        help="Optional GCS URI (gs://bucket/path) to upload the trained model.")
    args = parser.parse_args()
    main(args.model_path, getattr(args, "model_gcs_uri", None)) # Pass the parsed arguments to main