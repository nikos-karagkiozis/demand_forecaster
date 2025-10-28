# config.py
"""
Pydantic-based configuration management for the sales forecasting project.

This module centralizes all configuration parameters, allowing them to be
loaded from environment variables, which is ideal for cloud environments
like Vertex AI and Kubeflow.
"""

from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BigQueryConfig(BaseSettings):
    """BigQuery and GCS configuration settings."""

    # To allow loading from a .env file, we specify the file here.
    # In a cloud environment, these would be set as environment variables directly.
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    PROJECT_ID: str = "my-forecast-project-18870"
    DATASET: str = "gk2_takeaway_sales"
    STAGING_TABLE: str = "staging_daily_sales"
    FINAL_TABLE: str = "daily_sales_features"
    LOCATION: str = "US"
    GCS_URI: str = f"gs://{PROJECT_ID}-staging/input/{FINAL_TABLE}.csv"


class FeatureConfig(BaseSettings):
    """Feature engineering configuration settings."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    TARGET_COL: str = "total"
    LAGS: List[int] = Field(default_factory=lambda: [1, 2, 3, 7, 14])
    WINDOWS: List[int] = Field(default_factory=lambda: [2, 3, 4, 5, 6, 7])


class AppConfig(BaseSettings):
    """Main application configuration that aggregates all other configs."""

    bq: BigQueryConfig = BigQueryConfig()
    features: FeatureConfig = FeatureConfig()


# Instantiate the main config object so it can be imported and used across the project.
config = AppConfig()

# Example of how to use it in other files:
# from sales_forecast.config import config
#
# project_id = config.bq.PROJECT_ID
# target_col = config.features.TARGET_COL
#
# print(f"Project ID from config: {project_id}")
# print(f"Target column from config: {target_col}")