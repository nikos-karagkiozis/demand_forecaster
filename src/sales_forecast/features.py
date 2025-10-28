# features.py
"""
This module contains functions for feature engineering based on the daily sales data.
It generates temporal, lagged, and rolling window features to prepare the data
for a sales forecasting model.
"""

import pandas as pd
import numpy as np

def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts temporal features from the 'Date' column.

    Args:
        df (pd.DataFrame): Input DataFrame with a 'Date' column (datetime type).

    Returns:
        pd.DataFrame: DataFrame with added temporal features.
    """
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['day_of_week'] = df['Date'].dt.dayofweek  # Monday=0, Sunday=6
    df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)
    df['day_of_year'] = df['Date'].dt.dayofyear
    df['quarter'] = df['Date'].dt.quarter
    return df

def create_lagged_features(df: pd.DataFrame, target_col: str = 'total', lags: list = None) -> pd.DataFrame:
    """
    Creates lagged features for the target variable.

    Args:
        df (pd.DataFrame): Input DataFrame, sorted by 'Date'.
        target_col (str): Name of the target column to create lags for.
        lags (list): List of integers representing the number of periods to lag.
                     Defaults to [1, 7, 14] (yesterday, last week, two weeks ago).

    Returns:
        pd.DataFrame: DataFrame with added lagged features.
    """
    if lags is None:
        lags = [1, 2, 3, 7, 14]

    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    return df

def create_rolling_features(df: pd.DataFrame, target_col: str = 'total', windows: list = None) -> pd.DataFrame:
    """
    Creates rolling window features (mean and std) for the target variable.

    Args:
        df (pd.DataFrame): Input DataFrame, sorted by 'Date'.
        target_col (str): Name of the target column to create rolling features for.
        windows (list): List of integers representing the window sizes.
                        Defaults to [3, 7] (3-day and 7-day rolling).

    Returns:
        pd.DataFrame: DataFrame with added rolling features.
    """
    if windows is None:
        windows = [2, 3, 4, 5, 6, 7]

    for window in windows:
        df[f'{target_col}_rolling_mean_{window}'] = df[target_col].shift(1).rolling(window=window).mean()
        df[f'{target_col}_rolling_std_{window}'] = df[target_col].shift(1).rolling(window=window).std()
    return df

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical features using one-hot encoding.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with one-hot encoded categorical features.
    """
    # 'holiday_name' is the only explicit string categorical feature from data_ingest.py
    if 'holiday_name' in df.columns:
        df = pd.get_dummies(df, columns=['holiday_name'], prefix='holiday', drop_first=True)
    return df

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrates the creation of all features.

    Args:
        df (pd.DataFrame): Raw input DataFrame. Assumes 'Date' is already datetime.

    Returns:
        pd.DataFrame: DataFrame with all engineered features.
    """
    df = df.sort_values('Date').reset_index(drop=True)

    df = create_temporal_features(df)
    df = create_lagged_features(df)
    df = create_rolling_features(df)
    df = encode_categorical_features(df)

    # Drop rows with NaN values introduced by lagging/rolling (at the beginning of the series)
    df.dropna(inplace=True)

    return df

def prepare_dataset_for_modeling(df: pd.DataFrame, target_col: str = 'total') -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepares the dataset for modeling by separating features (X) and target (y).

    Args:
        df (pd.DataFrame): DataFrame with engineered features.
        target_col (str): Name of the target column.

    Returns:
        tuple[pd.DataFrame, pd.Series]: X (features) and y (target) DataFrames.
    """
    # Exclude 'Date' and the original target column from features
    features = df.drop(columns=['Date', target_col], errors='ignore')
    target = df[target_col]

    return features, target
