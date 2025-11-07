import pandas as pd
from datetime import datetime, timedelta

from sales_forecast.features import generate_features, prepare_dataset_for_modeling


def test_generate_features_and_prepare_dataset_for_modeling():
    # Build a minimal synthetic dataset with Date and target 'total'
    start = datetime(2020, 1, 1)
    rows = 60
    df = pd.DataFrame({
        "Date": [start + timedelta(days=i) for i in range(rows)],
        "total": [i % 100 for i in range(rows)],
        # Optional known columns used by feature code may be absent; code should handle gracefully
    })

    featured = generate_features(df)
    # After lag/rolling and dropna, we should still have rows
    assert len(featured) > 0, "Feature generation produced no rows"
    # A known lag column should exist
    assert any(c.startswith("total_lag_") for c in featured.columns), "Expected lag features missing"

    X, y = prepare_dataset_for_modeling(featured, "total")
    assert len(X) == len(y) > 0, "Prepared dataset is empty or misaligned"


