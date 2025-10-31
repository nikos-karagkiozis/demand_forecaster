# data_ingest.py
"""
Safe ingestion: load CSV into staging table, then create final table.
Run in Cloud Shell or Vertex AI Workbench where Application Default Credentials are available.
Requires: google-cloud-bigquery
pip install google-cloud-bigquery
"""

import os
from google.cloud import bigquery
from google.api_core.exceptions import NotFound

PROJECT = os.environ.get("PROJECT_ID", "my-forecast-project-18870")
DATASET = os.environ.get("DATASET", "gk2_takeaway_sales")
STAGING_TABLE = os.environ.get("STAGING_TABLE", "staging_daily_sales")
FINAL_TABLE = os.environ.get("FINAL_TABLE", "daily_sales_features")
GCS_URI = os.environ.get("GCS_URI", f"gs://{PROJECT}-staging/input/{FINAL_TABLE}.csv")
LOCATION = os.environ.get("LOCATION", "US")

client = bigquery.Client(project=PROJECT, location=LOCATION)

dataset_ref = f"{PROJECT}.{DATASET}"
staging_table_ref = f"{PROJECT}.{DATASET}.{STAGING_TABLE}"
final_table_ref = f"{PROJECT}.{DATASET}.{FINAL_TABLE}"

# 1) Ensure dataset exists
try:
    client.get_dataset(f"{PROJECT}.{DATASET}")
    print(f"Dataset {PROJECT}.{DATASET} exists.")
except NotFound:
    print(f"Creating dataset {PROJECT}.{DATASET} in location {LOCATION}.")
    dataset = bigquery.Dataset(f"{PROJECT}.{DATASET}")
    dataset.location = LOCATION
    client.create_dataset(dataset)
    print("Dataset created.")

# 2) Define explicit schema for staging load (adjust types if your CSV differs)
schema = [
    bigquery.SchemaField("date", "STRING"),
    bigquery.SchemaField("Total", "FLOAT"),
    bigquery.SchemaField("tavg", "FLOAT"),
    bigquery.SchemaField("tmin", "FLOAT"),
    bigquery.SchemaField("tmax", "FLOAT"),
    bigquery.SchemaField("prcp", "FLOAT"),
    bigquery.SchemaField("wspd", "FLOAT"),
    bigquery.SchemaField("pres", "FLOAT"),
    bigquery.SchemaField("is_holiday", "BOOL"),
    bigquery.SchemaField("holiday_name", "STRING"),
    bigquery.SchemaField("is_school_break", "BOOL"),
    bigquery.SchemaField("dinner_impact_score", "FLOAT"),
    bigquery.SchemaField("importance_score", "FLOAT"),
    bigquery.SchemaField("is_evening", "INT64"),
    bigquery.SchemaField("is_weekend", "INT64"),
    bigquery.SchemaField("is_home", "INT64"),
    bigquery.SchemaField("is_delhi", "INT64"),
    bigquery.SchemaField("is_t20i", "INT64"),
    bigquery.SchemaField("is_cricket_day", "BOOL"),
]

load_config = bigquery.LoadJobConfig(
    schema=schema,
    source_format=bigquery.SourceFormat.CSV,
    skip_leading_rows=1,
    field_delimiter=",",
    write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    autodetect=False,
)

print(f"Starting load job from {GCS_URI} to staging table {staging_table_ref} ...")
load_job = client.load_table_from_uri(GCS_URI, staging_table_ref, job_config=load_config, location=LOCATION)
load_job.result()
print("Staging load finished.")

# 3) Create final table from staging
create_sql = f"""
CREATE OR REPLACE TABLE `{final_table_ref}`
AS SELECT
  PARSE_DATE('%Y-%m-%d', SAFE_CAST(date AS STRING)) AS Date,
  CAST(Total AS INT64) AS total,
  tavg,
  tmin,
  tmax,
  prcp,
  wspd,
  pres,
  CAST(is_holiday AS INT64) AS is_holiday,
  CAST(is_school_break AS INT64) AS is_school_break,
  dinner_impact_score,
  importance_score,
  is_evening,
  is_weekend,
  is_home,
  is_delhi,
  is_t20i,
  CAST(is_cricket_day AS INT64) AS is_cricket_day
FROM `{staging_table_ref}`
WHERE SAFE_CAST(date AS STRING) IS NOT NULL
ORDER BY Date;
"""

print("Creating final table from staging...")
query_job = client.query(create_sql, location=LOCATION)
query_job.result()
print(f"Final table {final_table_ref} created.")

# 4) Optional: drop staging table to save costs (uncomment if you want)
# client.delete_table(staging_table_ref, not_found_ok=True)
# print(f"Staging table {staging_table_ref} deleted.")
