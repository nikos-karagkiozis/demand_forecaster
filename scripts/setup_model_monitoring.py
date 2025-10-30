#!/usr/bin/env python
"""
Create a Vertex AI Model Deployment Monitoring Job for a deployed endpoint (tabular).

This monitors feature distribution drift and prediction drift on a schedule.

Usage:
  python scripts/setup_model_monitoring.py \
    --project-id my-forecast-project-18870 \
    --region us-central1 \
    --endpoint-id 1234567890123456789 \
    --display-name sales-forecast-monitor \
    --sampling-rate 0.5 \
    --alert-email your@email.com
"""

from __future__ import annotations

import argparse
from typing import Optional

from google.cloud import aiplatform


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup Vertex Model Deployment Monitoring Job")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--endpoint-id", required=True)
    parser.add_argument("--display-name", default="sales-forecast-monitor")
    parser.add_argument("--sampling-rate", type=float, default=0.5, help="Traffic sampling rate [0,1]")
    parser.add_argument("--alert-email", required=False)
    args = parser.parse_args()

    aiplatform.init(project=args.project_id, location=args.region)

    endpoint_name = args.endpoint_id
    if not endpoint_name.startswith("projects/"):
        endpoint_name = f"projects/{args.project_id}/locations/{args.region}/endpoints/{args.endpoint_id}"

    # Minimal monitoring config (tabular drift)
    schedule_config = aiplatform.gapic.ModelDeploymentMonitoringScheduleConfig(
        monitor_interval=3600  # seconds; run hourly
    )

    objective_config = aiplatform.gapic.ModelDeploymentMonitoringObjectiveConfig(
        # Use default drift configs; for production, specify per-feature thresholds
        training_dataset=aiplatform.gapic.ModelDeploymentMonitoringObjectiveConfig.TrainingDataset(
            # Optionally reference a static dataset for baselines
        )
    )

    alert_config = None
    if args.alert_email:
        alert_config = aiplatform.gapic.ModelMonitoringAlertConfig(
            email_alert_config=aiplatform.gapic.ModelMonitoringAlertConfig.EmailAlertConfig(
                user_emails=[args.alert_email]
            )
        )

    logging_sampling_strategy = aiplatform.gapic.SamplingStrategy(
        random_sample_config=aiplatform.gapic.SamplingStrategy.RandomSampleConfig(
            sample_rate=args.sampling_rate
        )
    )

    job = aiplatform.ModelDeploymentMonitoringJob.create(
        display_name=args.display_name,
        endpoint=endpoint_name,
        schedule_config=schedule_config,
        alert_config=alert_config,
        logging_sampling_strategy=logging_sampling_strategy,
        model_deployment_monitoring_objective_configs=[objective_config],
        sync=True,
    )

    print("Created monitoring job:", job.resource_name)


if __name__ == "__main__":
    main()


