# Copyright 2025 Google LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#      https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module for creating and managing ML runs."""

from typing import Any

from google_cloud_mldiagnostics.core import create_mlrun
from google_cloud_mldiagnostics.custom_types import exceptions
from google_cloud_mldiagnostics.custom_types import mlrun_types


# List of supported regions for MLRun - New supported regions can be added here
# to expand the region list as required in future.
SUPPORTED_REGIONS = ["us-central1", "us-east5", "europe-west4"]


# Main SDK function - this is the primary interface users will import
def machinelearning_run(
    name: str,
    run_group: str = "",
    configs: dict[str, Any] | None = None,
    gcs_path: str | None = None,
    project: str | None = None,
    region: str | None = "us-central1",
    metrics_record_interval_sec: float = 10.0,
    on_demand_xprof: bool = False,
    environment: str = "prod",
) -> mlrun_types.MLRun:
  """Create a new machine learning run.

  This is the main entry point for the SDK that users will call to create ML
  runs.

  Args:
      name: The name of the run. For GKE workloads, a GKE timestamp will be
        added as a suffix to this name and is used as the display name.
      run_group: The run set this run belongs to.
      configs: dict of configuration parameters.
      gcs_path: GCS path for storing run artifacts.
      project: The Google Cloud project ID.
      region: The Google Cloud region. Default is us-central1.
      metrics_record_interval_sec: The interval in seconds for recording system
        metrics backend (tpu duty cycle, tpu tensorcore utilization, hbm
        utilization, host cpu utilization, host memory utilization).
      on_demand_xprof: Whether to start an on-demand xprof profiling server. By
        default, on-demand xprof is not enabled. If enabled, the port is set to
        9999.
      environment: The environment to use for the control plane client
        (autopush, staging, prod). Default is prod.

  Returns:
      MLRun: A new ML run instance

  Example:
      my_run = machinelearning_run(
            name="experiment_1",
            run_group="training_set_v1",
            configs={"epochs": 100, "batch_size": 32},
            gcs_path="gs://my-bucket/experiments"
        )

      # Update configs using dict methods
      my_run.configs.update({"epochs": 300, "optimizer": "adam"})

      # Update configs using attribute notation
      my_run.configs.batch_size = 64
  """
  if not name:
    raise exceptions.MLRunConfigurationError(
        "name is required and must be provided. The name parameter"
        " helps identify and organize your ML workloads. Please provide a"
        " meaningful name (e.g., 'training_v1',"
        " 'experiment_batch_1')."
    )
  if environment not in ["autopush", "staging", "prod"]:
    raise exceptions.MLRunConfigurationError(
        "environment must be one of 'autopush', 'staging', or 'prod'."
    )

  if region not in SUPPORTED_REGIONS:
    raise exceptions.MLRunConfigurationError(
        f"region must be one of {SUPPORTED_REGIONS} for now."
    )

  return create_mlrun.initialize_mlrun(
      name=name,
      on_demand_xprof=on_demand_xprof,
      environment=environment,
      run_group=run_group,
      configs=configs,
      gcs_path=gcs_path,
      project=project,
      region=region,
      metrics_record_interval_sec=metrics_record_interval_sec,
  )
