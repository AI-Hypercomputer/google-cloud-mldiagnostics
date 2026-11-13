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

from collections.abc import Mapping
import hashlib
import logging
import os
import re
from typing import Any
import uuid

from google_cloud_mldiagnostics.core import create_mlrun
from google_cloud_mldiagnostics.custom_types import exceptions
from google_cloud_mldiagnostics.custom_types import mlrun_types
import requests

logger = logging.getLogger(__name__)


def _generate_self_hosted_run_identifier(seed: str) -> str:
  """Generates a deterministic RFC 4122 v4 UUID from `seed` via SHA-256."""
  digest = hashlib.sha256(seed.encode("utf-8")).digest()
  byte_arr = bytearray(digest[:16])
  byte_arr[6] = (byte_arr[6] & 0x0F) | 0x40
  byte_arr[8] = (byte_arr[8] & 0x3F) | 0x80
  return str(uuid.UUID(bytes=bytes(byte_arr)))


def normalize_gcs_path(gcs_path):
  """Normalizes a Google Cloud Storage (GCS) path.

  This function ensures that a GCS path:
  1.  Retains the "gs://" prefix if present.
  2.  Replaces multiple consecutive slashes with a single slash.
  3.  Removes any trailing slash.

  Args:
      gcs_path: The GCS path string to normalize.

  Returns:
      The normalized GCS path.
  """
  if not gcs_path:
    return gcs_path

  if gcs_path.startswith("gs://"):
    prefix = "gs://"
    path_part = gcs_path[len(prefix) :]
  else:
    prefix = ""
    path_part = gcs_path

  path_part = re.sub("/+", "/", path_part)
  path_part = path_part.rstrip("/")

  return prefix + path_part


_GKE_WORKLOAD_DETAIL_VALUE_PATTERN = re.compile(r"^[A-Za-z0-9._/-]+$")
_GKE_FILTERABLE_DETAIL_KEYS = (
    "id",
    "kind",
    "namespace",
    "cluster",
    "parent_workload",
)


def _validate_workload_details(workload_details: Any) -> None:
  """Validates user-supplied GKE workload_details at the public API boundary."""
  if not isinstance(workload_details, Mapping):
    raise exceptions.MLRunConfigurationError(
        f"Invalid workload_details: {workload_details!r}. Expected dict."
    )
  unknown_keys = (
      set(workload_details.keys())
      - mlrun_types.ALLOWED_GKE_WORKLOAD_DETAIL_KEYS
  )
  if unknown_keys:
    raise exceptions.MLRunConfigurationError(
        "Unsupported keys in workload_details:"
        f" {sorted(unknown_keys)}. Supported keys are:"
        f" {sorted(mlrun_types.ALLOWED_GKE_WORKLOAD_DETAIL_KEYS)}."
    )
  for key in _GKE_FILTERABLE_DETAIL_KEYS:
    val = workload_details.get(key)
    if val is not None:
      if not isinstance(
          val, str
      ) or not _GKE_WORKLOAD_DETAIL_VALUE_PATTERN.match(val):
        raise exceptions.MLRunConfigurationError(
            f"Invalid workload_details[{key!r}]: {val!r}. Expected a"
            " non-empty string matching '^[A-Za-z0-9._/-]+$'."
        )
  labels = workload_details.get("labels")
  if labels is not None and not isinstance(labels, Mapping):
    raise exceptions.MLRunConfigurationError(
        f"Invalid workload_details['labels']: {labels!r}. Expected dict."
    )


# Main SDK function - this is the primary interface users will import
def machinelearning_run(
    name: str,
    run_group: str = "",
    configs: dict[str, Any] | None = None,
    gcs_path: str | None = None,
    base_url: str | None = None,
    cluster_id: str | None = None,
    project: str | None = None,
    region: str | None = None,
    metrics_record_interval_sec: float = 10.0,
    on_demand_xprof: bool = False,
    log_system_metrics: bool = False,
    environment: str = "prod",
    framework: mlrun_types.Framework = mlrun_types.Framework.JAX,
    serving_engine: mlrun_types.ServingEngine = mlrun_types.ServingEngine.NONE,
    run_workload_id: str | None = None,
    metric_only_run: bool = False,
    metrics_exporter_config: dict[str, Any] | None = None,
    accelerator_orchestrator: mlrun_types.AcceleratorOrchestrator = (
        mlrun_types.AcceleratorOrchestrator.NONE
    ),
    application_framework: mlrun_types.ApplicationFramework | str = (
        mlrun_types.ApplicationFramework.NONE
    ),
    rl_orchestrator: mlrun_types.RlOrchestrator | str = (
        mlrun_types.RlOrchestrator.NONE
    ),
    workload_details: Mapping[str, Any] | None = None,
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
      base_url: Optional URL for the self-hosted Diagon Control Plane.
      cluster_id: Optional cluster identifier for self-hosted runs.
      project: The Google Cloud project ID.
      region: The Google Cloud region. If None, auto-detected from the GCP
        environment.
      metrics_record_interval_sec: The interval in seconds for recording system
        metrics backend (tpu duty cycle, tpu tensorcore utilization, hbm
        utilization, host cpu utilization, host memory utilization).
      on_demand_xprof: Whether to start an on-demand xprof profiling server. By
        default, on-demand xprof is not enabled. If enabled, the port is set to
        9999.
      log_system_metrics: Whether to log system metrics to Cloud Logging. By
        default, system metrics are not logged to Cloud Logging.
      environment: The environment to use for the control plane client
        (autopush, staging, prod). Default is prod.
      framework: The framework used for the ML run. Default is JAX.
      serving_engine: The serving engine used for the ML run. Default is NONE.
      run_workload_id: Optional shared workload identifier for CUSTOM
        Orchestrator workloads.
      metric_only_run: Whether to create a metric-only run. Default is False.
      metrics_exporter_config: Optional configuration for metrics exporter.
      accelerator_orchestrator: The orchestrator managing the ML run workload.
        Default is NONE, but auto-detected if pathways is used.
      application_framework: The application framework used for the ML run
        (e.g., MAXTEXT, AXLEARN, MAXDIFFUSION). Default is NONE.
      rl_orchestrator: The RL orchestrator used for the ML run (e.g., TUNIX).
        Default is NONE.
      workload_details: Optional mapping of GKE workload metadata when running
        without the GKE Diagon operator webhook (supported for GKE workloads
        only). Required keys: 'id', plus 'cluster' and 'namespace' when not
        auto-detected in-cluster ('kind' defaults to 'JobSet'). Optional keys:
        'parent_workload', 'creation-timestamp', 'labels'. Bare cluster names
        are supported in-cluster only; otherwise provide the full cluster
        resource path.

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
  if workload_details is not None:
    _validate_workload_details(workload_details)

  if application_framework is None:
    application_framework = mlrun_types.ApplicationFramework.NONE
  elif isinstance(application_framework, str):
    try:
      application_framework = mlrun_types.ApplicationFramework[
          application_framework.upper()
      ]
    except KeyError as exc:
      raise exceptions.MLRunConfigurationError(
          f"Invalid application_framework: '{application_framework}'. Supported"
          " values are:"
          f" {[e.value for e in mlrun_types.ApplicationFramework]}"
      ) from exc
  elif not isinstance(application_framework, mlrun_types.ApplicationFramework):
    raise exceptions.MLRunConfigurationError(
        f"Invalid application_framework: {application_framework!r}. Expected"
        " mlrun_types.ApplicationFramework or str."
    )

  if rl_orchestrator is None:
    rl_orchestrator = mlrun_types.RlOrchestrator.NONE
  elif isinstance(rl_orchestrator, str):
    try:
      rl_orchestrator = mlrun_types.RlOrchestrator[rl_orchestrator.upper()]
    except KeyError as exc:
      raise exceptions.MLRunConfigurationError(
          f"Invalid rl_orchestrator: '{rl_orchestrator}'. Supported values"
          f" are: {[e.value for e in mlrun_types.RlOrchestrator]}"
      ) from exc
  elif not isinstance(rl_orchestrator, mlrun_types.RlOrchestrator):
    raise exceptions.MLRunConfigurationError(
        f"Invalid rl_orchestrator: {rl_orchestrator!r}. Expected"
        " mlrun_types.RlOrchestrator or str."
    )

  base_url = base_url or os.environ.get("DIAGON_BASE_URL")
  agent_url = os.environ.get("DIAGON_AGENT_URL")
  self_contained = bool(base_url)

  ml_run_id = None
  namespace = None
  workload_kind = None
  workload_id = run_workload_id

  if self_contained:
    cluster_id = cluster_id or os.environ.get("DIAGON_CLUSTER_ID", "")
    if not cluster_id and agent_url:
      try:
        resp = requests.get(f"{agent_url.rstrip('/')}/v1/agent-info", timeout=5)
        resp.raise_for_status()
        cluster_id = resp.json().get("cluster_id", "")
      except requests.RequestException as e:
        logger.warning(
            "Failed to fetch cluster_id from Diagon Agent (%s); continuing.", e
        )
    slurm_job_id = (os.environ.get("SLURM_JOB_ID") or "").strip()
    if slurm_job_id:
      raw_start_time = (os.environ.get("SLURM_JOB_START_TIME") or "").strip()
      try:
        slurm_start_time = int(raw_start_time)
      except ValueError as exc:
        raise exceptions.MLRunConfigurationError(
            "SLURM_JOB_START_TIME must be a positive Unix timestamp when"
            " SLURM_JOB_ID is set."
        ) from exc
      if slurm_start_time <= 0:
        raise exceptions.MLRunConfigurationError(
            "SLURM_JOB_START_TIME must be a positive Unix timestamp when"
            " SLURM_JOB_ID is set."
        )
      workload_id = slurm_job_id
      workload_kind = mlrun_types.SLURM_WORKLOAD_KIND
      ml_run_id = _generate_self_hosted_run_identifier(
          f"{slurm_job_id}/{slurm_start_time}/{cluster_id}"
      )
    else:
      namespace = os.environ.get("POD_NAMESPACE") or "default"
      env_workload_id = (
          os.environ.get("DIAGON_WORKLOAD_ID")
          or os.environ.get("WORKLOAD_ID")
          or ""
      )
      workload_id = run_workload_id or env_workload_id or name
      if env_workload_id and workload_id != env_workload_id:
        workload_kind = ""
      else:
        workload_kind = mlrun_types.normalize_k8s_workload_kind(
            os.environ.get("DIAGON_WORKLOAD_KIND")
        )
      ml_run_id = _generate_self_hosted_run_identifier(
          f"{namespace}/{workload_id}/{cluster_id}"
      )

  gcs_path = normalize_gcs_path(gcs_path)

  extra_kwargs: dict[str, Any] = {}
  if self_contained:
    extra_kwargs["workload_kind"] = workload_kind

  return create_mlrun.initialize_mlrun(
      name=name,
      on_demand_xprof=on_demand_xprof,
      log_system_metrics=log_system_metrics,
      metric_only_run=metric_only_run,
      environment=environment,
      run_group=run_group,
      configs=configs,
      gcs_path=gcs_path,
      project=project,
      region=region,
      metrics_record_interval_sec=metrics_record_interval_sec,
      framework=framework,
      serving_engine=serving_engine,
      run_workload_id=run_workload_id,
      metrics_exporter_config=metrics_exporter_config,
      accelerator_orchestrator=accelerator_orchestrator,
      application_framework=application_framework,
      rl_orchestrator=rl_orchestrator,
      base_url=base_url,
      cluster_id=cluster_id,
      ml_run_id=ml_run_id,
      workload_id=workload_id,
      namespace=namespace,
      workload_details=workload_details,
      **extra_kwargs,
  )
