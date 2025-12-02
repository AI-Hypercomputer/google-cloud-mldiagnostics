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

"""Module for registering and managing ML runs."""

import datetime
import logging
import threading
from typing import Any

from google_cloud_mldiagnostics.core import global_manager
from google_cloud_mldiagnostics.core import metrics
from google_cloud_mldiagnostics.core import xprof
from google_cloud_mldiagnostics.custom_types import metric_types
from google_cloud_mldiagnostics.custom_types import mlrun_types
from google_cloud_mldiagnostics.utils import config_utils
from google_cloud_mldiagnostics.utils import gcp
from google_cloud_mldiagnostics.utils import host_utils
from google_cloud_mldiagnostics.utils import metric_utils
from google_cloud_mldiagnostics.utils import orchestrator_utils
from google_cloud_mldiagnostics.utils import run_phase_utils

_METRICS_RECORDER_THREAD_LOCK = threading.Lock()
_METRICS_RECORDER_THREAD_STARTED = False


def initialize_mlrun(
    name: str,
    environment: str,
    on_demand_xprof: bool,
    run_group: str | None = None,
    configs: dict[str, Any] | None = None,
    gcs_path: str | None = None,
    project: str | None = None,
    region: str | None = None,
    metrics_record_interval_sec: float = 10.0,
) -> mlrun_types.MLRun:
  """Initializes a new ML run.

  Args:
      name: The name of the run.
      environment: The environment to use for the control plane client
        (autopush, staging, prod).
      on_demand_xprof: Whether to start an on-demand xprof profiling server. 
        If enabled, the port is set to 9999.
      run_group: The run set this run belongs to.
      configs: Dictionary of configuration parameters.
      gcs_path: GCS path for storing run artifacts.
      project: The Google Cloud project ID.
      region: The Google Cloud region.
      metrics_record_interval_sec: The metrics record interval in seconds.

  Returns:
      The initialized ML run object.
  """
  # Combine default configs with user configs.
  software_configs = config_utils.get_software_config()
  hardware_configs = config_utils.get_hardware_config()
  user_configs = configs if configs else {}
  configs = mlrun_types.ConfigDict({
      "softwareConfigs": software_configs,
      "hardwareConfigs": hardware_configs,
      "userConfigs": config_utils.sanitize_config(user_configs),
  })

  if region is None:
    region = gcp.get_instance_region()
  if project is None:
    project = gcp.get_project_id()

  # TODO([INTERNAL]): Add support for checking the repetitive registered ML
  # name in Spanner after control plane client ready.
  # Otherwise, generate new UUID.

  created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
  run_phase = mlrun_types.RunPhase.PHASE_ACTIVE
  workload_details = host_utils.get_workload_details()
  orchestrator = orchestrator_utils.detect_orchestrator()

  # Generate display name and name for the MLRun.
  # TODO: [INTERNAL] - Add support for non-GKE workloads.
  if orchestrator == "GKE":
    display_name = name + "-" + workload_details["creation-timestamp"]
    name = host_utils.get_identifier(workload_details)
  else:
    display_name = name

  # sanitize the name and use it as the MLRun name for the control plane.
  sanitized_name = host_utils.sanitize_identifier(name)

  ml_run = mlrun_types.MLRun(
      run_group=run_group,
      name=sanitized_name,
      configs=configs,
      gcs_path=gcs_path,
      location=region,
      project=project,
      run_phase=run_phase,
      created_at=created_at,
      workload_details=workload_details,
      orchestrator=orchestrator,
      display_name=display_name,
      on_demand_xprof=on_demand_xprof,
      environment=environment,
  )

  # register the run to global manager.
  manager = global_manager.get_global_run_manager()
  manager.initialize(ml_run)

  diagon_url = create_diagnostics_url(region, project, sanitized_name)
  xprof_url = create_xprof_url(diagon_url)
  logging.info(
      "Diagon SDK is in experimental mode, to visualize the run, please go to\n"
      "%s\n",
      diagon_url,
  )
  logging.info(
      "Xprof profiling url:\n%s\n",
      xprof_url,
  )

  run_phase_monitor = run_phase_utils.RunPhaseMonitor()
  run_phase_monitor.start()

  global _METRICS_RECORDER_THREAD_STARTED
  if not _METRICS_RECORDER_THREAD_STARTED:
    with _METRICS_RECORDER_THREAD_LOCK:
      if not _METRICS_RECORDER_THREAD_STARTED:
        # Avoid starting the metrics recorder thread repeatedly if the run is
        # already initialized.
        default_metrics_recorder = metrics.MetricsRecorderThread(
            metric_collectors=[
                (
                    metric_types.MetricType.TPU_DUTY_CYCLE.value,
                    metric_utils.get_tpu_duty_cycle,
                    {
                        "hostname": host_utils.get_hostname(),
                        "process_index": str(host_utils.get_process_index()),
                        "unit": "percent per chip",
                    },
                ),
                (
                    metric_types.MetricType.TPU_TENSORCORE_UTILIZATION.value,
                    metric_utils.get_tpu_tensorcore_utilization,
                    {
                        "hostname": host_utils.get_hostname(),
                        "process_index": str(host_utils.get_process_index()),
                        "unit": "percent per chip",
                    },
                ),
                (
                    metric_types.MetricType.HBM_UTILIZATION.value,
                    metric_utils.get_hbm_utilization,
                    {
                        "hostname": host_utils.get_hostname(),
                        "process_index": str(host_utils.get_process_index()),
                        "unit": "percent per core",
                    },
                ),
                (
                    metric_types.MetricType.HOST_CPU_UTILIZATION.value,
                    metric_utils.get_host_cpu_utilization,
                    {
                        "hostname": host_utils.get_hostname(),
                        "process_index": str(host_utils.get_process_index()),
                        "unit": "percent per host",
                    },
                ),
                (
                    metric_types.MetricType.HOST_MEMORY_UTILIZATION.value,
                    metric_utils.get_host_memory_utilization,
                    {
                        "hostname": host_utils.get_hostname(),
                        "process_index": str(host_utils.get_process_index()),
                        "unit": "percent per host",
                    },
                ),
            ],
            interval_seconds=metrics_record_interval_sec,
        )
        default_metrics_recorder.start()
        _METRICS_RECORDER_THREAD_STARTED = True
        run_phase_monitor.register_cleanup_handler(
            default_metrics_recorder.stop
        )

  if on_demand_xprof:
    # LINT.IfChange(xprof_port)
    xprof_port = 9999
    # LINT.ThenChange(//depot/google3/cloud/hosted/hypercomputecluster/clh/diagnostics/consumerservice/profilersession.go:defaultCapturePort)
    xprof.start_on_demand_xprof(port=xprof_port)

  return ml_run


def create_diagnostics_url(region: str, project: str, name: str) -> str:
  return f"https://console.cloud.google.com/cluster-director/diagnostics/details/{region}/{name}?project={project}"


def create_xprof_url(diagon_url: str) -> str:
  return diagon_url + "&pageState=(%22nav%22:(%22section%22:%22profiles%22))"
