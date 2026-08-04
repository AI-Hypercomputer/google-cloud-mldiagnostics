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

from collections.abc import Mapping
import datetime
import logging
import threading
from typing import Any

from google_cloud_mldiagnostics.core import global_manager
from google_cloud_mldiagnostics.core import metrics
from google_cloud_mldiagnostics.custom_types import metric_types
from google_cloud_mldiagnostics.custom_types import mlrun_types
from google_cloud_mldiagnostics.utils import config_utils
from google_cloud_mldiagnostics.utils import gcp
from google_cloud_mldiagnostics.utils import host_utils
from google_cloud_mldiagnostics.utils import metric_utils
from google_cloud_mldiagnostics.utils import orchestrator_utils
from google_cloud_mldiagnostics.utils import run_phase_utils
from google_cloud_mldiagnostics.utils.gpu_utils import gpu_metric

_METRICS_RECORDER_THREAD_LOCK = threading.Lock()
_METRICS_RECORDER_THREAD_STARTED = False

logger = logging.getLogger(__name__)


def _create_metric_collector(
    metric_name: str,
    collect_func: Any,
    framework: mlrun_types.Framework,
    accelerator_type: str | None = None,
) -> tuple[str, Any, dict[str, str]]:
  """Creates a metric collector tuple with standardized labels."""
  labels = {
      "hostname": host_utils.get_hostname(),
      "process_index": str(host_utils.get_process_index(framework)),
      "unit": "%",
  }
  if accelerator_type is not None:
    labels["accelerator_type"] = accelerator_type
  return (metric_name, collect_func, labels)


def initialize_mlrun(
    name: str,
    environment: str,
    on_demand_xprof: bool,
    log_system_metrics: bool = False,
    metric_only_run: bool = False,
    run_group: str | None = None,
    configs: Mapping[str, Any] | None = None,
    gcs_path: str | None = None,
    project: str | None = None,
    region: str | None = None,
    metrics_record_interval_sec: float = 10.0,
    framework: mlrun_types.Framework = mlrun_types.Framework.JAX,
    serving_engine: mlrun_types.ServingEngine = mlrun_types.ServingEngine.NONE,
    run_workload_id: str | None = None,
) -> mlrun_types.MLRun:
  """Initializes a new ML run.

  Args:
      name: The name of the run.
      environment: The environment to use for the control plane client
        (autopush, staging, prod).
      on_demand_xprof: Whether to start an on-demand xprof profiling server. If
        enabled, the port is set to 9999.
      log_system_metrics: Whether to log system metrics to Cloud Logging. By
        default, system metrics are logged to Cloud Logging.
      run_group: The run set this run belongs to.
      configs: Dictionary of configuration parameters.
      gcs_path: GCS path for storing run artifacts.
      project: The Google Cloud project ID.
      region: The Google Cloud region.
      metrics_record_interval_sec: The metrics record interval in seconds.
      framework: The framework used for the run.
      serving_engine: The serving engine used for the run.
      run_workload_id: Optional shared workload identifier for GCE/Custom
        Orchestrator workloads.

  Returns:
      The initialized ML run object.
  """
  # Combine default configs with user configs.
  software_configs = config_utils.get_software_config(framework, serving_engine)
  hardware_configs = config_utils.get_hardware_config(framework, serving_engine)
  user_configs = configs if configs else {}
  configs = mlrun_types.ConfigDict({
      "softwareConfigs": software_configs,
      "hardwareConfigs": hardware_configs,
      "userConfigs": config_utils.sanitize_config(user_configs),
  })

  if region is None:
    region = gcp.get_instance_region()
  gcp.validate_region(region)
  if project is None:
    project = gcp.get_project_id()

  # TODO([INTERNAL]): Add support for checking the repetitive registered ML
  # name in Spanner after control plane client ready.
  # Otherwise, generate new UUID.

  created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
  run_phase = mlrun_types.RunPhase.PHASE_ACTIVE
  orchestrator = orchestrator_utils.detect_orchestrator()
  workload_details = host_utils.get_workload_details(
      orchestrator, run_workload_id=run_workload_id
  )

  # Generate display name and name for the MLRun.
  display_name = name
  if orchestrator == "GKE":
    if not workload_details:
      raise ValueError(
          "Detected GKE environment but GKE metadata is missing. This might"
          " be because environment variables 'GKE_DIAGON_IDENTIFIER' or"
          " 'GKE_DIAGON_METADATA' are not set or are incomplete. Please"
          " ensure you are running SDK in a GKE environment with the GKE"
          " diagon operator webhook enabled. For more details on GKE"
          " configuration, please see"
          " https://github.com/AI-Hypercomputer/google-cloud-mldiagnostics?tab=readme-ov-file#configure-gke-cluster."
      )
    name = host_utils.get_identifier(orchestrator, workload_details)
  elif orchestrator == "SLURM":
    if not workload_details:
      raise ValueError(
          "Detected Slurm environment but Slurm workload details are missing."
      )
    name = host_utils.get_identifier(orchestrator, workload_details)
  elif orchestrator == "GCE":
    if not workload_details:
      raise ValueError(
          "Detected GCE environment but GCE workload details are missing."
      )
    name = host_utils.get_identifier(orchestrator, workload_details)
  else:
    name = name + "-" + str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))

  # sanitize the name and use it as the MLRun name for the control plane.
  sanitized_name = host_utils.sanitize_identifier(name)

  workload_targets = host_utils.get_workload_targets(
      orchestrator, workload_details
  )

  ml_run = mlrun_types.MLRun(
      run_group=run_group,  # pyrefly: ignore[bad-argument-type]
      name=sanitized_name,
      configs=configs,
      gcs_path=gcs_path,
      location=region,  # pyrefly: ignore[bad-argument-type]
      project=project,  # pyrefly: ignore[bad-argument-type]
      run_phase=run_phase,
      created_at=created_at,
      workload_details=workload_details,
      workload_targets=workload_targets,
      orchestrator=orchestrator,
      display_name=display_name,
      on_demand_xprof=on_demand_xprof,
      log_system_metrics=log_system_metrics,
      metric_only_run=metric_only_run,
      environment=environment,
      framework=framework,
      serving_engine=serving_engine,
  )

  logger.debug("Initializing MLRun: %s", ml_run)

  # register the run to global manager.
  manager = global_manager.get_global_run_manager()
  manager.initialize(ml_run)

  ml_diagnostics_url = create_diagnostics_url(region, project, sanitized_name)  # pyrefly: ignore[bad-argument-type]
  xprof_url = create_xprof_url(ml_diagnostics_url)
  logging.info("MLRun '%s' created successfully.", ml_run.display_name)
  logging.info(
      "ML Diagnostics URL: %s : %s",
      ml_run.display_name,
      ml_diagnostics_url,
  )
  logging.info(
      "Xprof URL: %s : %s",
      ml_run.display_name,
      xprof_url,
  )

  if orchestrator == "GKE":
    gke_url = create_gke_url(region, project, sanitized_name)  # pyrefly: ignore[bad-argument-type]
    logging.info(
        "GKE detail view URL: %s : %s",
        ml_run.display_name,
        gke_url,
    )

  run_phase_monitor = run_phase_utils.RunPhaseMonitor()
  run_phase_monitor.start()

  global _METRICS_RECORDER_THREAD_STARTED
  if metrics_record_interval_sec > 0 and not _METRICS_RECORDER_THREAD_STARTED:
    framework = ml_run.framework
    with _METRICS_RECORDER_THREAD_LOCK:
      if not _METRICS_RECORDER_THREAD_STARTED:
        # Avoid starting the metrics recorder thread repeatedly if the run is
        # already initialized.
        metric_collectors = []
        if log_system_metrics:
          logger.debug("System metrics logging is enabled.")
          accelerator_type = config_utils.get_accelerator_type(framework)
          logger.debug("Accelerator type: %s", accelerator_type)
          if accelerator_type == metric_types.AcceleratorType.GPU.value:
            metric_collectors = [
                _create_metric_collector(
                    metric_types.MetricType.GPU_DUTY_CYCLE.value,
                    gpu_metric.get_gpu_duty_cycle,
                    framework,
                    metric_types.AcceleratorType.GPU.value,
                ),
                _create_metric_collector(
                    metric_types.MetricType.GPU_UTILIZATION.value,
                    gpu_metric.get_gpu_utilization,
                    framework,
                    metric_types.AcceleratorType.GPU.value,
                ),
                _create_metric_collector(
                    metric_types.MetricType.GPU_TENSORCORE_UTILIZATION.value,
                    gpu_metric.get_gpu_tensorcore_utilization,
                    framework,
                    metric_types.AcceleratorType.GPU.value,
                ),
                _create_metric_collector(
                    metric_types.MetricType.VRAM_UTILIZATION.value,
                    gpu_metric.get_vram_utilization,
                    framework,
                    metric_types.AcceleratorType.GPU.value,
                ),
                _create_metric_collector(
                    metric_types.MetricType.HOST_CPU_UTILIZATION.value,
                    metric_utils.get_host_cpu_utilization,
                    framework,
                ),
                _create_metric_collector(
                    metric_types.MetricType.HOST_MEMORY_UTILIZATION.value,
                    metric_utils.get_host_memory_utilization,
                    framework,
                ),
            ]
          else:
            metric_collectors = [
                _create_metric_collector(
                    metric_types.MetricType.TPU_DUTY_CYCLE.value,
                    metric_utils.get_tpu_duty_cycle,
                    framework,
                    metric_types.AcceleratorType.TPU.value,
                ),
                _create_metric_collector(
                    metric_types.MetricType.TPU_TENSORCORE_UTILIZATION.value,
                    metric_utils.get_tpu_tensorcore_utilization,
                    framework,
                    metric_types.AcceleratorType.TPU.value,
                ),
                _create_metric_collector(
                    metric_types.MetricType.HBM_UTILIZATION.value,
                    metric_utils.get_hbm_utilization,
                    framework,
                    metric_types.AcceleratorType.TPU.value,
                ),
                _create_metric_collector(
                    metric_types.MetricType.HOST_CPU_UTILIZATION.value,
                    metric_utils.get_host_cpu_utilization,
                    framework,
                ),
                _create_metric_collector(
                    metric_types.MetricType.HOST_MEMORY_UTILIZATION.value,
                    metric_utils.get_host_memory_utilization,
                    framework,
                ),
            ]
        else:
          logging.info("System metrics logging is disabled.")
        default_metrics_recorder = metrics.MetricsRecorderThread(
            metric_collectors=metric_collectors,
            interval_seconds=metrics_record_interval_sec,
        )
        default_metrics_recorder.start()
        _METRICS_RECORDER_THREAD_STARTED = True
        run_phase_monitor.register_cleanup_handler(
            default_metrics_recorder.stop
        )

  logger.info(
      "Check and start xprof server => on_demand_xprof: %s,"
      " metric_only_run: %s",
      on_demand_xprof,
      metric_only_run,
  )
  if on_demand_xprof and not metric_only_run:
    # LINT.IfChange(xprof_port)
    xprof_port = 9999
    # LINT.ThenChange(//depot/google3/cloud/hosted/hypercomputecluster/clh/diagnostics/consumerservice/utils.go:DefaultCapturePort)
    from google_cloud_mldiagnostics.core import xprof  # pylint: disable=g-import-not-at-top

    xprof.start_on_demand_xprof(port=xprof_port)
    run_phase_monitor.register_cleanup_handler(xprof.stop_on_demand_xprof)

  return ml_run


def create_gke_url(region: str, project: str, name: str) -> str:
  """Creates GKE detail view URL."""
  return f"https://console.cloud.google.com/kubernetes/aiml/run/{region}/{name}?project={project}"


def create_diagnostics_url(region: str, project: str, name: str) -> str:
  return f"https://console.cloud.google.com/cluster-director/diagnostics/details/{region}/{name}?project={project}"


def create_xprof_url(ml_diagnostics_url: str) -> str:
  return f"{ml_diagnostics_url}&pageState=(%22nav%22:(%22section%22:%22profiles%22))"
