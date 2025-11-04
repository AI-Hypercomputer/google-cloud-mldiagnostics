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

"""Module for recording metrics within ML runs."""

import logging
import threading
from typing import Any, Callable, List, Optional, Tuple, Union

from google_cloud_mldiagnostics.clients import control_plane_client
from google_cloud_mldiagnostics.clients import logging_client
from google_cloud_mldiagnostics.core import global_manager
from google_cloud_mldiagnostics.custom_types import exceptions
from google_cloud_mldiagnostics.custom_types import metric_types
from google_cloud_mldiagnostics.custom_types import mlrun_types
from google_cloud_mldiagnostics.utils import host_utils

logger = logging.getLogger(__name__)


# TODO([INTERNAL]): Create a module to cache and average key metric values.
class _MetricsRecorder:
  """Internal metrics recorder that uses singleton monitoring client."""

  def __init__(self):
    # keep track the metrics
    self._track_list = (
        metric_types.MetricType.STEP_TIME.value,
        metric_types.MetricType.MFU.value,
        metric_types.MetricType.THROUGHPUT.value,
        metric_types.MetricType.LATENCY.value,
    )
    self._metric_tracker: dict[str, dict[str, Any]] = {}
    self._ml_run_name = None

  def _reset_tracker(self):
    """Reset the metric tracker."""
    self._metric_tracker = {}

  def _get_active_run_and_client(
      self,
  ) -> tuple[
      mlrun_types.MLRun,
      logging_client.LoggingClient,
  ]:
    """Get the active run and the logging client.

    Returns:
        A tuple of (MLRun, client).

    Raises:
        NoActiveRunError: If there's no active run.
    """
    manager = global_manager.get_global_run_manager()

    if not manager.has_active_run():
      raise exceptions.NoActiveRunError(
          "No active ML run found. Please initialize a run first."
      )

    ml_run = manager.run
    logging_client_instance = manager.logging_client

    # If logging client is not configured, use a no-op client
    if logging_client_instance is None:
      logging_client_instance = logging_client.NoOpLoggingClient()

    if ml_run is None or logging_client_instance is None:
      raise exceptions.NoActiveRunError(
          "ML run or monitoring client is None despite active run check."
      )

    # Reset the tracker if the ml run name is changed
    if ml_run.name != self._ml_run_name:
      self._reset_tracker()
      self._ml_run_name = ml_run.name

    return ml_run, logging_client_instance

  def record(
      self,
      metric_name: str,
      value: int | float,
      step: int | None = None,
      labels: dict[str, str] | None = None,
      record_on_all_hosts: bool = False,
  ) -> None:
    """Record a single metric value.

    Args:
        metric_name: Name of metric to record.
        value: Metric value.
        step: Optional step number (no step label nor step metric if not
          provided). Note that step metric will be recorded as a separate
          metric, the later step metric will overwrite the previous one and step
          information is the same as previous one
        labels: additional labels.
        record_on_all_hosts: Whether to record metrics on all hosts.

    Raises:
        RecordingError: If recording fails (except for rate limiting errors).
    """
    if not value:
      return

    try:
      # Get active run and client from global manager
      current_mlrun, ml_logging_client = self._get_active_run_and_client()
      is_master_host = host_utils.is_master_host()
      if is_master_host or record_on_all_hosts:

        # Record the metric using logging client
        ml_logging_client.write_metric(
            metric_name=metric_name,
            value=value,
            run_id=current_mlrun.name,
            location=current_mlrun.location,
            step=step,
            labels=labels,
        )

      # Update the metric tracker
      if metric_name in self._track_list:
        if metric_name not in self._metric_tracker:
          self._metric_tracker[metric_name] = {"num_records": 1, "avg": value}
        else:
          num_records = self._metric_tracker[metric_name]["num_records"]
          avg = self._metric_tracker[metric_name]["avg"]
          # Update the averaged metric value
          avg = (avg * num_records + value) / (num_records + 1)
          self._metric_tracker[metric_name] = {
              "num_records": num_records + 1,
              "avg": avg,
          }

    except Exception as e:  # pylint: disable=broad-exception-caught
      raise exceptions.RecordingError(
          "Error recording metric %s: %s" % (metric_name, e)
      ) from e

  def get_metric_tracker(self) -> dict[str, dict[str, Any]]:
    """Get the metric tracker."""
    return self._metric_tracker


class MetricsRecorderThread:
  """Records specified metrics and update the averaged metrics in control plane in a background thread."""

  def __init__(
      self,
      metric_collectors: List[
          Tuple[str, Callable[[], Union[int, float]], Optional[dict[str, str]]]
      ],
      interval_seconds: float,
  ):
    """Initializes the metrics collector.

    Args:
      metric_collectors: A list of tuples, where each tuple contains a metric
        name (str), a callable function that returns the metric value (int or
        float), and labels (dict or None) to be added to the metric.
      interval_seconds: How often to collect metrics in seconds.

    For example:
        metric_collectors = [
            ("host_cpu_utilization", metric_utils.get_host_cpu_utilization,
              {"hostname": "host1"}),
            ("tpu_duty_cycle", metric_utils.get_tpu_duty_cycle, {"hostname":
              "host1"}),
        ]
        interval_seconds = 10.0
        This will start a background thread that collects the host CPU
        utilization and TPU duty cycle every 10 seconds and update the
        control plane averaged metrics every 10 seconds.
    """
    self._metric_collectors = metric_collectors
    self._interval_seconds = interval_seconds
    self._thread: Optional[threading.Thread] = None
    self._stop_event = threading.Event()
    self._is_master_host = host_utils.is_master_host()

  def _get_active_run_and_client(self) -> tuple[
      mlrun_types.MLRun,
      control_plane_client.ControlPlaneClient | None,
  ]:
    """Get the active run and the logging client.

    Returns:
        A tuple of (MLRun, client).

    Raises:
        NoActiveRunError: If there's no active run.
    """

    manager = global_manager.get_global_run_manager()

    if not manager.has_active_run():
      raise exceptions.NoActiveRunError(
          "No active ML run found. Please initialize a run first."
      )

    ml_run = manager.run
    if ml_run is None:
      raise exceptions.NoActiveRunError(
          "ML run is None. Metrics will not be updated."
      )

    control_plane_client_instance = manager.control_plane_client
    if self._is_master_host and control_plane_client_instance is None:
      raise exceptions.NoActiveRunError(
          "Control plane client is None on the master host."
      )

    return ml_run, control_plane_client_instance

  def start(self):
    """Starts the background metric collection."""
    if self._thread is not None:
      logger.warning("Metrics collection thread is already running.")
      return

    self._stop_event.clear()
    self._thread = threading.Thread(
        target=self._collect_loop,
        daemon=True,
        name="diagon-sdk-metrics-recorder-thread",
    )
    self._thread.start()
    metric_names = [item[0] for item in self._metric_collectors]
    logger.info(
        "Started collecting metrics (%s) with interval %d seconds.",
        ", ".join(metric_names),
        self._interval_seconds,
    )

  def stop(self):
    """Stops the background metric collection."""
    if self._thread is None:
      return

    self._stop_event.set()
    self._thread.join()
    self._thread = None
    metric_names = [item[0] for item in self._metric_collectors]
    logger.info(
        "Stopped metrics (%s) collection.",
        ", ".join(metric_names),
    )

  def _collect_loop(self):
    """Continuously collects and records metrics until stop event is set."""
    while not self._stop_event.is_set():
      self._collect_and_record()
      self._update_avg_metrics()
      # Wait for the specified interval, or until the stop event is set.
      self._stop_event.wait(self._interval_seconds)

  def _collect_and_record(self):
    """Iterates through metric collectors, calls them, and records results."""
    for metric_name, collect_func, labels in self._metric_collectors:
      try:
        value = collect_func()
        metrics_recorder.record(
            metric_name=metric_name,
            value=value,
            labels=labels,
            record_on_all_hosts=True,
        )
      except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(
            "Failed to collect or record metric '%s': %s", metric_name, e
        )

  def _update_avg_metrics(self):
    """Updates the averaged metrics in control plane."""
    ml_run, control_plane_client_instance = self._get_active_run_and_client()
    if self._is_master_host:
      if control_plane_client_instance is None:
        raise exceptions.NoActiveRunError(
            "Control plane client is None on the master host."
        )
      metrics_tracker = metrics_recorder.get_metric_tracker()
      metrics_avg = {}
      if metric_types.MetricType.STEP_TIME.value in metrics_tracker:
        step_time_avg = metrics_tracker[
            metric_types.MetricType.STEP_TIME.value
        ]["avg"]
        metrics_avg["avgStep"] = (
            str(round(step_time_avg, 9))
            + "s"
        )
      if metric_types.MetricType.MFU.value in metrics_tracker:
        metrics_avg["avgMfu"] = metrics_tracker[
            metric_types.MetricType.MFU.value
            ]["avg"]
      if metric_types.MetricType.THROUGHPUT.value in metrics_tracker:
        metrics_avg["avgThroughput"] = metrics_tracker[
            metric_types.MetricType.THROUGHPUT.value
        ]["avg"]
      if metric_types.MetricType.LATENCY.value in metrics_tracker:
        latency_avg = metrics_tracker[metric_types.MetricType.LATENCY.value][
            "avg"
        ]
        metrics_avg["avgLatency"] = (
            str(round(latency_avg, 9))
            + "s"
        )

      control_plane_client_instance.update_ml_run(
          name=ml_run.name,
          metrics=metrics_avg,
      )


# Global metrics recorder instance
metrics_recorder = _MetricsRecorder()
