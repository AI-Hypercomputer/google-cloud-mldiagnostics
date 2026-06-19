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

from __future__ import annotations

import collections
from collections.abc import Mapping, Sequence
import copy
import logging
import queue
import statistics
import threading
from typing import Any, Callable

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
        metric_types.MetricType.HBM_UTILIZATION.value,
        metric_types.MetricType.TPU_TENSORCORE_UTILIZATION.value,
        metric_types.MetricType.VRAM_UTILIZATION.value,
        metric_types.MetricType.GPU_TENSORCORE_UTILIZATION.value,
    )
    self._metric_tracker: dict[str, dict[str, Any]] = collections.defaultdict(
        lambda: {"num_records": 0, "avg": 0.0}
    )
    self._ml_run_name: str | None = None
    self._lock = threading.Lock()
    self._is_master_host: bool | None = None

    # Async metrics queue
    self._queue = queue.Queue(maxsize=10_000)
    self._stop_event = threading.Event()
    self._worker_thread = threading.Thread(
        target=self._flush_metrics_worker, daemon=True
    )
    self._worker_thread.start()

  def _extract_metric_value(
      self, metric_name: str, value: int | float | Sequence[float]
  ) -> float | None:
    """Extract a single float metric value from the input.

    Args:
      metric_name: The name of the metric.
      value: The raw value, which can be an int, float, or sequence of floats.

    Returns:
      A float representing the metric value, or None if extraction fails.
    """
    if isinstance(value, (list, tuple)):
      if not value:
        logger.warning(
            "Metric '%s' has an empty list value: %s", metric_name, value
        )
        return None
      try:
        return statistics.mean(value)
      except statistics.StatisticsError:
        logger.warning(
            "Could not calculate mean for metric %s with value %s",
            metric_name,
            value,
        )
        return None
    if isinstance(value, (int, float)):
      return float(value)
    logger.warning(
        "Unsupported metric value type %s for %s",
        type(value),
        metric_name,
    )
    return None

  def _process_single_metric_item(
      self, item: Mapping[str, Any], is_master_host: bool
  ) -> dict[str, Any] | None:
    """Processes a single metric item from the queue.

    Args:
      item: The metric item dictionary from the queue.
      is_master_host: Whether the current host is the master host.

    Returns:
      A dictionary representing the metric to be written if it should be
      recorded, otherwise None.
    """
    metric_info = item["metric_info"]
    record_on_all_hosts = item["record_on_all_hosts"]
    metric_name = metric_info.get("metric_name")
    value = metric_info.get("value")
    step = metric_info.get("step")
    labels = metric_info.get("labels")

    if metric_name is None or value is None:
      logger.warning(
          "Invalid metric data: metric_name or value is None in item: %s", item
      )
      return None

    metric_value = self._extract_metric_value(metric_name, value)
    if metric_value is None:
      return None

    # Update the metric tracker
    if metric_name in self._track_list:
      with self._lock:
        tracker = self._metric_tracker[metric_name]
        num_records = tracker["num_records"]
        avg = tracker["avg"]
        tracker["avg"] += (metric_value - avg) / (num_records + 1)
        tracker["num_records"] = num_records + 1

    if is_master_host or record_on_all_hosts:
      all_labels = labels.copy() if labels else {}
      unit = metric_types.METRIC_UNITS.get(metric_name, "1")
      all_labels.setdefault("unit", unit)
      return {
          "metric_name": metric_name,
          "value": metric_value,
          "step": step,
          "labels": all_labels,
      }
    return None

  def _flush_metrics_worker(self) -> None:
    """Continuously pop items from the queue and publish them safely."""
    raw_items = []
    while True:
      try:
        if not raw_items:
          first_item = self._queue.get()
          if first_item is None:
            self._queue.task_done()
            break
          raw_items.append(first_item)

          # Drain any other available items immediately without blocking
          while not self._queue.empty():
            raw_items.append(self._queue.get_nowait())

        try:
          ml_run, logging_client_instance = self._get_active_run_and_client()
          if self._is_master_host is None:
            self._is_master_host = host_utils.is_master_host(
                ml_run.framework, ml_run.serving_engine
            )
          is_master_host = self._is_master_host
        except exceptions.NoActiveRunError:
          # If no active run yet, sleep briefly and retry on next loop.
          self._stop_event.wait(0.5)
          if self._stop_event.is_set():
            for _ in raw_items:
              self._queue.task_done()
            break
        else:
          should_stop = False
          try:
            metrics_to_write = []
            for item in raw_items:
              if item is None:
                should_stop = True
                continue

              metric_to_write = self._process_single_metric_item(
                  item, is_master_host
              )
              if metric_to_write:
                metrics_to_write.append(metric_to_write)

            if metrics_to_write:
              try:
                logging_client_instance.write_metrics(
                    metrics=metrics_to_write,
                    run_id=ml_run.name,
                    location=ml_run.location,
                )
              except Exception:  # pylint: disable=broad-exception-caught
                logger.exception(
                    "Error publishing async metrics batch: %s", metrics_to_write
                )
          finally:
            for _ in raw_items:
              self._queue.task_done()
            raw_items = []

          if should_stop:
            break

      except Exception:  # pylint: disable=broad-exception-caught
        logger.exception(
            "Unhandled exception in metrics worker daemon, raw_items: %s",
            raw_items,
        )

  def _reset_tracker(self):
    """Reset the metric tracker."""
    self._metric_tracker = collections.defaultdict(
        lambda: {"num_records": 0, "avg": 0.0}
    )

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
      raise exceptions.NoActiveRunError("ML run is not fully initialized.")

    # Reset the tracker if the ml run name is changed
    if ml_run.name != self._ml_run_name:
      self._reset_tracker()
      self._ml_run_name = ml_run.name

    return ml_run, logging_client_instance

  def record(
      self,
      metric_name: str,
      value: int | float | Sequence[float] | None,
      step: int | None = None,
      labels: Mapping[str, str] | None = None,
      record_on_all_hosts: bool = False,
  ) -> None:
    """Record a single metric value, averaging lists if provided.

    Args:
      metric_name: Name of metric to record.
      value: Metric value.
      step: Optional step number (no step label nor step metric if not
        provided). Note that step metric will be recorded as a separate
        metric, the later step metric will overwrite the previous one and step
        information is the same as previous one.
      labels: Additional labels.
      record_on_all_hosts: Whether to record metrics on all hosts.
    """
    if value is None:
      logger.debug("Received None value for metric %s", metric_name)
      return

    self.record_metrics(
        metrics_data=[{
            "metric_name": metric_name,
            "value": value,
            "step": step,
            "labels": labels,
        }],
        record_on_all_hosts=record_on_all_hosts,
    )

  def get_metric_tracker(self) -> dict[str, dict[str, Any]]:
    """Get the metric tracker."""
    with self._lock:
      return copy.deepcopy(self._metric_tracker)

  def record_metrics(
      self,
      metrics_data: Sequence[Mapping[str, Any]],
      record_on_all_hosts: bool = False,
  ) -> None:
    """Record multiple metric values.

    Args:
      metrics_data: A list of dictionaries, where each dictionary
        represents a metric and contains 'metric_name' (str) and 'value'
        (int, float, or list), and optionally 'step' (int) and 'labels'
        (dict).
      record_on_all_hosts: Whether to record metrics on all hosts.
    """
    dropped_count = 0
    for metric_info in metrics_data:
      try:
        self._queue.put_nowait({
            "metric_info": metric_info,
            "record_on_all_hosts": record_on_all_hosts,
        })
      except queue.Full:
        dropped_count += 1

    if dropped_count > 0:
      logger.warning(
          "Async metrics queue is full. Dropped %d metrics from batch.",
          dropped_count,
      )

  def stop(self) -> None:
    """Stop the background worker daemon."""
    if self._stop_event.is_set():
      return
    self._stop_event.set()
    self._queue.put(None)
    self._worker_thread.join()


class MetricsRecorderThread:
  """Records and updates averaged metrics in a background thread."""

  def __init__(
      self,
      metric_collectors: Sequence[
          tuple[
              str,
              Callable[[], int | float | Sequence[float] | None],
              Mapping[str, str] | None,
          ]
      ],
      interval_seconds: float,
  ):
    """Initialize the metrics collector.

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
    self._thread: threading.Thread | None = None
    self._stop_event = threading.Event()

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
    if (
        host_utils.is_master_host(ml_run.framework, ml_run.serving_engine)
        and control_plane_client_instance is None
    ):
      raise exceptions.ControlPlaneClientNotInitializedError(
          "Required services are not initialized on the master host."
      )

    return ml_run, control_plane_client_instance

  def start(self):
    """Start the background metric collection."""
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
    """Stop the background metric collection."""
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
    """Continuously collect and record metrics until stop event is set."""
    while not self._stop_event.is_set():
      try:
        self._collect_and_record()
        self._update_control_plane_time()
      except Exception:  # pylint: disable=broad-exception-caught
        logger.exception("Failed to collect or record metrics")
      finally:
        # Wait for the specified interval, or until the stop event is set.
        self._stop_event.wait(self._interval_seconds)

  def _collect_and_record(self):
    """Iterate through metric collectors, call them, and record results."""
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

  def _update_control_plane_time(self):
    """Update the time metric in control plane."""
    # Only update control plane time from the master host. This avoids
    # unnecessary client fetches and updates on worker hosts.
    ml_run, control_plane_client_instance = self._get_active_run_and_client()
    if host_utils.is_master_host(ml_run.framework, ml_run.serving_engine):
      if control_plane_client_instance is None:
        raise exceptions.ControlPlaneClientNotInitializedError(
            "Required services are not initialized on the master host."
        )
      logger.info("Updating control plane time stamp.")
      control_plane_client_instance.update_ml_run(
          name=ml_run.name,
          force=True,
          run_phase="ACTIVE",
      )

# Global metrics recorder instance
metrics_recorder = _MetricsRecorder()
