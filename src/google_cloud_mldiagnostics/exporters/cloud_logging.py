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

"""Concrete implementations of telemetry exporters using Google Cloud Logging."""

from collections.abc import Sequence
import logging
from typing import Any, Mapping, Optional

from google_cloud_mldiagnostics.clients import logging_client
from google_cloud_mldiagnostics.exporters import base_exporter

logger = logging.getLogger(__name__)


class CloudLoggingMetricsExporter(base_exporter.BaseMetricsExporter):
  """Exports metrics to Google Cloud Logging."""
  _run_id: str
  _location: str

  def __init__(
      self,
      resource_attributes: Mapping[str, Any],
      client: Optional[logging_client.LoggingClient] = None,
  ):
    """Initializes the metrics exporter.

    Args:
      resource_attributes: Attributes identifying the resource, must include
        'project_id', 'run_id', and 'location'.
      client: Optional existing LoggingClient instance to reuse.
    """
    super().__init__(resource_attributes)
    project_id = resource_attributes.get("project_id")
    if not project_id:
      raise ValueError("project_id is required in resource_attributes")

    run_id = resource_attributes.get("run_id")
    if not run_id:
      raise ValueError("run_id is required in resource_attributes")

    # Fallback to 'global' to match legacy behavior and avoid breaking changes
    location = resource_attributes.get("location", "global")

    if client:
      self._client = client
    else:
      self._client = logging_client.LoggingClient(project_id=project_id)
    self._run_id = run_id
    self._location = location

  def export(self, batch: Sequence[base_exporter.MetricPoint]) -> None:
    """Exports a batch of metric points to Cloud Logging.

    Args:
      batch: A sequence of MetricPoint objects to export.
    """
    logger.debug(
        "CloudLoggingMetricsExporter: Export requested for batch of size %d",
        len(batch),
    )
    if not batch:
      return

    metrics_payload = []
    for point in batch:
      metrics_payload.append({
          "metric_name": point.name,
          "value": point.value,
          "step": point.step,
          "labels": point.labels,
      })

    logger.debug(
        "CloudLoggingMetricsExporter: Writing %d metrics to Cloud Logging",
        len(metrics_payload),
    )
    self._client.write_metrics(
        metrics=metrics_payload,
        run_id=self._run_id,
        location=self._location,
    )


class CloudLoggingLogsExporter(base_exporter.BaseLogsExporter):
  """Exports logs to Google Cloud Logging."""
  _run_id: str
  _location: str

  def __init__(
      self,
      resource_attributes: Mapping[str, Any],
      client: Optional[logging_client.LoggingClient] = None,
  ):
    """Initializes the logs exporter.

    Args:
      resource_attributes: Attributes identifying the resource, must include
        'project_id', 'run_id', and 'location'.
      client: Optional existing LoggingClient instance to reuse.
    """
    super().__init__(resource_attributes)
    project_id = resource_attributes.get("project_id")
    if not project_id:
      raise ValueError("project_id is required in resource_attributes")

    run_id = resource_attributes.get("run_id")
    if not run_id:
      raise ValueError("run_id is required in resource_attributes")

    # Fallback to 'global' to match legacy behavior and avoid breaking changes
    location = resource_attributes.get("location", "global")

    if client:
      self._client = client
    else:
      self._client = logging_client.LoggingClient(project_id=project_id)
    self._run_id = run_id
    self._location = location

  def export(self, batch: Sequence[base_exporter.LogEntry]) -> None:
    """Exports a batch of log entries to Cloud Logging.

    Args:
      batch: A sequence of LogEntry objects to export.
    """
    logger.debug(
        "CloudLoggingLogsExporter: Export requested for batch of size %d",
        len(batch),
    )
    if not batch:
      return

    metrics_payload = []
    for entry in batch:
      # Determine custom namespace (metric_name in LoggingClient) if provided
      # in labels
      namespace = "sdk_logs"
      if entry.labels and "namespace" in entry.labels:
        namespace = entry.labels["namespace"]

      metrics_payload.append({
          "metric_name": namespace,
          "value": entry.body,
          "step": entry.step,
          "labels": entry.labels,
          "severity": entry.severity,
      })

    logger.debug(
        "CloudLoggingLogsExporter: Writing %d log entries to Cloud Logging",
        len(metrics_payload),
    )
    self._client.write_metrics(
        metrics=metrics_payload,
        run_id=self._run_id,
        location=self._location,
    )
