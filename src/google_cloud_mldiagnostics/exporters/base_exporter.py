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

"""Base classes and data structures for telemetry exporters."""

import abc
import dataclasses
import datetime
import logging
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class MetricPoint:
  """Represents a single metric point.

  Attributes:
    name: The name of the metric.
    value: The value of the metric.
    step: The step number associated with the metric.
    labels: Key-value pairs providing additional context.
    timestamp: The timestamp of the metric point.
  """

  name: str
  value: float | int
  step: int | None = None
  labels: Mapping[str, str] | None = None
  timestamp: datetime.datetime | None = None


@dataclasses.dataclass(frozen=True)
class LogEntry:
  """Represents a single log entry.

  Attributes:
    body: The log message body, either a string or a structured dictionary.
    severity: The severity level of the log (e.g., INFO, WARNING, ERROR).
    step: The step number associated with the log.
    labels: Key-value pairs providing additional context.
  """

  body: str | Mapping[str, Any]
  severity: str = "INFO"
  step: int | None = None
  labels: Mapping[str, str] | None = None


class BaseMetricsExporter(abc.ABC):
  """Abstract base class for metrics exporters."""

  def __init__(self, resource_attributes: Mapping[str, Any]):
    """Initializes the exporter.

    Args:
      resource_attributes: Attributes identifying the resource emitting metrics.
    """
    self.resource_attributes = resource_attributes

  @abc.abstractmethod
  def export(self, batch: Sequence[MetricPoint]) -> None:
    """Exports a batch of metric points.

    Args:
      batch: A sequence of MetricPoint objects to export.
    """

  def shutdown(self) -> None:
    """Cleans up and flushes exporter resources."""


class BaseLogsExporter(abc.ABC):
  """Abstract base class for logs exporters."""

  def __init__(self, resource_attributes: Mapping[str, Any]):
    """Initializes the exporter.

    Args:
      resource_attributes: Attributes identifying the resource emitting logs.
    """
    self.resource_attributes = resource_attributes

  @abc.abstractmethod
  def export(self, batch: Sequence[LogEntry]) -> None:
    """Exports a batch of log entries.

    Args:
      batch: A sequence of LogEntry objects to export.
    """

  def shutdown(self) -> None:
    """Cleans up and flushes exporter resources."""

  def force_flush(self) -> None:
    """Flushes buffered log entries."""


class CompositeMetricsExporter(BaseMetricsExporter):
  """Composite exporter that multiplexes metrics to multiple exporters."""

  def __init__(
      self,
      exporters: Sequence[BaseMetricsExporter],
      resource_attributes: Mapping[str, Any],
  ):
    super().__init__(resource_attributes)
    self._exporters = exporters

  def export(self, batch: Sequence[MetricPoint]) -> None:
    logger.debug(
        "CompositeMetricsExporter: Exporting batch of size %d to %d exporters",
        len(batch),
        len(self._exporters),
    )
    for exporter in self._exporters:
      try:
        logger.debug(
            "CompositeMetricsExporter: Dispatching to %s",
            exporter.__class__.__name__,
        )
        exporter.export(batch)
      except Exception as e:  # pylint: disable=broad-exception-caught
        logger.exception(
            "Error exporting metrics to %s: %s",
            exporter.__class__.__name__,
            e,
        )

  def shutdown(self) -> None:
    logger.debug("CompositeMetricsExporter: Shutting down underlying exporters")
    for exporter in self._exporters:
      try:
        exporter.shutdown()
      except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(
            "Error shutting down exporter %s: %s",
            exporter.__class__.__name__,
            e,
            exc_info=True,
        )


class CompositeLogsExporter(BaseLogsExporter):
  """Composite exporter that multiplexes logs to multiple exporters."""

  def __init__(
      self,
      exporters: Sequence[BaseLogsExporter],
      resource_attributes: Mapping[str, Any],
  ):
    super().__init__(resource_attributes)
    self._exporters = exporters

  def export(self, batch: Sequence[LogEntry]) -> None:
    logger.debug(
        "CompositeLogsExporter: Exporting batch of size %d to %d exporters",
        len(batch),
        len(self._exporters),
    )
    for exporter in self._exporters:
      try:
        logger.debug(
            "CompositeLogsExporter: Dispatching to %s",
            exporter.__class__.__name__,
        )
        exporter.export(batch)
      except Exception as e:  # pylint: disable=broad-exception-caught
        logger.exception(
            "Error exporting logs to %s: %s",
            exporter.__class__.__name__,
            e,
        )

  def shutdown(self) -> None:
    logger.debug("CompositeLogsExporter: Shutting down underlying exporters")
    for exporter in self._exporters:
      try:
        exporter.shutdown()
      except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(
            "Error shutting down exporter %s: %s",
            exporter.__class__.__name__,
            e,
            exc_info=True,
        )

  def force_flush(self) -> None:
    logger.debug("CompositeLogsExporter: Flushing underlying exporters")
    for exporter in self._exporters:
      try:
        exporter.force_flush()
      except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(
            "Error flushing exporter %s: %s",
            exporter.__class__.__name__,
            e,
            exc_info=True,
        )
