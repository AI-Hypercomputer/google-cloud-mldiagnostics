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

"""Concrete implementations of telemetry exporters using OpenTelemetry."""
# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring,protected-access,unused-import,g-import-not-at-top,invalid-name,broad-exception-caught

from collections.abc import Mapping, Sequence
import logging
import time

from typing import Any

from google_cloud_mldiagnostics.exporters import base_exporter

logger = logging.getLogger(__name__)


class OTelMetricsExporter(base_exporter.BaseMetricsExporter):
  """Exports metrics to OpenTelemetry.

  Assumes that standard OpenTelemetry MeterProvider is configured externally
  and available globally via OpenTelemetry API.
  """

  _is_available = None

  @classmethod
  def is_available(cls) -> bool:
    """Checks if OpenTelemetry metrics dependencies are available."""
    if cls._is_available is not None:
      return cls._is_available
      
    try:
      from opentelemetry import metrics  # pytype: disable=import-error
      from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter  # pytype: disable=import-error
      from opentelemetry.sdk.metrics import MeterProvider  # pytype: disable=import-error
      from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader  # pytype: disable=import-error

      cls._is_available = True
    except ImportError:
      cls._is_available = False
    return cls._is_available

  def __init__(self, resource_attributes: Mapping[str, Any]):
    """Initializes the OpenTelemetry metrics exporter.

    Args:
      resource_attributes: Attributes identifying the resource emitting metrics.
    """
    super().__init__(resource_attributes)
    # We avoid importing OTel at module level to not break non-OTel users.
    # Instruments will be created lazily on first export.
    self._instruments = {}
    self._meter = None

  def _get_meter(self):
    """Lazily retrieves and caches the OTel Meter."""
    if self._meter is None:
      try:
        from opentelemetry import metrics  # pytype: disable=import-error

        # Auto-initialize provider if none is set (Global No-Op)
        # This allows zero-config usage when DIAGON_ENABLE_OTEL is true
        try:
          from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter  # pytype: disable=import-error
          from opentelemetry.sdk.metrics import MeterProvider  # pytype: disable=import-error
          from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader  # pytype: disable=import-error
          from opentelemetry.sdk.resources import Resource  # pytype: disable=import-error

          # Check if global provider is No-Op
          current_provider = metrics.get_meter_provider()
          # A simple heuristic to check if it's the default _ProxyMeterProvider
          if "Proxy" in type(current_provider).__name__:
            # Build custom resource with run_id
            resource_attrs = {
                "service.name": "diagon_sdk_workload",
            }
            if (
                hasattr(self, "resource_attributes")
                and self.resource_attributes
            ):
              if "run_id" in self.resource_attributes:
                # Map run_id to host.name for legacy parity with GCP Monitored Resources, despite semantic mismatch.
                resource_attrs["host.name"] = self.resource_attributes[
                    "run_id"
                ]  # Map to node_id
              if "project_id" in self.resource_attributes:
                resource_attrs["project_id"] = self.resource_attributes[
                    "project_id"
                ]
              if "location" in self.resource_attributes:
                resource_attrs["cloud.region"] = self.resource_attributes[
                    "location"
                ]  # Map to location

            custom_resource = Resource(attributes=resource_attrs)

            # OTLPMetricExporter automatically picks up OTEL_EXPORTER_OTLP_ENDPOINT env var
            reader = PeriodicExportingMetricReader(OTLPMetricExporter())
            provider = MeterProvider(
                metric_readers=[reader], resource=custom_resource
            )
            metrics.set_meter_provider(provider)
            logger.info(
                "Auto-initialized OTel MeterProvider with custom Resource."
            )
        except ImportError as e:
          logger.warning(
              "Failed to auto-initialize OTel MeterProvider (dependencies"
              " missing): %s",
              e,
          )
        except Exception as e:  # pylint: disable=broad-exception-caught
          logger.warning("Failed to auto-initialize OTel MeterProvider: %s", e)

        self._meter = metrics.get_meter(__name__)
      except ImportError as e:
        logger.error(
            "Failed to import OpenTelemetry metrics API."
            " Ensure 'opentelemetry-api' is installed."
        )
        raise e
    return self._meter

  def export(self, batch: Sequence[base_exporter.MetricPoint]) -> None:
    """Exports a batch of metric points to OpenTelemetry.

    Args:
      batch: A sequence of MetricPoint objects to export.
    """
    logger.debug(
        "OTelMetricsExporter: Export requested for batch of size %d", len(batch)
    )
    if not batch:
      return

    try:
      meter = self._get_meter()
    except ImportError:
      # Skip export if OTel is not available
      return

    for point in batch:
      # CRITICAL Performance Optimization: Strictly process MetricPoint.
      # If by any chance a dictionary lands here, skip it gracefully.
      if not isinstance(point.value, (int, float)):
        logger.warning(
            "Skipping non-numeric metric point: name=%s, type=%s",
            point.name,
            type(point.value),
        )
        continue

      attributes: dict[str, Any] = dict(point.labels) if point.labels else {}
      attributes.pop("step", None)  # Ensure step is not in labels

      # Inject node_id for explicit dimension tagging in localized Prometheus
      if hasattr(self, "resource_attributes") and self.resource_attributes:
        if "run_id" in self.resource_attributes:
          attributes["node_id"] = self.resource_attributes["run_id"]
        if "location" in self.resource_attributes:
          attributes["location"] = self.resource_attributes["location"]

      instrument = self._instruments.get(point.name)
      if instrument is None:
        try:
          # Assuming standard Gauge for MetricPoint as it represents a point in time value.
          # We cache instruments by name.
          instrument = meter.create_gauge(name=point.name)
          self._instruments[point.name] = instrument
        except Exception as e:
          logger.exception(
              "Failed to create OTel instrument for %s: %s", point.name, e
          )
          continue

      try:
        # OTel Gauge takes value and optional attributes.

        # Merge with resource attributes if needed, but usually resource attributes
        # are handled by the MeterProvider/Resource.
        # We can add them as metric attributes if explicitly desired.

        logger.debug(
            "OTelMetricsExporter: Setting value %s for instrument %s",
            point.value,
            point.name,
        )
        instrument.set(point.value, attributes=attributes)
      except Exception as e:
        logger.exception(
            "Failed to set value for OTel instrument %s: %s", point.name, e
        )

      # Emit dedicated training_step metric
      if point.step is not None:
        training_step_instrument = self._instruments.get("training_step")
        if training_step_instrument is None:
          try:
            training_step_instrument = meter.create_gauge(name="training_step")
            self._instruments["training_step"] = training_step_instrument
          except Exception as e:
            logger.exception(
                "Failed to create OTel instrument for training_step: %s", e
            )

        if training_step_instrument is not None:
          try:
            logger.debug(
                "OTelMetricsExporter: Setting value %s for instrument"
                " training_step",
                point.step,
            )
            training_step_instrument.set(point.step, attributes=attributes)
          except Exception as e:
            logger.exception(
                "Failed to set value for OTel instrument training_step: %s", e
            )


class OTelLogsExporter(base_exporter.BaseLogsExporter):
  """Exports logs to OpenTelemetry.

  Assumes that standard OpenTelemetry LoggerProvider is configured externally
  and available globally via OpenTelemetry API/Internal.
  """

  _is_available = None

  @classmethod
  def is_available(cls) -> bool:
    """Checks if OpenTelemetry logs dependencies are available."""
    if cls._is_available is not None:
      return cls._is_available
      
    try:
      from opentelemetry import _logs  # pytype: disable=import-error
      from opentelemetry.sdk._logs import LoggerProvider  # pytype: disable=import-error
      from opentelemetry.sdk._logs.export import BatchLogRecordProcessor  # pytype: disable=import-error

      try:
        from opentelemetry.exporter.otlp.proto.grpc.log_exporter import OTLPLogExporter  # pytype: disable=import-error
      except ImportError:
        from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter  # pytype: disable=import-error
      cls._is_available = True
    except ImportError:
      cls._is_available = False
    return cls._is_available

  def __init__(self, resource_attributes: Mapping[str, Any]):
    """Initializes the OpenTelemetry logs exporter.

    Args:
      resource_attributes: Attributes identifying the resource emitting logs.
    """
    super().__init__(resource_attributes)
    self._logger = None

  def _get_logger(self):
    """Lazily retrieves and caches the OTel Logger."""
    if self._logger is None:
      try:
        from opentelemetry import _logs  # pytype: disable=import-error

        current_provider = _logs.get_logger_provider()
        if "Proxy" in type(current_provider).__name__:
          try:
            from opentelemetry.sdk._logs import LoggerProvider  # pytype: disable=import-error
            from opentelemetry.sdk._logs.export import BatchLogRecordProcessor  # pytype: disable=import-error
            from opentelemetry.sdk.resources import Resource  # pytype: disable=import-error

            # Build custom resource with run_id
            resource_attrs = {
                "service.name": "diagon_sdk_workload",
            }
            if (
                hasattr(self, "resource_attributes")
                and self.resource_attributes
            ):
              if "run_id" in self.resource_attributes:
                # Map run_id to host.name for legacy parity with GCP Monitored Resources, despite semantic mismatch.
                resource_attrs["host.name"] = self.resource_attributes[
                    "run_id"
                ]  # Map to node_id
              if "project_id" in self.resource_attributes:
                resource_attrs["project_id"] = self.resource_attributes[
                    "project_id"
                ]
              if "location" in self.resource_attributes:
                resource_attrs["cloud.region"] = self.resource_attributes[
                    "location"
                ]  # Map to location

            custom_resource = Resource(attributes=resource_attrs)

            try:
              from opentelemetry.exporter.otlp.proto.grpc.log_exporter import OTLPLogExporter  # pytype: disable=import-error
            except ImportError:
              from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter  # pytype: disable=import-error

            # OTLPLogExporter automatically picks up OTEL_EXPORTER_OTLP_ENDPOINT env var
            exporter = OTLPLogExporter()
            processor = BatchLogRecordProcessor(exporter)
            provider = LoggerProvider(resource=custom_resource)
            provider.add_log_record_processor(processor)
            _logs.set_logger_provider(provider)
            logger.info(
                "OTelLogsExporter: Global LoggerProvider auto-initialized via"
                " OTLP Exporter"
            )
          except ImportError as e:
            logger.warning(
                "Failed to auto-initialize OTel LoggerProvider (dependencies"
                " missing): %s",
                e,
            )
          except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(
                "Failed to auto-initialize OTel LoggerProvider: %s", e
            )

        self._logger = _logs.get_logger(__name__)
      except ImportError as e:
        logger.error(
            "Failed to import OpenTelemetry logs API."
            " Ensure 'opentelemetry-api' is installed and supports logs."
        )
        raise e
    return self._logger

  _severity_map = None
  _severity_number = None

  @classmethod
  def _get_severity_info(cls):
    if cls._severity_map is None:
      try:
        from opentelemetry._logs import SeverityNumber  # pytype: disable=import-error

        cls._severity_number = SeverityNumber
        cls._severity_map = {
            "DEBUG": SeverityNumber.DEBUG,
            "INFO": SeverityNumber.INFO,
            "WARNING": SeverityNumber.WARN,
            "WARN": SeverityNumber.WARN,
            "ERROR": SeverityNumber.ERROR,
            "FATAL": SeverityNumber.FATAL,
            "CRITICAL": SeverityNumber.FATAL,
        }
      except ImportError:
        cls._severity_number = None
        cls._severity_map = {}
    return cls._severity_map, cls._severity_number

  def export(self, batch: Sequence[base_exporter.LogEntry]) -> None:
    """Exports a batch of log entries to OpenTelemetry.

    Args:
      batch: A sequence of LogEntry objects to export.
    """
    logger.debug("OTelLogsExporter: Exporting batch of size %d", len(batch))
    if not batch:
      return

    try:
      otel_logger = self._get_logger()
    except ImportError:
      # Skip export if OTel is not available
      return

    severity_map, severity_number_cls = self._get_severity_info()

    for entry in batch:
      attributes: dict[str, Any] = dict(entry.labels) if entry.labels else {}
      if entry.step is not None:
        attributes["step"] = entry.step

      severity_number = severity_number_cls.UNSPECIFIED if severity_number_cls else 0
      if entry.severity in severity_map:
        severity_number = severity_map[entry.severity]
      else:
        # Try to match by prefix or default
        for k, v in severity_map.items():
          if entry.severity.upper().startswith(k):
            severity_number = v
            break

      try:
        # Map LogEntry to OTel LogRecord fields
        logger.debug(
            "OTelLogsExporter: Attempting to emit OTel log body=%s", entry.body
        )

        otel_logger.emit(
            body=entry.body,
            severity_text=entry.severity,
            severity_number=severity_number,
            attributes=attributes,
            # Timestamp is generated automatically if not provided,
            # but we could add entry timestamp if LogEntry had one.
            # BaseLogEntry does not have timestamp in base_exporter.py.
            timestamp=time.time_ns(),  # nanoseconds
        )
      except Exception as e:
        logger.exception("Failed to emit OTel log: %s", e)

  def force_flush(self) -> None:
    """Flushes buffered logs in OTel LoggerProvider."""
    try:
      from opentelemetry import _logs  # pytype: disable=import-error

      provider = _logs.get_logger_provider()

      if provider is not None:
        if hasattr(provider, "force_flush") and callable(provider.force_flush):
          provider.force_flush()
        else:
          logger.warning(
              "OTel logs provider has no force_flush method: %s", type(provider)
          )
      else:
        logger.warning("No OTel logs provider available to flush.")

    except Exception as e:  # pylint: disable=broad-except
      logger.exception("Failed to force_flush OTel logs: %s", e)
