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

"""Client for writing metrics directly to Google Cloud Logging."""

import datetime
import logging
import os
from typing import List, Optional

from google.auth import credentials
from google.cloud import logging as cloud_logging
from google.cloud.logging_v2 import resource
from google_cloud_mldiagnostics.custom_types import exceptions


logger = logging.getLogger(__name__)


class LoggingClient:
  """Client for writing metrics directly to Google Cloud Logging."""

  def __init__(
      self,
      project_id: str,
      log_name: str = "ml_diagnostics_metric",
      credentials_path: Optional[str] = None,
      user_credentials: Optional[credentials.Credentials] = None,
  ):
    """Initialize the logging client.

    Args:
        project_id: GCP project ID
        log_name: Name of the log to write to
        credentials_path: Optional path to service account credentials
        user_credentials: Optional explicit credentials object.
    """
    self.project_id = project_id
    self.log_name = log_name

    try:
      if user_credentials:
        # Use explicit credentials if provided
        self.client = cloud_logging.Client(
            project=project_id, credentials=user_credentials
        )
      elif credentials_path:
        # Fall back to credentials file
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        self.client = cloud_logging.Client(project=project_id)
      else:
        # Use default credentials
        self.client = cloud_logging.Client(project=project_id)

      # Get the logger for this log name
      self.logger = self.client.logger(log_name)

    except Exception as e:
      raise exceptions.MLDiagnosticError(
          f"Failed to initialize logging client: {e}"
      ) from e

  def write_metric(
      self,
      metric_name: str,
      value: float | int | List[float | int],
      run_id: str,
      location: str,
      step: Optional[int] = None,
      labels: Optional[dict[str, str]] = None,
  ):
    """Write a single metric point to Cloud Logging.

    Args:
        metric_name: Name of the metric
        value: Metric value. Can be a single int or float, or a list of ints
            and floats.
        run_id: ML run identifier
        location: ML run region
        step: Optional step number
        labels: Optional additional labels

    Returns:
        True if the metric was written successfully, False otherwise.
    """
    try:
      # Create timestamp
      current_time = datetime.datetime.now(datetime.timezone.utc)

      # Define the resource for the log entry
      metric_resource = resource.Resource(
          type="generic_node",
          labels={
              "project_id": self.project_id,
              "location": location,
              "namespace": metric_name,
              "node_id": run_id,
          },
      )

      # Build the log payload structure
      if isinstance(value, (int, float)):
        value = [value]
      payload = {"values": value}

      # Add optional fields
      if step is not None:
        payload["step_index"] = step
      if labels:
        payload.update(labels)

      # Write to Cloud Logging
      self.logger.log_struct(
          payload,
          severity="INFO",
          timestamp=current_time,
          resource=metric_resource,
      )

      logger.info("Successfully written metric to log: %s", metric_name)

    except Exception as e:
      raise exceptions.MLDiagnosticError(
          f"Failed to write to Cloud Logging: {e}"
      ) from e


class NoOpLoggingClient(LoggingClient):
  """A logging client that performs no operations."""

  def __init__(self):  # pylint: disable=super-init-not-called
    """Initializes the NoOp client. Does nothing."""
    pass

  def write_metric(
      self,
      metric_name: str,
      value: float | int | List[float | int],
      run_id: str,
      location: str,
      step: Optional[int] = None,
      labels: Optional[dict[str, str]] = None,
  ):
    """This is a no-op and does not write any metrics."""
    pass
