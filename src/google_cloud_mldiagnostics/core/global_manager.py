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

"""Module for managing global state."""

import logging
import threading
from typing import Optional

from google_cloud_mldiagnostics import _version
from google_cloud_mldiagnostics.clients import control_plane_client
from google_cloud_mldiagnostics.clients import logging_client
from google_cloud_mldiagnostics.custom_types import mlrun_types
from google_cloud_mldiagnostics.utils import host_utils


logger = logging.getLogger(__name__)


class GlobalRunManager:
  """Manages the global active run state using singleton pattern."""

  _instance: Optional["GlobalRunManager"] = None
  _lock = threading.RLock()

  def __new__(cls) -> "GlobalRunManager":
    """Ensure only one instance is created (thread-safe singleton)."""
    if cls._instance is None:
      with cls._lock:
        if cls._instance is None:
          cls._instance = super(GlobalRunManager, cls).__new__(cls)
          cls._instance._initialized = False
          cls._instance._ml_run: Optional[mlrun_types.MLRun] = None
          cls._instance._current_logging_client: Optional[
              logging_client.LoggingClient
          ] = None
          cls._instance._control_plane_client: Optional[
              control_plane_client.ControlPlaneClient
          ] = None
    return cls._instance

  def initialize(self, mlrun: mlrun_types.MLRun) -> None:
    """Initialize or update the singleton with new run information.

    Args:
        mlrun: The ML run to initialize.
    """
    with self._lock:
      if self._initialized:
        logger.info(
            "GlobalRunManager already initialized. Updating with new run"
            " information."
        )

      self._ml_run = mlrun
      self._current_logging_client = logging_client.LoggingClient(
          project_id=mlrun.project
      )

      # Initialize ControlPlaneClient with project and location from MLRun
      # Only initialize on master host to avoid duplicate MLRun creation.
      if host_utils.is_master_host():
        self._control_plane_client = control_plane_client.ControlPlaneClient(
            project_id=mlrun.project,
            location=mlrun.location,
            environment=mlrun.environment,
        )

        # Prepare artifacts configuration if gcs_path is provided
        artifacts = None
        if mlrun.gcs_path:
          artifacts = {"gcsPath": mlrun.gcs_path}

        # Prepare default tools (XProf is commonly used)
        tools = [{"xprof": {}}]
        # Create the ML run with mapped parameters
        try:
          response = self._control_plane_client.create_ml_run(
              name=mlrun.name,
              display_name=mlrun.name,
              run_phase=str(mlrun.run_phase.value),
              run_group=mlrun.run_group,
              configs=mlrun.configs,
              tools=tools,
              artifacts=artifacts,
              labels={
                  "created_by": "diagon_sdk",
                  # Request provision xprof tool, could be removed when Control
                  # Plane will do that by default.
                  "create-tool-mode": "regular",
                  "diagon_sdk_version": (
                      _version.get_version().replace(".", "-")
                  ),
                  "on_demand_xprof": (
                      "enabled" if mlrun.on_demand_xprof else "disabled"
                  ),
              },
              orchestrator=mlrun.orchestrator,
              workload_details=mlrun.workload_details,
          )
          logger.info(
              "Successfully created ML run: %s", response.get("name", "unknown")
          )

        except Exception as e:
          logger.error("Failed to create ML run: %s", e)
          raise
      else:
        logger.info(
            "Skipping ML run creation on control plane (run_group=%s, name=%s):"
            " Current host is not the master host.",
            mlrun.run_group,
            mlrun.name,
        )

      self._initialized = True

  def has_active_run(self) -> bool:
    """Check if there's an active run.

    Returns:
        True if there's an active run, False otherwise.
    """
    with self._lock:
      return self._initialized and self._ml_run is not None

  def is_initialized(self) -> bool:
    """Check if the manager has been initialized."""
    with self._lock:
      return self._initialized

  @property
  def run(self) -> Optional[mlrun_types.MLRun]:
    """Get the currently active MLRun object."""
    with self._lock:
      logger.debug("current run details: %s", self._ml_run)
      return self._ml_run

  @property
  def run_group(self) -> Optional[str]:
    """Get the current run set."""
    with self._lock:
      ml_run = self._ml_run
      if ml_run is None:
        return None
      return ml_run.run_group

  @property
  def run_id(self) -> Optional[str]:
    """Get the currently active run ID."""
    with self._lock:
      ml_run = self._ml_run
      if ml_run is None:
        return None
      return ml_run.name

  @property
  def location(self) -> Optional[str]:
    """Get the currently active run location."""
    with self._lock:
      ml_run = self._ml_run
      if ml_run is None:
        return None
      return ml_run.location

  @property
  def project_id(self) -> Optional[str]:
    """Get the currently active run project ID."""
    with self._lock:
      if self._ml_run is None:
        return None
      # Try both 'project' and 'project_id' attributes
      return getattr(self._ml_run, "project_id", None) or getattr(
          self._ml_run, "project", None
      )

  @property
  def logging_client(self) -> Optional[logging_client.LoggingClient]:
    """Get the current logging client."""
    with self._lock:
      return self._current_logging_client

  @property
  def control_plane_client(
      self,
  ) -> Optional[control_plane_client.ControlPlaneClient]:
    """Get the current control plane client."""
    with self._lock:
      return self._control_plane_client

  def clear(self) -> None:
    """Clear the current run state."""
    with self._lock:
      self._ml_run = None
      self._current_logging_client = None
      self._control_plane_client = None
      self._initialized = False

  @classmethod
  def get_instance(cls) -> "GlobalRunManager":
    """Get the singleton instance.

    Returns:
        The singleton GlobalRunManager instance.
    """
    return cls()


# Module-level convenience functions
def get_global_run_manager() -> GlobalRunManager:
  """Get the global run manager instance.

  Returns:
      The GlobalRunManager singleton instance.
  """
  return GlobalRunManager.get_instance()


def initialize_with_mlrun(mlrun: mlrun_types.MLRun) -> GlobalRunManager:
  """Initialize the global manager with an MLRun instance.

  Args:
      mlrun: The MLRun instance to register.

  Returns:
      The initialized GlobalRunManager instance.
  """
  manager = get_global_run_manager()
  manager.initialize(mlrun)
  return manager


def register_run(mlrun: mlrun_types.MLRun) -> None:
  """Register an MLRun instance with the global manager.

  Args:
      mlrun: The MLRun instance to register.
  """
  manager = get_global_run_manager()
  manager.initialize(mlrun)


def get_current_run() -> Optional[mlrun_types.MLRun]:
  """Get the current MLRun from the global manager.

  Returns:
      The current MLRun or None if not initialized.
  """
  manager = get_global_run_manager()
  return manager.run


def get_current_run_id() -> Optional[str]:
  """Get the current run ID from the global manager.

  Returns:
      The current run ID or None if not initialized.
  """
  manager = get_global_run_manager()
  return manager.run_id


def get_logging_client() -> Optional[logging_client.LoggingClient]:
  """Get the logging client from the global manager.

  Returns:
      The logging client or None if not initialized.
  """
  manager = get_global_run_manager()
  return manager.logging_client
