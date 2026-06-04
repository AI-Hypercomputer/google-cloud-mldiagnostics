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
import time
from typing import Optional

from google_cloud_mldiagnostics import _version
from google_cloud_mldiagnostics.clients import control_plane_client
from google_cloud_mldiagnostics.clients import logging_client
from google_cloud_mldiagnostics.custom_types import metric_types
from google_cloud_mldiagnostics.custom_types import mlrun_types
from google_cloud_mldiagnostics.utils import host_utils
import requests


logger = logging.getLogger(__name__)


class GlobalRunManager:
  """Manages the global active run state using singleton pattern."""

  _instance: Optional["GlobalRunManager"] = None
  _lock = threading.RLock()
  _PROFILER_TARGET_CREATION_TIMEOUT_SEC = 20
  _PROFILER_SESSION_CREATION_TIMEOUT_SEC = 20

  def __new__(cls, *args, **kwargs) -> "GlobalRunManager":
    """Ensure only one instance is created (thread-safe singleton)."""
    del args, kwargs
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
          cls._instance._timer_pt_creation: threading.Timer | None = None
          cls._instance._pt_creation_start_time: float | None = None
          cls._instance._timer_ps_creation: threading.Timer | None = None
          cls._instance._ps_creation_start_time: float | None = None
          cls._instance._profiler_target: Optional[str] = None
    return cls._instance

  def __init__(
      self, accelerator_type: metric_types.AcceleratorType | None = None
  ):
    """Initialize the instance.

    Args:
        accelerator_type: An optional accelerator type enum. If not provided,
            it will default to retrieving it from host_utils.
    """
    if (
        not hasattr(self, "_initialized_constructor")
        or accelerator_type is not None
    ):
      self._initialized_constructor = True
      self._accelerator_type = (
          accelerator_type or host_utils.get_accelerator_type()
      )

  def initialize(self, mlrun: mlrun_types.MLRun) -> None:
    """Initialize or update the singleton with new run information.

    Args:
        mlrun: The ML run to initialize.
    """
    if mlrun.environment != "prod":
      logger.info(
          "Non-prod environment %r detected. Profiler target creation will"
          " be attempted.",
          mlrun.environment,
      )
      # Check and register ML host as Profiler Target, run this before acquiring
      # the lock to avoid deadlock.
      self.create_profiler_target()

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
      self._control_plane_client = control_plane_client.ControlPlaneClient(
          project_id=mlrun.project,
          location=mlrun.location,
          environment=mlrun.environment,
      )

      if not host_utils.is_master_host():
        logger.info(
            "Skipping ML run initialization on control plane (run_group=%s,"
            " name=%s): Current host is not the master host.",
            mlrun.run_group,
            mlrun.name,
        )
        self._initialized = True
        return

      # Write userConfigs to Cloud Logging if available.
      if (
          mlrun.configs
          and isinstance(mlrun.configs, dict)
          and "userConfigs" in mlrun.configs
          and self._current_logging_client
      ):
        try:
          self._current_logging_client.write_metric(
              metric_name="mlrun_configs",
              value={"userConfigs": mlrun.configs.get("userConfigs")},
              run_id=mlrun.name,
              location=mlrun.location,
          )
        except Exception:  # pylint: disable=broad-exception-caught
          logger.exception(
              "Failed to write configs to Cloud Logging for run: %s",
              mlrun.name,
          )
        del mlrun.configs["userConfigs"]

      try:
        logger.info("Checking for existing ML run with name: %s", mlrun.name)
        response = self._control_plane_client.get_ml_run(mlrun.name)
        logger.info(
            "Found existing ML run: %s.",
            response.get("name", "unknown"),
        )
        if response.get("runPhase") == mlrun_types.RunPhase.PHASE_FAILED.value:
          logger.info(
              "Existing ML run %r is in FAILED state, updating to ACTIVE.",
              mlrun.name,
          )
          self._control_plane_client.update_ml_run(
              name=mlrun.name,
              run_phase=mlrun_types.RunPhase.PHASE_ACTIVE.value,
          )
        else:
          logger.info(
              "ML run %r with status %s already exists, skipping creation.",
              mlrun.name,
              response.get("runPhase"),
          )
      except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
          logger.info("ML run %r not found, creating a new one.", mlrun.name)
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
                display_name=mlrun.display_name,
                run_phase=str(mlrun.run_phase.value),
                run_group=mlrun.run_group,
                configs=mlrun.configs,
                tools=tools,
                artifacts=artifacts,
                labels={
                    "created_by": "diagon_sdk",
                    # Request provision xprof tool, can be removed when
                    # Control Plane does this by default.
                    "create-tool-mode": "regular",
                    "diagon_sdk_version": (
                        _version.get_version().replace(".", "-")
                    ),
                    "on_demand_xprof": (
                        "enabled" if mlrun.on_demand_xprof else "disabled"
                    ),
                    "accelerator_type": self._accelerator_type.value,
                },
                orchestrator=mlrun.orchestrator,
                workload_details=mlrun.workload_details,
            )
            logger.info(
                "Successfully created ML run: %s",
                response.get("name") if response else "unknown",
            )
            if response and "name" in response:
              self._ml_run.name = response.get("name", "unknown").split("/")[-1]

          except requests.exceptions.HTTPError as e_create:
            if (
                e_create.response is not None
                and e_create.response.status_code == 409
            ):
              logger.info(
                  "ML run %r already exists, skipping creation.", mlrun.name
              )
            else:
              logger.error("Failed to create ML run: %s", e_create)
              raise
          except Exception as e_create:
            logger.error("Failed to create ML run: %s", e_create)
            raise
        else:
          # HTTPError with status other than 404, or no response
          logger.error("Failed to get ML run %r: %s", mlrun.name, e)
          raise
      except Exception as e_get:
        logger.error("Failed to get ML run %r: %s", mlrun.name, e_get)
        raise

      self._initialized = True

  def _start_profiler_target_creation_timer(self, wait_time_sec: float) -> None:
    """Start profiler target creation timer.

    Args:
        wait_time_sec: The time in seconds to wait before attempting to create
          the profiler target again.

    Raises:
        TimeoutError: If the profiler target creation has exceeded the maximum
          allowed timeout.
    """
    if (
        self._pt_creation_start_time is not None
        and time.time() - self._pt_creation_start_time
        > self._PROFILER_TARGET_CREATION_TIMEOUT_SEC
    ):
      raise TimeoutError(
          "Profiler target creation time exceeded wait time of"
          f" {self._PROFILER_TARGET_CREATION_TIMEOUT_SEC} seconds."
      )
    logger.info(
        "Starting profiler target creation timer with wait time: %s",
        wait_time_sec,
    )
    self._timer_pt_creation = threading.Timer(
        wait_time_sec, self.create_profiler_target
    )
    self._timer_pt_creation.start()

  def create_profiler_target(self) -> None:
    """Create profiler targets for the ML run.

    Raises:
      TimeoutError: If profiler target creation time exceeds the defined
        timeout.
      requests.exceptions.HTTPError: If an HTTP error occurs during API calls
        to the control plane.
      Exception: For other unexpected errors during profiler target creation.
    """
    with self._lock:
      timer = self._timer_pt_creation
      if timer is not None:
        logger.info("Cancelling profiler target creation timer.")
        timer.cancel()
        self._timer_pt_creation = None

      if self._pt_creation_start_time is None:
        logger.info(
            "Starting profiler target creation timer. Current time: %s",
            time.time(),
        )
        self._pt_creation_start_time = time.time()

      if (
          not self._initialized
          or self._ml_run is None
          or self._control_plane_client is None
      ):
        logger.warning(
            "Prerequisites not met. initialized: %r, run_id: %r,"
            " control_plane_client: %r, retrying profiler target creation after"
            " 0.2 seconds.",
            self._initialized,
            self._ml_run.name if self._ml_run else None,
            self._control_plane_client,
        )
        self._start_profiler_target_creation_timer(0.2)
        return

      client = self._control_plane_client
      try:
        client.get_ml_run(self._ml_run.name)
      except requests.exceptions.HTTPError as e_get:
        if e_get.response is not None and (
            e_get.response.status_code == 404
            or 500 <= e_get.response.status_code < 600
        ):
          logger.info(
              "ML run %r not found, waiting for master node to create it.",
              self._ml_run.name,
          )
          self._start_profiler_target_creation_timer(0.5)
          return

        logger.error("Failed to get ML run '%s': %s", self._ml_run.name, e_get)
        raise

      try:
        instance_id = host_utils.get_instance_id()
        node_index = host_utils.get_process_index()
        client.create_profiler_target(
            ml_run_name=self._ml_run.name,
            name=instance_id,
            is_master=host_utils.is_master_host(),
            hostname=instance_id,
            node_index=node_index,
        )
        logger.info(
            "Successfully created profiler target for ML run: %s",
            self._ml_run.name,
        )
        # Save the profiler target resource name
        parent = (
            f"projects/{client.project_id}/locations/{client.location}/"
            f"machineLearningRuns/{self._ml_run.name}"
        )
        self._profiler_target = f"{parent}/profilerTargets/{instance_id}"
        # Clear the start time after successful creation
        self._pt_creation_start_time = None
      except Exception:
        logger.exception("Failed to create profiler target.")
        raise RuntimeError("Failed to create profiler target.") from None

  def _start_profiler_session_creation_timer(
      self,
      wait_time_sec: float,
      session_id: str,
      duration: str,
      context_msg: str,
  ) -> None:
    """Starts a timer to retry profiler session creation."""
    if (
        self._ps_creation_start_time is not None
        and time.time() - self._ps_creation_start_time
        > self._PROFILER_SESSION_CREATION_TIMEOUT_SEC
    ):
      raise TimeoutError(
          "Profiler session creation time exceeded wait time of"
          f" {self._PROFILER_SESSION_CREATION_TIMEOUT_SEC} seconds."
      )
    logger.info(
        "Starting profiler session creation timer with wait time: %s",
        wait_time_sec,
    )
    self._timer_ps_creation = threading.Timer(
        wait_time_sec,
        self.create_profiler_session,
        args=(session_id, duration, context_msg),
    )
    self._timer_ps_creation.start()

  def create_profiler_session(
      self, session_id: str, duration: str, context_msg: str
  ) -> None:
    """Create profiler session for the ML run.

    Args:
        session_id: The session ID to use for the profiling session.
        duration: Requested duration of the profile (e.g., "10s").
        context_msg: Context message for logging (e.g., "on stop").
    """
    with self._lock:
      timer = self._timer_ps_creation
      if timer is not None:
        logger.info("Cancelling profiler session creation timer.")
        timer.cancel()
        self._timer_ps_creation = None

      if self._ps_creation_start_time is None:
        self._ps_creation_start_time = time.time()

      if (
          not self._initialized
          or self._ml_run is None
          or self._control_plane_client is None
          or self._profiler_target is None
      ):
        logger.warning(
            "Prerequisites not met for session creation. Retrying after 0.2"
            " seconds."
        )
        self._start_profiler_session_creation_timer(
            0.2, session_id, duration, context_msg
        )
        return

      client = self._control_plane_client
      try:
        # TODO([INTERNAL]): Handle case of existing sessions and append to
        # existing TARGET list.
        resp = client.create_profiler_session(
            ml_run_id=self._ml_run.name,
            profiler_session_id=session_id,
            profiler_targets=[self._profiler_target],
            duration=duration,
            kind="KIND_PROGRAMMATIC",
            host_tracer_level="HOST_TRACER_LEVEL_INFO",
            device_tracer_level="DEVICE_TRACER_LEVEL_ENABLED",
            python_tracer_level="PYTHON_TRACER_LEVEL_DISABLED",
        )
        if resp == {"done": True}:
          logger.info(
              "Programmatic session lifecycle not enabled on server, session"
              " not persisted."
          )
        else:
          logger.info(
              "Successfully reported profiler session to Control Plane %s.",
              context_msg,
          )
        self._ps_creation_start_time = None
      except requests.exceptions.HTTPError as e:
        response = e.response
        if response is not None and response.status_code == 404:
          logger.warning(
              "ML Run or Target not found, retrying profiler session"
              " creation..."
          )
          self._start_profiler_session_creation_timer(
              0.5, session_id, duration, context_msg
          )
          return

        logger.exception("Failed to create profiler session")
        raise
      except Exception:
        logger.exception("Unexpected error reporting profiler session")
        self._ps_creation_start_time = None
        raise

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
  def profiler_target(self) -> Optional[str]:
    """Get the current profiler target resource name."""
    with self._lock:
      return self._profiler_target

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
      if host_utils.is_master_host():
        return self._control_plane_client
      return None

  def clear(self) -> None:
    """Clear the current run state."""
    with self._lock:
      if self._timer_pt_creation is not None:
        logger.info("Cancelling profiler target creation timer during clear.")
        self._timer_pt_creation.cancel()
        self._timer_pt_creation = None
      self._pt_creation_start_time = None
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
