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
  _PROFILER_SESSION_CREATION_TIMEOUT_SEC = 20
  _MAX_GET_ML_RUN_ATTEMPTS = 2

  def __new__(cls, *args, **kwargs) -> "GlobalRunManager":
    """Ensure only one instance is created (thread-safe singleton)."""
    del args, kwargs
    if cls._instance is None:
      with cls._lock:
        if cls._instance is None:
          cls._instance = super(GlobalRunManager, cls).__new__(cls)
          cls._instance._initialized = False
          cls._instance._ml_run: Optional[mlrun_types.MLRun] = None  # pyrefly: ignore[bad-assignment]
          cls._instance._current_logging_client: Optional[  # pyrefly: ignore[bad-assignment]
              logging_client.LoggingClient
          ] = None
          cls._instance._control_plane_client: Optional[  # pyrefly: ignore[bad-assignment]
              control_plane_client.ControlPlaneClient
          ] = None
          cls._instance._timer_ps_creation: threading.Timer | None = None  # pyrefly: ignore[bad-assignment]
          cls._instance._ps_creation_start_time: float | None = None  # pyrefly: ignore[bad-assignment]
    return cls._instance

  def __init__(
      self, accelerator_type: metric_types.AcceleratorType | None = None
  ):
    """Initialize the instance.

    Args:
        accelerator_type: An optional accelerator type enum. If not provided, it
          will default to retrieving it from host_utils.
    """
    if not hasattr(self, "_initialized_constructor"):
      self._initialized_constructor = True
      self._initialized: bool = False
      self._ml_run: Optional[mlrun_types.MLRun] = None
      self._current_logging_client: Optional[logging_client.LoggingClient] = (
          None
      )
      self._control_plane_client: Optional[
          control_plane_client.ControlPlaneClient
      ] = None
      self._timer_ps_creation: threading.Timer | None = None
      self._ps_creation_start_time: float | None = None

    if not hasattr(self, "_accelerator_type") or accelerator_type is not None:
      self._accelerator_type = (
          accelerator_type or host_utils.get_accelerator_type(framework=None)
      )

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
      if self._accelerator_type == metric_types.AcceleratorType.UNKNOWN:
        self._accelerator_type = host_utils.get_accelerator_type(
            mlrun.framework, mlrun.serving_engine
        )
      self._current_logging_client = logging_client.LoggingClient(
          project_id=mlrun.project
      )
      self._control_plane_client = control_plane_client.ControlPlaneClient(
          project_id=mlrun.project,
          location=mlrun.location,
          environment=mlrun.environment,
      )

      if not host_utils.is_master_host(mlrun.framework, mlrun.serving_engine):
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

      self._create_ml_run_on_control_plane(mlrun)
      self._initialized = True

  def _create_ml_run_on_control_plane(self, mlrun: mlrun_types.MLRun) -> None:
    """Helper to call create_ml_run on control plane client."""
    artifacts = None
    if mlrun.gcs_path:
      artifacts = {"gcsPath": mlrun.gcs_path}

    tools = [{"xprof": {}}]
    labels = {
        "created_by": "diagon_sdk",
        "create-tool-mode": "regular",
        "diagon_sdk_version": _version.get_version().replace(".", "-"),
        "on_demand_xprof": "enabled" if mlrun.on_demand_xprof else "disabled",
        "sdk_report_system_metrics": (
            "true" if mlrun.log_system_metrics else "false"
        ),
        "accelerator_type": self._accelerator_type.value,
        "framework": mlrun.framework.value.lower(),
        "serving_engine": (
            mlrun.serving_engine.value.lower()
            if mlrun.serving_engine != mlrun_types.ServingEngine.NONE
            else ""
        ),
    }
    try:
      response = self._control_plane_client.create_ml_run(
          name=mlrun.name,
          display_name=mlrun.display_name,
          run_phase=str(mlrun.run_phase.value),
          run_group=mlrun.run_group,
          configs=mlrun.configs,
          tools=tools,
          artifacts=artifacts,
          labels=labels,
          orchestrator=mlrun.orchestrator,
          workload_details=mlrun.workload_details,
          workload_targets=mlrun.workload_targets,
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
            "ML run %r already exists. Updating existing run details.", mlrun.name
        )
        self._control_plane_client.update_ml_run(
            name=mlrun.name,
            display_name=mlrun.display_name,
            tools=tools,
            artifacts=artifacts,
            run_phase=mlrun_types.RunPhase.PHASE_ACTIVE.value,
            labels=labels,
            configs=mlrun.configs,
        )
      else:
        logger.error("Failed to create ML run: %s", e_create)
        raise
    except Exception as e_create:
      logger.error("Failed to create ML run: %s", e_create)
      raise

  def _start_report_profiler_session_timer(
      self,
      wait_time_sec: float,
      create_new_session: bool,
      session_id: str,
      start_time: float,
      end_time: Optional[float],
      session_phase: str,
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
        self.create_or_update_profiler_session,
        args=(
            create_new_session,
            session_id,
            start_time,
            end_time,
            session_phase,
            context_msg,
        ),
    )
    self._timer_ps_creation.start()

  def create_or_update_profiler_session(
      self,
      create_new_session: bool,
      session_id: str,
      start_time: float,
      end_time: Optional[float],
      session_phase: str,
      context_msg: str,
  ) -> None:
    """Create profiler session for the ML run.

    Args:
        create_new_session: Whether to create a new session or update an
          existing one.
        session_id: The session ID to use for the profiling session.
        start_time: Requested start time of the profile.
        end_time: Requested end time of the profile.
        session_phase: The phase of the session (e.g., "SUCCEEDED").
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
      ):
        logger.warning(
            "Prerequisites not met for session creation. Retrying after 0.2"
            " seconds."
        )
        self._start_report_profiler_session_timer(
            0.2,
            create_new_session,
            session_id,
            start_time,
            end_time,
            session_phase,
            context_msg,
        )
        return

      client = self._control_plane_client
      try:
        hostname = host_utils.get_hostname()
        workload_details = self._ml_run.workload_details or {}
        if not workload_details.get("targets", None):
          # Attempt to fetch targets, and if missing, try updating
          # workload_details and fetch again as the backend might populate it.
          for i in range(self._MAX_GET_ML_RUN_ATTEMPTS):
            resp = client.get_ml_run(self._ml_run.name)
            workload_details = resp.get("workloadDetails", {})
            self._ml_run.workload_details = workload_details

            if len(workload_details.get("targets", [])) > 0:
              break

            if i == 0:
              logger.info(
                  "No targets found in ML Run workload details, trying to"
                  " update ML Run."
              )
              client.update_ml_run(
                  name=self._ml_run.name,
                  force=True,
                  run_phase=mlrun_types.RunPhase.PHASE_ACTIVE.value,
                  update_mask="workload_details"
              )
              continue

            logger.error(
                "No targets found in ML Run workload details even after"
                " update ML Run, aborting session creation."
            )
            return

        profiler_target = None
        for target in workload_details.get("targets", []):
          # In GKE, hostname is the pod name without unique suffix.
          if target.get("displayName", "").startswith(hostname):
            profiler_target = target.get("displayName", None)
            break

        if profiler_target is None:
          logger.error(
              "No profiler target found for hostname %r, aborting session"
              " creation.",
              hostname,
          )
          return

        report_function = (
            client.create_profiler_session
            if create_new_session
            else client.update_profiler_session
        )
        resp = report_function(
            ml_run_id=self._ml_run.name,
            profiler_session_id=session_id,
            gsc_file_path=self._ml_run.gcs_path + "/" + session_id,  # pyrefly: ignore[unsupported-operation]
            profiler_target=profiler_target,
            start_time=start_time,
            end_time=end_time,
            session_phase=session_phase,
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
          self._start_report_profiler_session_timer(
              0.5,
              create_new_session,
              session_id,
              start_time,
              end_time,
              session_phase,
              context_msg,
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
      if self._ml_run and host_utils.is_master_host(
          self._ml_run.framework, self._ml_run.serving_engine
      ):
        return self._control_plane_client
      return None

  def clear(self) -> None:
    """Clear the current run state."""
    with self._lock:
      if self._timer_ps_creation is not None:
        logger.info("Cancelling profiler session creation timer during clear.")
        self._timer_ps_creation.cancel()
        self._timer_ps_creation = None
      self._ps_creation_start_time = None
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
