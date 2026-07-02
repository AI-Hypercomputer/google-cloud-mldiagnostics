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

"""JAX profiling SDK wrapper for Google Cloud ML Diagnostics."""

import logging
import threading
import time

from google_cloud_mldiagnostics.core import global_manager
from google_cloud_mldiagnostics.custom_types import exceptions
from google_cloud_mldiagnostics.custom_types import mlrun_types
from google_cloud_mldiagnostics.utils import host_utils
import jax

logger = logging.getLogger(__name__)


# Wrapper for programmatic jax profiling
class Xprof:
  """Wrapper for JAX profiling with Google Cloud ML Diagnostics.

  Supports:
  - Object-oriented API (prof.start(), prof.stop())
  - Context manager (with Xprof() as prof:)
  - Decorator (@Xprof())
  """

  def __init__(
      self,
      run: mlrun_types.MLRun | None = None,
      process_index_list: list[int] | None = None,
  ):
    """Initializes the xprof profiler.

    Args:
        run: An instance of machinelearning_run to associate the profile with.
          If None, retrieve from global manager when needed.
        process_index_list: A list of process indices to profile. If None,
          profile all hosts. Default is profiling on all the hosts.
    """
    # Store input run but don't resolve until needed (lazy initialization)
    self._input_run = run
    self._resolved_run = None

    self._current_session_id = None
    self._is_profiling = False
    self._gcs_profile_dir = None
    self._initialized = False
    self._process_index_list = process_index_list

    self._start_time = None
    self._end_time = None
    self._session_phase = None

  def _ensure_initialized(self):
    """Lazy initialization - resolve run and setup directories when needed."""
    if self._initialized:
      return

    # Resolve the run now (at usage time, not construction time)
    self._resolved_run = (
        self._input_run
        if self._input_run is not None
        else global_manager.get_current_run()
    )

    if self._resolved_run is None:
      raise exceptions.ProfilingError(
          "No active ML run found for profiling. Please initialize an ML run"
          " or provide a valid ML run with a configured GCS path."
      )

    if self._resolved_run.gcs_path is None:
      raise exceptions.ProfilingError(
          "No GCS path found for profiling. Please provide a valid ML run with"
          " a GCS path."
      )

    # Set up the GCS directory path
    identifier = self._resolved_run.name
    self._gcs_profile_dir = f"{self._resolved_run.gcs_path}/{identifier}"

    logger.info(
        "xprof initialized. Profiling output path set to: %s",
        self._gcs_profile_dir,
    )

    self._initialized = True

  def _should_profile(self):
    if self._process_index_list is None or (
        self._resolved_run is not None
        and host_utils.get_process_index(self._resolved_run.framework)
        in self._process_index_list
    ):
      return True
    return False

  def _report_profiler_session(
      self, create_new_session: bool, context_msg: str
  ) -> None:
    """Reports the profiler session to the Control Plane."""
    if self._resolved_run and self._resolved_run.environment == "prod":
      return

    if (
        self._start_time is None
        or self._session_phase is None
    ):
      logger.error(
          "Profiler session not set start time or session phase,"
          " skipping reporting."
      )
      return

    try:
      logger.info(
          "Scheduling programmatic profiler session report in background"
          " for %r",
          self._current_session_id,
      )
      global_manager.GlobalRunManager.get_instance().create_or_update_profiler_session(
          create_new_session=create_new_session,
          session_id=self._current_session_id,  # pyrefly: ignore[bad-argument-type]
          start_time=self._start_time,
          end_time=self._end_time,
          session_phase=self._session_phase,
          context_msg=context_msg,
      )
    except Exception:  # pylint: disable=broad-exception-caught
      logger.exception(
          "Failed to report programmatic profiler session for %r %r state %r",
          self._current_session_id,
          context_msg,
          self._session_phase,
      )

  def start(self, session_id: str | None = None) -> None:
    """Starts the JAX profiler.

    Args:
        session_id: The session ID to use for the profiling session. If None,
          use the current timestamp.
    """
    # Ensure initialization happens before starting
    self._ensure_initialized()

    if self._is_profiling:
      logger.warning("Profiling is already active. Call stop() first.")
      return

    if not self._should_profile():
      logger.info("profiling_status: skipped")
      return

    logger.info("Starting JAX profiling to: %s", self._gcs_profile_dir)
    try:
      options = jax.profiler.ProfileOptions()
      self._current_session_id = host_utils.effective_session_id(session_id)
      options.session_id = self._current_session_id
      self._start_time = time.time()
      self._end_time = None
      jax.profiler.start_trace(self._gcs_profile_dir, profiler_options=options)  # pyrefly: ignore[bad-argument-type]
      self._session_phase = "ACTIVE"
      self._is_profiling = True

      logger.info("profiling_status: started")
    except exceptions.ProfilingError as e:
      self._session_phase = "FAILED"
      logger.error("Error starting JAX profiler: %s", e)
      self._is_profiling = False

    self._report_profiler_session(
        create_new_session=True, context_msg="on_start"
    )

  def stop(self):
    """Stops the JAX profiler."""
    if not self._is_profiling:
      logger.warning("No active profiling session to stop.")
      return

    logger.info("Stopping JAX profiling for: %s", self._gcs_profile_dir)
    try:
      self._end_time = time.time()
      jax.profiler.stop_trace()
      self._session_phase = "SUCCEEDED"
      self._is_profiling = False
      logger.info("profiling_status: stopped")
      logger.info(
          "profiling traces should be available at: %s", self._gcs_profile_dir
      )
    except exceptions.ProfilingError as e:
      self._session_phase = "FAILED"
      logger.error("Error stopping JAX profiler: %s", e)

    self._report_profiler_session(
        create_new_session=False, context_msg="on_stop"
    )

  def __enter__(self):
    """Context manager entry point."""
    # Ensure initialization happens before entering context
    self._ensure_initialized()
    if not self._should_profile():
      logger.info("profiling_status: skipped")
      return self

    self._current_session_id = host_utils.effective_session_id(None)

    options = jax.profiler.ProfileOptions()
    options.session_id = self._current_session_id

    self._trace_context_manager = jax.profiler.trace(
        self._gcs_profile_dir, profiler_options=options  # pyrefly: ignore[bad-argument-type]
    )
    logger.info("Entering xprof context for: %s", self._gcs_profile_dir)
    try:
      self._start_time = time.time()
      self._end_time = None
      self._trace_context_manager.__enter__()
      self._session_phase = "ACTIVE"
      self._is_profiling = True
      logger.info("profiling_status: context_started")
    except exceptions.ProfilingError as e:
      self._session_phase = "FAILED"
      logger.error("Error starting JAX profiler in context manager: %s", e)
      self._is_profiling = False

    self._report_profiler_session(
        create_new_session=True, context_msg="on_context_enter"
    )

    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    """Context manager exit point."""
    if self._is_profiling:
      logger.info("Exiting xprof context for: %s", self._gcs_profile_dir)
      self._end_time = time.time()
      self._trace_context_manager.__exit__(exc_type, exc_val, exc_tb)
      if exc_type is None:
        self._session_phase = "SUCCEEDED"
      else:
        self._session_phase = "FAILED"
      self._is_profiling = False
      logger.info("profiling_status: context_stopped")
      logger.info(
          "profiling traces should be available at: %s",
          self._gcs_profile_dir,
      )

      self._report_profiler_session(
          create_new_session=False, context_msg="on_context_exit"
      )

  def __call__(self, func):
    """Decorator for profiling a function."""

    def wrapper(*args, **kwargs):
      # Ensure initialization happens when the decorated function is called,
      # not when the decorator is applied
      self._ensure_initialized()

      logger.info(
          "Profiling function '%s' with xprof decorator.", func.__name__
      )
      self.start()
      try:
        result = func(*args, **kwargs)
      finally:
        self.stop()
      return result

    return wrapper


# Wrappers for on-demand xprof profiling
class _OnDemandXprofManager:
  """Manages the state of the on-demand xprof server to ensure thread safety."""

  def __init__(self):
    self._started = False
    self._lock = threading.Lock()

  def start(self, port: int = 9999):
    """Starts the on-demand xprof server if not already running."""
    with self._lock:
      if not self._started:
        logger.info(
            "Starting on-demand xprof profiling session on port %s.", port
        )
        jax.profiler.start_server(port)
        self._started = True
        logger.info("On-demand xprof profiling session started.")
      else:
        logger.warning("On-demand xprof profiling session already started.")

  def stop(self):
    """Stops the on-demand xprof server if running."""
    with self._lock:
      if self._started:
        logger.info("Stopping on-demand xprof profiling session.")
        jax.profiler.stop_server()
        self._started = False
        logger.info("On-demand xprof profiling session stopped.")


_ondemand_xprof_manager = _OnDemandXprofManager()


def start_on_demand_xprof(port):
  """Starts an xprofz to allow on-demand profiling.

  Args:
    port: The port to start the on-demand xprof profiling session on. Default is
      9999.
  """
  _ondemand_xprof_manager.start(port)


def stop_on_demand_xprof():
  """Stops an xprofz to allow on-demand profiling."""
  _ondemand_xprof_manager.stop()
