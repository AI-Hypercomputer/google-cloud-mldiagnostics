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

"""Profiling SDK wrapper for Google Cloud ML Diagnostics."""

import abc
import logging
import threading
import time
import typing

from google_cloud_mldiagnostics.core import global_manager
from google_cloud_mldiagnostics.custom_types import exceptions
from google_cloud_mldiagnostics.custom_types import mlrun_types
from google_cloud_mldiagnostics.utils import host_utils


logger = logging.getLogger(__name__)


# ==============================================================================
# Profiler Engine Strategy / Factory Pattern
# ==============================================================================


class BaseProfilerEngine(abc.ABC):
  """Abstract interface for framework-specific profiler engines."""

  # --- Programmatic Profiling Methods ---
  @abc.abstractmethod
  def start_trace(self, gcs_dir: str, session_id: str) -> typing.Any:
    """Starts programmatic trace profiling and returns profiler handle."""
    pass

  @abc.abstractmethod
  def stop_trace(self, handle: typing.Any, gcs_dir: str = "") -> None:
    """Stops programmatic trace profiling."""
    pass

  @abc.abstractmethod
  def get_trace_context(
      self, gcs_dir: str, session_id: str
  ) -> typing.ContextManager[typing.Any]:
    """Returns framework context manager instance for trace profiling."""
    pass

  # --- On-Demand Profiling Server Methods ---
  @abc.abstractmethod
  def start_server(self, port: int) -> None:
    """Starts the on-demand profiler server on the specified port."""
    pass

  @abc.abstractmethod
  def stop_server(self) -> None:
    """Stops the on-demand profiler server."""
    pass


class JaxProfilerEngine(BaseProfilerEngine):
  """Profiler engine implementation for JAX framework."""

  def start_trace(self, gcs_dir: str, session_id: str) -> typing.Any:
    import jax  # pylint: disable=g-import-not-at-top

    try:
      options = jax.profiler.ProfileOptions()
      options.session_id = session_id
      jax.profiler.start_trace(gcs_dir, profiler_options=options)  # pyrefly: ignore[bad-argument-type]
      return None
    except exceptions.ProfilingError as e:
      logger.error("Error starting JAX profiler: %s", e)
      raise

  def stop_trace(self, handle: typing.Any, gcs_dir: str = "") -> None:
    import jax  # pylint: disable=g-import-not-at-top

    try:
      jax.profiler.stop_trace()
    except exceptions.ProfilingError as e:
      logger.error("Error stopping JAX profiler: %s", e)
      raise

  def get_trace_context(
      self, gcs_dir: str, session_id: str
  ) -> typing.ContextManager[typing.Any]:
    import jax  # pylint: disable=g-import-not-at-top

    options = jax.profiler.ProfileOptions()
    options.session_id = session_id
    return jax.profiler.trace(gcs_dir, profiler_options=options)  # pyrefly: ignore[bad-argument-type]

  def start_server(self, port: int) -> None:
    logger.info("Defaulting to JAX framework for on-demand profiling.")
    import jax  # pylint: disable=g-import-not-at-top

    jax.profiler.start_server(port)

  def stop_server(self) -> None:
    import jax  # pylint: disable=g-import-not-at-top

    jax.profiler.stop_server()


class PyTorchProfilerEngine(BaseProfilerEngine):
  """Profiler engine implementation for PyTorch framework."""

  def start_trace(self, gcs_dir: str, session_id: str) -> typing.Any:
    import torch  # pylint: disable=g-import-not-at-top
    from torch_tpu._internal.profiler import TpuProfilerConfig  # pylint: disable=g-import-not-at-top # pytype: disable=import-error

    config = TpuProfilerConfig(
        host_tracer_level=2,
        device_tracer_level=1,
        run_dir=gcs_dir,
    )
    prof = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.PrivateUse1,
        ],
        experimental_config=config,
    )
    prof.start()
    return prof

  def stop_trace(self, handle: typing.Any, gcs_dir: str = "") -> None:
    if handle is not None:
      handle.stop()
    else:
      logger.warning("No active PyTorch profiler instance found to stop.")

  def get_trace_context(
      self, gcs_dir: str, session_id: str
  ) -> typing.ContextManager[typing.Any]:
    import torch  # pylint: disable=g-import-not-at-top
    from torch_tpu._internal.profiler import TpuProfilerConfig  # pylint: disable=g-import-not-at-top # pytype: disable=import-error

    config = TpuProfilerConfig(
        host_tracer_level=2,
        device_tracer_level=1,
        run_dir=gcs_dir,
    )
    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.PrivateUse1,
        ],
        experimental_config=config,
    )

  def start_server(self, port: int) -> None:
    logger.info("Detected PyTorch framework for on-demand profiling.")
    from torch_tpu._internal.profiler import _impl as profiler  # type: ignore  # pylint: disable=g-import-not-at-top # pytype: disable=import-error

    profiler.start_server(port)
    logger.info(
        "Started PyTorch TPU on-demand profiler server on port %d.", port
    )

  def stop_server(self) -> None:
    from torch_tpu._internal.profiler import _impl as profiler  # type: ignore  # pylint: disable=g-import-not-at-top # pytype: disable=import-error

    profiler.stop_server()
    logger.info("Stopped PyTorch TPU on-demand profiler server.")


def get_profiler_engine(
    framework: mlrun_types.Framework | None,
) -> BaseProfilerEngine:
  """Returns the appropriate profiler engine based on framework."""
  if framework == mlrun_types.Framework.PYTORCH:
    logger.info("Using PyTorchProfilerEngine for profiling.")
    return PyTorchProfilerEngine()
  logger.info("Using JaxProfilerEngine for profiling.")
  return JaxProfilerEngine()


# ==============================================================================
# Main Xprof Wrapper Class
# ==============================================================================

class Xprof:
  """Wrapper for profiling with Google Cloud ML Diagnostics.

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
    self._profiler_handle = None
    self._trace_context_manager = None
    self._engine = None

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
    self._engine = get_profiler_engine(self._resolved_run.framework)

    if self._engine is None:
      raise exceptions.ProfilingError("Failed to initialize profiling engine.")

    logger.info(
        "xprof initialized. Profiling output path set to: %s",
        self._gcs_profile_dir,
    )

    self._initialized = True

  def _should_profile(self):
    """Determines if profiling should be enabled based on the run and host info."""
    if (
        self._process_index_list is None
        or (
            self._resolved_run is not None
            and host_utils.get_process_index(self._resolved_run.framework)
            in self._process_index_list
        )
    ) and (
        self._resolved_run is not None
        and not self._resolved_run.metric_only_run
    ):
      return True
    return False

  def _report_profiler_session(
      self, create_new_session: bool, context_msg: str
  ) -> None:
    """Reports the profiler session to the Control Plane."""
    if self._resolved_run and self._resolved_run.environment == "prod":
      return

    if self._start_time is None or self._session_phase is None:
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

  def _update_session_state(
      self,
      session_phase: str,
      log_msg: str | None = None,
  ) -> None:
    """Updates profiler session state, timestamps, and optional status log."""
    self._session_phase = session_phase
    self._is_profiling = session_phase == "ACTIVE"
    if self._is_profiling:
      if self._start_time is None:
        self._start_time = time.time()
      self._end_time = None
    else:
      self._end_time = time.time()
    if log_msg:
      logger.info(
          "profiling_status: %s (session_phase: %s)" % (log_msg, session_phase)
      )

  def start(self, session_id: str | None = None) -> None:
    """Starts the profiler.

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

    if self._engine is None or self._gcs_profile_dir is None:
      raise exceptions.ProfilingError("Profiler is not properly initialized.")

    self._start_time = time.time()
    self._end_time = None
    try:
      self._current_session_id = host_utils.effective_session_id(session_id)
      self._profiler_handle = self._engine.start_trace(
          self._gcs_profile_dir, self._current_session_id
      )
      self._update_session_state(
          session_phase="ACTIVE",
          log_msg="started",
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.error("Error starting JAX profiler: %s", e)
      self._update_session_state(
          session_phase="FAILED",
      )
      self._report_profiler_session(
          create_new_session=True, context_msg="on_start"
      )
      return

    self._report_profiler_session(
        create_new_session=True, context_msg="on_start"
    )

  def stop(self):
    """Stops the profiler."""
    if not self._is_profiling:
      logger.warning("No active profiling session to stop.")
      return

    if self._engine is None or self._gcs_profile_dir is None:
      logger.warning("Profiler engine or GCS directory is not initialized.")
      return

    try:
      self._engine.stop_trace(self._profiler_handle, self._gcs_profile_dir)
      self._update_session_state(
          session_phase="SUCCEEDED",
          log_msg="stopped",
      )
      logger.info(
          "profiling traces should be available at: %s", self._gcs_profile_dir
      )
    except Exception:  # pylint: disable=broad-exception-caught
      self._update_session_state(
          session_phase="FAILED",
      )

    self._report_profiler_session(
        create_new_session=False, context_msg="on_stop"
    )

  def __enter__(self):
    """Context manager entry point."""
    self._ensure_initialized()
    if not self._should_profile():
      logger.info("profiling_status: skipped")
      return self

    if self._engine is None or self._gcs_profile_dir is None:
      raise exceptions.ProfilingError("Profiler is not properly initialized.")

    self._current_session_id = host_utils.effective_session_id(None)
    self._start_time = time.time()
    self._end_time = None

    try:
      self._trace_context_manager = self._engine.get_trace_context(
          self._gcs_profile_dir, self._current_session_id
      )
      self._trace_context_manager.__enter__()
      self._update_session_state(
          session_phase="ACTIVE",
          log_msg="context_started",
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.error("Error starting JAX profiler in context manager: %s", e)
      self._update_session_state(
          session_phase="FAILED",
      )
      self._report_profiler_session(
          create_new_session=True, context_msg="on_context_enter"
      )
      return self

    self._report_profiler_session(
        create_new_session=True, context_msg="on_context_enter"
    )

    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    """Context manager exit point."""
    if self._is_profiling:
      if self._trace_context_manager is not None:
        self._trace_context_manager.__exit__(exc_type, exc_val, exc_tb)

      phase = "SUCCEEDED" if exc_type is None else "FAILED"
      self._update_session_state(
          session_phase=phase,
          log_msg="context_stopped",
      )
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


# ==============================================================================
# Wrappers for On-Demand Xprof Profiling Server
# ==============================================================================

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

        current_run = global_manager.get_current_run()
        framework = current_run.framework if current_run else None

        engine = get_profiler_engine(framework)
        engine.start_server(port)

        self._started = True
        logger.info("On-demand xprof profiling session started.")
      else:
        logger.warning("On-demand xprof profiling session already started.")

  def stop(self):
    """Stops the on-demand xprof server if running."""
    with self._lock:
      if self._started:
        logger.info("Stopping on-demand xprof profiling session.")

        current_run = global_manager.get_current_run()
        framework = current_run.framework if current_run else None

        engine = get_profiler_engine(framework)
        engine.stop_server()

        self._started = False
        logger.info("On-demand xprof profiling session stopped.")


_ondemand_xprof_manager = _OnDemandXprofManager()


def start_on_demand_xprof(port: int = 9999):
  """Starts an xprofz to allow on-demand profiling."""
  _ondemand_xprof_manager.start(port)


def stop_on_demand_xprof():
  """Stops an xprofz to allow on-demand profiling."""
  _ondemand_xprof_manager.stop()
