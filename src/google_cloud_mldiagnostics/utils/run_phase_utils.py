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

"""Utilities for monitoring and reporting application status.

This module provides a RunPhaseMonitor class that hooks into `sys.excepthook`
and `atexit` to monitor application lifecycle events. When an unhandled
exception occurs or the program exits normally, it performs cleanup operations
and reports the application's final status (e.g., FAILED, COMPLETED) to a
control plane.

SIGTERM handling never overrides a framework's own handler (b/567346696):

- If JAX's preemption service is active (`jax.distributed.initialize()`), JAX
  owns SIGTERM. The SDK only watches the preemption notice that JAX's handler
  publishes to the coordination service and reports the run as FAILED.
- Otherwise, on the master host, the SDK installs a fallback SIGTERM handler
  that reports the run as FAILED and then hands SIGTERM back to whatever
  handled it before: the previous Python handler, the previous native handler
  (e.g. libtpu's crash handler, restored with its original flags and re-raised)
  or the default disposition. It is not installed if SIGTERM is ignored. If
  JAX installs its handler later, JAX's handler replaces the fallback at the
  OS level and the watcher takes over.
"""

from __future__ import annotations

import atexit
import ctypes
import logging
import os
import signal
import sys
import threading
import time
from typing import Any, Callable, Optional

from google_cloud_mldiagnostics.core import global_manager
from google_cloud_mldiagnostics.custom_types import exceptions
from google_cloud_mldiagnostics.custom_types import mlrun_types
from google_cloud_mldiagnostics.utils import host_utils

logger = logging.getLogger(__name__)

# Key that JAX's PreemptionSyncManager writes to the coordination service when
# any task receives SIGTERM. The value is the time the SIGTERM was caught. See
# xla/pjrt/distributed/preemption/preemption_sync_manager.cc.
_JAX_PREEMPTION_NOTICE_KEY = "RECEIVED_PREEMPTION_NOTICE"
_JAX_DISTRIBUTED_MODULE = "jax._src.distributed"
_JAX_PREEMPTION_POLL_INTERVAL_SEC = 2.0
# The watcher blocks on the notice key so it sees the notice as soon as JAX
# writes it (before any task reaches the preemption sync point). The wait
# returns immediately when the key is written; this timeout only bounds how
# long the watcher stays inside the native call, so that it can be joined
# quickly at exit.
_JAX_PREEMPTION_WAIT_TIMEOUT_SEC = 1.0
# How long the atexit hook and excepthook wait for the watcher thread to stop.
# The watcher must not be inside a native call (GIL released) when the
# interpreter finalizes: on CPython <= 3.13 a daemon thread re-acquiring the
# GIL during finalization is pthread_exit()-ed inside C++ frames, which aborts
# the process (gh-87135).
_JAX_PREEMPTION_WATCHER_JOIN_TIMEOUT_SEC = 3.0


def _get_jax_preemption_client() -> Optional[Any]:
  """Returns the JAX coordination client if JAX preemption sync is active.

  Returns None if JAX is not imported, `jax.distributed.initialize()` has not
  been called (or has been shut down), or the JAX preemption service is
  disabled. In those cases JAX does not install its SIGTERM handler.
  """
  jax_distributed = sys.modules.get(_JAX_DISTRIBUTED_MODULE)
  state = getattr(jax_distributed, "global_state", None)
  if getattr(state, "preemption_sync_manager", None) is None:
    return None
  return getattr(state, "client", None)


def _read_jax_preemption_notice(client: Any) -> Optional[str]:
  """Returns the JAX preemption notice (death time), or None if not set."""
  try:
    notice = client.key_value_try_get(_JAX_PREEMPTION_NOTICE_KEY)
  except Exception:  # pylint: disable=broad-exception-caught
    # NOT_FOUND until a task is preempted; also raised if the client is
    # shutting down.
    return None
  return notice or None


def _wait_for_jax_preemption_notice(client: Any) -> Optional[str]:
  """Waits for the JAX preemption notice; returns None on timeout or error."""
  try:
    notice = client.blocking_key_value_get(
        _JAX_PREEMPTION_NOTICE_KEY, int(_JAX_PREEMPTION_WAIT_TIMEOUT_SEC * 1000)
    )
  except Exception:  # pylint: disable=broad-exception-caught
    # DEADLINE_EXCEEDED until a task is preempted; also raised if the client is
    # shutting down.
    return None
  return notice or None


def _has_non_python_sigterm_handler() -> bool:
  """Returns True if a native (non-Python) SIGTERM handler is installed.

  `signal.getsignal()` reports SIG_DFL for native handlers installed after
  interpreter startup (e.g. via `std::signal`), so this reads the OS-level
  disposition through the CPython C API. It never modifies the disposition.
  Only meaningful when `signal.getsignal(SIGTERM)` is SIG_DFL.
  """
  try:
    pyos_getsig = ctypes.pythonapi.PyOS_getsig
    pyos_getsig.argtypes = [ctypes.c_int]
    pyos_getsig.restype = ctypes.c_void_p
    handler = pyos_getsig(signal.SIGTERM)
  except (AttributeError, OSError, TypeError, ValueError):
    return False
  return handler not in (None, int(signal.SIG_DFL), int(signal.SIG_IGN))


# Opaque storage for a `struct sigaction` (152 bytes on x86-64/aarch64 glibc).
# The SDK never parses it; it only round-trips it through sigaction(2) so the
# native handler is restored with its original flags (e.g. SA_SIGINFO).
_SIGACTION_BUFFER_SIZE = 512


def _sigaction(
    action: Optional[ctypes.Array[ctypes.c_char]],
    old_action: Optional[ctypes.Array[ctypes.c_char]],
) -> bool:
  """Calls libc sigaction(SIGTERM, action, old_action); returns success."""
  try:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.sigaction.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    libc.sigaction.restype = ctypes.c_int
    return libc.sigaction(signal.SIGTERM, action, old_action) == 0
  except (AttributeError, OSError, TypeError, ValueError):
    return False


def _save_sigterm_action() -> Optional[ctypes.Array[ctypes.c_char]]:
  """Returns the current OS-level SIGTERM action, or None on failure."""
  saved = ctypes.create_string_buffer(_SIGACTION_BUFFER_SIZE)
  return saved if _sigaction(None, saved) else None


def _restore_sigterm_action(saved: ctypes.Array[ctypes.c_char]) -> bool:
  """Reinstalls a SIGTERM action saved by `_save_sigterm_action`."""
  return _sigaction(saved, None)


class RunPhaseMonitor:
  """Encapsulates monitoring state and logic."""

  _cleanup_handlers: list[Callable[[], None]] = []
  _cleanup_handlers_lock = threading.Lock()
  _active_instance: Optional[RunPhaseMonitor] = None
  _active_instance_lock = threading.Lock()

  @classmethod
  def register_cleanup_handler(cls, handler: Callable[[], None]):
    """Registers a function to be called on exit."""
    with cls._cleanup_handlers_lock:
      cls._cleanup_handlers.append(handler)

  @classmethod
  def get_active_monitor(cls) -> Optional[RunPhaseMonitor]:
    """Gets the currently active RunPhaseMonitor instance."""
    with cls._active_instance_lock:
      return cls._active_instance

  def __init__(self):
    self._monitoring_started = False
    self._ml_run_failed = False
    # True once a FAILED update has actually been delivered (or no update is
    # expected from this host). `_ml_run_failed` only records the decision.
    self._failed_reported = False
    self._original_excepthook = sys.excepthook
    self._lock = threading.Lock()
    self._stop_watcher = threading.Event()
    self._watcher_thread: Optional[threading.Thread] = None
    self._prev_sigterm_handler: Any = None
    self._native_sigterm_action: Optional[ctypes.Array[ctypes.c_char]] = (
        None
    )
    self._manager = global_manager.get_global_run_manager()
    if not self._manager.has_active_run() or self._manager.run is None:
      raise exceptions.NoActiveRunError(
          "No active ML run found. Please initialize an ML run before"
          " monitoring."
      )
    self._is_master_host = host_utils.is_master_host(
        self._manager.run.framework, self._manager.run.serving_engine
    )
    self._control_plane_client = self._manager.control_plane_client
    if self._is_master_host and self._control_plane_client is None:
      raise exceptions.ControlPlaneClientNotInitializedError(
          "Required services are not initialized on the master host."
      )

  def _handle_unhandled_exception(self, exc_type, exc_val, exc_tb):
    """Custom exception hook to send 'FAILED' signal."""
    self._stop_jax_preemption_watcher()
    with self._lock:
      if self._monitoring_started:
        logger.error(
            "Unhandled exception detected!",
            exc_info=(exc_type, exc_val, exc_tb),
        )
        self.exit_cleanup()
        self.update_ml_run_with_phase(mlrun_types.RunPhase.PHASE_FAILED)
        self._ml_run_failed = True
        if self._original_excepthook:
          self._original_excepthook(exc_type, exc_val, exc_tb)

  def _on_normal_exit(self):
    """atexit handler to send 'COMPLETED' signal."""
    self._stop_jax_preemption_watcher()
    with self._lock:
      if self._monitoring_started:
        logger.info("Program exiting normally.")
        self.exit_cleanup()
        if self._ml_run_failed:
          if not self._failed_reported:
            # FAILED was decided (e.g. by the watcher) but its update failed or
            # was cut off; re-send it once.
            logger.warning("Re-sending undelivered 'FAILED' signal.")
            try:
              self.update_ml_run_with_phase(mlrun_types.RunPhase.PHASE_FAILED)
            except Exception:  # pylint: disable=broad-exception-caught
              logger.exception("Failed to re-send 'FAILED' signal.")
          return
        # Last-chance check in case the process exits right after a JAX
        # preemption notice, before the watcher thread observed it.
        client = _get_jax_preemption_client()
        notice = _read_jax_preemption_notice(client) if client else None
        if notice:
          logger.warning(
              "JAX preemption notice detected (SIGTERM at %s). Sending"
              " 'FAILED' signal instead of 'COMPLETED'.",
              notice,
          )
          self.update_ml_run_with_phase(mlrun_types.RunPhase.PHASE_FAILED)
        else:
          logger.info("Sending 'COMPLETED' signal.")
          self.update_ml_run_with_phase(mlrun_types.RunPhase.PHASE_COMPLETED)

  def _report_jax_preemption(self, notice: str) -> None:
    """Marks the run FAILED after JAX received a preemption notice."""
    with self._lock:
      if not self._monitoring_started or self._ml_run_failed:
        return
      # Mark FAILED under the lock so atexit never reports COMPLETED, but send
      # the RPC outside it so the SIGTERM and atexit hooks never wait on it.
      self._ml_run_failed = True
    logger.warning(
        "JAX preemption notice detected (SIGTERM at %s). Sending 'FAILED'"
        " signal; JAX handles checkpointing and shutdown.",
        notice,
    )
    try:
      self.update_ml_run_with_phase(mlrun_types.RunPhase.PHASE_FAILED)
    except Exception:  # pylint: disable=broad-exception-caught
      logger.exception("Failed to report JAX preemption.")

  def _watch_jax_preemption(self) -> None:
    """Waits for the JAX preemption notice until found or stopped.

    Runs on a daemon thread on the master host. It blocks on the notice key so
    the notice is seen as soon as JAX writes it, i.e. before any task reaches
    the preemption sync point and possibly calls `jax.distributed.shutdown()`.
    It only reads the notice and never calls into JAX's preemption sync
    protocol, so JAX's SIGTERM handling and checkpointing are unaffected.
    """
    while not self._stop_watcher.is_set():
      client = _get_jax_preemption_client()
      if client is None:
        # JAX distributed not (yet) initialized, or shut down.
        self._stop_watcher.wait(_JAX_PREEMPTION_POLL_INTERVAL_SEC)
        continue
      started = time.monotonic()
      notice = _wait_for_jax_preemption_notice(client)
      if notice:
        self._report_jax_preemption(notice)
        return
      if time.monotonic() - started < _JAX_PREEMPTION_WAIT_TIMEOUT_SEC / 2:
        # Failed fast rather than timing out (e.g. client shutting down).
        self._stop_watcher.wait(_JAX_PREEMPTION_POLL_INTERVAL_SEC)

  def _start_jax_preemption_watcher(self) -> None:
    """Starts the JAX preemption watcher on the master host."""
    if not self._is_master_host:
      return
    self._watcher_thread = threading.Thread(
        target=self._watch_jax_preemption,
        name="mldiagnostics-jax-preemption-watcher",
        daemon=True,
    )
    self._watcher_thread.start()

  def _stop_jax_preemption_watcher(self) -> None:
    """Stops the watcher and waits for it, before interpreter finalization.

    Must be called without holding `self._lock`: the watcher takes it in
    `_report_jax_preemption`.
    """
    self._stop_watcher.set()
    thread = self._watcher_thread
    if thread is None or thread is threading.current_thread():
      return
    thread.join(_JAX_PREEMPTION_WATCHER_JOIN_TIMEOUT_SEC)
    if thread.is_alive():
      logger.warning(
          "JAX preemption watcher did not stop within %.1fs.",
          _JAX_PREEMPTION_WATCHER_JOIN_TIMEOUT_SEC,
      )

  def _handle_sigterm(self, signum, frame) -> None:
    """Fallback SIGTERM handler used when no framework handles SIGTERM.

    Reports the run as FAILED, then hands SIGTERM back to what handled it
    before the SDK: the previous Python handler, the previous native handler
    (restored and re-raised), or the default disposition (re-raised).

    Args:
      signum: The signal number.
      frame: The current stack frame, forwarded to the previous handler.
    """
    prev_handler = self._prev_sigterm_handler
    if _get_jax_preemption_client() is not None:
      # JAX's preemption handler was installed after this one. Internal
      # (PLATFORM_GOOGLE) JAX builds before cl/991302183 register it with
      # thread::signal flags=0, which chains to this handler. JAX drives
      # checkpointing and shutdown, and the watcher reports the run, so never
      # re-raise or terminate here.
      logger.info("SIGTERM received; JAX preemption handling is active.")
      if callable(prev_handler):
        prev_handler(signum, frame)
      return
    # Signal handlers run on the main thread; never block on a lock that the
    # main thread may already hold (e.g. inside `_on_normal_exit`).
    if self._lock.acquire(blocking=False):
      try:
        if self._monitoring_started and not self._ml_run_failed:
          logger.warning(
              "SIGTERM received (signum: %s). Sending 'FAILED' signal.", signum
          )
          try:
            self.update_ml_run_with_phase(mlrun_types.RunPhase.PHASE_FAILED)
          except Exception:  # pylint: disable=broad-exception-caught
            logger.exception("Failed to report SIGTERM.")
      finally:
        self._lock.release()
    else:
      logger.warning(
          "SIGTERM received while run phase monitor is busy; not reporting."
      )
    if not callable(prev_handler):
      # The process is about to terminate without running atexit hooks.
      self._try_exit_cleanup()

    if callable(prev_handler):
      prev_handler(signum, frame)
      return
    native_action = self._native_sigterm_action
    if native_action is not None and _restore_sigterm_action(native_action):
      # The kernel delivers the re-raised signal to the native handler with its
      # original disposition and flags (e.g. SA_SIGINFO). Its siginfo reflects
      # a self-sent SIGTERM (si_code=SI_USER), not the original sender.
      os.kill(os.getpid(), signum)
      return
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    os.kill(os.getpid(), signum)

  def _maybe_install_sigterm_handler(self) -> None:
    """Installs the fallback SIGTERM handler if no framework handles SIGTERM."""
    if not self._is_master_host:
      return
    if threading.current_thread() is not threading.main_thread():
      logger.info("Not on the main thread; not installing SIGTERM handler.")
      return
    if _get_jax_preemption_client() is not None:
      logger.info(
          "JAX preemption service is active; JAX handles SIGTERM and the SDK"
          " reports preemption from its notice."
      )
      return
    prev_handler = signal.getsignal(signal.SIGTERM)
    if prev_handler is None or prev_handler == signal.SIG_IGN:
      logger.info(
          "SIGTERM is ignored or handled outside Python; not installing SIGTERM"
          " handler."
      )
      return
    native_action = None
    if prev_handler == signal.SIG_DFL and _has_non_python_sigterm_handler():
      native_action = _save_sigterm_action()
      if native_action is None:
        logger.info(
            "Native SIGTERM handler detected but could not be saved; not"
            " installing SIGTERM handler."
        )
        return
    self._prev_sigterm_handler = prev_handler
    self._native_sigterm_action = native_action
    signal.signal(signal.SIGTERM, self._handle_sigterm)
    logger.info(
        "Installed SDK SIGTERM handler (hands SIGTERM back to the %s handler).",
        "native" if native_action is not None else "previous",
    )

  def update_ml_run_with_phase(self, run_phase: mlrun_types.RunPhase):
    """Sends a signal to the control plane if on master host."""
    if run_phase == mlrun_types.RunPhase.PHASE_FAILED:
      self._ml_run_failed = True
    if (
        self._is_master_host
        and self._control_plane_client
        and self._manager.run
    ):
      logger.info("Sending '%s' signal to control plane.", run_phase)
      self._control_plane_client.update_ml_run(
          name=self._manager.run.name, force=True, run_phase=run_phase.value
      )
    if run_phase == mlrun_types.RunPhase.PHASE_FAILED:
      # Only reached if the update above succeeded (or none was due).
      self._failed_reported = True

  def start(self):
    """Registers the hooks if not already registered."""
    with self._lock:
      if self._monitoring_started:
        logger.warning("Run phase monitoring is already started.")
        return
      logger.info("Starting run phase monitoring...")
      sys.excepthook = self._handle_unhandled_exception
      atexit.register(self._on_normal_exit)
      self._monitoring_started = True
      self._ml_run_failed = False
      with RunPhaseMonitor._active_instance_lock:
        RunPhaseMonitor._active_instance = self
      self._start_jax_preemption_watcher()
      self._maybe_install_sigterm_handler()
      logger.info("Run phase monitoring is active.")

  def exit_cleanup(self):
    """Cleans up background threads and servers started during the MLRun.

    This function is called on exit to ensure graceful termination of
    any background processes, such as the on-demand Xprof server.
    """
    with RunPhaseMonitor._cleanup_handlers_lock:
      self._run_cleanup_handlers()

  def _try_exit_cleanup(self) -> None:
    """Like `exit_cleanup`, but never blocks; used from the SIGTERM handler.

    The handler runs on the main thread, which may already be inside
    `exit_cleanup` (holding the non-reentrant cleanup lock) when SIGTERM
    arrives; in that case cleanup is already in progress and is skipped here.
    """
    if not RunPhaseMonitor._cleanup_handlers_lock.acquire(blocking=False):
      logger.warning("Cleanup already in progress; skipping on SIGTERM.")
      return
    try:
      self._run_cleanup_handlers()
    finally:
      RunPhaseMonitor._cleanup_handlers_lock.release()

  def _run_cleanup_handlers(self) -> None:
    """Runs cleanup handlers; caller must hold `_cleanup_handlers_lock`."""
    for handler in RunPhaseMonitor._cleanup_handlers:
      try:
        handler()
      except Exception:  # pylint: disable=broad-exception-caught
        logger.exception("Exception during cleanup handler execution.")
