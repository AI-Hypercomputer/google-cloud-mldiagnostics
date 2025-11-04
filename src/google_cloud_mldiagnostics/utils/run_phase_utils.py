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
"""

import atexit
import logging
import sys
import threading
from typing import Callable

from google_cloud_mldiagnostics.core import global_manager
from google_cloud_mldiagnostics.core import xprof
from google_cloud_mldiagnostics.custom_types import exceptions
from google_cloud_mldiagnostics.custom_types import mlrun_types
from google_cloud_mldiagnostics.utils import host_utils


logger = logging.getLogger(__name__)


class RunPhaseMonitor:
  """Encapsulates monitoring state and logic."""
  _cleanup_handlers: list[Callable[[], None]] = [xprof.stop_on_demand_xprof]
  _cleanup_handlers_lock = threading.Lock()

  @classmethod
  def register_cleanup_handler(cls, handler: Callable[[], None]):
    """Registers a function to be called on exit."""
    with cls._cleanup_handlers_lock:
      cls._cleanup_handlers.append(handler)

  def __init__(self):
    self._monitoring_started = False
    self._original_excepthook = sys.excepthook
    self._is_master_host = host_utils.is_master_host()
    self._lock = threading.Lock()
    self._manager = global_manager.get_global_run_manager()
    if not self._manager.has_active_run():
      raise exceptions.NoActiveRunError(
          "Internal error: Active ML run is notfound."
      )
    self._control_plane_client = self._manager.control_plane_client
    if self._is_master_host and self._control_plane_client is None:
      raise exceptions.NoActiveRunError(
          "Internal error: Control plane client is None on the master host."
      )

  def _handle_unhandled_exception(self, exc_type, exc_val, exc_tb):
    """Custom exception hook to send 'FAILED' signal."""
    with self._lock:
      if self._monitoring_started:
        logger.error(
            "Unhandled exception detected!",
            exc_info=(exc_type, exc_val, exc_tb),
        )
        self.exit_cleanup()
        self.update_ml_run_with_phase(mlrun_types.RunPhase.PHASE_FAILED)
        if self._original_excepthook:
          self._original_excepthook(exc_type, exc_val, exc_tb)

  def _on_normal_exit(self):
    """atexit handler to send 'COMPLETED' signal."""
    with self._lock:
      if self._monitoring_started:
        logger.info("Program exiting normally. Sending 'COMPLETED' signal.")
        self.exit_cleanup()
        self.update_ml_run_with_phase(mlrun_types.RunPhase.PHASE_COMPLETED)

  def update_ml_run_with_phase(self, run_phase: mlrun_types.RunPhase):
    """Sends a signal to the control plane if on master host."""
    if (
        self._is_master_host
        and self._control_plane_client
        and self._manager.run
    ):
      logger.info("Sending '%s' signal to control plane.", run_phase)
      self._control_plane_client.update_ml_run(
          name=self._manager.run.name, run_phase=run_phase.value
      )

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
      logger.info("Run phase monitoring is active.")

  def exit_cleanup(self):
    """Cleans up background threads and servers started during the MLRun.

    This function is called on exit to ensure graceful termination of
    any background processes, such as the on-demand Xprof server.
    """
    with RunPhaseMonitor._cleanup_handlers_lock:
      for handler in RunPhaseMonitor._cleanup_handlers:
        try:
          handler()
        except Exception as e:  # pylint: disable=broad-except-catching
          logger.exception(
              "Exception during cleanup handler execution: %s.", e
          )
