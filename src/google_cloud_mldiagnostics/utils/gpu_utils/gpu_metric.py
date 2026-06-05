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

"""Utility functions for working with GPU metrics using pynvml."""

import atexit
from collections.abc import Sequence
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

# NVML Hopper/Ampere Tensor Pipe Utilization Field ID.
# Reference: nvml.h field identifiers.
NVML_FI_DEV_GPU_UTIL_TENSOR = 156

pynvml = None
_initialized = False
_init_lock = threading.RLock()  # Reentrant lock for thread-safety.


def _initialize() -> None:
  """Initializes pynvml SDK thread-safely."""
  global pynvml, _initialized
  if _initialized:
    return
  with _init_lock:
    # Double-checked locking pattern to prevent race conditions.
    if _initialized:
      return
    _initialized = True
    try:
      # pylint: disable=g-import-not-at-top
      import pynvml as pynvml_imported  # pytype: disable=import-error

      pynvml = pynvml_imported
      pynvml.nvmlInit()
      logger.info("Successfully initialized PyNVML drivers.")
    except Exception:  # pylint: disable=broad-exception-caught
      pynvml = None
      logger.warning(
          "PyNVML metrics are not available. Please make sure pynvml is"
          " installed and NVIDIA driver is accessible.",
          exc_info=True,
      )


def _shutdown() -> None:
  """Cleanly shuts down PyNVML driver connection on process exit."""
  global pynvml, _initialized
  with _init_lock:
    if _initialized and pynvml:
      try:
        pynvml.nvmlShutdown()
        logger.info("Successfully shut down PyNVML driver connection.")
      except Exception:  # pylint: disable=broad-exception-caught
        logger.exception("Failed to shut down PyNVML connection.")
      finally:
        pynvml = None
        _initialized = False


# Register the shutdown handler to execute automatically on Python exit.
atexit.register(_shutdown)


def _get_single_gpu_tensorcore_utilization(handle: Any) -> float:
  """Returns Tensor Core utilization for a single GPU handle."""
  if not pynvml:
    return 0.0
  try:
    # Query direct Tensor hardware pipeline counter.
    field_values = pynvml.nvmlDeviceGetFieldValues(
        handle, [NVML_FI_DEV_GPU_UTIL_TENSOR]
    )
    field_value, = field_values
    return float(field_value.value)
  except Exception:  # pylint: disable=broad-exception-caught
    # Fallback if direct Tensor Core counter is inaccessible.
    rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
    val = getattr(rates, "tensorcore", getattr(rates, "gpu", 0.0))
    return float(val)


def _get_single_vram_utilization(handle: Any) -> float:
  """Returns VRAM utilization for a single GPU handle."""
  if not pynvml:
    return 0.0
  mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
  total = getattr(mem_info, "total", 0)
  used = getattr(mem_info, "used", 0)
  if total > 0:
    return (float(used) / float(total)) * 100.0
  return 0.0


def get_gpu_utilization() -> Sequence[float]:
  """Returns the GPU SM core utilization from pynvml."""
  if not _initialized:
    _initialize()
  if not pynvml:
    # Gracefully return simulated baseline telemetry in test environments.
    return [0.0]
  try:
    device_count = pynvml.nvmlDeviceGetCount()
    return [
        float(
            pynvml.nvmlDeviceGetUtilizationRates(
                pynvml.nvmlDeviceGetHandleByIndex(i)
            ).gpu
        )
        for i in range(device_count)
    ]
  except Exception:  # pylint: disable=broad-exception-caught
    logger.warning("Failed to get GPU utilization.", exc_info=True)
    return [0.0]


def get_gpu_tensorcore_utilization() -> Sequence[float]:
  """Returns the GPU Tensor Core utilization from NVML performance fields."""
  if not _initialized:
    _initialize()
  if not pynvml:
    # Gracefully return simulated baseline telemetry in test environments.
    return [0.0]
  try:
    device_count = pynvml.nvmlDeviceGetCount()
    return [
        _get_single_gpu_tensorcore_utilization(
            pynvml.nvmlDeviceGetHandleByIndex(i)
        )
        for i in range(device_count)
    ]
  except Exception:  # pylint: disable=broad-exception-caught
    logger.warning("Failed to get GPU tensorcore utilization.", exc_info=True)
    return [0.0]


def get_vram_utilization() -> Sequence[float]:
  """Returns the VRAM utilization from pynvml."""
  if not _initialized:
    _initialize()
  if not pynvml:
    # Gracefully return simulated baseline telemetry in test environments.
    return [0.0]
  try:
    device_count = pynvml.nvmlDeviceGetCount()
    return [
        _get_single_vram_utilization(
            pynvml.nvmlDeviceGetHandleByIndex(i)
        )
        for i in range(device_count)
    ]
  except Exception:  # pylint: disable=broad-exception-caught
    logger.warning("Failed to get VRAM utilization.", exc_info=True)
    return [0.0]

