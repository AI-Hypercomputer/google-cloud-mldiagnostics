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

"""Utility functions for JAX host-related checks."""

import logging
from google_cloud_mldiagnostics.custom_types import metric_types
import jax

_logger = logging.getLogger(__name__)


def get_jax_process_index() -> int:
  """Returns the process index."""
  return jax.process_index()


def get_accelerator_type() -> metric_types.AcceleratorType:
  """Returns the accelerator type by inspecting JAX devices."""
  try:
    devices = jax.devices()
  except RuntimeError:
    _logger.warning("Failed to call jax.devices()", exc_info=True)
    return metric_types.AcceleratorType.UNKNOWN
  # We catch a broad Exception here because jax.devices() can raise various
  # exceptions depending on the environment and underlying issues. Since this
  # function is for diagnostics, failing to determine the accelerator type
  # should not crash the program. We log the error and return UNKNOWN.
  # Disabling the pylint warning is justified because gracefully handling
  # unexpected failures from jax.devices() is the primary goal.
  except Exception:  # pylint: disable=broad-exception-caught
    _logger.warning("Unexpected error calling jax.devices()", exc_info=True)
    return metric_types.AcceleratorType.UNKNOWN

  if not devices:
    return metric_types.AcceleratorType.UNKNOWN
  # For more details on how we determine the accelerator type, see
  # https://jax.readthedocs.io/en/latest/core/device-types.html.
  device = devices[0]
  platform = getattr(device, "platform", "").lower()
  device_kind = getattr(device, "device_kind", "").lower()

  if "tpu" in platform or "tpu" in device_kind:
    return metric_types.AcceleratorType.TPU
  if "gpu" in platform or "gpu" in device_kind or "cuda" in device_kind:
    return metric_types.AcceleratorType.GPU
  if "cpu" in platform or "cpu" in device_kind:
    return metric_types.AcceleratorType.CPU

  return metric_types.AcceleratorType.UNKNOWN
