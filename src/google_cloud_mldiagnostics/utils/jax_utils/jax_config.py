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

"""Utility functions for configurations in JAX framework."""

import jax


# Jax Software Configs.
def jax_version() -> str:
  """Returns the JAX version."""
  return jax.__version__


# JAX Hardware Configs.
class JaxHardwareConfig:
  """A class to hold and query JAX device configuration."""

  def __init__(self):
    self._devices = jax.devices()
    if not self._devices:
      raise ValueError('No JAX devices found.')
    self._is_multi_slice_tpu = hasattr(self._devices[0], 'slice_index')

  @property
  def _device_type(self) -> str:
    """Returns the device type used for ML workload."""
    return self._devices[0].device_kind

  @property
  def _num_slices(self) -> int:
    """Returns the number of TPU slices for ML workload."""
    if self._is_multi_slice_tpu:
      slice_indices = set()
      for device in self._devices:
        slice_indices.add(device.slice_index)
      return len(slice_indices)
    else:
      return 1

  @property
  def _devices_per_slice(self) -> int:
    """Returns the number of devices per TPU slice for ML workload."""
    return jax.device_count() // self._num_slices

  def get_config(self) -> dict[str, str]:
    """Returns the default configuration for JAX framework."""
    return {
        'device_type': self._device_type,
        'num_slices': str(self._num_slices),
        'devices_per_slice': str(self._devices_per_slice),
    }
