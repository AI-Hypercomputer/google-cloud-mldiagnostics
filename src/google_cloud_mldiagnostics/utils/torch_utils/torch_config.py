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

"""Utility functions for configurations in PyTorch framework."""

import os

from google_cloud_mldiagnostics.custom_types import metric_types
from google_cloud_mldiagnostics.utils.torch_utils import torch_host
import torch


# PyTorch Software Configs.
def torch_version() -> str:
  """Returns the PyTorch version."""
  return torch.__version__


# PyTorch Hardware Configs.
class TorchHardwareConfig:
  """A class to hold and query PyTorch device configuration."""

  def __init__(self):
    # Query accelerator type directly from torch_host single source of truth
    self._accelerator_type = torch_host.get_accelerator_type()
    self._is_tpu = self._accelerator_type == metric_types.AcceleratorType.TPU

  @property
  def _device_type(self) -> str:
    """The device type used for ML workload."""
    if self._is_tpu:
      try:
        return torch.tpu.get_device_name()
      except Exception:  # pylint: disable=broad-exception-caught
        return "TPU"
    return "CPU"

  @property
  def _num_slices(self) -> int:
    """The number of TPU slices for ML workload.

    Checks:
    - TPU_NUM_SLICES: Exported by Google Cloud TPU VM standalone runtime.
    - SLICE_COUNT: Exported by GKE JobSet / xpk multi-slice orchestrator.
    """
    if self._is_tpu:
      # TPU_NUM_SLICES: Cloud TPU VM runtime variable.
      # SLICE_COUNT: GKE JobSet / xpk multi-slice replica count variable.
      for env_var in ("TPU_NUM_SLICES", "SLICE_COUNT"):
        val = os.environ.get(env_var)
        if val is not None:
          try:
            return int(val)
          except ValueError:
            pass
      return 1
    return 1

  @property
  def _devices_per_slice(self) -> int:
    """Number of devices per TPU slice (or local devices) for ML workload."""
    if self._is_tpu:
      # 1. Distributed mode (Single-Host or Multi-Host): compute total devices
      # per slice
      if (
          torch.distributed.is_available()
          and torch.distributed.is_initialized()
      ):
        return torch.distributed.get_world_size() // self._num_slices

      # 2. Single-Host Non-Distributed mode: fallback to local device_count()
      try:
        return torch.tpu.device_count()
      except Exception:  # pylint: disable=broad-exception-caught
        return 0
    return 1

  def get_config(self) -> dict[str, str]:
    """Returns the default configuration for PyTorch framework."""
    return {
        "device_type": self._device_type,
        "num_slices": str(self._num_slices),
        "devices_per_slice": str(self._devices_per_slice),
        "accelerator_type": self._accelerator_type.value,
    }
