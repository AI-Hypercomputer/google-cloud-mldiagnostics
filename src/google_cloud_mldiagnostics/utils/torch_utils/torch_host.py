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

"""Utility functions for PyTorch host-related checks."""

import logging
import os
from google_cloud_mldiagnostics.custom_types import metric_types
import torch

_logger = logging.getLogger(__name__)


def get_torch_process_index() -> int | None:
  """Returns the process index (rank) for PyTorch."""
  # If distributed is initialized, use it.
  if torch.distributed.is_initialized():
    return torch.distributed.get_rank()

  return None


def get_torch_local_process_index() -> int:
  """Returns the host-local process index for PyTorch (defaults to 0)."""
  val = os.environ.get("LOCAL_RANK")
  if val is not None:
    try:
      return int(val)
    except ValueError:
      _logger.warning("Invalid integer value for LOCAL_RANK: %r", val)
  return 0


def is_torch_local_master_process() -> bool:
  """Checks if the current process is the local master (local rank 0) on this host."""
  return get_torch_local_process_index() == 0


def get_torch_local_world_size() -> int:
  """Returns the number of processes on this host for PyTorch (defaults to 1)."""
  val = os.environ.get("LOCAL_WORLD_SIZE")
  if val is not None:
    try:
      return int(val)
    except ValueError:
      _logger.warning("Invalid integer value for LOCAL_WORLD_SIZE: %r", val)
  return 1


def get_accelerator_type() -> metric_types.AcceleratorType:
  """Returns the accelerator type by inspecting PyTorch devices."""
  if (
      hasattr(torch, "tpu")
      and torch.accelerator.is_available()
      and torch.accelerator.current_accelerator().type == "tpu"  # pyrefly: ignore[missing-attribute]
  ):
    return metric_types.AcceleratorType.TPU

  return metric_types.AcceleratorType.CPU
