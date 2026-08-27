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
from google_cloud_mldiagnostics.custom_types import metric_types
import torch

_logger = logging.getLogger(__name__)


def get_torch_process_index() -> int | None:
  """Returns the process index (rank) for PyTorch."""
  # If distributed is initialized, use it.
  if torch.distributed.is_initialized():
    return torch.distributed.get_rank()

  return None


def get_accelerator_type() -> metric_types.AcceleratorType:
  """Returns the accelerator type by inspecting PyTorch devices."""
  if hasattr(torch, "tpu") and torch.tpu.is_available():
    return metric_types.AcceleratorType.TPU

  return metric_types.AcceleratorType.CPU
