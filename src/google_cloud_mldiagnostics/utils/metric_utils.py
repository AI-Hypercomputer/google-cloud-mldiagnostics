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

"""Utility functions for working with fetching metrics."""

import logging
from typing import List, Optional
from google_cloud_mldiagnostics.utils.libtpu_utils import libtpu_metric
import psutil

logger = logging.getLogger(__name__)


def get_host_cpu_utilization() -> float | None:
  """Returns the host CPU utilization from psutil."""
  try:
    return psutil.cpu_percent(interval=0.1)
  except Exception as e:  # pylint: disable=broad-except
    logger.warning("Failed to get CPUutilization: %s", e)
    return None


def get_host_memory_utilization() -> float | None:
  """Returns the host memory utilization from psutil."""
  try:
    return psutil.virtual_memory().percent
  except Exception as e:  # pylint: disable=broad-except
    logger.warning("Failed to get memory utilization: %s", e)
    return None


def get_tpu_duty_cycle() -> Optional[List[float]]:
  """Returns the TPU duty cycle from libtpu sdk."""
  return libtpu_metric.get_tpu_duty_cycle()


def get_tpu_tensorcore_utilization() -> Optional[List[float]]:
  """Returns the TPU tensorcore utilization from libtpu sdk."""
  return libtpu_metric.get_tpu_tensorcore_utilization()


def get_hbm_utilization() -> Optional[List[float]]:
  """Returns the HBM utilization from libtpu sdk."""
  return libtpu_metric.get_hbm_utilization()
