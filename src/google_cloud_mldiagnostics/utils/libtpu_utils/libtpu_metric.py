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

"""Utility functions for working with libtpu metrics."""

from importlib import metadata
import logging

logger = logging.getLogger(__name__)

libtpu_sdk = None
_monitoring_module = None
_initialized = False


def _initialize():
  """Initializes libtpu sdk and monitoring module."""
  global libtpu_sdk, _monitoring_module, _initialized
  if _initialized:
    return

  _initialized = True
  try:
    # pylint: disable=g-import-not-at-top
    from libtpu import sdk as libtpu_sdk_imported  # pytype: disable=import-error

    libtpu_sdk = libtpu_sdk_imported
    if hasattr(libtpu_sdk, "tpumonitoring"):
      _monitoring_module = libtpu_sdk.tpumonitoring
    elif hasattr(libtpu_sdk, "monitoring"):
      _monitoring_module = libtpu_sdk.monitoring
    else:
      _monitoring_module = None
  except ImportError:
    libtpu_sdk = None
    _monitoring_module = None
    logger.warning(
        "LibTPU metrics are not available. Please make sure libtpu is"
        " installed."
    )


def get_libtpu_version() -> str:
  """Returns libtpu version if available, otherwise 'n/a'."""
  if not _initialized:
    _initialize()
  if not libtpu_sdk:
    return "n/a"
  try:
    return metadata.version("libtpu")
  except metadata.PackageNotFoundError:
    try:
      return metadata.version("libtpu-nightly")
    except metadata.PackageNotFoundError:
      return "n/a"


def get_tpu_duty_cycle() -> list[float] | None:
  """Returns the TPU duty cycle from libtpu sdk."""
  if not _initialized:
    _initialize()
  if not _monitoring_module:
    return None
  try:
    tpu_duty_cycle_str = _monitoring_module.get_metric("duty_cycle_pct").data()
    tpu_duty_cycle = [float(value) for value in tpu_duty_cycle_str]
    return tpu_duty_cycle
  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.warning("Failed to get TPU duty cycle: %s", e)
    return None


def get_tpu_tensorcore_utilization() -> list[float] | None:
  """Returns the TPU tensorcore utilization from libtpu sdk."""
  if not _initialized:
    _initialize()
  if not _monitoring_module:
    return None
  try:
    tensorcore_util_str = _monitoring_module.get_metric(
        "tensorcore_util"
    ).data()
    tensorcore_util = [float(value) for value in tensorcore_util_str]
    return tensorcore_util
  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.warning("Failed to get TPU tensorcore utilization: %s", e)
    return None


def get_hbm_utilization() -> list[float] | None:
  """Returns the HBM utilization from libtpu sdk."""
  if not _initialized:
    _initialize()
  if not _monitoring_module:
    return None
  try:
    hbm_capacity_usage = _monitoring_module.get_metric(
        "hbm_capacity_usage"
    ).data()
    hbm_capacity_total = _monitoring_module.get_metric(
        "hbm_capacity_total"
    ).data()
    hbm_utilization = [
        int(usage) / int(total) * 100.0 if int(total) > 0 else 0.0
        for usage, total in zip(hbm_capacity_usage, hbm_capacity_total)
    ]
    return hbm_utilization
  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.warning("Failed to get HBM utilization: %s", e)
    return None
