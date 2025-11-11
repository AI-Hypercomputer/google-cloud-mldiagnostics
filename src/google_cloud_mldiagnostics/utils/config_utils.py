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

"""Utility functions for configurations."""

import logging
import os
from typing import Any

from google_cloud_mldiagnostics.utils.jax_utils import jax_config
from google_cloud_mldiagnostics.utils.libtpu_utils import libtpu_metric

_config_instance = None


# Software configs.
def _get_framework() -> str:
  """Returns the framework used for ML workload."""
  # TODO: [INTERNAL] - Add support for non-jax workloads.
  return "JAX"
  # The following way is not robust enough to detect the framework.
  # return os.environ.get("TPU_ML_PLATFORM", "unknown")


def _get_framework_version() -> str:
  """Returns the framework version used for ML workload."""
  # TODO: [INTERNAL] - Add support for non-jax workloads.
  if _get_framework() == "JAX":
    return jax_config.jax_version()
  else:
    return "unknown"
  # The following way is not robust enough to detect the version.
  # return os.environ.get("TPU_ML_PLATFORM_VERSION", "unknown")


def _get_xla_flags() -> str:
  """Returns the XLA flags used for ML workload."""
  return os.environ.get("XLA_FLAGS", "default")


def get_software_config() -> dict[str, str]:
  """Returns the software configuration for ML workload."""
  return {
      "framework": _get_framework(),
      "framework_version": _get_framework_version(),
      "xla_flags": _get_xla_flags(),
      "libtpu_version": libtpu_metric.get_libtpu_version(),
  }


# Hardware configs.
def _get_framework_config_instance():
  """Initializes and returns a framework-specific config object.

  The framework-specific config object is used for querying hardware
  configuration. Currently only JAX is supported. If the framework is not JAX,
  it will issue a warning.

  Returns:
    A framework-specific config object instance, or None if not supported.
  """
  framework = _get_framework()
  global _config_instance
  if _config_instance is None:
    if framework == "JAX":
      _config_instance = jax_config.JaxHardwareConfig()
    else:
      logging.warning(
          "Hardware configuration for framework '%s' is not supported.",
          framework,
      )
  return _config_instance


def get_hardware_config() -> dict[str, str]:
  """Returns the hardware configuration for ML workload."""
  config_instance = _get_framework_config_instance()
  if config_instance:
    framework_specific_config = config_instance.get_config()
  else:
    framework_specific_config = {}
  hardware_config = {}
  framework_required_keys = [
      "device_type",
      "num_slices",
      "devices_per_slice",
  ]
  for key in framework_required_keys:
    if key not in framework_specific_config:
      hardware_config[key] = "unknown"
    else:
      hardware_config[key] = framework_specific_config[key]
  return hardware_config


# Common functions.
def sanitize_config(config: dict[str, Any]) -> dict[str, str]:
  """Converts all values in a dictionary to strings."""
  return {str(k): str(v) for k, v in config.items()}
