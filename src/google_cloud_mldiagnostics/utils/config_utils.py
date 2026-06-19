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

from collections.abc import Mapping
import logging
import os
from typing import Any

from google_cloud_mldiagnostics.custom_types import metric_types
from google_cloud_mldiagnostics.custom_types import mlrun_types

_config_instance = None
_jax_config_module_cache = None
_libtpu_metric_module_cache = None


def _import_jax_config_module():
  """Lazy load jax_config module and cache result."""
  global _jax_config_module_cache
  if _jax_config_module_cache is not None:
    return _jax_config_module_cache

  from google_cloud_mldiagnostics.utils.jax_utils import jax_config  # pylint: disable=g-import-not-at-top

  _jax_config_module_cache = jax_config
  return _jax_config_module_cache


def _get_libtpu_version(
    serving_engine: mlrun_types.ServingEngine = mlrun_types.ServingEngine.NONE,
):
  """Lazy load libtpu_metric module and cache result."""
  if serving_engine != mlrun_types.ServingEngine.NONE:
    return "n/a"
  global _libtpu_metric_module_cache
  if _libtpu_metric_module_cache is not None:
    return _libtpu_metric_module_cache.get_libtpu_version()

  from google_cloud_mldiagnostics.utils.libtpu_utils import libtpu_metric  # pylint: disable=g-import-not-at-top

  _libtpu_metric_module_cache = libtpu_metric
  return _libtpu_metric_module_cache.get_libtpu_version()


def _get_framework_version(
    framework: mlrun_types.Framework,
    serving_engine: mlrun_types.ServingEngine = mlrun_types.ServingEngine.NONE,
) -> str:
  """Returns the framework version used for ML workload."""
  if serving_engine != mlrun_types.ServingEngine.NONE:
    return "unknown"
  if framework == mlrun_types.Framework.JAX:
    return _import_jax_config_module().jax_version()
  else:
    return "unknown"


def _get_xla_flags() -> str:
  """Returns the XLA flags used for ML workload."""
  return os.environ.get("XLA_FLAGS", "default")


def get_software_config(
    framework: mlrun_types.Framework = mlrun_types.Framework.JAX,
    serving_engine: mlrun_types.ServingEngine = mlrun_types.ServingEngine.NONE,
) -> dict[str, str]:
  """Returns the software configuration for ML workload."""
  framework_val = framework.value
  if serving_engine != mlrun_types.ServingEngine.NONE:
    framework_val = serving_engine.value
  return {
      "framework": framework_val,
      "framework_version": _get_framework_version(framework, serving_engine),
      "xla_flags": _get_xla_flags(),
      "libtpu_version": _get_libtpu_version(serving_engine),
  }


# Hardware configs.
def _get_framework_config_instance(
    framework: mlrun_types.Framework = mlrun_types.Framework.JAX,
    serving_engine: mlrun_types.ServingEngine = mlrun_types.ServingEngine.NONE,
):
  """Initializes and returns a framework-specific config object.

  The framework-specific config object is used for querying hardware
  configuration. Currently only JAX is supported. If the framework is not JAX,
  it will issue a warning.

  Args:
    framework: The framework to get the config instance for.
    serving_engine: The serving engine to check.

  Returns:
    A framework-specific config object instance, or None if not supported.
  """
  global _config_instance
  if _config_instance is None:
    if serving_engine != mlrun_types.ServingEngine.NONE:
      return None
    if framework == mlrun_types.Framework.JAX:
      _config_instance = _import_jax_config_module().JaxHardwareConfig()
    else:
      logging.warning(
          "Hardware configuration for framework '%s' is not supported.",
          framework,
      )
  return _config_instance


def get_hardware_config(
    framework: mlrun_types.Framework = mlrun_types.Framework.JAX,
    serving_engine: mlrun_types.ServingEngine = mlrun_types.ServingEngine.NONE,
) -> dict[str, str]:
  """Returns the hardware configuration for ML workload."""
  config_instance = _get_framework_config_instance(framework, serving_engine)
  if config_instance:
    framework_specific_config = config_instance.get_config()
  else:
    framework_specific_config = {}
  hardware_config = {}
  framework_required_keys = [
      "device_type",
      "num_slices",
      "devices_per_slice",
      "accelerator_type",
  ]
  for key in framework_required_keys:
    if key not in framework_specific_config:
      hardware_config[key] = "unknown"
    else:
      hardware_config[key] = framework_specific_config[key]
  return hardware_config


def get_accelerator_type(
    framework: mlrun_types.Framework = mlrun_types.Framework.JAX,
    serving_engine: mlrun_types.ServingEngine = mlrun_types.ServingEngine.NONE,
) -> str:
  """Returns the accelerator type (tpu or gpu) for ML workload."""
  config_instance = _get_framework_config_instance(framework, serving_engine)
  if config_instance and hasattr(config_instance, "accelerator_type"):
    return config_instance.accelerator_type

  from . import host_utils  # pylint: disable=g-import-not-at-top
  detected = host_utils.get_accelerator_type(framework, serving_engine)
  if detected != metric_types.AcceleratorType.UNKNOWN:
    return detected.value

  return metric_types.AcceleratorType.TPU.value


# Common functions.
def sanitize_config(config: Mapping[str, Any]) -> dict[str, str]:
  """Converts all values in a dictionary to strings."""
  return {str(k): str(v) for k, v in config.items()}
