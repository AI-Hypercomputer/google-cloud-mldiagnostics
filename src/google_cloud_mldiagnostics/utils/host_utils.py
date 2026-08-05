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

"""Utility functions for host- related checks."""

import datetime
import hashlib
import json
import logging
import os
import re
import socket
from typing import Any

from google_cloud_mldiagnostics.custom_types import metric_types
from google_cloud_mldiagnostics.custom_types import mlrun_types
from google_cloud_mldiagnostics.utils import gcp
import requests

logger = logging.getLogger(__name__)


# GKE related functions
def _get_gke_diagon_identifier() -> dict[str, Any] | None:
  """Returns GKE Diagon identifier as a dictionary if set, otherwise None."""
  diagon_identifier_str = os.environ.get("GKE_DIAGON_IDENTIFIER")
  if not diagon_identifier_str:
    logger.info("GKE_DIAGON_IDENTIFIER environment variable not set.")
    return None

  try:
    diagon_identifier = json.loads(diagon_identifier_str)
    return diagon_identifier
  except json.JSONDecodeError:
    logger.exception(
        "Failed to parse GKE_DIAGON_IDENTIFIER: %s", diagon_identifier_str
    )
    return None


def _get_gke_diagon_metadata() -> dict[str, Any] | None:
  """Returns GKE Diagon metadata as a dictionary if set, otherwise None."""
  diagon_metadata_str = os.environ.get("GKE_DIAGON_METADATA")
  if not diagon_metadata_str:
    logger.info("GKE_DIAGON_METADATA environment variable not set.")
    return None

  try:
    diagon_metadata = json.loads(diagon_metadata_str)
    return diagon_metadata
  except json.JSONDecodeError:
    logger.exception(
        "Failed to parse GKE_DIAGON_METADATA: %s", diagon_metadata_str
    )
    return None


def _get_gke_workload_details() -> dict[str, Any] | None:
  """Returns workload details if available, otherwise None."""
  identifier = _get_gke_diagon_identifier() or {}
  metadata = _get_gke_diagon_metadata() or {}

  details = {
      "id": identifier.get("metadata.name", ""),
      "kind": identifier.get("metadata.kind", ""),
      "cluster": identifier.get("clustername", ""),
      "namespace": identifier.get("namespace", ""),
      "parent_workload": metadata.get("parent-workload", None),
      "creation-timestamp": metadata.get("creation-timestamp", ""),
  }
  # Parse labels from metadata.
  labels_str = metadata.get("associated-labels", None)
  if labels_str:
    gke_labels = {}
    for pair in labels_str.split(","):
      if "=" in pair:
        key, value = pair.split("=", 1)
        gke_labels[key.strip()] = value.strip()
    details["labels"] = gke_labels
  else:
    details["labels"] = None

  if all(not v for v in details.values()):
    return None

  return details


def _get_gce_workload_details(
    run_workload_id: str | None = None,
) -> dict[str, Any] | None:
  """Returns workload details if available, otherwise None."""
  workload_id = run_workload_id or os.environ.get("RUN_WORKLOAD_ID")
  instance_id = get_instance_id()
  hostname = get_hostname()
  details = {
      "id": workload_id if workload_id else instance_id,
      "display_name": workload_id if workload_id else hostname,
      "create_time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
      "_instance_id": instance_id,
      "_is_workload_id_set": bool(workload_id),
      "_hostname": hostname,
  }
  return details


def _gce_workload_targets(
    workload_details: dict[str, Any] | None,
) -> list[dict[str, Any]] | None:
  """Returns workload targets if available, otherwise None."""
  if not workload_details:
    return None
  hostname = workload_details.get("_hostname")
  instance_id = workload_details.get("_instance_id")
  if not hostname and not instance_id:
    return None
  details = [{
      "display_name": hostname or "",
      "instance_id": instance_id,
      "hostname": hostname or "",
      "zone": gcp.get_instance_zone(),
      "state": "RUNNING",
  }]
  return details


def _format_slurm_start_time(raw_time: str | None) -> str | None:
  if not raw_time:
    return None
  if raw_time.isdigit():
    return datetime.datetime.fromtimestamp(
        int(raw_time), tz=datetime.timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
  return raw_time


def _get_slurm_workload_details() -> dict[str, Any] | None:
  """Returns workload details if available, otherwise None."""
  job_id = os.getenv("SLURM_JOB_ID")
  cluster_name = os.getenv("SLURM_CLUSTER_NAME")
  if not job_id or not cluster_name:
    return None

  cluster_path = (
      f"/projects/{gcp.get_project_id()}/locations/{gcp.get_instance_zone()}/clusters/{cluster_name}"
  )

  job_name = os.getenv("SLURM_JOB_NAME")
  display_name = job_name if job_name else f"slurm-job-{job_id}"

  return {
      "job_id": job_id,
      "cluster": cluster_path,
      "display_name": display_name,
      "submit_time": _format_slurm_start_time(
          os.getenv("SLURM_JOB_START_TIME")
      ),
  }


def _slurm_workload_targets() -> list[dict[str, Any]] | None:
  """Returns workload targets if available, otherwise None."""
  hostname = get_hostname()
  details = [{
      "display_name": hostname,
      "instance_id": get_instance_id(),
      "hostname": hostname,
      "zone": gcp.get_instance_zone(),
      "state": "RUNNING",
  }]
  return details


def _get_sha256_hash(input_string: str) -> str:
  """Calculates the SHA-256 hash of a given string."""
  encoded_string = input_string.encode("utf-8")
  sha256_hash = hashlib.sha256()
  sha256_hash.update(encoded_string)
  hex_digest = sha256_hash.hexdigest()
  logger.debug(
      "Input string: %s ML Run ID (SHA-256 hash): %s", input_string, hex_digest
  )
  return hex_digest


def _gke_run_identifier(workload_details: dict[str, Any]) -> str:
  """Returns the unique identifier for the gke workload.

  Args:
    workload_details: A dictionary containing workload details.

  Example output (SHA-256 hash of the raw string identifier):
  4d1c1bd2a83e01fcb88a8d05e326071ab8ca9ec58f4a187a27ebf9dcd87b3225

  Raw string identifier format used for hashing:
  clustername_namespace_kind_workloadid_YYYYMMDD-HHmmss
  """

  if not workload_details:
    raise ValueError(
        "Could not generate GKE workload identifier due to missing workload"
        " details. This might be because environment variables"
        " 'GKE_DIAGON_IDENTIFIER' or 'GKE_DIAGON_METADATA' are not set or are"
        " incomplete. Please ensure you are running SDK in a GKE environment"
        " with the GKE diagon operator webhook enabled."
    )

  identifier_keys = ["namespace", "cluster", "kind", "id"]
  metadata_keys = ["creation-timestamp"]
  missed_identifier_keys = []
  missed_metadata_keys = []
  for key in identifier_keys:
    if not workload_details.get(key):
      missed_identifier_keys.append(key)
  for key in metadata_keys:
    if not workload_details.get(key):
      missed_metadata_keys.append(key)

  if missed_identifier_keys or missed_metadata_keys:
    missing_keys_str = ", ".join(missed_identifier_keys + missed_metadata_keys)
    error_message = (
        "Could not generate GKE workload identifier due to missing"
        f" properties: {missing_keys_str}."
    )
    if missed_identifier_keys:
      error_message += (
          " Please check if 'GKE_DIAGON_IDENTIFIER' environment variable is"
          " set correctly."
      )
    if missed_metadata_keys:
      error_message += (
          " Please check if 'GKE_DIAGON_METADATA' environment variable is set"
          " correctly."
      )
    error_message += (
        " Ensure you are running SDK in a GKE environment with the GKE diagon"
        " operator webhook enabled."
    )
    raise ValueError(error_message)

  # Preprocess cluster name and timestamp.
  cluster = workload_details["cluster"].split("/")[-1]
  transformed_timestamp = workload_details["creation-timestamp"]
  transformed_timestamp = transformed_timestamp[:-1] + "+00:00"
  transformed_timestamp = (
      datetime.datetime.fromisoformat(transformed_timestamp)
      .astimezone(datetime.timezone.utc)
      .strftime("%Y%m%d-%H%M%S")
  )

  identifier = (
      f"{cluster}"
      f"_{workload_details['namespace']}"
      f"_{workload_details['kind']}_{workload_details['id']}"
      f"_{transformed_timestamp}"
  )
  return _get_sha256_hash(identifier)


def _gce_run_identifier(workload_details: dict[str, Any]) -> str:
  """Returns the unique identifier for the gce workload."""
  if not workload_details:
    raise ValueError(
        "Could not generate GCE workload identifier due to missing workload"
        " details."
    )
  if workload_details.get("_is_workload_id_set"):
    return _get_sha256_hash(str(workload_details["id"]))

  required_keys = ["id", "create_time"]
  missing_keys = [k for k in required_keys if not workload_details.get(k)]
  if missing_keys:
    raise ValueError(
        "Could not generate GCE workload identifier due to missing properties:"
        f" {', '.join(missing_keys)}."
    )
  identifier = f"{workload_details['id']}_{workload_details['create_time']}"
  return _get_sha256_hash(identifier)


def _slurm_run_identifier(workload_details: dict[str, Any]) -> str:
  """Returns the unique identifier for the slurm workload."""
  if not workload_details:
    raise ValueError(
        "Could not generate Slurm workload identifier due to missing workload"
        " details."
    )
  required_keys = ["job_id", "cluster", "submit_time"]
  missing_keys = [k for k in required_keys if not workload_details.get(k)]
  if missing_keys:
    raise ValueError(
        "Could not generate Slurm workload identifier due to missing"
        f" properties: {', '.join(missing_keys)}."
    )

  cluster = workload_details["cluster"].split("/")[-1]
  raw_submit_time = workload_details["submit_time"]
  iso_str = (
      raw_submit_time.replace("Z", "+00:00")
      if raw_submit_time.endswith("Z")
      else raw_submit_time
  )
  transformed_timestamp = (
      datetime.datetime.fromisoformat(iso_str)
      .astimezone(datetime.timezone.utc)
      .strftime("%Y%m%d-%H%M%S")
  )
  identifier = f"{cluster}_{workload_details['job_id']}_{transformed_timestamp}"
  return _get_sha256_hash(identifier)


# Public functions
def get_hostname() -> str:
  """Returns hostname or pod name of the current machine."""
  # HOSTNAME is set in GCE and GKE.
  # Fallback to socket.gethostname() for non-containerized environments.
  return os.environ.get("HOSTNAME") or socket.gethostname()


def get_instance_id() -> str:
  """Returns the VM instance ID if available, otherwise raises RuntimeError."""
  headers = {"Metadata-Flavor": "Google"}
  with requests.get(
      "http://metadata.google.internal/computeMetadata/v1/instance/id",
      headers=headers,
      timeout=2.0,
  ) as response:
    if response.status_code != 200:
      raise RuntimeError(
          f"Failed to fetch instance ID. Status code: {response.status_code}"
      )
    instance_id = response.text.strip()
    if not instance_id:
      raise RuntimeError("Metadata server returned an empty instance ID")
    return instance_id


_jax_host_module_cache = None


def _import_jax_host_module():
  """Lazy load jax_host module and cache result to avoid loading jax."""
  global _jax_host_module_cache
  if _jax_host_module_cache is not None:
    return _jax_host_module_cache

  from google_cloud_mldiagnostics.utils.jax_utils import (  # pylint: disable=g-import-not-at-top
      jax_host,
  )

  _jax_host_module_cache = jax_host
  return _jax_host_module_cache


def get_process_index(
    framework: mlrun_types.Framework = mlrun_types.Framework.JAX,
    serving_engine: mlrun_types.ServingEngine = mlrun_types.ServingEngine.NONE,
) -> int:
  """Returns host index."""
  if (
      framework == mlrun_types.Framework.JAX
      and serving_engine == mlrun_types.ServingEngine.NONE
  ):
    if os.environ.get("MLRUN_SKIP_LIBTPU", "False").lower() != "true":
      # TODO: [INTERNAL] - Add support for non-jax workloads.
      return _import_jax_host_module().get_jax_process_index()

  # For non-JAX distributed frameworks (like vLLM/PyTorch), check standard env vars.
  for env_var in ("NODE_RANK", "GROUP_RANK", "RANK", "JOB_COMPLETION_INDEX"):
    val = os.environ.get(env_var)
    if val is not None:
      try:
        return int(val)
      except ValueError:
        pass
  return 0


def get_accelerator_type(
    framework: mlrun_types.Framework | None = mlrun_types.Framework.JAX,
    serving_engine: mlrun_types.ServingEngine = mlrun_types.ServingEngine.NONE,
) -> metric_types.AcceleratorType:
  """Returns the accelerator type of the current host."""
  # 1. Inspect JAX devices first if framework is JAX and engine is NONE.
  if (
      framework == mlrun_types.Framework.JAX
      and serving_engine == mlrun_types.ServingEngine.NONE
  ):
    if os.environ.get("MLRUN_SKIP_LIBTPU", "False").lower() != "true":
      try:
        jax_acc = _import_jax_host_module().get_accelerator_type()
        if jax_acc != metric_types.AcceleratorType.UNKNOWN:
          return jax_acc
      except Exception:  # pylint: disable=broad-exception-caught
        pass

  # 2. Check environment variables
  if any(
      var in os.environ
      for var in ["TPU_NAME", "TPU_ACCELERATOR_TYPE", "JAX_FORCE_TPU_INIT"]
  ):
    return metric_types.AcceleratorType.TPU
  if any(
      var in os.environ
      for var in [
          "CUDA_VISIBLE_DEVICES",
          "NVIDIA_VISIBLE_DEVICES",
          "CUDA_VERSION",
      ]
  ):
    return metric_types.AcceleratorType.GPU

  # 3. Check device nodes & libraries
  import glob  # pylint: disable=g-import-not-at-top

  if (
      glob.glob("/dev/accel/tpu_*")
      or os.path.exists("/dev/accel")
      or os.path.exists("/usr/lib/libtpu.so")
      or os.path.exists("/lib/libtpu.so")
  ):
    return metric_types.AcceleratorType.TPU
  if glob.glob("/dev/nvidia*") or os.path.exists("/dev/dri/renderD128"):
    return metric_types.AcceleratorType.GPU

  return metric_types.AcceleratorType.UNKNOWN


def is_master_host(
    framework: mlrun_types.Framework = mlrun_types.Framework.JAX,
    serving_engine: mlrun_types.ServingEngine = mlrun_types.ServingEngine.NONE,
) -> bool:
  """Checks if the current host is the master host."""
  process_index = get_process_index(framework, serving_engine)
  logger.info(
      "framework: %s, serving_engine: %s, process_index: %s",
      framework,
      serving_engine,
      process_index,
  )
  return process_index == 0


def get_workload_details(
    orchestrator: str = "GKE", run_workload_id: str | None = None
) -> dict[str, Any] | None:
  """Returns workload details if available, otherwise None."""
  if orchestrator == "SLURM":
    return _get_slurm_workload_details()

  if orchestrator == "GCE":
    return _get_gce_workload_details(run_workload_id)

  return _get_gke_workload_details()


def get_identifier(
    orchestrator: str = "GKE", workload_details: dict[str, Any] | None = None
) -> str:
  """Returns a unique SHA-256 identifier for the workload."""
  if orchestrator == "SLURM":
    return _slurm_run_identifier(workload_details)  # pyrefly: ignore[bad-argument-type]

  if orchestrator == "GCE":
    return _gce_run_identifier(workload_details)  # pyrefly: ignore[bad-argument-type]

  return _gke_run_identifier(workload_details)  # pyrefly: ignore[bad-argument-type]


def get_workload_targets(
    orchestrator: str = "GKE", workload_details: dict[str, Any] | None = None
) -> list[dict[str, Any]] | None:
  if orchestrator == "SLURM":
    return _slurm_workload_targets()

  if orchestrator == "GCE":
    return _gce_workload_targets(workload_details)

  return None


def sanitize_identifier(identifier: str) -> str:
  """Sanitize the identifier for the MLRun."""
  sanitized_id = re.sub(r"[^a-z0-9]+", "-", identifier.lower())
  # Remove leading/trailing hyphens
  sanitized_id = sanitized_id.strip("-")
  return sanitized_id


def effective_session_id(session_id: str | None = None) -> str:
  """Returns the effective session ID."""
  if not session_id:
    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    logger.debug(
        "Profiling session_id not provided, generated"
        " session_id using current timestamp: %s",
        session_id,
    )

  return session_id
