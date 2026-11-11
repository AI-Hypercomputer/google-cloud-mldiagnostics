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

"""Utility functions for GCP related operations."""

import json
import logging
import os
import re
import subprocess
import textwrap
import urllib.error
import urllib.request

_REGION_REGEX = re.compile(
    textwrap.dedent("""\
        [a-z]+          # Prefix (e.g., 'us', 'europe')
        (-[a-z]+)*      # Optional middle parts (e.g., '-central', '-gov-west')
        -[a-z]+         # Name part (e.g., '-west', '-east')
        [0-9]+          # Number part (e.g., '1', '3')
    """),
    re.VERBOSE,
)
_ZONE_REGEX = re.compile(
    textwrap.dedent("""\
        [a-z]+          # Prefix
        (-[a-z]+)*      # Optional middle parts
        -[a-z]+         # Name part
        [0-9]+          # Number part (e.g., '1')
        -[a-z]+         # Zone suffix (e.g., '-a', '-b')
    """),
    re.VERBOSE,
)
_LOCATION_PATH_REGEX = re.compile(
    r"(?:^|/)(?:locations|zones|regions)/([^/\s]+)"
)
_REGION_ENV_VARS = (
    "CLOUD_ML_REGION",
    "GOOGLE_CLOUD_REGION",
    "CLOUDSDK_COMPUTE_REGION",
    "REGION",
)
_ZONE_ENV_VARS = (
    "CLOUDSDK_COMPUTE_ZONE",
    "ZONE",
)


def extract_region_from_location(location: str | None) -> str | None:
  """Extracts and normalizes a GCP region from a region, zone, or resource path.

  Args:
      location: A region (e.g. 'us-central1'), zone (e.g. 'us-central1-a'), or
        GCP resource path (e.g.
        'projects/p/locations/us-central1-a/clusters/c').

  Returns:
      The normalized region string (e.g. 'us-central1') if valid, otherwise
      None.
  """
  if not isinstance(location, str):
    return None
  candidate = location.strip()

  match = _LOCATION_PATH_REGEX.search(candidate)
  if match:
    candidate = match.group(1)

  if _REGION_REGEX.fullmatch(candidate):
    return candidate
  if _ZONE_REGEX.fullmatch(candidate):
    return "-".join(candidate.split("-")[:-1])
  return None


def _get_region_from_env() -> str | None:
  """Resolves the GCP region from environment variables or GKE metadata."""
  for env_var in _REGION_ENV_VARS + _ZONE_ENV_VARS:
    region = extract_region_from_location(os.environ.get(env_var))
    if region:
      return region

  diagon_id_raw = os.environ.get("GKE_DIAGON_IDENTIFIER")
  if diagon_id_raw:
    try:
      diagon_id = json.loads(diagon_id_raw)
      if isinstance(diagon_id, dict):
        loc_val = diagon_id.get("location")
        if isinstance(loc_val, str):
          region = extract_region_from_location(loc_val)
          if region:
            return region
        cluster_val = diagon_id.get("clustername")
        if isinstance(cluster_val, str) and "/" in cluster_val:
          region = extract_region_from_location(cluster_val)
          if region:
            return region
    except (json.JSONDecodeError, TypeError, ValueError):
      # Ignore malformed JSON or unexpected schema in env var; fall through.
      pass

  return None


def _get_gcloud_region(timeout: int = 5) -> str | None:
  """Resolves the GCP region from gcloud CLI configuration."""
  for prop in ("compute/region", "compute/zone"):
    try:
      proc = subprocess.run(
          ["gcloud", "config", "get-value", prop],
          capture_output=True,
          text=True,
          timeout=timeout,
          check=False,
      )
      if proc.returncode == 0 and proc.stdout:
        for line in reversed(proc.stdout.splitlines()):
          region = extract_region_from_location(line)
          if region:
            return region
    except (OSError, subprocess.SubprocessError, ValueError):
      # gcloud CLI unavailable, timed out, or misconfigured; try next property.
      continue
  return None


def get_project_id(timeout: int = 5) -> str | None:
  """Get the GCP project ID from the metadata server.

  Args:
      timeout: Request timeout in seconds (default: 5)

  Returns:
      Project ID string if successful, None if failed
  """
  url = "http://metadata.google.internal/computeMetadata/v1/project/project-id"

  try:
    req = urllib.request.Request(url)
    req.add_header("Metadata-Flavor", "Google")

    with urllib.request.urlopen(req, timeout=timeout) as response:
      return response.read().decode("utf-8").strip()

  except (urllib.error.URLError, urllib.error.HTTPError, ValueError) as e:
    logging.warning("Failed to get project ID in Diagon SDK: %s", e)
    return None


def get_instance_metadata(path: str, timeout: int = 5) -> str | None:
  """Get GCE instance metadata from the metadata server.

  Args:
      path: Path under computeMetadata/v1/instance/ (e.g. 'zone', 'machine-type',
        or 'attributes/accelerator-type').
      timeout: Request timeout in seconds (default: 5)

  Returns:
      Metadata value string if successful, None if failed
  """
  url = f"http://metadata.google.internal/computeMetadata/v1/instance/{path}"

  try:
    req = urllib.request.Request(url)
    req.add_header("Metadata-Flavor", "Google")

    with urllib.request.urlopen(req, timeout=timeout) as response:
      return response.read().decode("utf-8").strip()

  except (urllib.error.URLError, urllib.error.HTTPError, ValueError) as e:
    logging.warning("Failed to get instance metadata '%s': %s", path, e)
    return None


def get_instance_zone(timeout: int = 5) -> str | None:
  """Get the GCE instance zone from the metadata server.

  Args:
      timeout: Request timeout in seconds (default: 5)

  Returns:
      Zone name (e.g., 'us-central1-a') if successful, None if failed
  """
  zone_path = get_instance_metadata("zone", timeout=timeout)
  if zone_path:
    # Extract zone name from path, e.g.
    # "projects/123456789/zones/us-central1-a"
    return zone_path.split("/")[-1]
  return None


def get_instance_region(timeout: int = 5) -> str | None:
  """Get the GCP region from environment, GKE/GCE metadata, or gcloud config.

  Args:
      timeout: Request timeout in seconds (default: 5)

  Returns:
      Region name (e.g., 'us-central1') if successful, None if failed
  """
  env_region = _get_region_from_env()
  if env_region:
    return env_region

  if os.environ.get("KUBERNETES_SERVICE_HOST"):
    cluster_location = get_instance_attribute(
        "cluster-location", timeout=timeout
    )
    region = extract_region_from_location(cluster_location)
    if region:
      return region

  zone = get_instance_zone(timeout)
  if zone:
    # Extract region from zone (e.g., 'us-central1-a' -> 'us-central1')
    parts = zone.split("-")
    if len(parts) >= 3:
      return "-".join(parts[:-1])

  gcloud_region = _get_gcloud_region(timeout=timeout)
  if gcloud_region:
    return gcloud_region

  return None


def get_instance_attribute(attribute_name: str, timeout: int = 5) -> str | None:
  """Get a custom GCE instance attribute from the metadata server.

  Args:
      attribute_name: The attribute key to fetch (e.g. 'accelerator-type').
      timeout: Request timeout in seconds (default: 5)

  Returns:
      Attribute value string if successful, None if failed
  """
  return get_instance_metadata(f"attributes/{attribute_name}", timeout=timeout)


def get_tpu_accelerator_type(timeout: int = 5) -> str | None:
  """Get TPU accelerator type from GCE metadata attributes or machine type."""
  accel_type = get_instance_attribute("accelerator-type", timeout=timeout)
  if accel_type:
    return accel_type.split("/")[-1]

  # Check tpu-env attribute (YAML/key-value format)
  tpu_env = get_instance_attribute("tpu-env", timeout=timeout)
  if tpu_env:
    for line in tpu_env.splitlines():
      match = re.search(
          r"^\s*(?:ACCELERATOR_TYPE|TPU_ACCELERATOR_TYPE|TPU_TYPE)\s*:\s*['\"]?([^'\"\s]+)['\"]?",
          line,
      )
      if match:
        return match.group(1).strip()

  # Check machine-type for TPU VM naming (e.g. ct5lp-hightpu-4t -> v5litepod-4)
  machine_type = get_instance_metadata("machine-type", timeout=timeout)
  if machine_type:
    mt = machine_type.split("/")[-1]
    if "tpu" in mt:
      return mt

  return None


def validate_region(region: str | None) -> None:
  """Validates that the given string is a valid region and not a zone.

  Args:
      region: The region string to validate.

  Raises:
      ValueError: If the region is invalid or is a zone.
  """
  if region is None:
    raise ValueError("Region value is required.")

  if _REGION_REGEX.fullmatch(region):
    return

  if _ZONE_REGEX.fullmatch(region):
    raise ValueError(
        f"Invalid region {region!r}. Expected a region format like"
        " 'us-central1', but received a zone. Please provide a valid region."
    )

  raise ValueError(
      f"Invalid region {region!r}. Expected a region format like"
      " 'us-central1'. Please provide a valid region."
  )
