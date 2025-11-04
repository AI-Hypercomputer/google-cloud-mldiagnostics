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

import logging
import urllib.error
import urllib.request


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


def get_instance_zone(timeout: int = 5) -> str | None:
  """Get the GCE instance zone from the metadata server.

  Args:
      timeout: Request timeout in seconds (default: 5)

  Returns:
      Zone name (e.g., 'us-central1-a') if successful, None if failed
  """
  url = "http://metadata.google.internal/computeMetadata/v1/instance/zone"

  try:
    req = urllib.request.Request(url)
    req.add_header("Metadata-Flavor", "Google")

    with urllib.request.urlopen(req, timeout=timeout) as response:
      zone_path = response.read().decode("utf-8").strip()
      # Extract zone name from path, e.g.
      # "projects/123456789/zones/us-central1-a"
      return zone_path.split("/")[-1]

  except (urllib.error.URLError, urllib.error.HTTPError, ValueError) as e:
    print(f"Failed to get instance zone: {e}")
    return None


def get_instance_region(timeout: int = 5) -> str | None:
  """Get the GCE instance region from the metadata server.

  Args:
      timeout: Request timeout in seconds (default: 5)

  Returns:
      Region name (e.g., 'us-central1') if successful, None if failed
  """
  zone = get_instance_zone(timeout)
  if zone:
    # Extract region from zone (e.g., 'us-central1-a' -> 'us-central1')
    parts = zone.split("-")
    if len(parts) >= 3:
      return "-".join(parts[:-1])
  return None
