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

"""Utility functions for identifying orchestrator."""

import os

from google_cloud_mldiagnostics.custom_types import mlrun_types
import requests


def _fetch_gcp_metadata(path: str) -> requests.Response | None:
  """Fetches metadata from the GCP metadata server.

  Args:
    path: The metadata path to query (e.g., 'instance/id').

  Returns:
    The response object if successful, otherwise None.
  """
  headers = {'Metadata-Flavor': 'Google'}
  try:
    return requests.get(
        f'http://metadata.google.internal/computeMetadata/v1/{path}',
        headers=headers,
        timeout=0.1,
    )
  except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
    return None


def detect_orchestrator():
  """Detects the orchestrator the workload is running on."""
  orchestrator = None

  # Check for GCE Metadata Server to determine if running on GCP
  response = _fetch_gcp_metadata('instance/id')
  on_gcp = False
  if (
      response is not None
      and response.status_code == 200
      and response.headers.get('Metadata-Flavor') == 'Google'
  ):
    on_gcp = True

  if on_gcp:
    # First check if we are in *any* Kubernetes environment
    is_k8s = False
    if os.getenv('KUBERNETES_SERVICE_HOST') or os.path.exists(
        '/var/run/secrets/kubernetes.io/serviceaccount/token'
    ):
      is_k8s = True

    # Then check if it is Managed GKE by verifying the cluster-name attribute
    is_gke = False
    if is_k8s:
      gke_response = _fetch_gcp_metadata('instance/attributes/cluster-name')
      if gke_response is not None and gke_response.status_code == 200:
        is_gke = True

    if is_gke:
      orchestrator = mlrun_types.Orchestrator.GKE.value
    elif (
        os.getenv('SLURM_CLUSTER_NAME')
        and os.getenv('SLURM_JOB_ID')
        and (
            os.getenv('SLURM_NODELIST')
            or os.getenv('SLURM_JOB_START_TIME')
        )
    ):
      orchestrator = mlrun_types.Orchestrator.SLURM.value
    elif is_k8s:
      # TODO([INTERNAL]): Use a proper orchestrator type for self-hosted
      # K3s/K8s once design is approved.
      # Currently, self-hosted K3s/K8s falls back to GCE.
      orchestrator = mlrun_types.Orchestrator.GCE.value
    else:
      # Any non-GKE workload on GCE VMs
      orchestrator = mlrun_types.Orchestrator.GCE.value

  return orchestrator
