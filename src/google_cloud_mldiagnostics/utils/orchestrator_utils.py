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
import requests


def detect_orchestrator():
  """Detects the orchestrator the workload is running on."""
  orchestrator = None

  # Check for GCE Metadata Server to determine if running on GCP
  on_gcp = False
  try:
    headers = {'Metadata-Flavor': 'Google'}
    # Use a more specific endpoint to be sure
    response = requests.get(
        'http://metadata.google.internal/computeMetadata/v1/instance/id',
        headers=headers,
        timeout=0.1,
    )
    if (
        response.status_code == 200
        and response.headers.get('Metadata-Flavor') == 'Google'
    ):
      on_gcp = True
  except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
    pass

  if on_gcp:
    # Running on GCP, check if it's GKE or standard GCE
    if os.getenv('KUBERNETES_SERVICE_HOST') or os.path.exists(
        '/var/run/secrets/kubernetes.io/serviceaccount/token'
    ):
      orchestrator = 'GKE'
    else:
      orchestrator = 'GCE'

  return orchestrator
