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

"""Client for sending requests to Diagon Control Plane."""

import logging
import pprint
import random
import time
from typing import Any, Dict, List, Optional

import google.auth
from google.auth.transport import requests as google_auth_requests
from google_cloud_mldiagnostics.utils import host_utils
import requests


logger = logging.getLogger(__name__)


class ControlPlaneClient:
  """Client for communicating with Google Cloud Hypercompute Cluster ML Run service."""

  def __init__(
      self,
      project_id: str,
      environment: str,
      location: str = "us-central1",
  ):
    """Initializes a new ControlPlaneClient.

    Args:
        project_id: Google Cloud project ID
        environment: Environment to use (autopush, staging, prod)
        location: Google Cloud location/region
    """
    if environment == "prod":
      base_url = "https://hypercomputecluster.googleapis.com/v1alpha"
    else:
      base_url = (
          f"https://{environment}-hypercomputecluster.sandbox.googleapis.com/v1alpha"
      )
    self.project_id = project_id
    self.location = location
    self.base_url = base_url
    self.ml_runs_path = f"{base_url}/projects/{project_id}/locations/{location}/machineLearningRuns"

    # Initialize Google Cloud credentials
    self.credentials, _ = google.auth.default()

  def _get_access_token(self) -> str:
    """Get Google Cloud access token for authentication."""
    if not self.credentials.valid:
      self.credentials.refresh(google_auth_requests.Request())

    return self.credentials.token

  def _get_headers(self) -> Dict[str, str]:
    """Get HTTP headers with authentication."""
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {self._get_access_token()}",
    }

  def get_operation(self, operation_name: str) -> Dict[str, Any]:
    """Get an existing operation using the Google Cloud API.

    Args:
        operation_name: Name of the operation to retrieve.

    Returns:
        Response from the API as a dictionary.

    Raises:
        requests.exceptions.RequestException: If the HTTP request fails.
    """
    operation_url = f"{self.base_url}/{operation_name}"
    logger.debug("Get Operation request: url=%s", operation_url)
    response = requests.get(
        operation_url,
        headers=self._get_headers(),
    )
    try:
      response.raise_for_status()
    except requests.exceptions.HTTPError:
      logger.exception(
          "Get Operation request failed: status_code=%s, content=%s",
          response.status_code,
          response.text,
      )
      raise
    json_response = response.json()
    logger.debug("Get Operation response: %s", pprint.pformat(json_response))
    return json_response

  def _wait_for_operation(
      self,
      operation_name: str,
      polling_interval_sec: int = 1,
      timeout_sec: int = 300,
  ) -> Dict[str, Any]:
    """Waits for an operation to complete.

    Args:
        operation_name: The name of the operation to wait for.
        polling_interval_sec: The initial interval in seconds to poll the
          operation.
        timeout_sec: The maximum time in seconds to wait for the operation to
          complete.

    Returns:
        The completed operation.

    Raises:
        requests.exceptions.HTTPError: If the operation fails.
        TimeoutError: If the operation does not complete within the timeout.
    """
    start_time = time.time()
    delay = float(polling_interval_sec)
    while True:
      try:
        operation = self.get_operation(operation_name)
      except requests.exceptions.HTTPError:
        # Re-raise HTTP errors to fail fast.
        raise
      except requests.exceptions.RequestException as e:
        logger.warning(
            "Failed to get operation status for %s: %s", operation_name, e
        )
      else:
        if operation.get("done"):
          if operation.get("error"):
            raise requests.exceptions.HTTPError(
                f"Operation {operation_name} failed: {operation['error']}"
            )
          return operation

      if time.time() - start_time >= timeout_sec:
        raise TimeoutError(
            f"Timed out waiting for operation {operation_name} to complete."
        )

      # Operation not done or request failed, sleep with backoff
      time.sleep(delay * (0.5 + random.random() * 0.5))
      delay = min(delay * 2, 60.0)

  def create_ml_run(
      self,
      name: str,
      display_name: str,
      run_phase: str,
      configs: Optional[Dict[str, Any]] = None,
      tools: Optional[List[Dict[str, Any]]] = None,
      metrics: Optional[Dict[str, str]] = None,
      artifacts: Optional[Dict[str, str]] = None,
      run_group: Optional[str] = None,
      labels: Optional[Dict[str, str]] = None,
      orchestrator: Optional[str] = None,
      workload_details: Optional[Dict[str, Any]] = None,
  ) -> Dict[str, Any]:
    """Create a new ML run using the Google Cloud API.

    Args:
        name: Name of the run
        display_name: Display name for the run
        run_phase: Phase of the run (ACTIVE, COMPLETE, FAILED)
        configs: Configuration settings (userConfigs, softwareConfigs,
          hardwareConfigs)
        tools: List of tools to enable (e.g., XProf, NSys)
        metrics: Metrics for the run (e.g., avgStep, avgLatency)
        artifacts: Artifacts configuration (e.g., gcsPath)
        run_group: Run group grouping identifier
        labels: Custom labels for the run
        orchestrator: Orchestrator the workload is running on (e.g., GCE, GKE)
        workload_details: Details about the workload

    Returns:
        Response from the API as a dictionary

    Raises:
        requests.exceptions.RequestException: If the HTTP request fails
    """
    payload = {"displayName": display_name, "name": name}

    if configs:
      payload["configs"] = configs

    if metrics:
      payload["metrics"] = metrics

    if artifacts:
      payload["artifacts"] = artifacts

    if run_group:
      payload["runSet"] = run_group

    if labels:
      payload["labels"] = labels

    if run_phase:
      payload["runPhase"] = run_phase

    if tools:
      payload["tools"] = tools

    if orchestrator:
      payload["orchestrator"] = orchestrator
      if orchestrator == "GKE" and workload_details:
        gke_workload_details = {
            "id": workload_details["id"],
            "kind": workload_details["kind"],
            "cluster": workload_details["cluster"],
            "namespace": workload_details["namespace"],
        }
        if workload_details["parent_workload"]:
          gke_workload_details["parentWorkload"] = workload_details[
              "parent_workload"
          ]
        if workload_details["labels"]:
          gke_workload_details["labels"] = workload_details["labels"]
        payload["workloadDetails"] = {"gke": gke_workload_details}

    # Sanitize the name for machineLearningRunId
    sanitized_name = host_utils.sanitize_identifier(name)
    params = {"machine_learning_run_id": sanitized_name}

    logger.debug(
        "Create ML Run request: url=%s, params=%s, json=%s",
        self.ml_runs_path,
        params,
        payload,
    )
    response = requests.post(
        self.ml_runs_path,
        headers=self._get_headers(),
        params=params,
        json=payload,
    )

    try:
      response.raise_for_status()
    except requests.exceptions.HTTPError:
      logger.exception(
          "Create ML Run request failed: status_code=%s, content=%s",
          response.status_code,
          response.text,
      )
      raise
    json_response = response.json()
    logger.debug("Create ML Run response: %s", pprint.pformat(json_response))

    if not json_response.get("done"):
      operation = self._wait_for_operation(json_response["name"])
    else:
      operation = json_response

    logger.info("Create ML Run operation: %s", pprint.pformat(operation))

    if operation.get("error"):
      raise requests.exceptions.HTTPError(
          f"Operation {operation['name']} failed: {operation['error']}"
      )

    if operation.get("response"):
      return operation["response"]
    else:
      # If no response field, fetch mlrun using target in metadata
      metadata = operation.get("metadata", {})
      target = metadata.get("target")
      if not target:
        raise ValueError(
            f"Could not find target in operation metadata for operation"
            f" {operation.get('name')}"
        )
      mlrun_name = target.split("/")[-1]
      return self.get_ml_run(mlrun_name)

  def get_ml_run(self, name: str) -> Dict[str, Any]:
    """Get an existing ML run using the Google Cloud API.

    Args:
        name: Name of the run to retrieve.

    Returns:
        Response from the API as a dictionary.

    Raises:
        requests.exceptions.RequestException: If the HTTP request fails.
    """
    run_url = f"{self.ml_runs_path}/{name}"
    logger.debug("Get ML Run request: url=%s", run_url)
    response = requests.get(
        run_url,
        headers=self._get_headers(),
    )

    try:
      response.raise_for_status()
    except requests.exceptions.HTTPError:
      logger.exception(
          "Get ML Run request failed: status_code=%s, content=%s",
          response.status_code,
          response.text,
      )
      raise
    json_response = response.json()
    logger.debug("Get ML Run response: %s", pprint.pformat(json_response))
    return json_response

  def update_ml_run(
      self,
      name: str,
      run_phase: Optional[str] = None,
      metrics: Optional[Dict[str, str]] = None,
  ) -> Dict[str, Any]:
    """Update an existing ML run using the Google Cloud API by sending the full resource.

    Args:
        name: Name of the run to update
        run_phase: Phase of the run (ACTIVE, COMPLETE, FAILED)
        metrics: Metrics for the run (e.g., avgStep, avgLatency)

    Returns:
        Response from the API as a dictionary

    Raises:
        requests.exceptions.RequestException: If the HTTP request fails
    """
    payload = self.get_ml_run(name)
    need_update = False

    if metrics is not None and payload.get("metrics") != metrics:
      payload["metrics"] = metrics
      need_update = True

    if run_phase is not None and payload.get("runPhase") != run_phase:
      payload["runPhase"] = run_phase
      need_update = True

    if not need_update:
      return payload

    # Remove fields that are output-only
    for field in ["createTime", "updateTime", "endTime"]:
      payload.pop(field, None)

    run_url = f"{self.ml_runs_path}/{name}"
    params = {"update_mask": "*"}

    logger.debug(
        "Update ML Run request: url=%s, params=%s, json=%s",
        run_url,
        params,
        payload,
    )
    response = requests.patch(
        run_url,
        headers=self._get_headers(),
        params=params,
        json=payload,
    )

    try:
      response.raise_for_status()
    except requests.exceptions.HTTPError:
      logger.exception(
          "Update ML Run request failed: status_code=%s, content=%s",
          response.status_code,
          response.text,
      )
      raise
    json_response = response.json()
    logger.debug("Update ML Run response: %s", pprint.pformat(json_response))
    return json_response
