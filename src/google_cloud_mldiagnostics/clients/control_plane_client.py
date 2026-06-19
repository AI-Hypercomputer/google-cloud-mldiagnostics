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

import ast
import datetime
import logging
import pprint
import random
import time
from typing import Any, Dict, List, Optional

import google.auth
from google.auth.transport import requests as google_auth_requests
from google_cloud_mldiagnostics.utils import gcp
from google_cloud_mldiagnostics.utils import host_utils
import requests

logger = logging.getLogger(__name__)
_MAX_RETRIES = 3
_ERROR_CODE_ALREADY_EXISTS = 6


def _extract_run_id_from_error(err: Exception) -> Optional[str]:
  """Extract the existing run ID from HTTPError.

  This function parses the string representation of an HTTPError to find a
  previously created run ID when a resource already exists error (code 6) is
  returned by the API.

  Args:
    err: The exception raised, typically a requests.exceptions.HTTPError.

  Returns:
    The extracted run ID as a string if found, otherwise None.
  """
  try:
    err_str = str(err)
    if "failed: {" in err_str:
      dict_str = err_str.split("failed: ", 1)[1]
      err_dict = ast.literal_eval(dict_str)
      if err_dict.get("code") == _ERROR_CODE_ALREADY_EXISTS:
        for detail in err_dict.get("details", []):
          if (
              detail.get("@type")
              == "type.googleapis.com/google.rpc.ResourceInfo"
          ):
            resource_name = detail.get("resourceName")
            if resource_name:
              return resource_name.split("/")[-1]
  except Exception as e:
    logger.debug("Failed to extract run ID from error string: %s", e)
  return None


class ControlPlaneClient:
  """Client for communicating with the Hypercompute Cluster ML Run service."""

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
    gcp.validate_region(location)
    if environment == "prod":
      base_url = "https://hypercomputecluster.googleapis.com/v1alpha"
    else:
      base_url = f"https://{environment}-hypercomputecluster.sandbox.googleapis.com/v1alpha"
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
      logger.error(
          "Get Operation request failed: status_code=%s, content=%s",
          response.status_code,
          response.text,
      )
      raise
    json_response = response.json()
    if logger.isEnabledFor(logging.DEBUG):
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
        configs: Configuration settings (softwareConfigs, hardwareConfigs)
        tools: List of tools to enable (e.g., XProf, NSys)
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
        creation_timestamp = workload_details.get("creation-timestamp")
        if creation_timestamp:
          gke_workload_details["createTime"] = creation_timestamp
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
      logger.error(
          "Create ML Run request failed: status_code=%s, content=%s",
          response.status_code,
          response.text,
      )
      raise
    json_response = response.json()
    if logger.isEnabledFor(logging.DEBUG):
      logger.debug("Create ML Run response: %s", pprint.pformat(json_response))

    if not json_response.get("done"):
      try:
        operation = self._wait_for_operation(json_response["name"])
      except requests.exceptions.HTTPError as e_op:
        existing_run_id = _extract_run_id_from_error(e_op)
        if existing_run_id:
          logger.info(
              "ML run already exists. Recovering run ID %r.", existing_run_id
          )
          return self.update_ml_run(
              name=existing_run_id,
              display_name=display_name,
              tools=tools,
              artifacts=artifacts,
              run_phase=run_phase,
          )
        raise
    else:
      operation = json_response

    if logger.isEnabledFor(logging.INFO):
      logger.info("Create ML Run operation: %s", pprint.pformat(operation))

    if operation.get("error"):
      raise requests.exceptions.HTTPError(
          f"Operation {operation['name']!r} failed: {operation['error']!r}"
      )

    if operation.get("response"):
      return operation["response"]
    else:
      # If no response field, fetch mlrun using target in metadata
      metadata = operation.get("metadata", {})
      target = metadata.get("target")
      if not target:
        raise ValueError(
            "Could not find target in operation metadata for operation"
            f" {operation.get('name')}"
        )
      mlrun_name = target.split("/")[-1]
      return self.get_ml_run(mlrun_name)

  def create_profiler_session(
      self,
      ml_run_id: str,
      profiler_session_id: str,
      gsc_file_path: str,
      profiler_target: str,
      start_time: float,
      end_time: Optional[float],
      session_phase: str,
  ) -> Dict[str, Any]:
    """Create a ProfilerSession resource.

    Args:
        ml_run_id: The ID of the ML run.
        profiler_session_id: The ID for the profiler session.
        gsc_file_path: The GCS path for storing profiler session data.
        profiler_target: The profiler target resource name.
        start_time: Start time of the profiler session.
        end_time: End time of the profiler session.
        session_phase: The phase of the session (e.g., "SUCCEEDED").

    Returns:
        The response JSON dictionary.

    Raises:
        requests.exceptions.RequestException: If the HTTP request fails.
    """
    parent = f"projects/{self.project_id}/locations/{self.location}/machineLearningRuns/{ml_run_id}"
    url = f"{self.base_url}/{parent}/profilerSessions"

    if end_time is not None:
      duration_sec = end_time - start_time
      duration_str = f"{max(0.001, duration_sec):.3f}s"
      end_time_str = datetime.datetime.fromtimestamp(
          end_time, datetime.timezone.utc
      ).strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
      duration_str = "0.001s"
      end_time_str = "0001-01-01T00:00:00Z"

    start_time_str = datetime.datetime.fromtimestamp(
        start_time, datetime.timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    target_session = {
        "startTime": start_time_str,
        "sessionPhase": session_phase,
    }
    if end_time_str:
      target_session["endTime"] = end_time_str

    payload = {
        "profilerTargets": [profiler_target],
        "targetSessions": {
            profiler_target: target_session,
        },
        "storageFolderUri": gsc_file_path,
        "dashboardUri": "",
        "duration": duration_str,
        "kind": "KIND_PROGRAMMATIC",
        "hostTracerLevel": "HOST_TRACER_LEVEL_INFO",
        "deviceTracerLevel": "DEVICE_TRACER_LEVEL_ENABLED",
        "pythonTracerLevel": "PYTHON_TRACER_LEVEL_DISABLED",
    }

    params = {"profiler_session_id": profiler_session_id}

    if logger.isEnabledFor(logging.DEBUG):
      logger.debug(
          "Create Profiler Session request: url=%s, params=%s, json=%s",
          url,
          pprint.pformat(params),
          pprint.pformat(payload),
      )

    with requests.post(
        url,
        headers=self._get_headers(),
        params=params,
        json=payload,
    ) as response:
      try:
        response.raise_for_status()
      except requests.exceptions.HTTPError as e:
        if response.status_code == 409:
          err_dict = ast.literal_eval(response.text)
          logger.info("error dict: %s", err_dict)
          for detail in err_dict.get("error", {}).get("details", []):
            if (
                detail.get("@type", "")
                == "type.googleapis.com/google.rpc.ResourceInfo"
            ):
              resource_name = detail.get("resourceName", None)
              if resource_name:
                existing_session_id = resource_name.split("/")[-1]
                logger.info(
                    "Profiler session '%s' already exists, updating it.",
                    profiler_session_id,
                )
                return self.update_profiler_session(
                    ml_run_id=ml_run_id,
                    profiler_session_id=existing_session_id,
                    gsc_file_path=gsc_file_path,
                    profiler_target=profiler_target,
                    start_time=start_time,
                    end_time=end_time,
                    session_phase=session_phase,
                )

        logger.exception(
            "Create Profiler Session request failed: status_code=%s,"
            " content=%s",
            response.status_code,
            response.text,
        )
        raise

      json_response = response.json()

    logger.debug(
        "Create Profiler Session response: %s", pprint.pformat(json_response)
    )

    if not json_response.get("done"):
      operation = self._wait_for_operation(json_response["name"])
    else:
      operation = json_response

    return operation

  def update_profiler_session(
      self,
      ml_run_id: str,
      profiler_session_id: str,
      gsc_file_path: str,
      profiler_target: str,
      start_time: float,
      end_time: Optional[float],
      session_phase: str,
  ) -> Dict[str, Any]:
    """Update an existing ProfilerSession resource."""
    parent = f"projects/{self.project_id}/locations/{self.location}/machineLearningRuns/{ml_run_id}"
    url = f"{self.base_url}/{parent}/profilerSessions/{profiler_session_id}"

    if end_time is not None:
      duration_sec = end_time - start_time
      duration_str = f"{max(0.001, duration_sec):.3f}s"
      end_time_str = datetime.datetime.fromtimestamp(
          end_time, datetime.timezone.utc
      ).strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
      duration_str = "0.001s"
      end_time_str = "0001-01-01T00:00:00Z"

    start_time_str = datetime.datetime.fromtimestamp(
        start_time, datetime.timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    target_session = {
        "startTime": start_time_str,
        "sessionPhase": session_phase,
    }
    if end_time_str:
      target_session["endTime"] = end_time_str

    payload = {
        "profilerTargets": [profiler_target],
        "targetSessions": {
            profiler_target: target_session,
        },
        "storageFolderUri": gsc_file_path,
        "dashboardUri": "",
        "duration": duration_str,
        "kind": "KIND_PROGRAMMATIC",
        "hostTracerLevel": "HOST_TRACER_LEVEL_INFO",
        "deviceTracerLevel": "DEVICE_TRACER_LEVEL_ENABLED",
        "pythonTracerLevel": "PYTHON_TRACER_LEVEL_DISABLED",
    }
    params = {"update_mask": "target_sessions"}
    logger.debug(
        "Update Profiler Session request: url=%s, params=%s, json=%s",
        url,
        pprint.pformat(params),
        pprint.pformat(payload),
    )
    retry_count = 0
    while retry_count < _MAX_RETRIES:
      retry_count += 1
      with requests.patch(
          url,
          headers=self._get_headers(),
          params=params,
          json=payload,
      ) as response:
        try:
          response.raise_for_status()
        except requests.exceptions.HTTPError:
          logger.error(
              "Try %s: Update Profiler Session request failed: status_code=%s,"
              " content=%s",
              retry_count,
              response.status_code,
              response.text,
          )
          # This is possible when multiple host try to update the session
          # when its getting created.
          if response.status_code == 409:
            time.sleep(0.5)
            continue

          raise
        json_response = response.json()

      logger.debug(
          "Update Profiler Session response: %s",
          pprint.pformat(json_response),
      )
      if not json_response.get("done"):
        operation = self._wait_for_operation(json_response["name"])
      else:
        operation = json_response

      return operation

    raise RuntimeError(
        f"update_profiler_session failed to return a response and max retries"
        f" %s reached",
        _MAX_RETRIES,
    )

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
      if response.status_code == 404:
        logger.warning("ML run '%s' not found.", name)
      else:
        logger.error(
            "Get ML Run request failed: status_code=%s, content=%s",
            response.status_code,
            response.text,
        )
      raise
    json_response = response.json()
    if logger.isEnabledFor(logging.DEBUG):
      logger.debug("Get ML Run response: %s", pprint.pformat(json_response))
    return json_response

  def update_ml_run(
      self,
      name: str,
      force: bool = False,
      run_phase: Optional[str] = None,
      *,
      display_name: Optional[str] = None,
      tools: Optional[List[Dict[str, Any]]] = None,
      artifacts: Optional[Dict[str, str]] = None,
  ) -> Dict[str, Any]:
    """Update an existing ML run.

    This method updates the ML run by sending the full resource to the Google
    Cloud API. It retries on HTTP errors.

    Args:
        name: Name of the run to update
        force: If True, forces an update even if no fields have changed.
        run_phase: Phase of the run (ACTIVE, COMPLETE, FAILED)
        display_name: Optional new display name for the run
        tools: Optional new list of tools to enable (e.g., XProf, NSys)
        artifacts: Optional new artifacts configuration (e.g., gcsPath)

    Returns:
        Response from the API as a dictionary

    Raises:
        requests.exceptions.RequestException: If the HTTP request fails
        RuntimeError: If the update fails to return a response or raise an error
    """
    for attempt in range(_MAX_RETRIES):
      try:
        return self._attempt_update_ml_run(
            name,
            force,
            run_phase,
            display_name=display_name,
            tools=tools,
            artifacts=artifacts,
        )
      except requests.exceptions.HTTPError as e:
        logger.warning(
            "Update for ML run '%s' (phase: %s) failed. "
            "(Attempt %s/%s). Error: %s",
            name,
            run_phase,
            attempt + 1,
            _MAX_RETRIES,
            e,
        )
        if attempt == _MAX_RETRIES - 1:
          raise
        time.sleep(0.2)

    raise RuntimeError(
        "update_ml_run failed to return a response or raise an error"
    )

  def _attempt_update_ml_run(
      self,
      name: str,
      force: bool = False,
      run_phase: Optional[str] = None,
      *,
      display_name: Optional[str] = None,
      tools: Optional[List[Dict[str, Any]]] = None,
      artifacts: Optional[Dict[str, str]] = None,
  ) -> Dict[str, Any]:
    """Attempt to update an existing ML run once."""
    payload = self.get_ml_run(name)
    need_update = force

    if display_name is not None and payload.get("displayName") != display_name:
      payload["displayName"] = display_name
      need_update = True

    if run_phase is not None and payload.get("runPhase") != run_phase:
      payload["runPhase"] = run_phase
      need_update = True

    if tools is not None:
      existing_tools = payload.get("tools", [])
      if {"xprof": {}} in tools and {"xprof": {}} not in existing_tools:
        payload["tools"] = existing_tools + [{"xprof": {}}]
        need_update = True

    if artifacts is not None and payload.get("artifacts") != artifacts:
      payload["artifacts"] = artifacts
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
      logger.error(
          "Update ML Run request failed: status_code=%s, content=%s",
          response.status_code,
          response.text,
      )
      raise
    json_response = response.json()
    if logger.isEnabledFor(logging.DEBUG):
      logger.debug("Update ML Run response: %s", pprint.pformat(json_response))

    # If it's a resource (no "done" field), return it directly
    if "done" not in json_response:
      return json_response

    operation = (
        self._wait_for_operation(json_response["name"])
        if not json_response.get("done")
        else json_response
    )

    if operation.get("error"):
      err = operation["error"]
      raise requests.exceptions.HTTPError(f"Operation failed: {err}")

    return operation.get("response", operation)

  def create_profiler_target(
      self,
      *,
      ml_run_name: str,
      name: str,
      is_master: bool,
      hostname: str,
      node_index: int,
  ) -> None:
    """Create a profiler target for the ML run.

    Args:
        ml_run_name: The name of the ML run.
        name: Name of the profiler target
        is_master: Whether the target is the master host
        hostname: Hostname of the target
        node_index: Index of the node in the cluster

    Raises:
        requests.exceptions.RequestException: If the HTTP request fails
    """
    profiler_target_url = f"{self.ml_runs_path}/{ml_run_name}/profilerTargets"
    params = {"profiler_target_id": name}
    payload = {
        "name": name,
        "isMaster": is_master,
        "hostname": hostname,
        "nodeIndex": node_index,
    }

    logger.debug(
        "Create a profiler target: url=%s, params=%r, payload=%r",
        profiler_target_url,
        params,
        payload,
    )
    response = requests.post(
        profiler_target_url,
        headers=self._get_headers(),
        params=params,
        json=payload,
    )

    try:
      response.raise_for_status()
    except requests.exceptions.HTTPError:
      if response.status_code == 409:
        logger.warning(
            "Profiler target '%s/%s' already exists, ignoring the "
            "create request.",
            profiler_target_url,
            name,
        )
        return
      else:
        logger.error(
            "Create profiler target request failed: status_code=%s, content=%s",
            response.status_code,
            response.text,
        )
        raise
    json_response = response.json()
    if logger.isEnabledFor(logging.DEBUG):
      logger.debug(
          "Create profiler target response: %s", pprint.pformat(json_response)
      )

    if not json_response.get("done"):
      operation = self._wait_for_operation(json_response["name"])
    else:
      operation = json_response

    if logger.isEnabledFor(logging.INFO):
      logger.info(
          "Create profiler target operation: %s", pprint.pformat(operation)
      )

    if operation.get("error"):
      raise requests.exceptions.HTTPError(
          f"Operation {operation['name']!r} failed: {operation['error']!r}"
      )
