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

"""HTTP client for the Self-Hosted Diagon Control Plane."""

import logging
from typing import Any, Dict, List, Optional
import urllib.parse
from google_cloud_mldiagnostics.custom_types import mlrun_types
import requests

logger = logging.getLogger(__name__)

_RUN_PHASE_TO_WORKLOAD_STATE = {
    mlrun_types.RunPhase.PHASE_ACTIVE.value: "STATE_RUNNING",
    mlrun_types.RunPhase.PHASE_COMPLETED.value: "STATE_SUCCEEDED",
    mlrun_types.RunPhase.PHASE_FAILED.value: "STATE_FAILED",
}


class SelfHostedDiagonServiceClient:
  """REST client for synchronizing MLRun metadata with a self-hosted control plane.

  This client allows customers to bring their own backend to receive MLRun
  metadata.

  API Contract:
    - Endpoint: Expose `PATCH /v1/machineLearningRuns/{run_id}`.
    - Creations: Processed as upserts via the `allow_missing=true` query
    parameter.
    - Updates: Processed as partial patches via the `update_mask` query
    parameter.
  """

  def __init__(self, control_plane_url: str, cluster_id: Optional[str] = None):
    self.control_plane_url = control_plane_url.rstrip("/")
    parsed_url = urllib.parse.urlsplit(self.control_plane_url)
    if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
      raise ValueError(
          f"Invalid control_plane_url {control_plane_url!r}: must include"
          " http:// or https:// scheme and host."
      )
    self.cluster_id = cluster_id or ""
    self.session = requests.Session()

  def _patch(
      self,
      run_id: str,
      body: Dict[str, Any],
      update_mask: str,
      allow_missing: bool,
  ) -> Optional[Dict[str, Any]]:
    encoded_run_id = urllib.parse.quote(run_id, safe="") if run_id else ""
    base_url = f"{self.control_plane_url}/v1/machineLearningRuns"
    url = f"{base_url}/{encoded_run_id}" if encoded_run_id else base_url
    params = {
        "update_mask": update_mask,
        "allow_missing": str(allow_missing).lower(),
    }

    try:
      response = self.session.patch(url, params=params, json=body, timeout=5)
      response.raise_for_status()
      result = response.json()
      return result
    except requests.HTTPError as e:
      status_code = (
          e.response.status_code if e.response is not None else "unknown"
      )
      resp_text = e.response.text if e.response is not None else ""
      logger.warning(
          "Control plane returned HTTP %s at %s: %s",
          status_code,
          url,
          resp_text,
      )
      raise
    except (requests.RequestException, ValueError) as e:
      logger.warning(
          "Control plane unreachable or invalid JSON at %s: %s. Continuing"
          " without control plane.",
          url,
          e,
      )
      return None

  def create_ml_run(
      self,
      name: str,
      display_name: str,
      run_phase: mlrun_types.RunPhase | str,
      configs: Optional[Dict[str, Any]] = None,
      tools: Optional[List[Dict[str, Any]]] = None,
      artifacts: Optional[Dict[str, str]] = None,
      run_group: Optional[str] = None,
      labels: Optional[Dict[str, str]] = None,
      orchestrator: Optional[str] = None,
      workload_details: Optional[Dict[str, Any]] = None,
      workload_targets: Optional[List[Dict[str, Any]]] = None,
  ) -> Optional[Dict[str, Any]]:
    _ = (
        configs,
        tools,
        run_group,
        labels,
        orchestrator,
        workload_targets,
        run_phase,
    )
    workload_details = workload_details or {}
    workload_id = workload_details.get("id", "")
    if not workload_id:
      raise ValueError(
          "workload_details must contain a valid 'id' to create an ML Run."
      )

    body: Dict[str, Any] = {
        "logicalId": name,
        "displayName": display_name,
        "creator": "CREATOR_MANUAL",
        "clusterId": self.cluster_id,
        "workload": {
            "workloadId": workload_id,
            "displayName": display_name,
        },
    }

    update_mask_fields = [
        "display_name",
        "creator",
        "cluster_id",
        "workload.workload_id",
        "workload.display_name",
        "logical_id",
    ]

    if artifacts and artifacts.get("gcsPath"):
      body["baseBlobPath"] = artifacts["gcsPath"]
      update_mask_fields.append("base_blob_path")

    created_update_mask = ",".join(update_mask_fields)

    return self._patch(
        run_id="",
        body=body,
        update_mask=created_update_mask,
        allow_missing=True,
    )

  def update_ml_run(
      self,
      name: str,
      force: bool = False,
      run_phase: mlrun_types.RunPhase | str | None = None,
      *,
      display_name: Optional[str] = None,
      tools: Optional[List[Dict[str, Any]]] = None,
      artifacts: Optional[Dict[str, str]] = None,
      labels: Optional[Dict[str, str]] = None,
      configs: Optional[Dict[str, Any]] = None,
      workload_targets: Optional[List[Dict[str, Any]]] = None,
      update_mask: str = "*",
  ) -> Optional[Dict[str, Any]]:
    _ = (
        force,
        display_name,
        tools,
        artifacts,
        labels,
        configs,
        workload_targets,
        update_mask,
    )
    if not run_phase:
      return None

    run_phase_str = (
        str(run_phase.value) if hasattr(run_phase, "value") else str(run_phase)
    )
    state = _RUN_PHASE_TO_WORKLOAD_STATE.get(run_phase_str)
    if state is None:
      logger.warning("Unmapped run phase %r; skipping update.", run_phase)
      return None

    run_id = name.removeprefix("machineLearningRuns/").strip()
    if not run_id:
      logger.warning("Cannot update MLRun with empty run ID.")
      return None
    resource_name = f"machineLearningRuns/{run_id}"
    body: Dict[str, Any] = {"name": resource_name, "workload": {"state": state}}
    return self._patch(
        run_id=run_id,
        body=body,
        update_mask="workload.state",
        allow_missing=False,
    )
