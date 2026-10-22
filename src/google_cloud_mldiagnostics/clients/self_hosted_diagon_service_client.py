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

import datetime
import hashlib
import logging
import threading
from typing import Any
import urllib.parse
import uuid
from google_cloud_mldiagnostics.custom_types import mlrun_types
import requests

logger = logging.getLogger(__name__)

_RUN_PHASE_TO_WORKLOAD_STATE = {
    mlrun_types.RunPhase.PHASE_ACTIVE.value: "STATE_RUNNING",
    mlrun_types.RunPhase.PHASE_COMPLETED.value: "STATE_SUCCEEDED",
    mlrun_types.RunPhase.PHASE_FAILED.value: "STATE_FAILED",
}

_SESSION_PHASE_TO_STATES = {
    "ACTIVE": ("STATE_ACTIVE", "STATE_RUNNING"),
    "SUCCEEDED": ("STATE_SUCCEEDED", "STATE_SUCCEEDED"),
    "FAILED": ("STATE_FAILED", "STATE_FAILED"),
}


def _normalize_run_id(ml_run_id: str) -> str:
  """Strips the resource prefix and whitespace from an ML run ID."""
  return ml_run_id.removeprefix("machineLearningRuns/").strip()


def _format_timestamp(timestamp: float) -> str:
  """Formats a POSIX timestamp (seconds) as an RFC 3339 UTC string."""
  return datetime.datetime.fromtimestamp(
      timestamp, datetime.timezone.utc
  ).strftime("%Y-%m-%dT%H:%M:%SZ")


class SelfHostedDiagonServiceClient:
  """REST client for synchronizing MLRun metadata with a self-hosted control plane.

  This client allows customers to bring their own backend to receive MLRun and
  profiler session metadata.

  API Contract:
    - MLRuns Endpoint: Expose `PATCH /v1/machineLearningRuns/{run_id}`.
      Creations are processed as upserts via `allow_missing=true` and updates
      via the `update_mask` query parameter.
    - ProfilerSessions Endpoints: Expose
      `POST /v1/machineLearningRuns/{run_id}/profilerSessions` for creation and
      `PATCH /v1/machineLearningRuns/{run_id}/profilerSessions/{session_id}`
      with `update_mask` for lifecycle updates.
  """

  def __init__(self, control_plane_url: str, cluster_id: str | None = None):
    self.control_plane_url = control_plane_url.rstrip("/")
    parsed_url = urllib.parse.urlsplit(self.control_plane_url)
    if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
      raise ValueError(
          f"Invalid control_plane_url {control_plane_url!r}: must include"
          " http:// or https:// scheme and host."
      )
    self.cluster_id = cluster_id or ""
    self.session = requests.Session()
    self._session_id_map: dict[str, str] = {}
    self._session_id_lock = threading.Lock()

  def _resolve_session_id(self, run_id: str, session_id: str) -> str:
    """Resolves a profiler session ID to its server UUID."""
    cache_key = f"{run_id}/{session_id}"
    with self._session_id_lock:
      cached = self._session_id_map.get(cache_key)
    if cached:
      return cached
    try:
      return str(uuid.UUID(session_id))
    except ValueError:
      pass
    data = cache_key.encode("utf-8")
    digest = hashlib.sha256(data).digest()
    byte_arr = bytearray(digest[:16])
    byte_arr[6] = (byte_arr[6] & 0x0F) | 0x50
    byte_arr[8] = (byte_arr[8] & 0x3F) | 0x80
    resolved = str(uuid.UUID(bytes=bytes(byte_arr)))
    with self._session_id_lock:
      self._session_id_map[cache_key] = resolved
    return resolved

  def _send_request(
      self,
      method: str,
      url: str,
      params: dict[str, str],
      body: dict[str, Any],
  ) -> dict[str, Any] | None:
    """Sends an HTTP request and handles transport/JSON errors."""
    request_fn = self.session.post if method == "POST" else self.session.patch
    try:
      response = request_fn(url, params=params, json=body, timeout=5)
      response.raise_for_status()
      return response.json()
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

  def _patch(
      self,
      run_id: str,
      body: dict[str, Any],
      update_mask: str,
      allow_missing: bool,
  ) -> dict[str, Any] | None:
    encoded_run_id = urllib.parse.quote(run_id, safe="") if run_id else ""
    base_url = f"{self.control_plane_url}/v1/machineLearningRuns"
    url = f"{base_url}/{encoded_run_id}" if encoded_run_id else base_url
    params = {
        "update_mask": update_mask,
        "allow_missing": str(allow_missing).lower(),
    }
    return self._send_request("PATCH", url, params, body)

  def create_ml_run(
      self,
      name: str,
      display_name: str,
      run_phase: mlrun_types.RunPhase | str,
      configs: dict[str, Any] | None = None,
      tools: list[dict[str, Any]] | None = None,
      artifacts: dict[str, str] | None = None,
      run_group: str | None = None,
      labels: dict[str, str] | None = None,
      orchestrator: str | None = None,
      workload_details: dict[str, Any] | None = None,
      workload_targets: list[dict[str, Any]] | None = None,
  ) -> dict[str, Any] | None:
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

    body: dict[str, Any] = {
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
      display_name: str | None = None,
      tools: list[dict[str, Any]] | None = None,
      artifacts: dict[str, str] | None = None,
      labels: dict[str, str] | None = None,
      configs: dict[str, Any] | None = None,
      workload_targets: list[dict[str, Any]] | None = None,
      update_mask: str = "*",
  ) -> dict[str, Any] | None:
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

    run_id = _normalize_run_id(name)
    if not run_id:
      logger.warning("Cannot update MLRun with empty run ID.")
      return None
    resource_name = f"machineLearningRuns/{run_id}"
    body: dict[str, Any] = {"name": resource_name, "workload": {"state": state}}
    return self._patch(
        run_id=run_id,
        body=body,
        update_mask="workload.state",
        allow_missing=False,
    )

  def create_profiler_session(
      self,
      ml_run_id: str,
      profiler_session_id: str,
      gsc_file_path: str,
      profiler_target: str,
      start_time: float,
      end_time: float | None,
      session_phase: str,
  ) -> dict[str, Any] | None:
    """Creates a programmatic ProfilerSession on the self-hosted control plane.

    Args:
      ml_run_id: Resource name or UUID of the parent MLRun.
      profiler_session_id: Client-generated or user-specified session ID.
      gsc_file_path: GCS path where profiler trace artifacts are stored.
      profiler_target: Identifier of the workload target being profiled.
      start_time: Session start timestamp in POSIX seconds.
      end_time: Session end timestamp in POSIX seconds, or None if active.
      session_phase: Lifecycle phase ("ACTIVE", "SUCCEEDED", or "FAILED").

    Returns:
      The parsed JSON response dict from the control plane, or None if inputs
      are invalid or the control plane is unreachable.

    Raises:
      requests.HTTPError: If the control plane returns a non-409 HTTP error.
    """
    run_id = _normalize_run_id(ml_run_id)
    if not run_id:
      logger.warning("Cannot create profiler session with empty run ID.")
      return None

    session_id = profiler_session_id.strip()
    if not session_id:
      logger.warning("Cannot create profiler session with empty session ID.")
      return None

    encoded_run_id = urllib.parse.quote(run_id, safe="")
    url = (
        f"{self.control_plane_url}/v1/machineLearningRuns/{encoded_run_id}"
        "/profilerSessions"
    )
    params = {"profiler_session_id": session_id}

    if end_time is not None:
      duration_sec = end_time - start_time
      duration_str = f"{max(0.001, duration_sec):.3f}s"
    else:
      duration_str = "0.001s"

    body: dict[str, Any] = {
        "profilePath": gsc_file_path,
        "duration": duration_str,
        "kind": "KIND_PROGRAMMATIC",
        "isTraceEnabled": True,
        "hostTracerLevel": "HOST_TRACER_LEVEL_INFO",
        "deviceTracerLevel": "DEVICE_TRACER_LEVEL_ENABLED",
        "pythonTracerLevel": "PYTHON_TRACER_LEVEL_DISABLED",
        "profilerTargets": (
            [{"workloadTargetId": profiler_target}] if profiler_target else []
        ),
    }

    try:
      result = self._send_request("POST", url, params, body)
    except requests.HTTPError as e:
      if e.response is not None and e.response.status_code == 409:
        logger.info(
            "Profiler session %r already exists, updating it.",
            session_id,
        )
        return self.update_profiler_session(
            ml_run_id=ml_run_id,
            profiler_session_id=session_id,
            gsc_file_path=gsc_file_path,
            profiler_target=profiler_target,
            start_time=start_time,
            end_time=end_time,
            session_phase=session_phase,
        )
      raise

    if result is None:
      return None
    if isinstance(result, dict):
      resp_name = result.get("name", "")
      if isinstance(resp_name, str) and "/profilerSessions/" in resp_name:
        resolved_id = resp_name.rsplit("/profilerSessions/", 1)[-1].strip()
        if resolved_id:
          with self._session_id_lock:
            self._session_id_map[f"{run_id}/{session_id}"] = resolved_id
    if session_phase != "ACTIVE":
      return self.update_profiler_session(
          ml_run_id=ml_run_id,
          profiler_session_id=session_id,
          gsc_file_path=gsc_file_path,
          profiler_target=profiler_target,
          start_time=start_time,
          end_time=end_time,
          session_phase=session_phase,
      )
    return result

  def update_profiler_session(
      self,
      ml_run_id: str,
      profiler_session_id: str,
      gsc_file_path: str,
      profiler_target: str,
      start_time: float,
      end_time: float | None,
      session_phase: str,
  ) -> dict[str, Any] | None:
    """Updates an existing ProfilerSession on the self-hosted control plane.

    Args:
      ml_run_id: Resource name or UUID of the parent MLRun.
      profiler_session_id: Client-generated or user-specified session ID.
      gsc_file_path: GCS path where profiler trace artifacts are stored.
      profiler_target: Identifier of the workload target being profiled.
      start_time: Session start timestamp in POSIX seconds.
      end_time: Session end timestamp in POSIX seconds, or None if active.
      session_phase: Lifecycle phase ("ACTIVE", "SUCCEEDED", or "FAILED").

    Returns:
      The parsed JSON response dict from the control plane, or None if inputs
      are invalid or the control plane is unreachable.

    Raises:
      requests.HTTPError: If the control plane returns an HTTP error status.
    """
    run_id = _normalize_run_id(ml_run_id)
    if not run_id:
      logger.warning("Cannot update profiler session with empty run ID.")
      return None

    session_id = profiler_session_id.strip()
    if not session_id:
      logger.warning("Cannot update profiler session with empty session ID.")
      return None
    resolved_session_id = self._resolve_session_id(run_id, session_id)

    states = _SESSION_PHASE_TO_STATES.get(session_phase)
    if states is None:
      logger.warning(
          "Unmapped profiler session phase %r; skipping update.",
          session_phase,
      )
      return None
    session_state, target_state = states

    start_time_str = _format_timestamp(start_time)

    resource_name = (
        f"machineLearningRuns/{run_id}/profilerSessions/{resolved_session_id}"
    )
    body: dict[str, Any] = {
        "name": resource_name,
        "state": session_state,
        "startTime": start_time_str,
        "profilePath": gsc_file_path,
    }
    update_mask_fields = [
        "state",
        "start_time",
        "profile_path",
    ]

    end_time_str: str | None = None
    if end_time is not None:
      end_time_str = _format_timestamp(end_time)
      body["endTime"] = end_time_str
      update_mask_fields.append("end_time")

    if profiler_target:
      target_session: dict[str, Any] = {
          "state": target_state,
          "startTime": start_time_str,
      }
      if end_time_str is not None:
        target_session["endTime"] = end_time_str
      body["targetSessions"] = {profiler_target: target_session}
      update_mask_fields.append("target_sessions")

    encoded_run_id = urllib.parse.quote(run_id, safe="")
    encoded_session_id = urllib.parse.quote(resolved_session_id, safe="")
    url = (
        f"{self.control_plane_url}/v1/machineLearningRuns/{encoded_run_id}"
        f"/profilerSessions/{encoded_session_id}"
    )
    params = {"update_mask": ",".join(update_mask_fields)}

    return self._send_request("PATCH", url, params, body)
