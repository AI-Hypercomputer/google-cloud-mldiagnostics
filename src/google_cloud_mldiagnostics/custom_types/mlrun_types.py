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

"""Module for registering and managing ML runs."""

import dataclasses
import enum
from typing import Any


class RunPhase(enum.Enum):
  """Enumeration of possible run phases for machine learning runs."""

  # Phase is not specified.
  PHASE_UNSPECIFIED = "RUN_PHASE_UNSPECIFIED"
  # Run is active.
  PHASE_ACTIVE = "ACTIVE"
  # Run is completed.
  PHASE_COMPLETED = "COMPLETED"
  # Run is failed.
  PHASE_FAILED = "FAILED"


class ConfigDict(dict):
  """A dictionary that supports both dict-style and attribute-style access."""

  def __setattr__(self, name: str, value: Any) -> None:
    """Allow setting values using dot notation."""
    self[name] = value

  def __getattr__(self, name: str) -> Any:
    """Allow getting values using dot notation."""
    try:
      return self[name]
    except KeyError as exc:
      raise AttributeError(
          f"'{self.__class__.__name__}' object has no attribute '{name}'"
      ) from exc

  def update(self, *args, **kwargs):
    """Update the dictionary and maintain attribute access."""
    super().update(*args, **kwargs)
    # Ensure new keys are accessible as attributes
    for key in self.keys():
      if not hasattr(self, key):
        setattr(self, key, self[key])
    self.control_plane_client.update_ml_run(
        self
    )  # TODO([INTERNAL]): Add support for updating ML run after client ready.

  def __delattr__(self, name: str) -> None:
    """Allow deleting values using dot notation."""
    try:
      del self[name]
    except KeyError as exc:
      raise AttributeError(
          f"'{self.__class__.__name__}' object has no attribute '{name}'"
      ) from exc


@dataclasses.dataclass
class MLRun:
  """Represents a machine learning run with configurations and metadata."""

  # Required fields
  name: str

  # fields with defaults if not provided by users
  run_group: str = ""
  configs: dict[str, Any] | None = None
  gcs_path: str | None = None

  # Fields with default values before GKE integration
  location: str = "us-central1"
  project: str = "supercomputer-testing"

  # Fields assgined/updated by SDK
  run_phase: RunPhase = RunPhase.PHASE_ACTIVE

  # Fields computed during initialization (excluded from __init__)
  created_at: str | None = None
  workload_details: dict[str, Any] | None = None
  orchestrator: str | None = None
  display_name: str = ""
  on_demand_xprof: bool = False
  environment: str = ""
